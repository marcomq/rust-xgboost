use libc::{c_float, c_uint};
use std::ffi::CStr;
use std::{ffi, path::Path, ptr, slice};

use super::{XGBError, XGBResult};

static KEY_GROUP_PTR: &CStr = c"group_ptr";
static KEY_GROUP: &CStr = c"group";
static KEY_LABEL: &CStr = c"label";
static KEY_WEIGHT: &CStr = c"weight";
static KEY_BASE_MARGIN: &CStr = c"base_margin";

/// Creates a JSON-encoded array interface string for f32 data.
pub(crate) fn make_array_interface_f32(data: &[f32]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<f4","version":3}}"#,
        ptr, len
    )
}

/// Creates a JSON-encoded array interface string for u64 data.
///
/// This function is used for CSR/CSC indices and indptr arrays.
/// XGBoost's C API expects uint64_t (bst_ulong) for these arrays,
/// so we use fixed-width u64 to ensure cross-platform compatibility.
pub(crate) fn make_array_interface_u64(data: &[u64]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<u8","version":3}}"#,
        ptr, len
    )
}

/// Reject missing-value sentinels the JSON configs cannot represent.
///
/// Applied uniformly across the constructors (including the ones that pass the
/// float directly) so `missing` has one contract everywhere: finite or NaN.
pub(crate) fn validate_missing(missing: f32) -> XGBResult<()> {
    if missing.is_nan() || missing.is_finite() {
        Ok(())
    } else {
        Err(XGBError::new(format!(
            "missing value must be finite or NaN, got {missing}"
        )))
    }
}

/// Render a missing value as a literal XGBoost's JSON config parser accepts
/// (`NaN` for NaN — the same spelling the static configs use).
pub(crate) fn missing_json(missing: f32) -> XGBResult<String> {
    validate_missing(missing)?;
    Ok(if missing.is_nan() { "NaN".to_owned() } else { missing.to_string() })
}

/// Build the `XGDMatrixCreateFromCSR`/`CSC` ingestion config.
fn build_ingest_config(missing: f32, single_thread: bool) -> XGBResult<ffi::CString> {
    let nthread = if single_thread { r#", "nthread": 1"# } else { "" };
    let config = format!(r#"{{"missing": {}{}}}"#, missing_json(missing)?, nthread);
    Ok(ffi::CString::new(config).unwrap())
}

/// Creates a JSON-encoded array interface string for u32 data.
fn make_array_interface_u32(data: &[u32]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<u4","version":3}}"#,
        ptr, len
    )
}

/// Payload for a single batch pushed into a proxy DMatrix: either a dense array
/// interface, or the three CSR array interfaces plus the column count.
enum BatchData {
    Dense(ffi::CString),
    Csr {
        indptr: ffi::CString,
        indices: ffi::CString,
        values: ffi::CString,
        num_cols: usize,
    },
}

/// Single-batch data iterator used to build a `QuantileDMatrix` from an
/// in-memory matrix via XGBoost's callback API. It yields the whole matrix as a
/// single batch, then signals end-of-iteration.
struct BatchIter {
    /// Batch payload (array interfaces referencing caller data).
    data: BatchData,
    /// Optional labels as an array interface; the referenced data must outlive
    /// construction. Prebuilt so the FFI callback below stays panic-free.
    labels: Option<ffi::CString>,
    /// Proxy DMatrix the callbacks push data into.
    proxy: xgboost_sys::DMatrixHandle,
    /// Whether the single batch has already been yielded.
    yielded: bool,
}

/// `next` callback: push the one batch into the proxy, or report end-of-iteration.
/// Returns 1 while a batch is available, 0 at the end, -1 on failure. Must not
/// unwind across the FFI boundary, so it avoids any panicking operations.
unsafe extern "C" fn batch_iter_next(handle: xgboost_sys::DataIterHandle) -> i32 {
    let it = unsafe { &mut *(handle as *mut BatchIter) };
    if it.yielded {
        return 0;
    }
    let ret = match &it.data {
        BatchData::Dense(interface) => unsafe { xgboost_sys::XGProxyDMatrixSetDataDense(it.proxy, interface.as_ptr()) },
        BatchData::Csr {
            indptr,
            indices,
            values,
            num_cols,
        } => unsafe {
            xgboost_sys::XGProxyDMatrixSetDataCSR(
                it.proxy,
                indptr.as_ptr(),
                indices.as_ptr(),
                values.as_ptr(),
                *num_cols as xgboost_sys::bst_ulong,
            )
        },
    };
    if ret != 0 {
        return -1;
    }
    if let Some(interface) = &it.labels {
        let ret =
            unsafe { xgboost_sys::XGDMatrixSetInfoFromInterface(it.proxy, c"label".as_ptr(), interface.as_ptr()) };
        if ret != 0 {
            return -1;
        }
    }
    it.yielded = true;
    1
}

/// `reset` callback: rewind so the batch can be yielded again.
unsafe extern "C" fn batch_iter_reset(handle: xgboost_sys::DataIterHandle) {
    let it = unsafe { &mut *(handle as *mut BatchIter) };
    it.yielded = false;
}

/// Data matrix used throughout XGBoost for training/predicting [`Booster`](struct.Booster.html) models.
///
/// It's used as a container for both features (i.e. a row for every instance), and an optional true label for that
/// instance (as an `f32` value).
///
/// Can be created files, or from dense or sparse
/// ([CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
/// or [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrices.
///
/// # Examples
///
/// ## Load from file
///
/// Load matrix from file in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) or binary format.
///
/// ```should_panic
/// use xgb::DMatrix;
///
/// let dmat = DMatrix::load(r#"{"uri": "somefile.txt?format=csv"}"#).unwrap();
/// ```
///
/// ## Create from dense array
///
/// ```
/// use xgb::DMatrix;
///
/// let data = &[1.0, 0.5, 0.2, 0.2,
///              0.7, 1.0, 0.1, 0.1,
///              0.2, 0.0, 0.0, 1.0];
/// let num_rows = 3;
/// let mut dmat = DMatrix::from_dense(data, num_rows).unwrap();
/// assert_eq!(dmat.shape(), (3, 4));
///
/// // set true labels for each row
/// dmat.set_labels(&[1.0, 0.0, 1.0]);
/// ```
///
/// ## Create from sparse CSR matrix
///
/// Create from sparse representation of
/// ```text
/// [[1.0, 0.0, 2.0],
///  [0.0, 0.0, 3.0],
///  [4.0, 5.0, 6.0]]
/// ```
///
/// ```
/// use xgb::DMatrix;
///
/// let indptr: &[u64] = &[0, 1, 2, 6];
/// let indices: &[u64] = &[0, 2, 2, 0, 1, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dmat = DMatrix::from_csc(indptr, indices, data, None).unwrap();
/// assert_eq!(dmat.shape(), (3, 3));
/// ```
#[derive(Debug)]
pub struct DMatrix {
    pub(super) handle: xgboost_sys::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

// SAFETY: a `DMatrixHandle` has no thread affinity. As with `Booster` (see the
// safety comment there), the C API keeps per-call scratch buffers in
// thread-local storage keyed by the DMatrix pointer, so using or freeing the
// handle from a different thread than the one that created it is fine. Kept
// `!Sync`: XGBoost does not document the DMatrix getters/setters as safe for
// concurrent use on one handle.
unsafe impl Send for DMatrix {}

impl DMatrix {
    /// Construct a new instance from a DMatrixHandle created by the XGBoost C API.
    fn new(handle: xgboost_sys::DMatrixHandle) -> XGBResult<Self> {
        // number of rows/cols are frequently read throughout applications, so more convenient to pull them out once
        // when the matrix is created, instead of having to check errors each time XGDMatrixNum* is called
        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumRow(handle, &mut out))?;
        let num_rows = out as usize;

        let mut out = 0;
        xgb_call!(xgboost_sys::XGDMatrixNumCol(handle, &mut out))?;
        let num_cols = out as usize;

        trace!("Loaded DMatrix with shape: {}x{}", num_rows, num_cols);
        Ok(DMatrix {
            handle,
            num_rows,
            num_cols,
        })
    }

    /// Create a new `DMatrix` from dense array in row-major order.
    ///
    /// E.g. the matrix
    /// ```text
    /// [[1.0, 2.0],
    ///  [3.0, 4.0],
    ///  [5.0, 6.0]]
    /// ```
    /// would be represented converted into a `DMatrix` with
    /// ```
    /// use xgb::DMatrix;
    ///
    /// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let num_rows = 3;
    /// let dmat = DMatrix::from_dense(data, num_rows).unwrap();
    /// ```
    pub fn from_dense(data: &[f32], num_rows: usize) -> XGBResult<Self> {
        Self::from_dense_with_missing(data, num_rows, f32::NAN)
    }

    /// Like [`from_dense`](Self::from_dense), but treats `missing` as the
    /// missing-value sentinel instead of NaN.
    ///
    /// For data that encodes missing as e.g. `0.0` or `-999.0`, this lets
    /// XGBoost drop those entries during its own ingestion pass instead of the
    /// caller rewriting the whole array to NaN first (a full O(n) copy).
    /// `missing` must be finite or NaN.
    pub fn from_dense_with_missing(data: &[f32], num_rows: usize, missing: f32) -> XGBResult<Self> {
        // Three-way benchmark of the dense creation APIs
        // (benches/dmatrix_benchmark.rs, `from_dense` group):
        //   - XGDMatrixCreateFromMat: fastest below ~50k elements (no thread setup)
        //   - XGDMatrixCreateFromMat_omp: fastest above (~1.5x the array-interface
        //     XGDMatrixCreateFromDense at 500k+, ~5x plain CreateFromMat)
        // Neither is deprecated, so dispatch on the measured crossover.
        const PARALLEL_THRESHOLD: usize = 50_000;

        // CreateFromMat takes the float directly, so no JSON rendering is
        // involved — but reject infinities anyway to keep the missing-value
        // contract identical across the dense/CSR/CSC constructors.
        validate_missing(missing)?;

        // Guard the shape inference: without this, num_rows = 0 dies on a bare
        // integer division panic, and a non-divisible length silently drops the
        // trailing values and builds a wrong-shaped matrix.
        if num_rows == 0 || !data.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "data length {} does not divide into num_rows {}",
                data.len(),
                num_rows
            )));
        }

        let mut handle = ptr::null_mut();
        let num_cols = (data.len() / num_rows) as xgboost_sys::bst_ulong;

        if data.len() < PARALLEL_THRESHOLD {
            xgb_call!(xgboost_sys::XGDMatrixCreateFromMat(
                data.as_ptr(),
                num_rows as xgboost_sys::bst_ulong,
                num_cols,
                missing,
                &mut handle
            ))?;
        } else {
            xgb_call!(xgboost_sys::XGDMatrixCreateFromMat_omp(
                data.as_ptr(),
                num_rows as xgboost_sys::bst_ulong,
                num_cols,
                missing,
                &mut handle,
                0 // <=0 means use all available cores
            ))?;
        }
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a dense row-major `f64` slice.
    ///
    /// Uses the array-interface entry point (`XGDMatrixCreateFromDense`) with a
    /// `<f8` typestr, so the f64→f32 narrowing happens inside XGBoost's own
    /// mandatory ingestion copy — callers holding f64 data avoid converting
    /// the whole array to f32 in Rust first (an extra full-array copy). NaN is
    /// the missing value, matching [`from_dense`](Self::from_dense).
    pub fn from_dense_f64(data: &[f64], num_rows: usize) -> XGBResult<Self> {
        // Same shape guard as `from_dense`.
        if num_rows == 0 || !data.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "data length {} does not divide into num_rows {}",
                data.len(),
                num_rows
            )));
        }
        let num_cols = data.len() / num_rows;
        let interface = format!(
            r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f8","version":3}}"#,
            data.as_ptr() as usize,
            num_rows,
            num_cols
        );
        let interface_cstr = ffi::CString::new(interface).unwrap();
        let config: &CStr = cr#"{"missing": NaN, "nthread": 0}"#;

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGDMatrixCreateFromDense(
            interface_cstr.as_ptr(),
            config.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a `QuantileDMatrix` from a dense row-major matrix.
    ///
    /// A `QuantileDMatrix` stores the data pre-binned for the `hist` tree method
    /// (the XGBoost default), using far less memory than a regular `DMatrix`
    /// (~1 byte per feature value instead of 4) and skipping a separate sketching
    /// pass during training. Prefer it for large in-memory training sets.
    ///
    /// `max_bin` must match the booster's `max_bin` training parameter (default
    /// 256), otherwise training errors out. Labels, when given, are attached
    /// during construction (the supported path for quantile matrices) and must
    /// have length `num_rows`.
    ///
    /// Note: a `QuantileDMatrix` is intended for training with `hist`; unlike a
    /// regular `DMatrix` it cannot be serialised with [`save`](Self::save).
    pub fn from_dense_quantile(data: &[f32], num_rows: usize, labels: Option<&[f32]>, max_bin: u32) -> XGBResult<Self> {
        if let Some(l) = labels {
            assert_eq!(l.len(), num_rows, "labels length must equal num_rows");
        }
        // Same shape guard as `from_dense`.
        if num_rows == 0 || !data.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "data length {} does not divide into num_rows {}",
                data.len(),
                num_rows
            )));
        }
        let num_cols = data.len() / num_rows;
        let ptr = data.as_ptr() as usize;
        let data_interface = format!(
            r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f4","version":3}}"#,
            ptr, num_rows, num_cols
        );
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        Self::quantile_from_batch(BatchData::Dense(data_cstr), labels, max_bin, None)
    }

    /// Like [`from_dense_quantile`](Self::from_dense_quantile), but shares the
    /// bin cuts of `ref_matrix` instead of sketching them from `data`.
    ///
    /// This is what Python's `QuantileDMatrix(..., ref=dtrain)` does: build the
    /// training matrix without a ref, then build validation/eval matrices
    /// against it so all of them quantize identically — independently sketched
    /// cuts make eval metrics subtly incomparable, and holding eval sets as
    /// full `DMatrix` instead costs ~4x the memory. `ref_matrix` should be the
    /// quantile matrix used for training, built with the same `max_bin`.
    pub fn from_dense_quantile_ref(
        data: &[f32],
        num_rows: usize,
        labels: Option<&[f32]>,
        max_bin: u32,
        ref_matrix: &DMatrix,
    ) -> XGBResult<Self> {
        if let Some(l) = labels {
            assert_eq!(l.len(), num_rows, "labels length must equal num_rows");
        }
        // Same shape guard as `from_dense`.
        if num_rows == 0 || !data.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "data length {} does not divide into num_rows {}",
                data.len(),
                num_rows
            )));
        }
        let num_cols = data.len() / num_rows;
        let data_interface = format!(
            r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f4","version":3}}"#,
            data.as_ptr() as usize,
            num_rows,
            num_cols
        );
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        Self::quantile_from_batch(BatchData::Dense(data_cstr), labels, max_bin, Some(ref_matrix))
    }

    /// Create a `QuantileDMatrix` from a sparse CSR matrix, the sparse
    /// counterpart of [`from_dense_quantile`](Self::from_dense_quantile).
    ///
    /// Same CSR layout as [`from_csr`](Self::from_csr); `num_cols` is required
    /// (the proxy DMatrix cannot infer it). See `from_dense_quantile` for the
    /// `max_bin`/labels semantics and usage caveats.
    pub fn from_csr_quantile(
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
        labels: Option<&[f32]>,
        max_bin: u32,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        if let Some(l) = labels {
            assert_eq!(l.len(), indptr.len() - 1, "labels length must equal num_rows");
        }
        let batch = BatchData::Csr {
            indptr: ffi::CString::new(make_array_interface_u64(indptr)).unwrap(),
            indices: ffi::CString::new(make_array_interface_u64(indices)).unwrap(),
            values: ffi::CString::new(make_array_interface_f32(data)).unwrap(),
            num_cols,
        };
        Self::quantile_from_batch(batch, labels, max_bin, None)
    }

    /// Like [`from_csr_quantile`](Self::from_csr_quantile), but shares the bin
    /// cuts of `ref_matrix`; see
    /// [`from_dense_quantile_ref`](Self::from_dense_quantile_ref) for when and
    /// why.
    pub fn from_csr_quantile_ref(
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
        labels: Option<&[f32]>,
        max_bin: u32,
        ref_matrix: &DMatrix,
    ) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        if let Some(l) = labels {
            assert_eq!(l.len(), indptr.len() - 1, "labels length must equal num_rows");
        }
        let batch = BatchData::Csr {
            indptr: ffi::CString::new(make_array_interface_u64(indptr)).unwrap(),
            indices: ffi::CString::new(make_array_interface_u64(indices)).unwrap(),
            values: ffi::CString::new(make_array_interface_f32(data)).unwrap(),
            num_cols,
        };
        Self::quantile_from_batch(batch, labels, max_bin, Some(ref_matrix))
    }

    /// Shared `QuantileDMatrix` construction from a single prepared batch,
    /// optionally aligned to the bin cuts of a reference matrix.
    fn quantile_from_batch(
        data: BatchData,
        labels: Option<&[f32]>,
        max_bin: u32,
        ref_matrix: Option<&DMatrix>,
    ) -> XGBResult<Self> {
        let mut proxy = ptr::null_mut();
        xgb_call!(xgboost_sys::XGProxyDMatrixCreate(&mut proxy))?;

        let mut iter = BatchIter {
            data,
            labels: labels.map(|l| ffi::CString::new(make_array_interface_f32(l)).unwrap()),
            proxy,
            yielded: false,
        };

        let config = format!(r#"{{"missing": NaN, "max_bin": {}}}"#, max_bin);
        let config_cstr = ffi::CString::new(config).unwrap();

        let mut out = ptr::null_mut();
        // SAFETY: `iter` outlives the (synchronous) call; the callbacks only ever
        // dereference the handle we pass and never unwind across the boundary.
        // The `ref` argument is typed DataIterHandle in the C API but is
        // dereferenced as a DMatrixHandle (`shared_ptr<DMatrix>*`) — passing a
        // DMatrix handle here is exactly what the Python binding does.
        let ret = unsafe {
            xgboost_sys::XGQuantileDMatrixCreateFromCallback(
                &mut iter as *mut BatchIter as xgboost_sys::DataIterHandle,
                proxy,
                ref_matrix.map_or(ptr::null_mut(), |m| m.handle),
                Some(batch_iter_reset),
                Some(batch_iter_next),
                config_cstr.as_ptr(),
                &mut out,
            )
        };

        // Free the proxy regardless of outcome; surface the create error first.
        let proxy_free = unsafe { xgboost_sys::XGDMatrixFree(proxy) };
        XGBError::check_return_value(ret)?;
        XGBError::check_return_value(proxy_free)?;
        DMatrix::new(out)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_cols` is set to None, number of columns will be inferred from given data.
    ///
    /// Note: For small matrices (< 30000 non-zero elements), single-threaded creation is used
    /// to avoid thread synchronization overhead. For larger matrices, multi-threaded creation
    /// is used for better performance.
    pub fn from_csr(indptr: &[u64], indices: &[u64], data: &[f32], num_cols: Option<usize>) -> XGBResult<Self> {
        Self::from_csr_with_missing(indptr, indices, data, num_cols, f32::NAN)
    }

    /// Like [`from_csr`](Self::from_csr), but treats `missing` as the
    /// missing-value sentinel instead of NaN; see
    /// [`from_dense_with_missing`](Self::from_dense_with_missing) for the
    /// rationale. `missing` must be finite or NaN.
    pub fn from_csr_with_missing(
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: Option<usize>,
        missing: f32,
    ) -> XGBResult<Self> {
        // Threshold below which single-threaded is faster due to thread overhead.
        // Benchmarking shows crossover point is around 30k non-zeros on typical hardware.
        const SINGLE_THREAD_THRESHOLD: usize = 30000;

        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_cols = num_cols.unwrap_or(0) as xgboost_sys::bst_ulong;

        let indptr_interface = make_array_interface_u64(indptr);
        let indices_interface = make_array_interface_u64(indices);
        let data_interface = make_array_interface_f32(data);

        let indptr_cstr = ffi::CString::new(indptr_interface).unwrap();
        let indices_cstr = ffi::CString::new(indices_interface).unwrap();
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        // Use single thread for small matrices to avoid thread synchronization overhead
        let config_cstr = build_ingest_config(missing, data.len() < SINGLE_THREAD_THRESHOLD)?;

        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_cols,
            config_cstr.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSC](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))) matrix.
    ///
    /// Uses standard CSC representation where the row indices for column _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_rows` is set to None, number of rows will be inferred from given data.
    ///
    /// Note: For small matrices (< 30000 non-zero elements), single-threaded creation is used
    /// to avoid thread synchronization overhead. For larger matrices, multi-threaded creation
    /// is used for better performance.
    pub fn from_csc(indptr: &[u64], indices: &[u64], data: &[f32], num_rows: Option<usize>) -> XGBResult<Self> {
        Self::from_csc_with_missing(indptr, indices, data, num_rows, f32::NAN)
    }

    /// Like [`from_csc`](Self::from_csc), but treats `missing` as the
    /// missing-value sentinel instead of NaN; see
    /// [`from_dense_with_missing`](Self::from_dense_with_missing) for the
    /// rationale. `missing` must be finite or NaN.
    pub fn from_csc_with_missing(
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_rows: Option<usize>,
        missing: f32,
    ) -> XGBResult<Self> {
        // Threshold below which single-threaded is faster due to thread overhead.
        // Benchmarking shows crossover point is around 30k non-zeros on typical hardware.
        const SINGLE_THREAD_THRESHOLD: usize = 30000;

        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_rows = num_rows.unwrap_or(0) as xgboost_sys::bst_ulong;

        let indptr_interface = make_array_interface_u64(indptr);
        let indices_interface = make_array_interface_u64(indices);
        let data_interface = make_array_interface_f32(data);

        let indptr_cstr = ffi::CString::new(indptr_interface).unwrap();
        let indices_cstr = ffi::CString::new(indices_interface).unwrap();
        let data_cstr = ffi::CString::new(data_interface).unwrap();

        // Use single thread for small matrices to avoid thread synchronization overhead
        let config_cstr = build_ingest_config(missing, data.len() < SINGLE_THREAD_THRESHOLD)?;

        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSC(
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_rows,
            config_cstr.as_ptr(),
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from given file.
    ///
    /// Supports text files in [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) format, CSV,
    /// binary files written either by `save`, or from another XGBoost library.
    ///
    /// For more details on accepted formats, seem the
    /// [XGBoost input format](https://xgboost.readthedocs.io/en/latest/tutorials/input_format.html)
    /// documentation.
    ///
    /// # LIBSVM format
    ///
    /// Specified data in a sparse format as:
    /// ```text
    /// <label> <index>:<value> [<index>:<value> ...]
    /// ```
    ///
    /// E.g.
    /// ```text
    /// 0 1:1 9:0 11:0
    /// 1 9:1 11:0.375 15:1
    /// 0 1:0 8:0.22 11:1
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        let fname = crate::path_to_c_str(path);
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(fname.as_ptr(), &mut handle))?;
        DMatrix::new(handle)
    }

    /// Load a `DMatrix` from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if the path contains non-UTF8 characters.
    pub fn load_binary<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        // Use XGDMatrixCreateFromURI with a JSON config specifying the URI
        // Binary format is auto-detected, no format parameter needed
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| XGBError::new("Path contains non-UTF8 characters"))?;
        // Escape backslashes and quotes for valid JSON
        let escaped_path = path_str.replace('\\', "\\\\").replace('"', "\\\"");
        let config = format!(r#"{{"uri": "{}", "silent": 1}}"#, escaped_path);
        let config_cstr = ffi::CString::new(config).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(config_cstr.as_ptr(), &mut handle))?;
        DMatrix::new(handle)
    }

    /// Serialise this `DMatrix` as a binary file to given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing DMatrix to: {}", path.as_ref().display());
        let fname: ffi::CString = crate::path_to_c_str(path);
        let silent = true;
        xgb_call!(xgboost_sys::XGDMatrixSaveBinary(
            self.handle,
            fname.as_ptr(),
            silent as i32
        ))
    }

    /// Get the number of rows in this matrix.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Get the number of columns in this matrix.
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Get the shape (rows x columns) of this matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows(), self.num_cols())
    }

    /// Get a new DMatrix as a containing only given indices.
    pub fn slice(&self, indices: &[usize]) -> XGBResult<DMatrix> {
        debug!("Slicing {} rows from DMatrix", indices.len());
        // XGDMatrixSliceDMatrix does not bounds-check: an out-of-range index
        // reads outside the row array and can segfault. Validate here (also
        // covers the i32 cast below overflowing).
        if let Some(&bad) = indices.iter().find(|&&i| i >= self.num_rows) {
            return Err(XGBError::new(format!(
                "slice index {} out of bounds for DMatrix with {} rows",
                bad, self.num_rows
            )));
        }
        let mut out_handle = ptr::null_mut();
        let indices: Vec<i32> = indices.iter().map(|x| *x as i32).collect();
        xgb_call!(xgboost_sys::XGDMatrixSliceDMatrix(
            self.handle,
            indices.as_ptr(),
            indices.len() as xgboost_sys::bst_ulong,
            &mut out_handle
        ))?;
        DMatrix::new(out_handle)
    }

    /// Get ground truth labels for each row of this matrix.
    pub fn get_labels(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_LABEL)
    }

    /// Set ground truth labels for each row of this matrix.
    pub fn set_labels(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_LABEL, array)
    }

    /// Get weights of each instance.
    pub fn get_weights(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_WEIGHT)
    }

    /// Set weights of each instance.
    pub fn set_weights(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_WEIGHT, array)
    }

    /// Get base margin.
    pub fn get_base_margin(&self) -> XGBResult<&[f32]> {
        self.get_float_info(KEY_BASE_MARGIN)
    }

    /// Set base margin.
    ///
    /// If specified, xgboost will start from this margin, can be used to specify initial prediction to boost from.
    pub fn set_base_margin(&mut self, array: &[f32]) -> XGBResult<()> {
        self.set_float_info(KEY_BASE_MARGIN, array)
    }

    /// Set the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn set_group(&mut self, group: &[u32]) -> XGBResult<()> {
        // same as xgb_call!(xgboost_sys::XGDMatrixSetGroup(self.handle, group.as_ptr(), group.len() as u64))
        self.set_uint_info(KEY_GROUP, group)
    }

    /// Get the index for the beginning and end of a group.
    ///
    /// Needed when the learning task is ranking.
    ///
    /// See the XGBoost documentation for more information.
    pub fn get_group(&self) -> XGBResult<&[u32]> {
        self.get_uint_info(KEY_GROUP_PTR)
    }

    /// Set the type of each feature, enabling categorical splits for columns
    /// marked categorical.
    ///
    /// One entry per column: `"q"` (quantitative/float), `"c"` (categorical),
    /// `"int"` (integer) or `"i"` (boolean). For categorical columns the data
    /// values must be non-negative integers (encoded as `f32`, e.g. `0.0`,
    /// `1.0`, ...); XGBoost 3.3+ then uses categorical splits for them with the
    /// `hist` tree method — no one-hot encoding needed.
    ///
    /// **Call this before the DMatrix is first used for training.** XGBoost
    /// caches the quantized gradient index it builds on first use, and the
    /// cache key does not include feature types — changing them on a DMatrix
    /// that has already trained a booster silently reuses the old
    /// numeric-sketched index (this getter will still report the new types).
    pub fn set_feature_types(&mut self, types: &[&str]) -> XGBResult<()> {
        self.set_str_info(c"feature_type", types)
    }

    /// Get the per-column feature types previously set with
    /// [`set_feature_types`](Self::set_feature_types) (empty if never set).
    pub fn get_feature_types(&self) -> XGBResult<Vec<String>> {
        self.get_str_info(c"feature_type")
    }

    /// Set the name of each feature (one entry per column).
    pub fn set_feature_names(&mut self, names: &[&str]) -> XGBResult<()> {
        self.set_str_info(c"feature_name", names)
    }

    /// Get the per-column feature names (empty if never set).
    pub fn get_feature_names(&self) -> XGBResult<Vec<String>> {
        self.get_str_info(c"feature_name")
    }

    fn set_str_info(&mut self, field: &CStr, values: &[&str]) -> XGBResult<()> {
        // Owned CStrings must outlive the FFI call; the pointer array borrows them.
        let owned: Vec<ffi::CString> = values
            .iter()
            .map(|s| {
                ffi::CString::new(*s)
                    .map_err(|_| XGBError::new(format!("string info value contains an interior NUL byte: {s:?}")))
            })
            .collect::<XGBResult<_>>()?;
        let mut ptrs: Vec<*const libc::c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        xgb_call!(xgboost_sys::XGDMatrixSetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            ptrs.as_mut_ptr(),
            values.len() as xgboost_sys::bst_ulong
        ))
    }

    fn get_str_info(&self, field: &CStr) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_sys::XGDMatrixGetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out
        ))?;
        if out_len == 0 {
            return Ok(Vec::new());
        }
        let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
        out_ptr_slice
            .iter()
            .map(|str_ptr| {
                unsafe { ffi::CStr::from_ptr(*str_ptr) }
                    .to_str()
                    .map(str::to_owned)
                    // Possible on a DMatrix written by another binding.
                    .map_err(|e| XGBError::new(format!("string info value is not valid UTF-8: {e}")))
            })
            .collect()
    }

    fn get_float_info(&self, field: &CStr) -> XGBResult<&[f32]> {
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetFloatInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_float, out_len as usize) })
        } else {
            Ok(&[0.0; 0])
        }
    }

    fn set_float_info(&mut self, field: &CStr, array: &[f32]) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGDMatrixSetFloatInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    fn get_uint_info(&self, field: &CStr) -> XGBResult<&[u32]> {
        let mut out_len = 0;
        let mut out_dptr = ptr::null();
        xgb_call!(xgboost_sys::XGDMatrixGetUIntInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out_dptr
        ))?;

        if out_len > 0 {
            Ok(unsafe { slice::from_raw_parts(out_dptr as *mut c_uint, out_len as usize) })
        } else {
            Ok(&[0; 0])
        }
    }

    fn set_uint_info(&mut self, field: &CStr, array: &[u32]) -> XGBResult<()> {
        let array_interface = make_array_interface_u32(array);
        let data_cstr = ffi::CString::new(array_interface).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetInfoFromInterface(
            self.handle,
            field.as_ptr(),
            data_cstr.as_ptr()
        ))
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        xgb_call!(xgboost_sys::XGDMatrixFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#)
    }

    #[test]
    fn read_matrix() {
        assert!(read_train_matrix().is_ok());
    }

    #[test]
    fn read_num_rows() {
        assert_eq!(read_train_matrix().unwrap().num_rows(), 6513);
    }

    #[test]
    fn read_num_cols() {
        assert_eq!(read_train_matrix().unwrap().num_cols(), 127);
    }

    #[test]
    fn writing_and_reading() {
        let dmat = read_train_matrix().unwrap();

        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let out_path = tmp_dir.path().join("dmat.bin");
        dmat.save(&out_path).unwrap();

        let dmat2 = DMatrix::load_binary(out_path).unwrap();

        assert_eq!(dmat.num_rows(), dmat2.num_rows());
        assert_eq!(dmat.num_cols(), dmat2.num_cols());

        // Verify labels are preserved after save/load
        let labels1 = dmat.get_labels().unwrap();
        let labels2 = dmat2.get_labels().unwrap();
        assert_eq!(labels1.len(), labels2.len());
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn get_set_labels() {
        let mut dmat = read_train_matrix().unwrap();
        let labels = dmat.get_labels();
        assert!(labels.is_ok());
        let mut labels = labels.unwrap().to_vec();
        assert_eq!(labels.len(), 6513);

        labels[0] = 0.1;
        assert_ne!(dmat.get_labels().unwrap(), labels);
        assert!(dmat.set_labels(&labels).is_ok());
        assert_eq!(dmat.get_labels().unwrap(), labels);
    }

    #[test]
    fn get_set_weights() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_weights().unwrap().is_empty());

        let weight = [1.0, 10.0, 44.9555];
        assert!(dmat.set_weights(&weight).is_ok());
        assert_eq!(dmat.get_weights().unwrap(), weight);
    }

    #[test]
    fn get_set_base_margin() {
        let mut dmat = read_train_matrix().unwrap();
        let base_margin = dmat.get_base_margin();
        assert!(base_margin.is_ok());
        assert!(base_margin.unwrap().is_empty());

        let base_margin = vec![0.00001; dmat.num_rows()];
        assert!(dmat.set_base_margin(&base_margin).is_ok());
        assert_eq!(dmat.get_base_margin().unwrap(), base_margin);
    }

    #[test]
    fn get_set_group() {
        let mut dmat = read_train_matrix().unwrap();
        assert!(dmat.get_group().unwrap().is_empty());

        let group = [1];
        assert!(dmat.set_group(&group).is_ok());
        assert_eq!(dmat.get_group().unwrap(), &[0, 1]);
    }

    #[test]
    fn from_csr() {
        let indptr: [u64; 5] = [0, 2, 3, 6, 8];
        let indices: [u64; 8] = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 0); // https://github.com/dmlc/xgboost/pull/7265

        let dmat = DMatrix::from_csr(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 4);
        assert_eq!(dmat.num_cols(), 10);
    }

    #[test]
    fn from_csc() {
        let indptr: [u64; 5] = [0, 2, 3, 6, 8];
        let indices: [u64; 8] = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
    }

    /// Entries stored in the DMatrix (i.e. not dropped as missing).
    fn num_non_missing(dmat: &DMatrix) -> u64 {
        let mut out = 0;
        let ret = unsafe { xgboost_sys::XGDMatrixNumNonMissing(dmat.handle, &mut out) };
        XGBError::check_return_value(ret).unwrap();
        out
    }

    #[test]
    fn from_dense_with_missing_drops_sentinel() {
        let data = [1.0, -999.0, 3.0, 4.0, -999.0, 6.0];

        // Default NaN missing: all 6 entries present.
        let dmat = DMatrix::from_dense(&data, 2).unwrap();
        assert_eq!(num_non_missing(&dmat), 6);

        // Sentinel missing: the two -999 entries drop out during ingestion.
        let dmat = DMatrix::from_dense_with_missing(&data, 2, -999.0).unwrap();
        assert_eq!(dmat.shape(), (2, 3));
        assert_eq!(num_non_missing(&dmat), 4);

        // Infinities are unrepresentable in the shared config contract.
        assert!(DMatrix::from_dense_with_missing(&data, 2, f32::INFINITY).is_err());
    }

    #[test]
    fn from_csr_with_missing_drops_sentinel() {
        let indptr: [u64; 3] = [0, 2, 4];
        let indices: [u64; 4] = [0, 1, 0, 1];
        let data = [1.0, -999.0, -999.0, 4.0];

        let dmat = DMatrix::from_csr_with_missing(&indptr, &indices, &data, Some(2), -999.0).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(num_non_missing(&dmat), 2);
    }

    #[test]
    fn from_csc_with_missing_drops_sentinel() {
        let indptr: [u64; 3] = [0, 2, 4];
        let indices: [u64; 4] = [0, 1, 0, 1];
        let data = [1.0, -999.0, -999.0, 4.0];

        let dmat = DMatrix::from_csc_with_missing(&indptr, &indices, &data, None, -999.0).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(num_non_missing(&dmat), 2);
    }

    #[test]
    fn from_dense_f64_matches_f32_ingestion() {
        let data_f64 = [1.5f64, f64::NAN, 3.25, 4.0, 5.0, 6.0];
        let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();

        let dmat64 = DMatrix::from_dense_f64(&data_f64, 3).unwrap();
        let dmat32 = DMatrix::from_dense(&data_f32, 3).unwrap();

        assert_eq!(dmat64.shape(), (3, 2));
        assert_eq!(dmat64.shape(), dmat32.shape());
        // The NaN drops out in both paths; the remaining 5 entries survive.
        assert_eq!(num_non_missing(&dmat64), 5);
        assert_eq!(num_non_missing(&dmat64), num_non_missing(&dmat32));

        // Same shape guards as from_dense.
        assert!(DMatrix::from_dense_f64(&data_f64, 0).is_err());
        assert!(DMatrix::from_dense_f64(&data_f64, 4).is_err());
    }

    #[test]
    fn from_dense_quantile_ref_shares_cuts() {
        let num_rows = 50;
        let num_cols = 4;
        let mut data = vec![0f32; num_rows * num_cols];
        let mut labels = vec![0f32; num_rows];
        for i in 0..num_rows {
            for j in 0..num_cols {
                data[i * num_cols + j] = ((i * 3 + j) % 17) as f32;
            }
            labels[i] = (i % 2) as f32;
        }
        let train = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 64).unwrap();

        // Eval slice drawn from a narrower value range than the training data:
        // with independent sketching its cuts would differ; with `ref` they are
        // inherited from `train`. Constructing against the ref must succeed and
        // preserve the shape.
        let eval_rows = 10;
        let eval_data = &data[..eval_rows * num_cols];
        let eval =
            DMatrix::from_dense_quantile_ref(eval_data, eval_rows, Some(&labels[..eval_rows]), 64, &train).unwrap();
        assert_eq!(eval.shape(), (eval_rows, num_cols));
        assert_eq!(eval.get_labels().unwrap(), &labels[..eval_rows]);

        // CSR variant: one dense-as-sparse row set against the same ref.
        let indptr: Vec<u64> = (0..=eval_rows as u64).map(|i| i * num_cols as u64).collect();
        let indices: Vec<u64> = (0..eval_rows).flat_map(|_| 0..num_cols as u64).collect();
        let eval_csr =
            DMatrix::from_csr_quantile_ref(&indptr, &indices, eval_data, num_cols, None, 64, &train).unwrap();
        assert_eq!(eval_csr.shape(), (eval_rows, num_cols));
    }

    #[test]
    fn from_dense_quantile() {
        let num_rows = 50;
        let num_cols = 4;
        let mut data = vec![0f32; num_rows * num_cols];
        let mut labels = vec![0f32; num_rows];
        for i in 0..num_rows {
            for j in 0..num_cols {
                data[i * num_cols + j] = ((i * 3 + j) % 17) as f32;
            }
            labels[i] = (i % 2) as f32;
        }

        let qdm = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 64).unwrap();
        assert_eq!(qdm.shape(), (num_rows, num_cols));
        assert_eq!(qdm.get_labels().unwrap(), &labels[..]);

        // Labels are optional.
        let qdm = DMatrix::from_dense_quantile(&data, num_rows, None, 64).unwrap();
        assert_eq!(qdm.shape(), (num_rows, num_cols));
    }

    /// Meta info (base_margin, weights) must be settable AFTER quantile
    /// construction: only the feature data is frozen by the binning, not the
    /// float-info fields. Distributional-regression callers rely on this to
    /// inject start values as the base margin on a QuantileDMatrix.
    #[test]
    fn quantile_get_set_base_margin_and_weights() {
        let num_rows = 50;
        let num_cols = 4;
        let mut data = vec![0f32; num_rows * num_cols];
        for i in 0..num_rows {
            for j in 0..num_cols {
                data[i * num_cols + j] = ((i * 3 + j) % 17) as f32;
            }
        }
        let mut qdm = DMatrix::from_dense_quantile(&data, num_rows, None, 64).unwrap();

        assert!(qdm.get_base_margin().unwrap().is_empty());
        // Multi-target-style margin: 2 values per row, flattened row-major.
        let margin: Vec<f32> = (0..num_rows * 2).map(|i| i as f32 * 0.25).collect();
        assert!(qdm.set_base_margin(&margin).is_ok());
        assert_eq!(qdm.get_base_margin().unwrap(), &margin[..]);

        let weights: Vec<f32> = (0..num_rows).map(|i| 1.0 + (i % 3) as f32).collect();
        assert!(qdm.set_weights(&weights).is_ok());
        assert_eq!(qdm.get_weights().unwrap(), &weights[..]);
    }

    #[test]
    fn from_dense() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_rows = 2;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 2);
        assert_eq!(dmat.num_cols(), 3);

        let data = vec![1.0, 2.0, 3.0];
        let num_rows = 3;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 1);
    }

    #[test]
    fn from_dense_rejects_bad_shape() {
        // Length not divisible by num_rows: must error, not silently truncate.
        assert!(DMatrix::from_dense(&[1.0, 2.0, 3.0], 2).is_err());
        // Zero rows: must error, not panic on division by zero.
        assert!(DMatrix::from_dense(&[1.0], 0).is_err());
        assert!(DMatrix::from_dense_quantile(&[1.0, 2.0, 3.0], 2, None, 256).is_err());
    }

    #[test]
    fn slice_from_indices() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 2));

        assert_eq!(dmat.slice(&[]).unwrap().shape(), (0, 2));
        assert_eq!(dmat.slice(&[1]).unwrap().shape(), (1, 2));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 2));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 2));
        // Out-of-bounds indices are rejected before reaching the C API, which
        // does not bounds-check and can segfault on them.
        assert!(dmat.slice(&[10, 11, 12]).is_err());
        assert!(dmat.slice(&[0, 4]).is_err());
    }

    #[test]
    fn slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let num_rows = 4;

        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        assert_eq!(dmat.shape(), (4, 3));

        assert_eq!(dmat.slice(&[0, 1, 2, 3]).unwrap().shape(), (4, 3));
        assert_eq!(dmat.slice(&[0, 1]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[1, 0]).unwrap().shape(), (2, 3));
        assert_eq!(dmat.slice(&[0, 1, 2]).unwrap().shape(), (3, 3));
        assert_eq!(dmat.slice(&[3, 2, 1]).unwrap().shape(), (3, 3));
    }
}
