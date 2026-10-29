use libc::{c_float, c_uint};
use std::{ffi, path::Path, ptr, slice};

use super::{XGBError, XGBResult};

static KEY_GROUP_PTR: &str = "group_ptr";
static KEY_GROUP: &str = "group";
static KEY_LABEL: &str = "label";
static KEY_WEIGHT: &str = "weight";
static KEY_BASE_MARGIN: &str = "base_margin";

const USIZE_TYPESTR: &str = if cfg!(target_pointer_width = "64") { "<u8" } else { "<u4" };
const SPARSE_CONFIG: &str = r#"{"missing": NaN}"#;

/// JSON-encoded `__array_interface__` describing `data`, as XGBoost's array-based constructors expect.
/// The pointer is borrowed, so `data` must outlive the call it is passed to.
fn array_interface<T>(data: &[T], typestr: &str) -> ffi::CString {
    let json = format!(
        r#"{{"data":[{},true],"shape":[{}],"strides":null,"typestr":"{}","version":3}}"#,
        data.as_ptr() as usize,
        data.len(),
        typestr
    );
    ffi::CString::new(json).expect("array interface JSON has no NUL bytes")
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
/// let indptr = &[0, 1, 2, 6];
/// let indices = &[0, 2, 2, 0, 1, 2];
/// let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dmat = DMatrix::from_csc(indptr, indices, data, None).unwrap();
/// assert_eq!(dmat.shape(), (3, 3));
/// ```
// Not `Send`: XGBoost keeps per-thread entries keyed by the matrix (`DMatrixThreadLocal` in data.cc)
// and only clears the dropping thread's entry, so dropping on another thread leaks them.
#[derive(Debug)]
pub struct DMatrix {
    pub(super) handle: xgboost_sys::DMatrixHandle,
    num_rows: usize,
    num_cols: usize,
}

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

    /// Like [`from_dense`](Self::from_dense), but treats every value equal to `missing` as a missing
    /// value instead of `NaN`. `NaN` values are always treated as missing.
    ///
    /// Returns an error if `num_rows` is 0 or `data.len()` is not a multiple of `num_rows`.
    pub fn from_dense_with_missing(data: &[f32], num_rows: usize, missing: f32) -> XGBResult<Self> {
        if num_rows == 0 || !data.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "dense data of length {} cannot be split into {} rows",
                data.len(),
                num_rows
            )));
        }
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGDMatrixCreateFromMat(
            data.as_ptr(),
            num_rows as xgboost_sys::bst_ulong,
            (data.len() / num_rows) as xgboost_sys::bst_ulong,
            missing,
            &mut handle
        ))?;
        DMatrix::new(handle)
    }

    /// Create a new `DMatrix` from a sparse
    /// [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) matrix.
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are stored in
    /// `indices[indptr[i]:indptr[i+1]]` and their corresponding values are stored in
    /// `data[indptr[i]:indptr[i+1]`.
    ///
    /// If `num_cols` is set to None, number of columns will be inferred from given data.
    pub fn from_csr(indptr: &[usize], indices: &[usize], data: &[f32], num_cols: Option<usize>) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_cols = num_cols.unwrap_or(0); // infer from data if 0
        let config = ffi::CString::new(SPARSE_CONFIG).unwrap();
        let (indptr, indices) = (
            array_interface(indptr, USIZE_TYPESTR),
            array_interface(indices, USIZE_TYPESTR),
        );
        let data = array_interface(data, "<f4");
        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSR(
            indptr.as_ptr(),
            indices.as_ptr(),
            data.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
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
    pub fn from_csc(indptr: &[usize], indices: &[usize], data: &[f32], num_rows: Option<usize>) -> XGBResult<Self> {
        assert_eq!(indices.len(), data.len());
        let mut handle = ptr::null_mut();
        let num_rows = num_rows.unwrap_or(0); // infer from data if 0
        let config = ffi::CString::new(SPARSE_CONFIG).unwrap();
        let (indptr, indices) = (
            array_interface(indptr, USIZE_TYPESTR),
            array_interface(indices, USIZE_TYPESTR),
        );
        let data = array_interface(data, "<f4");
        xgb_call!(xgboost_sys::XGDMatrixCreateFromCSC(
            indptr.as_ptr(),
            indices.as_ptr(),
            data.as_ptr(),
            num_rows as xgboost_sys::bst_ulong,
            config.as_ptr(),
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

    pub fn load_binary<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading DMatrix from: {}", path.as_ref().display());
        let mut handle = ptr::null_mut();
        let mut config = br#"{"uri": ""#.to_vec();
        for &b in path.as_ref().as_os_str().as_encoded_bytes() {
            if matches!(b, b'\\' | b'"') {
                config.push(b'\\');
            }
            config.push(b);
        }
        config.extend_from_slice(br#"", "silent": 1}"#);
        let config = ffi::CString::new(config).map_err(|e| XGBError::new(e.to_string()))?;
        xgb_call!(xgboost_sys::XGDMatrixCreateFromURI(config.as_ptr(), &mut handle))?;
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
        let indices: Vec<i32> = indices
            .iter()
            .map(|&i| {
                i32::try_from(i).ok().filter(|_| i < self.num_rows).ok_or_else(|| {
                    XGBError::new(format!(
                        "slice index {i} out of bounds for DMatrix with {} rows",
                        self.num_rows
                    ))
                })
            })
            .collect::<XGBResult<_>>()?;
        let mut out_handle = ptr::null_mut();
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

    /// Set the type of each feature: "q" (quantitative), "int", "float" or "c" (categorical).
    ///
    /// Categorical columns hold the category codes as values; training needs `tree_method` "hist" or "approx".
    pub fn set_feature_types(&mut self, types: &[&str]) -> XGBResult<()> {
        self.set_str_info("feature_type", types)
    }

    /// Set the name of each feature.
    pub fn set_feature_names(&mut self, names: &[&str]) -> XGBResult<()> {
        self.set_str_info("feature_name", names)
    }

    fn set_str_info(&mut self, field: &str, values: &[&str]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        let owned: Vec<ffi::CString> = values
            .iter()
            .map(|s| ffi::CString::new(*s))
            .collect::<Result<_, _>>()
            .map_err(|e| XGBError::new(e.to_string()))?;
        let mut ptrs: Vec<*const libc::c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        xgb_call!(xgboost_sys::XGDMatrixSetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            ptrs.as_mut_ptr(),
            values.len() as u64
        ))
    }

    pub(crate) fn get_str_info(&self, field: &str) -> XGBResult<Vec<String>> {
        let field = ffi::CString::new(field).unwrap();
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
        let ptrs = unsafe { slice::from_raw_parts(out, out_len as usize) };
        Ok(ptrs
            .iter()
            .map(|p| unsafe { ffi::CStr::from_ptr(*p).to_string_lossy().into_owned() })
            .collect())
    }

    fn get_float_info(&self, field: &str) -> XGBResult<&[f32]> {
        let field = ffi::CString::new(field).unwrap();
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

    fn set_float_info(&mut self, field: &str, array: &[f32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetFloatInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
        ))
    }

    fn get_uint_info(&self, field: &str) -> XGBResult<&[u32]> {
        let field = ffi::CString::new(field).unwrap();
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

    fn set_uint_info(&mut self, field: &str, array: &[u32]) -> XGBResult<()> {
        let field = ffi::CString::new(field).unwrap();
        xgb_call!(xgboost_sys::XGDMatrixSetUIntInfo(
            self.handle,
            field.as_ptr(),
            array.as_ptr(),
            array.len() as u64
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
        // TODO: check contents as well, if possible
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
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
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
        let indptr = [0, 2, 3, 6, 8];
        let indices = [0, 2, 2, 0, 1, 2, 1, 2];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, None).unwrap();
        assert_eq!(dmat.num_rows(), 3);
        assert_eq!(dmat.num_cols(), 4);

        let dmat = DMatrix::from_csc(&indptr, &indices, &data, Some(10)).unwrap();
        assert_eq!(dmat.num_rows(), 10);
        assert_eq!(dmat.num_cols(), 4);
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
    fn categorical_feature_types_train_a_categorical_split() {
        use crate::{parameters, Booster};
        let data: Vec<f32> = (0..40).flat_map(|i| [(i % 4) as f32, i as f32]).collect();
        let labels: Vec<f32> = (0..40).map(|i| if i % 4 == 2 { 1.0 } else { 0.0 }).collect();
        let mut dmat = DMatrix::from_dense(&data, 40).unwrap();
        dmat.set_labels(&labels).unwrap();
        dmat.set_feature_names(&["colour", "n"]).unwrap();
        dmat.set_feature_types(&["c", "q"]).unwrap();

        let tree = parameters::tree::TreeBoosterParametersBuilder::default()
            .tree_method(parameters::tree::TreeMethod::Hist)
            .max_depth(1)
            .build()
            .unwrap();
        let learning = parameters::learning::LearningTaskParametersBuilder::default()
            .objective(parameters::learning::Objective::BinaryLogistic)
            .build()
            .unwrap();
        let booster_params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree))
            .learning_params(learning)
            .build()
            .unwrap();
        let params = parameters::TrainingParametersBuilder::default()
            .dtrain(&dmat)
            .boost_rounds(2)
            .booster_params(booster_params)
            .build()
            .unwrap();
        let bst = Booster::train(&params).unwrap();
        let json = String::from_utf8(bst.save_buffer(false).unwrap()).unwrap();
        assert!(json.contains("\"categories_nodes\":[0]"), "{json}");
        assert_eq!(bst.predict_contributions(&dmat).unwrap().1, (40, 3));
        let first = String::from_utf8(bst.slice(0, 1).unwrap().save_buffer(false).unwrap()).unwrap();
        assert!(first.contains("\"num_trees\":\"1\""), "{first}");
        assert!(bst.slice(0, 10).is_err());
        assert!(bst.slice(2, 1).is_err());
        assert!(dmat.set_feature_names(&["a\0b", "n"]).is_err());
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
        assert!(dmat.slice(&[10, 11, 12]).is_err());
        assert!(dmat.slice(&[4]).is_err());
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

    #[test]
    fn from_dense_rejects_invalid_row_count() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(DMatrix::from_dense(&data, 0).is_err());
        assert!(DMatrix::from_dense(&data, 4).is_err());
        assert_eq!(DMatrix::from_dense(&data, 3).unwrap().shape(), (3, 2));
    }

    #[test]
    fn from_dense_with_missing_drops_sentinel() {
        let data = [1.0, -999.0, -999.0, 4.0, 5.0, 6.0];
        let non_missing = |dmat: &DMatrix| {
            let mut out = 0;
            xgb_call!(xgboost_sys::XGDMatrixNumNonMissing(dmat.handle, &mut out)).unwrap();
            out
        };
        let with_sentinel = DMatrix::from_dense_with_missing(&data, 3, -999.0).unwrap();
        assert_eq!(with_sentinel.shape(), (3, 2));
        assert_eq!(non_missing(&with_sentinel), 4);
        assert_eq!(non_missing(&DMatrix::from_dense(&data, 3).unwrap()), 6);
    }
}
