use crate::dmatrix::DMatrix;
use crate::error::XGBError;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::os::raw;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{ffi, fmt, fs::File, ptr, slice};

use indexmap::IndexMap;

use super::XGBResult;
use crate::parameters::{BoosterParameters, CallbackEnv, TrainingParameters};

pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);

// The gradient/hessian array interface for `XGBoosterTrainOneIter` is built on
// the stack in `boost` via `write_interface_{1d,2d}` (see that call site). For
// multi-target boosters (num_target > 1, e.g. distributional models training
// mu/sigma in one booster) XGBoost requires the gradient/hessian to be declared
// as a 2D `[num_row, n_targets]` array. Passing a 1D `[len]` (where len =
// num_row * n_targets) trips the C-side check `i_grad.Shape<0>() ==
// p_fmat->Info().num_row_` because Shape<0> becomes num_row * n_targets.

#[derive(Default, Debug, Clone)]
pub enum PredictType {
    #[default]
    Normal = 0,
    OutputMargin = 1,
    PredictContribitions = 2,
    PredictApproximateContributions = 3,
    PredictFeatureInteractions = 4,
    PredictApproximateFeatureInteractions = 5,
    PredictLeafTraining = 6,
}

#[derive(Default)]
pub struct PredictConfig {
    pub _type: PredictType,
    pub training: bool,
    pub iteration_begin: i64,
    pub iteration_end: i64,
    pub strict_shape: bool,
}

impl PredictConfig {
    /// returns 0 terminated json of the config, mainly for usage in predict_matrix
    pub fn as_json(&self) -> String {
        format!(
            "{{\"type\":{},\"training\":{},\"iteration_begin\":{},\"iteration_end\":{},\"strict_shape\":{}}}\0",
            self._type.clone() as usize,
            self.training,
            self.iteration_begin,
            self.iteration_end,
            self.strict_shape
        )
    }
}

/// NUL-terminated `XGBoosterPredictFromDMatrix` config strings, one per prediction
/// type. These mirror what [`PredictConfig::as_json`] would produce, kept as
/// `&'static CStr` literals so the prediction paths (including the per-round
/// training hot path) allocate nothing. `iteration_end: 0` means "use all trees",
/// matching the old `ntree_limit = 0`; `strict_shape: false` preserves the legacy
/// output layouts that the return-shape calculations below depend on.
mod predict_config {
    use std::ffi::CStr;

    /// Normal prediction (type 0). Used by `predict` and the training hot path.
    pub const NORMAL: &CStr =
        cr#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#;
    /// Normal inplace prediction (type 0). Inplace predict additionally accepts a
    /// `missing` field; NaN matches the missing value used by `DMatrix`.
    pub const NORMAL_INPLACE: &CStr =
        cr#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false,"missing":NaN}"#;
    /// Output margin (type 1). `training: false`: this backs the public
    /// `predict_margin` inference API and the custom-eval path, matching
    /// Python's `predict(output_margin=True)`. Only DART models are sensitive
    /// to the flag — the legacy call this replaced passed `training=1`, which
    /// applied random tree-dropout per call and made DART margins
    /// nondeterministic.
    pub const MARGIN: &CStr =
        cr#"{"type":1,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#;
    /// SHAP feature contributions (type 2).
    pub const CONTRIBUTIONS: &CStr =
        cr#"{"type":2,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#;
    /// SHAP feature interactions (type 4).
    pub const INTERACTIONS: &CStr =
        cr#"{"type":4,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#;
    /// Predicted leaf indices (type 6).
    pub const LEAF: &CStr =
        cr#"{"type":6,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false}"#;
}

/// Fixed-capacity stack buffer for building short NUL-terminated FFI strings
/// (array-interface JSON) without heap allocation on prediction hot paths.
///
/// The array-interface templates plus three 20-digit integers (max `usize`)
/// total under 140 bytes, so the 192-byte buffers used below cannot overflow;
/// `write_str` still checks and errors rather than truncating, reserving the
/// final byte for the NUL terminator.
struct CBuf<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> CBuf<N> {
    fn new() -> Self {
        CBuf { buf: [0; N], len: 0 }
    }

    /// NUL-terminate the accumulated bytes and view them as a `&CStr`.
    fn as_cstr(&mut self) -> &ffi::CStr {
        // In-bounds: write_str reserves the final byte for this terminator.
        self.buf[self.len] = 0;
        ffi::CStr::from_bytes_with_nul(&self.buf[..=self.len]).expect("JSON written to CBuf contains no interior NUL")
    }
}

impl<const N: usize> fmt::Write for CBuf<N> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        // >= reserves the final byte for the NUL terminator
        if self.len + bytes.len() >= N {
            return Err(fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

/// Buffer size used for all array-interface JSON built on the stack.
const ARRAY_INTERFACE_BUF: usize = 192;

/// Write a 1D array-interface JSON (`typestr` per the NumPy spec, e.g. `"<f4"`,
/// `"<u8"`) into `buf` and return it NUL-terminated.
fn write_interface_1d<'a>(
    buf: &'a mut CBuf<ARRAY_INTERFACE_BUF>,
    ptr: usize,
    len: usize,
    typestr: &str,
) -> &'a ffi::CStr {
    write!(
        buf,
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"{}","version":3}}"#,
        ptr, len, typestr
    )
    .expect("array-interface JSON exceeds stack buffer");
    buf.as_cstr()
}

/// Write a 2D row-major f32 array-interface JSON into `buf` and return it
/// NUL-terminated.
fn write_interface_2d(buf: &mut CBuf<ARRAY_INTERFACE_BUF>, ptr: usize, rows: usize, cols: usize) -> &ffi::CStr {
    write!(
        buf,
        r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f4","version":3}}"#,
        ptr, rows, cols
    )
    .expect("array-interface JSON exceeds stack buffer");
    buf.as_cstr()
}

/// Assemble the borrowed (data, shape) output of an `XGBoosterPredict*` call.
///
/// XGBoost hands back a null data pointer when the prediction is empty (e.g. a
/// 0-row DMatrix); that must become an empty slice rather than either a panic
/// or a `slice::from_raw_parts(null, 0)` call, which is UB even for length 0.
/// A null pointer alongside a non-empty shape would be a C-API contract
/// violation and still asserts. The unconstrained return lifetime is bound by
/// each caller's signature to the booster owning the buffers.
fn predict_output_slices<'a>(
    out_shape: *const u64,
    out_shape_dim: xgboost_sys::bst_ulong,
    out_result: *const f32,
) -> (&'a [f32], &'a [u64]) {
    let shape: &[u64] = if out_shape.is_null() {
        &[]
    } else {
        unsafe { slice::from_raw_parts(out_shape, out_shape_dim as usize) }
    };
    let data_size: u64 = shape.iter().product();
    if out_result.is_null() {
        assert_eq!(data_size, 0, "XGBoost returned null predictions for a non-empty shape");
        (&[], shape)
    } else {
        (unsafe { slice::from_raw_parts(out_result, data_size as usize) }, shape)
    }
}

/// Core model in XGBoost, containing functions for training, evaluating and predicting.
///
/// Usually created through the [`train`](struct.Booster.html#method.train) function, which
/// creates and trains a Booster in a single call.
///
/// For more fine grained usage, can be created using [`new`](struct.Booster.html#method.new) or
/// [`new_with_cached_dmats`](struct.Booster.html#method.new_with_cached_dmats), then trained by calling
/// [`update`](struct.Booster.html#method.update) or [`update_custom`](struct.Booster.html#method.update_custom)
/// in a loop.
pub struct Booster {
    handle: xgboost_sys::BoosterHandle,
    /// Lazily-created proxy DMatrix reused across inplace prediction calls
    /// (`predict_from_dense`/`predict_from_csr`). Passing a proxy to the C API
    /// avoids it allocating a fresh internal one per call, which measures at
    /// ~18% of single-row inplace predict latency on a single-thread booster.
    /// `Booster` is `!Sync` (raw handle field), so `&self` calls cannot race on it.
    inplace_proxy: std::cell::OnceCell<xgboost_sys::DMatrixHandle>,
    /// Inplace-prediction config with a custom missing value, built once by
    /// [`set_inplace_predict_missing`](Self::set_inplace_predict_missing).
    /// `None` means the static NaN config — the hot path stays allocation-free
    /// either way.
    inplace_config: Option<ffi::CString>,
}

// SAFETY: a `BoosterHandle` has no thread affinity. All C-API return buffers
// are stored in a *thread-local* map keyed by learner pointer
// (`LearnerAPIThreadLocalStore` in src/learner.cc), so calls made from
// different threads never share scratch state, and `~LearnerImpl` only erases
// the destroying thread's entry — dropping on a different thread than the one
// that created or used the booster is fine (this is how the Python bindings
// use the C API, calling it from arbitrary threads with the GIL released).
// The cached `inplace_proxy` DMatrix handle moves with the booster and is
// only touched through `&self`/`&mut self`. `Booster` must stay `!Sync`:
// concurrent `&self` inplace predictions would race on the shared proxy (the
// `OnceCell` field enforces this automatically).
unsafe impl Send for Booster {}

impl Booster {
    /// Wrap a raw booster handle owned by this struct from now on.
    fn from_handle(handle: xgboost_sys::BoosterHandle) -> Self {
        Booster {
            handle,
            inplace_proxy: std::cell::OnceCell::new(),
            inplace_config: None,
        }
    }

    /// Set the value the inplace prediction paths (`predict_from_dense*`,
    /// `predict_from_csr*`) treat as missing, instead of the default NaN.
    ///
    /// For serving inputs that encode missing as a sentinel (e.g. `0.0` or
    /// `-999.0`), this avoids rewriting each row to NaN before predicting. The
    /// config string is built once here (one small allocation); the per-call
    /// hot path remains allocation-free. Pass NaN to restore the default.
    /// `missing` must be finite or NaN. Does not affect the DMatrix-based
    /// predict paths — a `DMatrix` applies its own missing value at
    /// construction (see `DMatrix::from_dense_with_missing`).
    pub fn set_inplace_predict_missing(&mut self, missing: f32) -> XGBResult<()> {
        if missing.is_nan() {
            self.inplace_config = None;
            return Ok(());
        }
        let json = format!(
            r#"{{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":false,"missing":{}}}"#,
            crate::dmatrix::missing_json(missing)?
        );
        self.inplace_config = Some(ffi::CString::new(json).expect("JSON built above contains no interior NUL"));
        Ok(())
    }

    /// Config used by the inplace prediction paths: the custom-missing config
    /// when set, else the static NaN one.
    fn inplace_config(&self) -> &ffi::CStr {
        self.inplace_config.as_deref().unwrap_or(predict_config::NORMAL_INPLACE)
    }

    /// Get the cached proxy DMatrix for inplace prediction, creating it on first use.
    fn inplace_proxy(&self) -> XGBResult<xgboost_sys::DMatrixHandle> {
        if let Some(&proxy) = self.inplace_proxy.get() {
            return Ok(proxy);
        }
        let mut proxy = ptr::null_mut();
        xgb_call!(xgboost_sys::XGProxyDMatrixCreate(&mut proxy))?;
        // Cannot collide: `!Sync` forbids concurrent calls, and this runs only
        // when the cell was empty with no other set between (single thread).
        self.inplace_proxy
            .set(proxy)
            .expect("inplace proxy cell set concurrently");
        Ok(proxy)
    }
    /// Create a new Booster model with given parameters.
    ///
    /// This model can then be trained using calls to update/boost as appropriate.
    ///
    /// The [`train`](struct.Booster.html#method.train)  function is often a more convenient way of constructing,
    /// training and evaluating a Booster in a single call.
    pub fn new(params: &BoosterParameters) -> XGBResult<Self> {
        Self::new_with_cached_dmats(params, &[])
    }

    /// Create a new Booster model with given parameters and list of DMatrix to cache.
    ///
    /// Cached DMatrix can sometimes be used internally by XGBoost to speed up certain operations.
    ///
    /// # Safety Note
    ///
    /// The DMatrix handles are only used during the `XGBoosterCreate` call to initialize
    /// internal caches. The booster does not retain references to the DMatrix objects after
    /// creation, so it is safe if the DMatrix objects are freed after this function returns.
    /// However, for training purposes, you should keep the DMatrix alive for the duration
    /// of training since you'll need to pass it to `update()` or `update_custom()`.
    pub fn new_with_cached_dmats(params: &BoosterParameters, dmats: &[&DMatrix]) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        let s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_sys::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster::from_handle(handle);
        booster.set_params(params)?;
        Ok(booster)
    }

    /// Save this Booster as a binary file at given path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing Booster to: {}", path.as_ref().display());
        let fname = crate::path_to_c_str(path);
        xgb_call!(xgboost_sys::XGBoosterSaveModel(self.handle, fname.as_ptr()))
    }

    /// Save this Booster to a buffer.
    /// Format is "ubj" when binary, otherwise "json"
    pub fn save_buffer(&self, binary: bool) -> XGBResult<Vec<u8>> {
        trace!("Writing Booster to buffer");
        // Static NUL-terminated configs: XGBoosterSaveModelToBuffer expects a C
        // string. The previous format!-built String passed its pointer without
        // a NUL terminator, so the C side read past the end of the allocation.
        let config: &ffi::CStr = if binary { cr#"{"format":"ubj"}"# } else { cr#"{"format":"json"}"# };
        let mut out_len: xgboost_sys::bst_ulong = 0;
        let mut out_buffer = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterSaveModelToBuffer(
            self.handle,
            config.as_ptr(),
            &mut out_len,
            &mut out_buffer
        ))?;
        let buffer = unsafe { slice::from_raw_parts(out_buffer as *const u8, out_len as usize).to_vec() };
        Ok(buffer)
    }

    /// Serialise this Booster's full state (model + internal configuration) to a
    /// memory snapshot buffer (`XGBoosterSerializeToBuffer`).
    ///
    /// Unlike [`save_buffer`](Self::save_buffer), which encodes the model alone
    /// in a stable on-disk format, this is a raw snapshot intended for
    /// checkpointing within the same XGBoost version; restore it with
    /// [`unserialize_from_buffer`](Self::unserialize_from_buffer).
    pub fn serialize_to_buffer(&self) -> XGBResult<Vec<u8>> {
        let mut out_len: xgboost_sys::bst_ulong = 0;
        let mut out_buffer = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterSerializeToBuffer(
            self.handle,
            &mut out_len,
            &mut out_buffer
        ))?;
        Ok(unsafe { slice::from_raw_parts(out_buffer as *const u8, out_len as usize).to_vec() })
    }

    /// Restore a Booster from a [`serialize_to_buffer`](Self::serialize_to_buffer)
    /// snapshot. Must be the same XGBoost version that produced the snapshot.
    pub fn unserialize_from_buffer(bytes: &[u8]) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterUnserializeFromBuffer(
            handle,
            bytes.as_ptr() as *const _,
            bytes.len() as u64
        ))?;
        Ok(Booster::from_handle(handle))
    }

    /// Load a Booster from a binary file at given path.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading Booster from: {}", path.as_ref().display());

        // gives more control over error messages, avoids stack trace dump from C++
        if !path.as_ref().exists() {
            return Err(XGBError::new(format!("File not found: {}", path.as_ref().display())));
        }

        let fname = crate::path_to_c_str(path);
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModel(handle, fname.as_ptr()))?;
        Ok(Booster::from_handle(handle))
    }

    /// Load a Booster directly from a buffer.
    pub fn load_buffer(bytes: &[u8]) -> XGBResult<Self> {
        debug!("Loading Booster from buffer (length = {})", bytes.len());

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModelFromBuffer(
            handle,
            bytes.as_ptr() as *const _,
            bytes.len() as u64
        ))?;
        Ok(Booster::from_handle(handle))
    }

    /// Convenience function for creating/training a new Booster.
    ///
    /// This does the following:
    ///
    /// 1. create a new Booster model with given parameters
    /// 2. train the model with given DMatrix
    /// 3. print out evaluation results for each training round
    /// 4. return trained Booster
    ///
    /// * `params` - training parameters
    /// * `dtrain` - matrix to train Booster with
    /// * `num_boost_round` - number of training iterations
    /// * `eval_sets` - list of datasets to evaluate after each boosting round
    pub fn train(params: &TrainingParameters) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![params.dtrain];
            if let Some(eval_sets) = params.evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let mut bst = Booster::new_with_cached_dmats(&params.booster_params, &cached_dmats)?;
        // Each evaluation is a full prediction pass per eval set; honor the
        // configured cadence (always evaluating the final round so training
        // never ends without a metric). 0 is documented as meaning 1.
        let eval_period = params.eval_period.max(1) as i32;
        let last_round = params.boost_rounds as i32 - 1;
        for i in 0..params.boost_rounds as i32 {
            debug!("Updating in round: {}", i);
            if let Some(objective_fn) = params.custom_objective_fn {
                bst.update_custom(params.dtrain, i, objective_fn)?;
            } else {
                bst.update(params.dtrain, i)?;
            }

            // Collect evaluation results if evaluation sets are provided and
            // this round is on the evaluation schedule
            let eval_this_round = i % eval_period == 0 || i == last_round;
            let evaluation_results = if let Some(eval_sets) = params.evaluation_sets.filter(|_| eval_this_round) {
                let mut dmat_eval_results = bst.eval_set(eval_sets, i)?;

                if let Some(eval_fn) = params.custom_evaluation_fn {
                    let eval_name = "custom";
                    for (dmat, dmat_name) in eval_sets {
                        // Borrow XGBoost's margin buffer; `eval_fn` consumes it
                        // before the next booster call, so no copy is needed.
                        let margin = bst.predict_margin_borrowed(dmat)?;
                        let eval_result = eval_fn(margin, dmat);
                        let eval_results = dmat_eval_results
                            .entry(eval_name.to_string())
                            .or_insert_with(IndexMap::new);
                        eval_results.insert(dmat_name.to_string(), eval_result);
                    }
                }

                if params.verbose_eval {
                    // convert to map of eval_name -> (dmat_name -> score)
                    let mut eval_dmat_results = BTreeMap::new();
                    for (dmat_name, eval_results) in &dmat_eval_results {
                        for (eval_name, result) in eval_results {
                            let dmat_results = eval_dmat_results.entry(eval_name).or_insert_with(BTreeMap::new);
                            dmat_results.insert(dmat_name, result);
                        }
                    }

                    // Build the line once and emit it with a single write, instead
                    // of taking the stdout lock several times per boosting round.
                    let mut line = format!("[{}]", i);
                    for (eval_name, dmat_results) in eval_dmat_results {
                        for (dmat_name, result) in dmat_results {
                            let _ = write!(line, "\t{}-{}:{}", dmat_name, eval_name, result);
                        }
                    }
                    println!("{}", line);
                }

                Some(dmat_eval_results)
            } else {
                None
            };

            // Invoke callbacks if any are registered
            if let Some(ref callbacks) = params.callbacks {
                let callback_env = CallbackEnv {
                    iteration: i,
                    total_rounds: params.boost_rounds,
                    evaluation_results,
                };

                for callback in callbacks {
                    if !callback(&callback_env) {
                        // Callback returned false, stop training early
                        debug!("Callback requested early stopping at iteration {}", i);
                        return Ok(bst);
                    }
                }
            }
        }

        Ok(bst)
    }

    /// Release the data caches XGBoost accumulated during training (XGBoost 3.0+).
    ///
    /// The trained model is unaffected; only internal training caches (gradient
    /// buffers, prediction caches for the training matrices) are freed. Call this
    /// after training when keeping the booster around for inference to reduce its
    /// resident memory.
    pub fn reset(&mut self) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterReset(self.handle))
    }

    /// Update this Booster's parameters.
    pub fn set_params(&mut self, p: &BoosterParameters) -> XGBResult<()> {
        for (key, value) in p.as_string_pairs() {
            debug!("Setting parameter: {}={}", &key, &value);
            self.set_param(&key, &value)?;
        }
        Ok(())
    }

    /// Update this model by training it for one round with given training matrix.
    ///
    /// Uses XGBoost's objective function that was specificed in this Booster's learning objective parameters.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))
    }

    /// Update this model by training it for one round with a custom objective function.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    /// * `objective_fn` - custom objective function that returns (gradient, hessian)
    pub fn update_custom(&mut self, dtrain: &DMatrix, iteration: i32, objective_fn: CustomObjective) -> XGBResult<()> {
        // Borrow XGBoost's internal prediction buffer directly rather than
        // copying it into a Vec. The buffer is valid until the next prediction
        // call on this booster, and `objective_fn` consumes it (returning owned
        // gradient/hessian Vecs) before `boost` runs, so no copy is needed.
        let pred = self.predict_borrowed(dtrain)?;
        let (gradient, hessian) = objective_fn(pred, dtrain);
        self.boost(dtrain, iteration, &gradient, &hessian)
    }

    /// Update this model by directly specifying the first and second order gradients.
    ///
    /// This is typically used instead of `update` when using a customised loss function.
    /// Prefer it over [`update_custom`](Self::update_custom) when the caller already
    /// has the gradients in hand (e.g. it carries the training margin across rounds
    /// and computes grad/hess from it): `update_custom` predicts on `dtrain` before
    /// invoking its callback, so a caller that ignores that prediction pays one
    /// wasted prediction-cache read per round.
    ///
    /// For multi-target boosters (`num_target > 1`), `gradient`/`hessian` are
    /// row-major `[num_rows, n_targets]` flattened; the target count is inferred
    /// from `gradient.len() / dtrain.num_rows()`.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    /// * `gradient` - first order gradient
    /// * `hessian` - second order gradient
    pub fn boost(&mut self, dtrain: &DMatrix, iteration: i32, gradient: &[f32], hessian: &[f32]) -> XGBResult<()> {
        if gradient.len() != hessian.len() {
            let msg = format!(
                "Mismatch between length of gradient and hessian arrays ({} != {})",
                gradient.len(),
                hessian.len()
            );
            return Err(XGBError::new(msg));
        }
        assert_eq!(gradient.len(), hessian.len());

        self.validate_features(dtrain)?;

        // Infer n_targets from the gradient buffer length. Single-target
        // boosters get a 1D shape `[num_rows]` (unchanged behavior); multi-
        // target distributional boosters get a 2D shape `[num_rows, n_targets]`.
        let num_rows = dtrain.num_rows();
        let n_targets = if num_rows > 0 && !gradient.is_empty() && gradient.len().is_multiple_of(num_rows) {
            gradient.len() / num_rows
        } else {
            1
        };

        // Build the gradient/hessian array-interface JSON on the stack, matching
        // the allocation-free predict paths. `write_interface_{1d,2d}` produce
        // byte-identical output to the old `make_array_interface`: a 1D `[len]`
        // shape for single-target boosters, a 2D `[num_rows, n_targets]` shape
        // for multi-target ones. Keeps the per-round custom-objective training
        // hot path free of the four heap allocations `format!` + `CString::new`
        // incurred here previously (see the `predict_config` module comment).
        let mut grad_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let mut hess_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let grad_ptr = gradient.as_ptr() as usize;
        let hess_ptr = hessian.as_ptr() as usize;
        let (grad_cstr, hess_cstr) = if n_targets > 1 {
            (
                write_interface_2d(&mut grad_buf, grad_ptr, num_rows, n_targets),
                write_interface_2d(&mut hess_buf, hess_ptr, num_rows, n_targets),
            )
        } else {
            (
                write_interface_1d(&mut grad_buf, grad_ptr, gradient.len(), "<f4"),
                write_interface_1d(&mut hess_buf, hess_ptr, hessian.len(), "<f4"),
            )
        };

        xgb_call!(xgboost_sys::XGBoosterTrainOneIter(
            self.handle,
            dtrain.handle,
            iteration,
            grad_cstr.as_ptr(),
            hess_cstr.as_ptr()
        ))
    }

    fn eval_set(
        &self,
        evals: &[(&DMatrix, &str)],
        iteration: i32,
    ) -> XGBResult<IndexMap<String, IndexMap<String, f32>>> {
        let (dmats, names) = {
            let mut dmats = Vec::with_capacity(evals.len());
            let mut names = Vec::with_capacity(evals.len());
            for (dmat, name) in evals {
                dmats.push(dmat);
                names.push(*name);
            }
            (dmats, names)
        };
        assert_eq!(dmats.len(), names.len());

        let mut s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        // build separate arrays of C strings and pointers to them to ensure they live long enough
        let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let mut evptrs: Vec<*const libc::c_char> = Vec::with_capacity(names.len());

        for name in &names {
            let cstr = ffi::CString::new(*name).unwrap();
            evptrs.push(cstr.as_ptr());
            evnames.push(cstr);
        }

        // shouldn't be necessary, but guards against incorrect array sizing
        evptrs.shrink_to_fit();

        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterEvalOneIter(
            self.handle,
            iteration,
            s.as_mut_ptr(),
            evptrs.as_mut_ptr(),
            dmats.len() as u64,
            &mut out_result
        ))?;
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
        Ok(Booster::parse_eval_string(&out, &names))
    }

    /// Evaluate given matrix against this model using metrics defined in this model's parameters.
    ///
    /// See parameter::learning::EvaluationMetric for a full list.
    ///
    /// Returns a map of evaluation metric name to score.
    pub fn evaluate(&self, dmat: &DMatrix) -> XGBResult<HashMap<String, f32>> {
        let name = "default";
        let mut eval = self.eval_set(&[(dmat, name)], 0)?;
        let mut result = HashMap::new();
        eval.swap_remove(name).unwrap().into_iter().for_each(|(k, v)| {
            result.insert(k.to_owned(), v);
        });

        Ok(result)
    }

    /// Get a string attribute that was previously set for this model.
    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_sys::XGBoosterGetAttr(
            self.handle,
            key.as_ptr(),
            &mut out_buf,
            &mut success
        ))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
    }

    /// Store a string attribute in this model with given key.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetAttr(self.handle, key.as_ptr(), value.as_ptr()))
    }

    /// Get names of all attributes stored in this model. Values can then be fetched with calls to `get_attribute`.
    pub fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterGetAttrNames(self.handle, &mut out_len, &mut out))?;
        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
            let out_vec = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
                .collect();
            Ok(out_vec)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get names of feature names stored in this model.
    pub fn get_feature_names(&self) -> XGBResult<Vec<String>> {
        self.get_feature_info("feature_name")
    }

    /// Get the number of features this model was configured for.
    ///
    /// Uses `XGBoosterGetNumFeature` directly rather than counting feature
    /// *names*: it is valid for models that never had names attached, and
    /// avoids materializing the name array on per-iteration paths.
    fn num_features(&self) -> XGBResult<usize> {
        let mut out: xgboost_sys::bst_ulong = 0;
        xgb_call!(xgboost_sys::XGBoosterGetNumFeature(self.handle, &mut out))?;
        Ok(out as usize)
    }

    /// Get names of features stored in this model.
    pub fn get_feature_info(&self, field: &str) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        let field: ffi::CString =
            ffi::CString::new(field).map_err(|_| XGBError::new("field contains an interior NUL byte"))?;
        xgb_call!(xgboost_sys::XGBoosterGetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            &mut out_len,
            &mut out
        ))?;
        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
            out_ptr_slice
                .iter()
                .map(|str_ptr| {
                    unsafe { ffi::CStr::from_ptr(*str_ptr) }
                        .to_str()
                        .map(str::to_owned)
                        // Possible on a model saved by another binding.
                        .map_err(|e| XGBError::new(format!("feature info value is not valid UTF-8: {e}")))
                })
                .collect()
        } else {
            Ok(Vec::new())
        }
    }

    /// Set names of features stored in this model.
    pub fn set_feature_names(&self, features: &Vec<&str>) -> XGBResult<()> {
        self.set_feature_info("feature_name", features)
    }

    /// Set names of features stored in this model.
    #[allow(clippy::unnecessary_cast)]
    pub fn set_feature_info(&self, field: &str, features: &Vec<&str>) -> XGBResult<()> {
        let field: ffi::CString =
            ffi::CString::new(field).map_err(|_| XGBError::new("field contains an interior NUL byte"))?;

        // We want zero terminated strings. Keep the owned `CString`s alive in
        // `c_temp_features` for the duration of the FFI call so the pointers in
        // `c_feature_ptr` remain valid; they are dropped (and freed) on return.
        let c_temp_features: Vec<ffi::CString> = features
            .iter()
            .map(|s| {
                ffi::CString::new(*s)
                    .map_err(|_| XGBError::new(format!("feature info value contains an interior NUL byte: {s:?}")))
            })
            .collect::<XGBResult<_>>()?;
        let mut c_feature_ptr: Vec<*const raw::c_char> = c_temp_features
            .iter()
            .map(|s| s.as_ptr() as *const raw::c_char)
            .collect();

        xgb_call!(xgboost_sys::XGBoosterSetStrFeatureInfo(
            self.handle,
            field.as_ptr(),
            c_feature_ptr.as_mut_ptr() as *mut *const raw::c_char,
            features.len() as u64
        ))
    }

    /// Validate that this booster's feature count is compatible with the given DMatrix.
    ///
    /// Checks the model's configured feature count against the number of columns in the
    /// DMatrix. Skipped when the count is unavailable (e.g. a booster created without
    /// cached DMatrices before its first update) or zero, and when the DMatrix has
    /// 0 columns (unknown dimensions from CSR/CSC sparse matrices).
    fn validate_features(&self, dmat: &DMatrix) -> XGBResult<()> {
        // Best-effort: an error here means the model has no feature count yet,
        // not that the inputs mismatch — let the real operation proceed.
        let num_features = self.num_features().unwrap_or(0);
        if num_features == 0 {
            return Ok(());
        }

        let num_cols = dmat.num_cols();
        if num_cols == 0 {
            // Column count unknown (e.g., inferred from CSR/CSC), skip validation
            return Ok(());
        }

        if num_features != num_cols {
            return Err(XGBError::new(format!(
                "Feature count mismatch: booster has {} features but DMatrix has {} columns",
                num_features, num_cols
            )));
        }

        Ok(())
    }

    /// Predict results for given data.
    ///
    /// config_json should be a 0 terminated string, preferred created by PredictConfig::as_json
    /// Returns an array containing one entry per row in the given data and its shape as array.
    pub fn predict_matrix(&self, dmat: &DMatrix, config_json: &str) -> XGBResult<(Vec<f32>, Vec<u64>)> {
        let str_buffer: ffi::CString;
        // from_bytes_with_nul (unlike a trailing-NUL check + from_ptr) rejects
        // interior NULs instead of silently truncating the config at the first
        // one; a string that merely lacks the trailing NUL falls through to the
        // allocating path, which reports interior NULs as an error.
        let cfg = match ffi::CStr::from_bytes_with_nul(config_json.as_bytes()) {
            Ok(cfg) => cfg,
            Err(_) => {
                str_buffer = ffi::CString::new(config_json)
                    .map_err(|_| XGBError::new("config_json contains an interior NUL byte"))?;
                str_buffer.as_c_str()
            }
        };
        let (data, shape) = self.predict_raw(dmat, cfg)?;
        Ok((data.to_vec(), shape.to_vec()))
    }

    /// Core prediction via `XGBoosterPredictFromDMatrix`.
    ///
    /// Returns the prediction data and its shape as slices borrowed from buffers
    /// owned by this booster; both are only valid until the next prediction call
    /// on it (XGBoost reuses the buffers). Callers that need the data to outlive
    /// subsequent calls must copy it. `config` must be a NUL-terminated JSON string.
    fn predict_raw(&self, dmat: &DMatrix, config: &ffi::CStr) -> XGBResult<(&[f32], &[u64])> {
        let mut out_shape = ptr::null();
        let mut out_shape_dim = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredictFromDMatrix(
            self.handle,
            dmat.handle,
            config.as_ptr(),
            &mut out_shape,
            &mut out_shape_dim,
            &mut out_result
        ))?;
        Ok(predict_output_slices(out_shape, out_shape_dim, out_result))
    }

    /// Predict directly from a dense, row-major `[num_rows, num_cols]` slice
    /// without constructing a [`DMatrix`] first (XGBoost "inplace" prediction).
    ///
    /// For online serving of small batches this avoids the DMatrix build, which
    /// dominates end-to-end latency at that scale. Returns the prediction data
    /// and its shape, mirroring [`predict_matrix`](Self::predict_matrix). NaN is
    /// treated as the missing value, matching `DMatrix::from_dense`.
    ///
    /// # Latency tuning for small batches
    ///
    /// Small-batch latency is dominated by OpenMP thread dispatch inside
    /// XGBoost, controlled by the booster's `nthread` parameter (default: all
    /// cores). Pinning the booster to one thread with
    /// `booster.set_param("nthread", "1")` is much faster for small inputs:
    /// measured on a 127-feature/50-tree binary model, ~11x for 1 row, ~5x for
    /// 16 rows, ~2x for 100 rows, while multi-threading wins again from roughly
    /// 1000 rows up. For latency-sensitive serving of single rows or small
    /// batches, set `nthread` to 1 once after loading the model.
    ///
    /// Note: if the booster is configured for a CUDA device, XGBoost falls back
    /// to the DMatrix path with a performance warning (irrelevant for CPU boosters).
    pub fn predict_from_dense(&self, values: &[f32], num_rows: usize) -> XGBResult<(Vec<f32>, Vec<u64>)> {
        let (data, shape) = self.inplace_predict_dense_raw(values, num_rows)?;
        Ok((data.to_vec(), shape.to_vec()))
    }

    /// Like [`predict_from_dense`](Self::predict_from_dense), but writes the
    /// predictions into `out` (cleared first) instead of returning a fresh `Vec`.
    ///
    /// Reusing one buffer across calls makes the steady-state serving loop
    /// allocation-free. The output length is `num_rows * n_groups` (`n_groups`
    /// is 1 for regression/binary models), so the per-row group count is
    /// recoverable as `out.len() / num_rows`.
    pub fn predict_from_dense_into(&self, values: &[f32], num_rows: usize, out: &mut Vec<f32>) -> XGBResult<()> {
        let (data, _shape) = self.inplace_predict_dense_raw(values, num_rows)?;
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    }

    /// Dense inplace prediction via `XGBoosterPredictFromDense`, returning
    /// slices borrowed from booster-owned buffers (same validity contract as
    /// [`predict_raw`](Self::predict_raw): valid until the next prediction call).
    fn inplace_predict_dense_raw(&self, values: &[f32], num_rows: usize) -> XGBResult<(&[f32], &[u64])> {
        // Guard the shape inference: without this, num_rows = 0 dies on a bare
        // integer division panic, and a non-divisible length silently drops the
        // trailing values and predicts on a wrong-shaped matrix.
        if num_rows == 0 || !values.len().is_multiple_of(num_rows) {
            return Err(XGBError::new(format!(
                "values length {} does not divide into num_rows {}",
                values.len(),
                num_rows
            )));
        }
        let num_cols = values.len() / num_rows;
        let mut values_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let values_cstr = write_interface_2d(&mut values_buf, values.as_ptr() as usize, num_rows, num_cols);

        let mut out_shape = ptr::null();
        let mut out_shape_dim = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredictFromDense(
            self.handle,
            values_cstr.as_ptr(),
            self.inplace_config().as_ptr(),
            self.inplace_proxy()?,
            &mut out_shape,
            &mut out_shape_dim,
            &mut out_result
        ))?;
        Ok(predict_output_slices(out_shape, out_shape_dim, out_result))
    }

    /// Predict directly from a sparse CSR matrix without constructing a
    /// [`DMatrix`] first (XGBoost "inplace" prediction), the sparse counterpart
    /// of [`predict_from_dense`](Self::predict_from_dense).
    ///
    /// Uses standard CSR representation where the column indices for row _i_ are
    /// stored in `indices[indptr[i]:indptr[i+1]]` with their values at the same
    /// positions in `data`. `num_cols` is the number of features and must match
    /// the model. Returns the prediction data and its shape.
    ///
    /// For small batches, see the latency note on
    /// [`predict_from_dense`](Self::predict_from_dense): setting the booster's
    /// `nthread` parameter to 1 is much faster below ~1000 rows.
    ///
    /// Note: if the booster is configured for a CUDA device, XGBoost falls back
    /// to the DMatrix path with a performance warning (irrelevant for CPU boosters).
    pub fn predict_from_csr(
        &self,
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
    ) -> XGBResult<(Vec<f32>, Vec<u64>)> {
        let (data, shape) = self.inplace_predict_csr_raw(indptr, indices, data, num_cols)?;
        Ok((data.to_vec(), shape.to_vec()))
    }

    /// Like [`predict_from_csr`](Self::predict_from_csr), but writes the
    /// predictions into `out` (cleared first) instead of returning a fresh `Vec`;
    /// see [`predict_from_dense_into`](Self::predict_from_dense_into) for the
    /// buffer-reuse rationale and output layout.
    pub fn predict_from_csr_into(
        &self,
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
        out: &mut Vec<f32>,
    ) -> XGBResult<()> {
        let (data, _shape) = self.inplace_predict_csr_raw(indptr, indices, data, num_cols)?;
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    }

    /// CSR inplace prediction via `XGBoosterPredictFromCSR`, returning slices
    /// borrowed from booster-owned buffers (same validity contract as
    /// [`predict_raw`](Self::predict_raw): valid until the next prediction call).
    fn inplace_predict_csr_raw(
        &self,
        indptr: &[u64],
        indices: &[u64],
        data: &[f32],
        num_cols: usize,
    ) -> XGBResult<(&[f32], &[u64])> {
        assert_eq!(indices.len(), data.len());
        let mut indptr_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let mut indices_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let mut data_buf = CBuf::<ARRAY_INTERFACE_BUF>::new();
        let indptr_cstr = write_interface_1d(&mut indptr_buf, indptr.as_ptr() as usize, indptr.len(), "<u8");
        let indices_cstr = write_interface_1d(&mut indices_buf, indices.as_ptr() as usize, indices.len(), "<u8");
        let data_cstr = write_interface_1d(&mut data_buf, data.as_ptr() as usize, data.len(), "<f4");

        let mut out_shape = ptr::null();
        let mut out_shape_dim = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredictFromCSR(
            self.handle,
            indptr_cstr.as_ptr(),
            indices_cstr.as_ptr(),
            data_cstr.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            self.inplace_config().as_ptr(),
            self.inplace_proxy()?,
            &mut out_shape,
            &mut out_shape_dim,
            &mut out_result
        ))?;
        Ok(predict_output_slices(out_shape, out_shape_dim, out_result))
    }

    /// Predict results for given data.
    ///
    /// Returns an array containing one entry per row in the given data for
    /// single-output models. Multi-output models (multiclass softprob,
    /// multi-quantile/expectile, multi-target) return `num_rows * n_outputs`
    /// entries in row-major order — e.g. one column per alpha for
    /// [`RegQuantile`](crate::parameters::learning::Objective::RegQuantile).
    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        Ok(self.predict_borrowed(dmat)?.to_vec())
    }

    /// Like [`predict`](Self::predict), but writes the predictions into `out`
    /// (cleared first) instead of returning a fresh `Vec`.
    ///
    /// The DMatrix-path counterpart of
    /// [`predict_from_dense_into`](Self::predict_from_dense_into): reusing one
    /// buffer across calls makes a steady-state batch-scoring loop
    /// allocation-free on the Rust side. The output length is
    /// `num_rows * n_outputs` (`n_outputs` is 1 for single-output regression/
    /// binary models, the class count for softprob, the alpha count for
    /// multi-quantile/expectile).
    pub fn predict_into(&self, dmat: &DMatrix, out: &mut Vec<f32>) -> XGBResult<()> {
        let data = self.predict_borrowed(dmat)?;
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    }

    /// Predict results for given data, borrowing XGBoost's internal output buffer.
    ///
    /// The returned slice points into a buffer owned by this booster and is only
    /// valid until the next prediction call on it. Callers must copy the data if
    /// they need it to outlive subsequent booster calls (see [`predict`]).
    fn predict_borrowed(&self, dmat: &DMatrix) -> XGBResult<&[f32]> {
        Ok(self.predict_raw(dmat, predict_config::NORMAL)?.0)
    }

    /// Predict margin for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    pub fn predict_margin(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        Ok(self.predict_margin_borrowed(dmat)?.to_vec())
    }

    /// Like [`predict_margin`](Self::predict_margin), but writes the margins
    /// into `out` (cleared first); see [`predict_into`](Self::predict_into)
    /// for the buffer-reuse rationale.
    pub fn predict_margin_into(&self, dmat: &DMatrix, out: &mut Vec<f32>) -> XGBResult<()> {
        let data = self.predict_margin_borrowed(dmat)?;
        out.clear();
        out.extend_from_slice(data);
        Ok(())
    }

    /// Margin prediction borrowing XGBoost's internal output buffer; same
    /// validity contract as [`predict_borrowed`](Self::predict_borrowed).
    fn predict_margin_borrowed(&self, dmat: &DMatrix) -> XGBResult<&[f32]> {
        Ok(self.predict_raw(dmat, predict_config::MARGIN)?.0)
    }

    /// Get predicted leaf index for each sample in given data.
    ///
    /// Returns an array of shape (number of samples, number of trees) as tuple of (data, num_rows).
    ///
    /// Note: the leaf index of a tree is unique per tree, so e.g. leaf 1 could be found in both tree 1 and tree 0.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let data = self.predict_raw(dmat, predict_config::LEAF)?.0.to_vec();
        let num_rows = dmat.num_rows();
        // 0-row matrices (e.g. from `slice(&[])`) must not panic on division.
        let num_cols = data.len().checked_div(num_rows).unwrap_or(0);
        Ok((data, (num_rows, num_cols)))
    }

    /// Get feature contributions (SHAP values) for each prediction.
    ///
    /// The sum of all feature contributions is equal to the run untransformed margin value of the
    /// prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1) as a tuple of
    /// (data, num_rows). The final column contains the bias term.
    pub fn predict_contributions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let data = self.predict_raw(dmat, predict_config::CONTRIBUTIONS)?.0.to_vec();
        let num_rows = dmat.num_rows();
        // 0-row matrices (e.g. from `slice(&[])`) must not panic on division.
        let num_cols = data.len().checked_div(num_rows).unwrap_or(0);
        Ok((data, (num_rows, num_cols)))
    }

    /// Get SHAP interaction values for each pair of features for each prediction.
    ///
    /// The sum of each row (or column) of the interaction values equals the corresponding SHAP
    /// value (from `predict_contributions`), and the sum of the entire matrix equals the raw
    /// untransformed margin value of the prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1, number of features + 1).
    /// The final row and column contain the bias terms.
    pub fn predict_interactions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize, usize))> {
        let data = self.predict_raw(dmat, predict_config::INTERACTIONS)?.0.to_vec();
        let num_rows = dmat.num_rows();

        // 0-row matrices (e.g. from `slice(&[])`) must not panic on division.
        let per_row = data.len().checked_div(num_rows).unwrap_or(0);
        let dim = (per_row as f64).sqrt() as usize;
        Ok((data, (num_rows, dim, dim)))
    }

    /// Get a dump of this model as a string.
    ///
    /// * `with_statistics` - whether to include statistics in output dump
    /// * `feature_map` - if given, map feature IDs to feature names from given map
    pub fn dump_model(&self, with_statistics: bool, feature_map: Option<&FeatureMap>) -> XGBResult<String> {
        if let Some(fmap) = feature_map {
            let tmp_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let file_path = tmp_dir.path().join("fmap.json");
            let file: File = match File::create(&file_path) {
                Ok(f) => f,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let mut writer = BufWriter::new(file);
            for (feature_num, (feature_name, feature_type)) in &fmap.0 {
                writeln!(writer, "{}\t{}\t{}", feature_num, feature_name, feature_type).unwrap();
            }
            writer.flush().unwrap();

            self.dump_model_fmap(with_statistics, Some(&file_path))
        } else {
            self.dump_model_fmap(with_statistics, None)
        }
    }

    pub fn dump_model_vec(&self, with_statistics: bool) -> XGBResult<Vec<String>> {
        self.dump_model_fmap_vec(with_statistics, None)
    }

    fn dump_model_fmap(&self, with_statistics: bool, feature_map_path: Option<&PathBuf>) -> XGBResult<String> {
        Ok(self.dump_model_fmap_vec(with_statistics, feature_map_path)?.join("\n"))
    }

    fn dump_model_fmap_vec(&self, with_statistics: bool, feature_map_path: Option<&PathBuf>) -> XGBResult<Vec<String>> {
        let fmap = if let Some(path) = feature_map_path {
            crate::path_to_c_str(path)
        } else {
            ffi::CString::new("").unwrap()
        };
        let format = ffi::CString::new("text").unwrap();
        let mut out_len = 0;
        let mut out_dump_array = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterDumpModelEx(
            self.handle,
            fmap.as_ptr(),
            with_statistics as i32,
            format.as_ptr(),
            &mut out_len,
            &mut out_dump_array
        ))?;

        if out_len > 0 {
            let out_ptr_slice = unsafe { slice::from_raw_parts(out_dump_array, out_len as usize) };
            let out_vec: Vec<String> = out_ptr_slice
                .iter()
                .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
                .collect();

            assert_eq!(out_len as usize, out_vec.len());
            Ok(out_vec)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))
    }

    fn parse_eval_string(eval: &str, evnames: &[&str]) -> IndexMap<String, IndexMap<String, f32>> {
        let mut result: IndexMap<String, IndexMap<String, f32>> = IndexMap::new();

        debug!("Parsing evaluation line: {}", &eval);
        for part in eval.split('\t').skip(1) {
            for evname in evnames {
                // Entries are `<name>-<metric>:<score>`; requiring the `-`
                // separator stops a name that is a prefix of another (e.g.
                // "val" / "val2") from also matching the longer name's entries.
                if part.starts_with(evname) && part[evname.len()..].starts_with('-') {
                    let metric_parts: Vec<&str> = part[evname.len() + 1..].split(':').collect();
                    assert_eq!(metric_parts.len(), 2);
                    let metric = metric_parts[0];
                    let score = metric_parts[1]
                        .parse::<f32>()
                        .unwrap_or_else(|_| panic!("Unable to parse XGBoost metrics output: {}", eval));

                    let metric_map = result.entry(evname.to_string()).or_default();
                    metric_map.insert(metric.to_owned(), score);
                }
            }
        }

        debug!("result: {:?}", &result);
        result
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        if let Some(&proxy) = self.inplace_proxy.get() {
            xgb_call!(xgboost_sys::XGDMatrixFree(proxy)).unwrap();
        }
        xgb_call!(xgboost_sys::XGBoosterFree(self.handle)).unwrap();
    }
}

/// Maps a feature index to a name and type, used when dumping models as text.
///
/// See [dump_model](struct.Booster.html#method.dump_model) for usage.
pub struct FeatureMap(BTreeMap<u32, (String, FeatureType)>);

impl FeatureMap {
    /// Read a `FeatureMap` from a file at given path.
    ///
    /// File should contain one feature definition per line, and be of the form:
    /// ```text
    /// <number>\t<name>\t<type>\n
    /// ```
    ///
    /// Type should be one of:
    /// * `i` - binary feature
    /// * `q` - quantitative feature
    /// * `int` - integer features
    ///
    /// E.g.:
    /// ```text
    /// 0   age int
    /// 1   is-parent?=yes  i
    /// 2   is-parent?=no   i
    /// 3   income  int
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<FeatureMap> {
        let file = File::open(path)?;
        let mut features: FeatureMap = FeatureMap(BTreeMap::new());

        for (i, line) in BufReader::new(&file).lines().enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                let msg = format!(
                    "Unable to parse features from line {}, expected 3 tab separated values",
                    i + 1
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            assert_eq!(parts.len(), 3);
            let feature_num: u32 = match parts[0].parse() {
                Ok(num) => num,
                Err(err) => {
                    let msg = format!(
                        "Unable to parse features from line {}, could not parse feature number: {}",
                        i + 1,
                        err
                    );
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };

            let feature_name = &parts[1];
            let feature_type = match FeatureType::from_str(parts[2]) {
                Ok(feature_type) => feature_type,
                Err(msg) => {
                    let msg = format!("Unable to parse features from line {}: {}", i + 1, msg);
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };
            features.0.insert(feature_num, (feature_name.to_string(), feature_type));
        }
        Ok(features)
    }
}

/// Indicates the type of a feature, used when dumping models as text.
pub enum FeatureType {
    /// Binary indicator feature.
    Binary,

    /// Quantitative feature (e.g. age, time, etc.), can be missing.
    Quantitative,

    /// Integer feature (when hinted, decision boundary will be integer).
    Integer,
}

impl FromStr for FeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i" => Ok(FeatureType::Binary),
            "q" => Ok(FeatureType::Quantitative),
            "int" => Ok(FeatureType::Integer),
            _ => Err(format!(
                "unrecognised feature type '{}', must be one of: 'i', 'q', 'int'",
                s
            )),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            FeatureType::Binary => "i",
            FeatureType::Quantitative => "q",
            FeatureType::Integer => "int",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::{self, learning, tree};

    fn read_train_matrix() -> XGBResult<DMatrix> {
        DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#)
    }

    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat]).expect("Creating Booster failed")
    }

    #[test]
    fn set_booster_param() {
        let mut booster = load_test_booster();
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn get_set_attr() {
        let mut booster = load_test_booster();
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn save_and_load_from_buffer() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let mut booster = Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test-xgboost-model");
        booster.save(&path).expect("saving booster");
        drop(booster);
        let bytes = std::fs::read(&path).expect("read saved booster file");
        let booster = Booster::load_buffer(&bytes[..]).expect("load booster from buffer");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let in_memory_bytes = booster.save_buffer(true).unwrap();
        let booster =
            Booster::load_buffer(&in_memory_bytes[..] as &[u8]).expect("load booster from memory only buffer");
        let attr = booster.get_attribute("foo").expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = load_test_booster();
        let attrs = booster.get_attribute_names().expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster.set_attribute("foo", "bar").expect("Setting attribute failed");
        booster
            .set_attribute("another", "another")
            .expect("Setting attribute failed");
        booster.set_attribute("4", "4").expect("Setting attribute failed");
        booster
            .set_attribute("an even longer attribute name?", "")
            .expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort();
        let mut attrs = booster.get_attribute_names().expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn get_set_feature_names() {
        let booster = load_test_booster();
        let attrs = booster.get_feature_names().expect("Getting features failed");
        assert_eq!(attrs, Vec::<String>::new());
        let mut expected = vec!["foo", "another", "4", "an even longer features name?"];
        expected.sort();
        booster.set_feature_names(&expected).expect("Setting features failed");
        let mut attrs = booster.get_feature_names().expect("Getting features failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn predict() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            // Pinned: the hardcoded expected values below were produced with a
            // fixed 0.5 intercept (the pre-2.0 default), not the estimated one.
            .base_score(0.5)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::MAPCutNegative(4),
                learning::EvaluationMetric::LogLoss,
                learning::EvaluationMetric::BinaryError,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let train_metrics = booster.evaluate(&dmat_train).unwrap();
        assert_eq!(*train_metrics.get("logloss").unwrap(), 0.006634271);
        assert_eq!(*train_metrics.get("map@4-").unwrap(), 1.0);

        let test_metrics = booster.evaluate(&dmat_test).unwrap();
        let diff = *test_metrics.get("logloss").unwrap() - 0.0069199526;
        assert_eq!(diff < 0.000001, diff > -0.000001);
        assert_eq!(*test_metrics.get("map@4-").unwrap(), 1.0);

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());

        // first 10 predictions
        let expected_start = [
            0.0050151693,
            0.9884467,
            0.0050151693,
            0.0050151693,
            0.026636455,
            0.11789363,
            0.9884467,
            0.01231471,
            0.9884467,
            0.00013656063,
        ];

        // last 10 predictions
        let expected_end = [
            0.002520344,
            0.00060917926,
            0.99881005,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.9981102,
            0.002855195,
            0.9981102,
        ];
        let eps = 1e-6;

        for (pred, expected) in v.iter().zip(&expected_start) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }

        for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
    }

    /// Deterministic regression data with spread: y depends on x plus a
    /// repeating offset, so different quantiles/expectiles of y|x separate.
    fn synthetic_spread(num_rows: usize) -> (Vec<f32>, Vec<f32>) {
        let mut data = Vec::with_capacity(num_rows);
        let mut labels = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let x = (i % 16) as f32;
            data.push(x);
            labels.push(x + (i % 7) as f32); // spread of 0..6 around x
        }
        (data, labels)
    }

    fn train_with_objective(objective: learning::Objective, dmat: &DMatrix, rounds: i32) -> Booster {
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(objective)
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[dmat]).unwrap();
        for i in 0..rounds {
            booster.update(dmat, i).unwrap();
        }
        booster
    }

    /// Multi-quantile / multi-expectile objectives: one output column per
    /// alpha, and the columns must be ordered (mean of the 0.1-quantile
    /// predictions below the mean of the 0.9-quantile predictions).
    #[test]
    fn quantile_and_expectile_objectives() {
        let num_rows = 256;
        let (data, labels) = synthetic_spread(num_rows);
        let mut dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        dmat.set_labels(&labels).unwrap();

        let booster = train_with_objective(learning::Objective::RegQuantile(vec![0.1, 0.5, 0.9]), &dmat, 20);
        let preds = booster.predict(&dmat).unwrap();
        assert_eq!(preds.len(), num_rows * 3, "one prediction column per quantile");
        let col_mean = |c: usize| preds.iter().skip(c).step_by(3).sum::<f32>() / num_rows as f32;
        assert!(col_mean(0) < col_mean(1), "q0.1 mean must be below q0.5 mean");
        assert!(col_mean(1) < col_mean(2), "q0.5 mean must be below q0.9 mean");

        // Expectile regression is new in XGBoost 3.3.
        let booster = train_with_objective(learning::Objective::RegExpectile(vec![0.25, 0.75]), &dmat, 20);
        let preds = booster.predict(&dmat).unwrap();
        assert_eq!(preds.len(), num_rows * 2, "one prediction column per expectile");
        let col_mean = |c: usize| preds.iter().skip(c).step_by(2).sum::<f32>() / num_rows as f32;
        assert!(col_mean(0) < col_mean(1), "e0.25 mean must be below e0.75 mean");

        // Builder validation: empty and out-of-range alphas are rejected.
        assert!(
            learning::LearningTaskParametersBuilder::default()
                .objective(learning::Objective::RegQuantile(vec![]))
                .build()
                .is_err()
        );
        assert!(
            learning::LearningTaskParametersBuilder::default()
                .objective(learning::Objective::RegExpectile(vec![1.5]))
                .build()
                .is_err()
        );
    }

    /// Categorical feature support: marking a column "c" makes XGBoost use
    /// categorical splits for it. y depends only on whether the categorical
    /// feature equals 2, which a categorical split can express exactly.
    #[test]
    fn categorical_feature_training() {
        let num_rows = 256;
        let mut data = Vec::with_capacity(num_rows * 2);
        let mut labels = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let cat = (i % 4) as f32; // categories 0..3
            data.push(cat);
            data.push((i % 10) as f32 / 10.0); // irrelevant quantitative feature
            labels.push(if cat == 2.0 { 1.0 } else { 0.0 });
        }
        let mut dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        dmat.set_labels(&labels).unwrap();
        dmat.set_feature_types(&["c", "q"]).unwrap();
        assert_eq!(dmat.get_feature_types().unwrap(), vec!["c", "q"]);

        let booster = train_with_objective(learning::Objective::RegLinear, &dmat, 20);
        let preds = booster.predict(&dmat).unwrap();
        assert_eq!(preds.len(), num_rows);
        let (mut cat2_sum, mut rest_sum, mut cat2_n, mut rest_n) = (0f32, 0f32, 0, 0);
        for (i, p) in preds.iter().enumerate() {
            if i % 4 == 2 {
                cat2_sum += p;
                cat2_n += 1;
            } else {
                rest_sum += p;
                rest_n += 1;
            }
        }
        let (cat2_mean, rest_mean) = (cat2_sum / cat2_n as f32, rest_sum / rest_n as f32);
        assert!(
            cat2_mean > 0.9 && rest_mean < 0.1,
            "categorical split must separate cat==2 (mean {}) from the rest (mean {})",
            cat2_mean,
            rest_mean
        );
    }

    /// XGBoost 3.3 deprecated `booster=dart` (remapped internally to gbtree
    /// with dropout params) and `booster=gblinear` (removal planned). Both
    /// must keep training and predicting until upstream actually removes
    /// them; this failing on a future submodule bump means the wrapper's
    /// `BoosterType` surface needs migrating, not just a re-pin.
    #[test]
    fn deprecated_booster_types_still_train() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();

        for booster_type in [
            parameters::BoosterType::Dart(Default::default()),
            parameters::BoosterType::Linear(Default::default()),
        ] {
            let params = parameters::BoosterParametersBuilder::default()
                .booster_type(booster_type)
                .verbose(false)
                .build()
                .unwrap();
            let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train]).unwrap();
            for i in 0..3 {
                booster
                    .update(&dmat_train, i)
                    .expect("deprecated booster type failed to train");
            }
            let preds = booster.predict(&dmat_train).unwrap();
            assert_eq!(preds.len(), dmat_train.num_rows());
            assert!(preds.iter().all(|p| p.is_finite()));
        }
    }

    #[test]
    fn predict_into_matches_predict() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();
        let mut booster = Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        for i in 0..5 {
            booster.update(&dmat_train, i).unwrap();
        }

        let mut out = Vec::new();
        booster.predict_into(&dmat_test, &mut out).unwrap();
        assert_eq!(out, booster.predict(&dmat_test).unwrap());

        // Reuse must clear stale contents from the previous call.
        let single = dmat_test.slice(&[0]).unwrap();
        booster.predict_into(&single, &mut out).unwrap();
        assert_eq!(out.len(), 1);

        booster.predict_margin_into(&dmat_test, &mut out).unwrap();
        assert_eq!(out, booster.predict_margin(&dmat_test).unwrap());
    }

    /// The distributional-regression training flow on a `QuantileDMatrix`:
    /// multi-target booster, start values injected as a post-construction base
    /// margin, custom gradients fed through the public `boost`, per-round
    /// cached `predict`. Verifies (a) predict on a quantile matrix includes the
    /// base margin (0-tree booster returns exactly margin + base_score), and
    /// (b) `boost` + `predict` round-trips build a real multi-target model.
    #[test]
    fn quantile_dmatrix_base_margin_boost_multi_target() {
        let num_rows = 60;
        let num_cols = 5;
        let n_targets = 2;
        let mut data = vec![0f32; num_rows * num_cols];
        let mut labels = vec![0f32; num_rows];
        for i in 0..num_rows {
            for j in 0..num_cols {
                data[i * num_cols + j] = ((i * 7 + j * 3) % 13) as f32;
            }
            labels[i] = (i % 4) as f32;
        }

        let mut qdm = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 256).unwrap();
        // Row-major [num_rows, n_targets] margin, constant per target.
        let margin: Vec<f32> = (0..num_rows).flat_map(|_| [0.5f32, -1.5f32]).collect();
        qdm.set_base_margin(&margin).unwrap();

        let mut booster = Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&qdm]).unwrap();
        booster.set_param("num_target", "2").unwrap();
        booster.set_param("base_score", "0.0").unwrap();
        booster.set_param("tree_method", "hist").unwrap();
        booster.set_param("disable_default_eval_metric", "true").unwrap();

        // 0 trees: predict must be exactly the base margin.
        let preds = booster.predict(&qdm).unwrap();
        assert_eq!(preds.len(), num_rows * n_targets);
        for i in 0..num_rows {
            assert_eq!(preds[i * n_targets], 0.5);
            assert_eq!(preds[i * n_targets + 1], -1.5);
        }

        // A few rounds of custom gradients on the carried margin (squared
        // error per target against labels / labels*2).
        let mut preds = preds;
        for round in 0..4 {
            let mut grad = vec![0f32; num_rows * n_targets];
            let hess = vec![1f32; num_rows * n_targets];
            for i in 0..num_rows {
                grad[i * n_targets] = preds[i * n_targets] - labels[i];
                grad[i * n_targets + 1] = preds[i * n_targets + 1] - 2.0 * labels[i];
            }
            booster.boost(&qdm, round, &grad, &hess).unwrap();
            preds = booster.predict(&qdm).unwrap();
        }

        // Trees were actually built and moved predictions toward the targets.
        let mse_before: f32 = (0..num_rows).map(|i| (0.5 - labels[i]).powi(2)).sum::<f32>() / num_rows as f32;
        let mse_after: f32 = (0..num_rows)
            .map(|i| (preds[i * n_targets] - labels[i]).powi(2))
            .sum::<f32>()
            / num_rows as f32;
        assert!(
            mse_after < mse_before,
            "boost rounds must reduce target-0 MSE ({mse_after} !< {mse_before})"
        );
    }

    /// `Booster` and `DMatrix` are `Send`: a model loaded on one thread can be
    /// moved to (used on, and dropped on) another, the pattern serving thread
    /// pools rely on. Predictions must be identical across threads.
    #[test]
    fn booster_and_dmatrix_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Booster>();
        assert_send::<DMatrix>();

        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();
        let mut booster = Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        for i in 0..5 {
            booster.update(&dmat_train, i).unwrap();
        }
        let expected = booster.predict(&dmat_test).unwrap();
        // Exercise the inplace path on this thread first so the cached proxy
        // DMatrix also crosses the thread boundary below.
        let row = vec![0.0f32; dmat_test.num_cols()];
        booster.predict_from_dense(&row, 1).unwrap();

        let from_thread = std::thread::spawn(move || {
            let preds = booster.predict(&dmat_test).unwrap();
            booster.predict_from_dense(&row, 1).unwrap();
            preds
            // booster and dmat_test drop on this thread
        })
        .join()
        .unwrap();
        assert_eq!(expected, from_thread);
    }

    #[test]
    fn predict_matrix() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            // Pinned: the hardcoded expected values below were produced with a
            // fixed 0.5 intercept (the pre-2.0 default), not the estimated one.
            .base_score(0.5)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::MAPCutNegative(4),
                learning::EvaluationMetric::LogLoss,
                learning::EvaluationMetric::BinaryError,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let train_metrics = booster.evaluate(&dmat_train).unwrap();
        assert_eq!(*train_metrics.get("logloss").unwrap(), 0.006634271);
        assert_eq!(*train_metrics.get("map@4-").unwrap(), 1.0);

        let test_metrics = booster.evaluate(&dmat_test).unwrap();
        let diff = *test_metrics.get("logloss").unwrap() - 0.0069199526;
        assert_eq!(diff < 0.000001, diff > -0.000001);
        assert_eq!(*test_metrics.get("map@4-").unwrap(), 1.0);

        let single_matrix = dmat_test.slice(&[0]).unwrap();
        let (v, shape) = booster
            .predict_matrix(&single_matrix, &PredictConfig::default().as_json())
            .unwrap();
        assert_eq!(shape, vec![1]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 0.0050151693);
        let cfg = PredictConfig::default();
        let (v, shape) = booster.predict_matrix(&dmat_test, &cfg.as_json()).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());
        assert_eq!(shape, vec![1611]);

        // first 10 predictions
        let expected_start = [
            0.0050151693,
            0.9884467,
            0.0050151693,
            0.0050151693,
            0.026636455,
            0.11789363,
            0.9884467,
            0.01231471,
            0.9884467,
            0.00013656063,
        ];

        // last 10 predictions
        let expected_end = [
            0.002520344,
            0.00060917926,
            0.99881005,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.00060917926,
            0.9981102,
            0.002855195,
            0.9981102,
        ];
        let eps = 1e-6;

        for (pred, expected) in v.iter().zip(&expected_start) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }

        for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
    }

    #[test]
    fn predict_leaf() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![learning::EvaluationMetric::LogLoss]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 15;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_leaf(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        assert_eq!(shape, (num_samples, num_rounds as usize));

        // 0-row matrices must not panic on the shape division; whether the C
        // API returns Ok or Err for them is its business.
        let empty = dmat_test.slice(&[]).unwrap();
        if let Ok((_, shape)) = booster.predict_leaf(&empty) {
            assert_eq!(shape.0, 0);
        }
        if let Ok((_, shape)) = booster.predict_contributions(&empty) {
            assert_eq!(shape.0, 0);
        }
        if let Ok((_, shape)) = booster.predict_interactions(&empty) {
            assert_eq!(shape.0, 0);
        }
    }

    #[test]
    fn predict_contributions() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![learning::EvaluationMetric::LogLoss]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_contributions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1));
    }

    #[test]
    fn predict_interactions() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();
        let dmat_test =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![learning::EvaluationMetric::LogLoss]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster = Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_interactions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1, num_features + 1));
    }

    /// Build a deterministic dense binary-classification dataset.
    fn synthetic_dense(num_rows: usize, num_cols: usize) -> (Vec<f32>, Vec<f32>) {
        let mut data = vec![0f32; num_rows * num_cols];
        let mut labels = vec![0f32; num_rows];
        let mut seed = 1u64;
        for i in 0..num_rows {
            let mut acc = 0f32;
            for j in 0..num_cols {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let v = (seed % 1000) as f32 / 1000.0;
                data[i * num_cols + j] = v;
                acc += v;
            }
            labels[i] = if acc > num_cols as f32 / 2.0 { 1.0 } else { 0.0 };
        }
        (data, labels)
    }

    fn hist_binary_params(max_bin: u32) -> BoosterParameters {
        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .tree_method(tree::TreeMethod::Hist)
            .max_depth(4)
            .max_bin(max_bin)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            // Pinned so `update_custom_matches_builtin_objective` compares like
            // with like: XGBoosterTrainOneIter (custom objective) never runs
            // the automatic intercept estimation that UpdateOneIter does.
            .base_score(0.5)
            .build()
            .unwrap();
        parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap()
    }

    /// Build a deterministic sparse CSR binary-classification dataset.
    fn synthetic_csr(num_rows: usize, num_cols: usize) -> (Vec<u64>, Vec<u64>, Vec<f32>, Vec<f32>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut labels = vec![0f32; num_rows];
        let mut seed = 7u64;
        let mut nnz = 0u64;
        for i in 0..num_rows {
            let mut acc = 0f32;
            for j in 0..num_cols {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // ~40% density
                if seed % 10 < 4 {
                    let v = (seed % 1000) as f32 / 1000.0;
                    indices.push(j as u64);
                    values.push(v);
                    acc += v;
                    nnz += 1;
                }
            }
            indptr.push(nnz);
            labels[i] = if acc > 1.0 { 1.0 } else { 0.0 };
        }
        (indptr, indices, values, labels)
    }

    #[test]
    fn predict_from_csr_matches_dmatrix() {
        let (num_rows, num_cols) = (256, 8);
        let (indptr, indices, values, labels) = synthetic_csr(num_rows, num_cols);

        let mut dm = DMatrix::from_csr(&indptr, &indices, &values, Some(num_cols)).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let via_dmatrix = bst.predict(&dm).unwrap();
        let (via_inplace, shape) = bst.predict_from_csr(&indptr, &indices, &values, num_cols).unwrap();

        assert_eq!(shape, vec![num_rows as u64]);
        assert_eq!(via_dmatrix.len(), via_inplace.len());
        for (a, b) in via_dmatrix.iter().zip(&via_inplace) {
            assert!((a - b).abs() < 1e-6, "inplace CSR predict mismatch: {} vs {}", a, b);
        }

        // Second call goes through the cached proxy DMatrix; results must not change.
        let (again, shape_again) = bst.predict_from_csr(&indptr, &indices, &values, num_cols).unwrap();
        assert_eq!(shape_again, shape);
        assert_eq!(again, via_inplace);
    }

    #[test]
    fn quantile_csr_matches_csr_training() {
        let (num_rows, num_cols) = (512, 8);
        let (indptr, indices, values, labels) = synthetic_csr(num_rows, num_cols);
        let params = hist_binary_params(256);

        // Regular CSR DMatrix + hist.
        let mut dm = DMatrix::from_csr(&indptr, &indices, &values, Some(num_cols)).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst_dm = Booster::new_with_cached_dmats(&params, &[&dm]).unwrap();
        for i in 0..10 {
            bst_dm.update(&dm, i).unwrap();
        }
        let pred_dm = bst_dm.predict(&dm).unwrap();

        // QuantileDMatrix built from the same CSR data.
        let qdm = DMatrix::from_csr_quantile(&indptr, &indices, &values, num_cols, Some(&labels), 256).unwrap();
        assert_eq!(qdm.shape(), (num_rows, num_cols));
        assert_eq!(qdm.get_labels().unwrap(), &labels[..]);
        let mut bst_q = Booster::new_with_cached_dmats(&params, &[&qdm]).unwrap();
        for i in 0..10 {
            bst_q.update(&qdm, i).unwrap();
        }
        let pred_q = bst_q.predict(&qdm).unwrap();

        assert_eq!(pred_dm.len(), pred_q.len());
        for (a, b) in pred_dm.iter().zip(&pred_q) {
            assert!((a - b).abs() < 1e-4, "quantile vs CSR pred mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn serialize_unserialize_roundtrip() {
        let (num_rows, num_cols) = (256, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let before = bst.predict(&dm).unwrap();
        let snapshot = bst.serialize_to_buffer().unwrap();
        let restored = Booster::unserialize_from_buffer(&snapshot).unwrap();
        let after = restored.predict(&dm).unwrap();
        assert_eq!(before, after, "snapshot restore must reproduce predictions exactly");
    }

    #[test]
    fn reset_keeps_model() {
        let (num_rows, num_cols) = (256, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let before = bst.predict(&dm).unwrap();
        bst.reset().unwrap();
        let after = bst.predict(&dm).unwrap();
        assert_eq!(before, after, "reset must not change the trained model");
    }

    #[test]
    fn predict_from_dense_matches_dmatrix() {
        let (num_rows, num_cols) = (256, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let via_dmatrix = bst.predict(&dm).unwrap();
        let (via_inplace, shape) = bst.predict_from_dense(&data, num_rows).unwrap();

        assert_eq!(shape, vec![num_rows as u64]);
        assert_eq!(via_dmatrix.len(), via_inplace.len());
        for (a, b) in via_dmatrix.iter().zip(&via_inplace) {
            assert!((a - b).abs() < 1e-6, "inplace predict mismatch: {} vs {}", a, b);
        }

        // Second call goes through the cached proxy DMatrix; results must not change.
        let (again, shape_again) = bst.predict_from_dense(&data, num_rows).unwrap();
        assert_eq!(shape_again, shape);
        assert_eq!(again, via_inplace);
    }

    #[test]
    fn inplace_predict_missing_sentinel_matches_nan() {
        let (num_rows, num_cols) = (256, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        // A row with a genuinely missing feature, encoded both ways.
        let mut row_nan = data[..num_cols].to_vec();
        row_nan[3] = f32::NAN;
        let mut row_sentinel = data[..num_cols].to_vec();
        row_sentinel[3] = -999.0;

        let (expected, _) = bst.predict_from_dense(&row_nan, 1).unwrap();

        // Without the custom missing value, -999 is an ordinary feature value
        // and (for this model/row) should not match the NaN prediction. Guard
        // against a vacuous test where the tree never splits on feature 3.
        let (as_value, _) = bst.predict_from_dense(&row_sentinel, 1).unwrap();

        bst.set_inplace_predict_missing(-999.0).unwrap();
        let (as_missing, _) = bst.predict_from_dense(&row_sentinel, 1).unwrap();
        assert_eq!(
            as_missing, expected,
            "sentinel-encoded missing must predict identically to NaN-encoded"
        );
        if as_value != expected {
            // The interesting case: the sentinel actually changed the path,
            // proving the config took effect rather than being ignored.
            assert_ne!(as_missing, as_value);
        }

        // NaN restores the default config (and the static-config path).
        bst.set_inplace_predict_missing(f32::NAN).unwrap();
        let (restored, _) = bst.predict_from_dense(&row_nan, 1).unwrap();
        assert_eq!(restored, expected);

        // Infinity is rejected, matching the DMatrix constructors.
        assert!(bst.set_inplace_predict_missing(f32::INFINITY).is_err());
    }

    #[test]
    fn predict_from_dense_into_matches_and_reuses_buffer() {
        let (num_rows, num_cols) = (64, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let (owned, _shape) = bst.predict_from_dense(&data, num_rows).unwrap();

        let mut out = Vec::new();
        bst.predict_from_dense_into(&data, num_rows, &mut out).unwrap();
        assert_eq!(out, owned);

        // A smaller second batch must overwrite (not append) and reuse capacity.
        let half_rows = num_rows / 2;
        let cap_before = out.capacity();
        bst.predict_from_dense_into(&data[..half_rows * num_cols], half_rows, &mut out)
            .unwrap();
        assert_eq!(out.len(), half_rows);
        assert_eq!(out.capacity(), cap_before, "buffer must be reused, not reallocated");
        assert_eq!(
            out[..],
            owned[..half_rows],
            "row predictions are independent of batch size"
        );
    }

    #[test]
    fn predict_from_csr_into_matches_and_reuses_buffer() {
        let (num_rows, num_cols) = (64, 8);
        let (indptr, indices, values, labels) = synthetic_csr(num_rows, num_cols);

        let mut dm = DMatrix::from_csr(&indptr, &indices, &values, Some(num_cols)).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst = Booster::new_with_cached_dmats(&hist_binary_params(256), &[&dm]).unwrap();
        for i in 0..10 {
            bst.update(&dm, i).unwrap();
        }

        let (owned, _shape) = bst.predict_from_csr(&indptr, &indices, &values, num_cols).unwrap();

        let mut out = Vec::new();
        bst.predict_from_csr_into(&indptr, &indices, &values, num_cols, &mut out)
            .unwrap();
        assert_eq!(out, owned);

        // CSR prefix (first half of the rows) into the same buffer.
        let half_rows = num_rows / 2;
        let nnz_half = indptr[half_rows] as usize;
        let cap_before = out.capacity();
        bst.predict_from_csr_into(
            &indptr[..=half_rows],
            &indices[..nnz_half],
            &values[..nnz_half],
            num_cols,
            &mut out,
        )
        .unwrap();
        assert_eq!(out.len(), half_rows);
        assert_eq!(out.capacity(), cap_before, "buffer must be reused, not reallocated");
        assert_eq!(
            out[..],
            owned[..half_rows],
            "row predictions are independent of batch size"
        );
    }

    #[test]
    fn update_custom_matches_builtin_objective() {
        // Exercises `boost()` — XGBoosterTrainOneIter with the stack-built
        // gradient/hessian array interfaces — end to end: a hand-rolled
        // binary-logistic objective must reproduce the built-in
        // `binary:logistic` update on the same data.
        fn logistic_obj(preds: &[f32], dtrain: &DMatrix) -> (Vec<f32>, Vec<f32>) {
            let labels = dtrain.get_labels().unwrap();
            let mut grad = Vec::with_capacity(preds.len());
            let mut hess = Vec::with_capacity(preds.len());
            for (&p, &y) in preds.iter().zip(labels) {
                grad.push(p - y);
                hess.push(p * (1.0 - p));
            }
            (grad, hess)
        }

        let (num_rows, num_cols) = (256, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);
        let params = hist_binary_params(256);

        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();

        let mut bst_builtin = Booster::new_with_cached_dmats(&params, &[&dm]).unwrap();
        let mut bst_custom = Booster::new_with_cached_dmats(&params, &[&dm]).unwrap();
        for i in 0..10 {
            bst_builtin.update(&dm, i).unwrap();
            bst_custom.update_custom(&dm, i, logistic_obj).unwrap();
        }

        let pred_builtin = bst_builtin.predict(&dm).unwrap();
        let pred_custom = bst_custom.predict(&dm).unwrap();
        assert_eq!(pred_builtin.len(), pred_custom.len());
        for (a, b) in pred_builtin.iter().zip(&pred_custom) {
            assert!(
                (a - b).abs() < 1e-5,
                "custom objective diverged from builtin: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn quantile_dmatrix_matches_dense_training() {
        let (num_rows, num_cols) = (512, 8);
        let (data, labels) = synthetic_dense(num_rows, num_cols);
        let params = hist_binary_params(256);

        // Regular DMatrix + hist (which bins internally with max_bin=256).
        let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
        dm.set_labels(&labels).unwrap();
        let mut bst_dm = Booster::new_with_cached_dmats(&params, &[&dm]).unwrap();
        for i in 0..10 {
            bst_dm.update(&dm, i).unwrap();
        }
        let pred_dm = bst_dm.predict(&dm).unwrap();

        // QuantileDMatrix carrying the same binning.
        let qdm = DMatrix::from_dense_quantile(&data, num_rows, Some(&labels), 256).unwrap();
        let mut bst_q = Booster::new_with_cached_dmats(&params, &[&qdm]).unwrap();
        for i in 0..10 {
            bst_q.update(&qdm, i).unwrap();
        }
        let pred_q = bst_q.predict(&qdm).unwrap();

        assert_eq!(pred_dm.len(), pred_q.len());
        for (a, b) in pred_dm.iter().zip(&pred_q) {
            assert!((a - b).abs() < 1e-4, "quantile vs dense pred mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn parse_eval_string() {
        let s = "[0]\ttrain-map@4-:0.5\ttrain-logloss:1.0\ttest-map@4-:0.25\ttest-logloss:0.75";
        let mut metrics = IndexMap::new();

        let mut train_metrics = IndexMap::new();
        train_metrics.insert("map@4-".to_owned(), 0.5);
        train_metrics.insert("logloss".to_owned(), 1.0);

        let mut test_metrics = IndexMap::new();
        test_metrics.insert("map@4-".to_owned(), 0.25);
        test_metrics.insert("logloss".to_owned(), 0.75);

        metrics.insert("train".to_owned(), train_metrics);
        metrics.insert("test".to_owned(), test_metrics);
        assert_eq!(Booster::parse_eval_string(s, &["train", "test"]), metrics);
    }

    #[test]
    fn parse_eval_string_prefix_names() {
        // "val" is a prefix of "val2"; entries must not cross-match.
        let s = "[0]\tval-logloss:1.0\tval2-logloss:0.5";
        let metrics = Booster::parse_eval_string(s, &["val", "val2"]);
        assert_eq!(metrics["val"].len(), 1);
        assert_eq!(metrics["val"]["logloss"], 1.0);
        assert_eq!(metrics["val2"].len(), 1);
        assert_eq!(metrics["val2"]["logloss"], 0.5);
    }

    #[test]
    fn predict_from_dense_rejects_bad_shape() {
        let bst = load_test_booster();
        // Length not divisible by num_rows: must error, not silently truncate.
        assert!(bst.predict_from_dense(&[1.0, 2.0, 3.0], 2).is_err());
        // Zero rows: must error, not panic on division by zero.
        assert!(bst.predict_from_dense(&[1.0], 0).is_err());
    }

    #[test]
    fn dump_model() {
        let dmat_train =
            DMatrix::load(r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#).unwrap();

        println!("{:?}", dmat_train.shape());

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            // Pinned: the hardcoded model dump below was produced with a fixed
            // 0.5 intercept (the pre-2.0 default), not the estimated one.
            .base_score(0.5)
            .build()
            .unwrap();
        let booster_params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();

        let training_params = parameters::TrainingParametersBuilder::default()
            .booster_params(booster_params)
            .dtrain(&dmat_train)
            .boost_rounds(10)
            .build()
            .unwrap();
        let booster = Booster::train(&training_params).unwrap();

        assert_eq!(
            booster.dump_model(true, None).unwrap(),
            "0:[f29<2.00001001] yes=1,no=2,missing=2,gain=4000.53101,cover=1628.25
	1:[f109<2.00001001] yes=3,no=4,missing=4,gain=198.173828,cover=703.75
		3:leaf=1.85964918,cover=13.25
		4:leaf=-1.94070864,cover=690.5
	2:[f56<2.00001001] yes=5,no=6,missing=6,gain=1158.21204,cover=924.5
		5:leaf=-1.70044053,cover=112.5
		6:leaf=1.71217716,cover=812

0:[f60<2.00001001] yes=1,no=2,missing=2,gain=832.544983,cover=788.852051
	1:leaf=-6.23624468,cover=20.462389
	2:[f29<2.00001001] yes=3,no=4,missing=4,gain=569.725098,cover=768.389709
		3:leaf=-0.968530357,cover=309.45282
		4:leaf=0.78471756,cover=458.936859

0:[f102<2.00001001] yes=1,no=2,missing=2,gain=368.744568,cover=457.069458
	1:[f111<2.00001001] yes=3,no=4,missing=4,gain=258.184326,cover=236.018005
		3:leaf=-9.421422,cover=2.53038669
		4:leaf=-0.791407049,cover=233.487625
	2:[f67<2.00001001] yes=5,no=6,missing=6,gain=226.336975,cover=221.051468
		5:leaf=5.77228642,cover=8.05200672
		6:leaf=0.658725023,cover=212.999451

0:[f27<2.00001001] yes=1,no=2,missing=2,gain=140.486053,cover=364.119354
	1:leaf=1.07747853,cover=90.0174103
	2:[f39<2.00001001] yes=3,no=4,missing=4,gain=139.860519,cover=274.101959
		3:leaf=-0.877905607,cover=178.241974
		4:leaf=0.614153326,cover=95.8599854

0:[f109<2.00001001] yes=1,no=2,missing=2,gain=112.605019,cover=189.202194
	1:leaf=2.92190909,cover=11.4303684
	2:[f36<2.00001001] yes=3,no=4,missing=4,gain=66.4029999,cover=177.771835
		3:leaf=0.152607277,cover=135.494431
		4:leaf=-1.26934469,cover=42.277401

0:[f23<2.00001001] yes=1,no=2,missing=2,gain=52.5610313,cover=170.612762
	1:[f36<2.00001001] yes=3,no=4,missing=4,gain=12.4420547,cover=19.731596
		3:leaf=-1.02315068,cover=16.0739021
		4:leaf=-3.02413678,cover=3.65769386
	2:[f24<2.00001001] yes=5,no=6,missing=6,gain=67.3869553,cover=150.881165
		5:leaf=-1.53846073,cover=18.9789505
		6:leaf=0.431742132,cover=131.902222

0:[f29<2.00001001] yes=1,no=2,missing=2,gain=66.2389145,cover=142.360611
	1:[f109<2.00001001] yes=3,no=4,missing=4,gain=12.1987419,cover=69.6048737
		3:leaf=0.836115122,cover=3.48375821
		4:leaf=-0.912605286,cover=66.1211166
	2:[f24<2.00001001] yes=5,no=6,missing=6,gain=31.229435,cover=72.7557373
		5:leaf=-1.19710124,cover=8.22473907
		6:leaf=0.777142286,cover=64.5309982

0:[f39<2.00001001] yes=1,no=2,missing=2,gain=20.6531773,cover=79.4027634
	1:[f27<2.00001001] yes=3,no=4,missing=4,gain=22.1144371,cover=44.4738464
		3:leaf=0.890622675,cover=7.49097395
		4:leaf=-0.908311546,cover=36.982872
	2:[f112<2.00001001] yes=5,no=6,missing=6,gain=16.0703697,cover=34.9289207
		5:leaf=1.4361918,cover=9.89693928
		6:leaf=-0.0180106498,cover=25.0319824

0:[f23<2.00001001] yes=1,no=2,missing=2,gain=11.7128553,cover=53.3251991
	1:leaf=-1.01502442,cover=9.02525806
	2:[f102<2.00001001] yes=3,no=4,missing=4,gain=12.5461531,cover=44.299942
		3:leaf=0.56883812,cover=28.5100231
		4:leaf=-0.515293062,cover=15.7899179

0:[f115<2.00001001] yes=1,no=2,missing=2,gain=14.8892794,cover=45.9312019
	1:[f61<2.00001001] yes=3,no=4,missing=4,gain=19.3462334,cover=2.87474418
		3:leaf=-0.609474957,cover=1.53319895
		4:leaf=3.63442755,cover=1.34154534
	2:[f29<2.00001001] yes=5,no=6,missing=6,gain=10.1308861,cover=43.0564575
		5:leaf=-0.734555721,cover=20.7280827
		6:leaf=0.217203051,cover=22.3283749
"
        );
    }
}
