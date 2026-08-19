//! Builders for parameters that control various aspects of training.
//!
//! Configuration is based on the documented
//! [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html), see those for
//! more details.
//!
//! Parameters are generally created through builders that provide sensible defaults, and ensure that
//! any given settings are valid when built.
use std::fmt::{self, Display};

mod booster;
pub mod dart;
pub mod learning;
pub mod linear;
pub mod tree;

pub use self::booster::BoosterType;
use super::DMatrix;
use super::booster::CustomObjective;

/// Device for training and prediction (XGBoost 3.x `device` parameter).
///
/// XGBoost 2.0 removed the `gpu_hist`/`gpu_exact` tree methods; GPU selection
/// happens exclusively through this parameter, combined with a regular tree
/// method (`hist` for GPU training). On large datasets GPU hist training is
/// commonly 5-20x faster than CPU.
///
/// Note: the bundled C++ must be compiled with CUDA support (this crate's
/// `cuda` feature) for the CUDA variants to work — on a CPU-only build XGBoost
/// rejects them at configure time with a clear error.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Device {
    /// CPU (XGBoost default).
    #[default]
    Cpu,
    /// The first visible CUDA device.
    Cuda,
    /// A specific CUDA device by ordinal (`cuda:<n>`).
    CudaOrdinal(u32),
}

impl Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda => write!(f, "cuda"),
            Device::CudaOrdinal(n) => write!(f, "cuda:{}", n),
        }
    }
}

/// Parameters for training boosters.
/// Created using [`BoosterParametersBuilder`](struct.BoosterParametersBuilder.html).
#[derive(Builder, Clone, Default)]
#[builder(default)]
pub struct BoosterParameters {
    /// Type of booster (tree, linear or DART) along with its parameters.
    ///
    /// *default*: [`GbTree`](enum.BoosterType.html#variant.GbTree)
    booster_type: booster::BoosterType,

    /// Configuration for the learning objective.
    pub(crate) learning_params: learning::LearningTaskParameters,

    /// Whether to print XGBoost's C library's messages or not.
    ///
    /// *default*: `false`
    verbose: bool,

    /// Number of parallel threads XGboost will use (if compiled with multiprocessing support).
    ///
    /// *default*: `None` (XGBoost will automatically determing max threads to use)
    threads: Option<u32>,

    /// Device to train and predict on (see [`Device`]).
    ///
    /// *default*: [`Device::Cpu`]
    device: Device,
}

impl BoosterParameters {
    /// Get type of booster (tree, linear or DART) along with its parameters.
    pub fn booster_type(&self) -> &booster::BoosterType {
        &self.booster_type
    }

    /// Set type of booster (tree, linear or DART) along with its parameters.
    pub fn set_booster_type<T: Into<booster::BoosterType>>(&mut self, booster_type: T) {
        self.booster_type = booster_type.into();
    }

    /// Get configuration for the learning objective.
    pub fn learning_params(&self) -> &learning::LearningTaskParameters {
        &self.learning_params
    }

    /// Set configuration for the learning objective.
    pub fn set_learning_params<T: Into<learning::LearningTaskParameters>>(&mut self, learning_params: T) {
        self.learning_params = learning_params.into();
    }

    /// Check whether verbose output is enabled or not.
    pub fn verbose(&self) -> bool {
        self.verbose
    }

    /// Set to `true` to enable verbose output from XGBoost's C library.
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Get number of parallel threads XGboost will use (if compiled with multiprocessing support).
    ///
    /// If `None`, XGBoost will determine the number of threads to use automatically.
    pub fn threads(&self) -> Option<u32> {
        self.threads
    }

    /// Set number of parallel threads XGBoost will use (if compiled with multiprocessing support).
    ///
    /// If `None`, XGBoost will determine the number of threads to use automatically.
    pub fn set_threads<T: Into<Option<u32>>>(&mut self, threads: T) {
        self.threads = threads.into();
    }

    /// Get the device XGBoost trains and predicts on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Set the device XGBoost trains and predicts on.
    pub fn set_device(&mut self, device: Device) {
        self.device = device;
    }

    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();

        v.extend(self.booster_type.as_string_pairs());
        v.extend(self.learning_params.as_string_pairs());

        // `silent` was removed as a learner parameter in XGBoost 1.0 and is
        // silently ignored by 3.x; `verbosity` (0=silent, 1=warning, 2=info,
        // 3=debug) is the working equivalent.
        v.push(("verbosity".to_owned(), if self.verbose { "2" } else { "0" }.to_owned()));

        if let Some(nthread) = self.threads {
            v.push(("nthread".to_owned(), nthread.to_string()));
        }

        // Only emitted when non-default: "cpu" is XGBoost's default, and not
        // sending it keeps the parameter stream identical for existing users.
        if self.device != Device::Cpu {
            v.push(("device".to_owned(), self.device.to_string()));
        }

        v
    }
}

type CustomEvaluation = fn(&[f32], &DMatrix) -> f32;

/// Information passed to callbacks after each training round.
#[derive(Debug, Clone)]
pub struct CallbackEnv {
    /// Current iteration number (0-indexed).
    pub iteration: i32,
    /// Total number of boosting rounds.
    pub total_rounds: u32,
    /// Evaluation results for this round, if evaluation sets were provided.
    /// Maps dataset name -> (metric name -> score).
    pub evaluation_results: Option<indexmap::IndexMap<String, indexmap::IndexMap<String, f32>>>,
}

/// Callback function type for training events.
///
/// Returns `true` to continue training, `false` to stop early.
pub type TrainingCallback = fn(&CallbackEnv) -> bool;

/// Parameters used by the [`Booster::train`](../struct.Booster.html#method.train) method for training new models.
/// Created using [`TrainingParametersBuilder`](struct.TrainingParametersBuilder.html).
#[derive(Builder, Clone)]
pub struct TrainingParameters<'a> {
    /// Matrix used for training model.
    pub(crate) dtrain: &'a DMatrix,

    /// Number of boosting rounds to use during training.
    ///
    /// *default*: `10`
    #[builder(default = "10")]
    pub(crate) boost_rounds: u32,

    /// Configuration for the booster model that will be trained.
    ///
    /// *default*: `BoosterParameters::default()`
    #[builder(default = "BoosterParameters::default()")]
    pub(crate) booster_params: BoosterParameters,

    #[builder(default = "None")]
    /// Optional list of DMatrix to evaluate against after each boosting round.
    ///
    /// Supplied as a list of tuples of (DMatrix, description). The description is used to differentiate between
    /// different evaluation datasets when output during training.
    ///
    /// *default*: `None`
    pub(crate) evaluation_sets: Option<&'a [(&'a DMatrix, &'a str)]>,

    /// Optional custom objective function to use for training.
    ///
    /// *default*: `None`
    #[builder(default = "None")]
    pub(crate) custom_objective_fn: Option<CustomObjective>,

    /// Optional custom evaluation function to use during training.
    ///
    /// *default*: `None`
    #[builder(default = "None")]
    pub(crate) custom_evaluation_fn: Option<CustomEvaluation>,

    /// Optional list of callback functions to call after each training round.
    ///
    /// Each callback receives a `CallbackEnv` with information about the current training state.
    /// If any callback returns `false`, training will stop early.
    ///
    /// *default*: `None`
    #[builder(default = "None")]
    pub(crate) callbacks: Option<Vec<TrainingCallback>>,

    /// Evaluate `evaluation_sets` only every `eval_period` rounds (plus always
    /// on the final round), like Python's `verbose_eval=<int>`.
    ///
    /// Each evaluation is a full prediction pass over every evaluation set —
    /// with an eval set comparable in size to the training data, evaluating
    /// every round can approach half of total training time. Rounds that skip
    /// evaluation pass `evaluation_results: None` to callbacks. `0` is treated
    /// as `1`.
    ///
    /// *default*: `1` (evaluate every round)
    #[builder(default = "1")]
    pub(crate) eval_period: u32,

    /// Whether to print evaluation results to stdout on rounds that evaluate.
    ///
    /// When `false`, evaluation still runs on schedule and the results still
    /// reach callbacks — only the printing is suppressed.
    ///
    /// *default*: `true`
    #[builder(default = "true")]
    pub(crate) verbose_eval: bool,
}

impl<'a> TrainingParameters<'a> {
    pub fn dtrain(&self) -> &'a DMatrix {
        self.dtrain
    }

    pub fn set_dtrain(&mut self, dtrain: &'a DMatrix) {
        self.dtrain = dtrain;
    }

    pub fn boost_rounds(&self) -> u32 {
        self.boost_rounds
    }

    pub fn set_boost_rounds(&mut self, boost_rounds: u32) {
        self.boost_rounds = boost_rounds;
    }

    pub fn booster_params(&self) -> &BoosterParameters {
        &self.booster_params
    }

    pub fn set_booster_params<T: Into<BoosterParameters>>(&mut self, booster_params: T) {
        self.booster_params = booster_params.into();
    }

    pub fn evaluation_sets(&self) -> Option<&'a [(&'a DMatrix, &'a str)]> {
        self.evaluation_sets
    }

    pub fn set_evaluation_sets(&mut self, evaluation_sets: Option<&'a [(&'a DMatrix, &'a str)]>) {
        self.evaluation_sets = evaluation_sets;
    }

    pub fn custom_objective_fn(&self) -> Option<CustomObjective> {
        self.custom_objective_fn
    }

    pub fn set_custom_objective_fn(&mut self, custom_objective_fn: Option<CustomObjective>) {
        self.custom_objective_fn = custom_objective_fn;
    }

    pub fn custom_evaluation_fn(&self) -> Option<CustomEvaluation> {
        self.custom_evaluation_fn
    }

    pub fn set_custom_evaluation_fn(&mut self, custom_evaluation_fn: Option<CustomEvaluation>) {
        self.custom_evaluation_fn = custom_evaluation_fn;
    }

    pub fn callbacks(&self) -> Option<&Vec<TrainingCallback>> {
        self.callbacks.as_ref()
    }

    pub fn set_callbacks(&mut self, callbacks: Option<Vec<TrainingCallback>>) {
        self.callbacks = callbacks;
    }

    pub fn eval_period(&self) -> u32 {
        self.eval_period
    }

    pub fn set_eval_period(&mut self, eval_period: u32) {
        self.eval_period = eval_period;
    }

    pub fn verbose_eval(&self) -> bool {
        self.verbose_eval
    }

    pub fn set_verbose_eval(&mut self, verbose_eval: bool) {
        self.verbose_eval = verbose_eval;
    }
}

enum Inclusion {
    Open,
    Closed,
}

struct Interval<T> {
    min: T,
    min_inclusion: Inclusion,
    max: T,
    max_inclusion: Inclusion,
}

impl<T: Display> Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let lower = match self.min_inclusion {
            Inclusion::Closed => '[',
            Inclusion::Open => '(',
        };
        let upper = match self.max_inclusion {
            Inclusion::Closed => ']',
            Inclusion::Open => ')',
        };
        write!(f, "{}{}, {}{}", lower, self.min, self.max, upper)
    }
}

impl<T: PartialOrd + Display> Interval<T> {
    fn new(min: T, min_inclusion: Inclusion, max: T, max_inclusion: Inclusion) -> Self {
        Interval {
            min,
            min_inclusion,
            max,
            max_inclusion,
        }
    }

    fn new_open_closed(min: T, max: T) -> Self {
        Interval::new(min, Inclusion::Open, max, Inclusion::Closed)
    }

    fn new_closed_closed(min: T, max: T) -> Self {
        Interval::new(min, Inclusion::Closed, max, Inclusion::Closed)
    }

    fn contains(&self, val: &T) -> bool {
        // If any comparison returns None, treat as uncomparable (e.g., NaN for floats)
        let min_cmp = match self.min_inclusion {
            Inclusion::Closed => val.partial_cmp(&self.min).map(|o| o >= std::cmp::Ordering::Equal),
            Inclusion::Open => val.partial_cmp(&self.min).map(|o| o == std::cmp::Ordering::Greater),
        };
        if min_cmp.is_none() || min_cmp == Some(false) {
            return false; // Uncomparable or less than min
        }
        let max_cmp = match self.max_inclusion {
            Inclusion::Closed => val.partial_cmp(&self.max).map(|o| o <= std::cmp::Ordering::Equal),
            Inclusion::Open => val.partial_cmp(&self.max).map(|o| o == std::cmp::Ordering::Less),
        };
        if max_cmp.is_none() || max_cmp == Some(false) {
            return false; // Uncomparable or less than min
        }
        true
    }

    fn validate(&self, val: &Option<T>, name: &str) -> Result<(), String> {
        if let Some(val) = val {
            if self.contains(val) {
                Ok(())
            } else {
                Err(format!(
                    "Invalid value for '{}' parameter, {} is not in range {}.",
                    name, val, self
                ))
            }
        } else {
            Ok(())
        }
    }
}
