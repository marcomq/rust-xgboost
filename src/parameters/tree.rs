//! BoosterParameters for controlling tree boosters.
//!
//!
use std::default::Default;

use super::Interval;

/// The tree construction algorithm used in XGBoost (see description in the
/// [reference paper](http://arxiv.org/abs/1603.02754)).
#[derive(Clone, Default)]
pub enum TreeMethod {
    /// Resolves to [`Hist`](TreeMethod::Hist) (since XGBoost 2.0; the bundled
    /// 3.3 dispatches `auto` straight to the quantile histogram updater).
    /// There is no small-data heuristic anymore — `auto` and `hist` are the
    /// same fast method.
    #[default]
    Auto,

    /// Exact greedy algorithm. Legacy: enumerates every split candidate, is
    /// easily 5-10x slower than `hist` on wide data, and does not support
    /// `QuantileDMatrix` or categorical features. Prefer
    /// [`Hist`](TreeMethod::Hist) (or the default) unless you specifically
    /// need exact split enumeration.
    Exact,

    /// Approximate greedy algorithm using sketching and histogram, re-sketched
    /// per iteration. Mainly useful for distributed setups; `hist` is faster
    /// for single-machine training.
    Approx,

    /// Fast histogram optimized approximate greedy algorithm. It uses some performance improvements
    /// such as bins caching.
    ///
    /// For GPU training combine with `device=cuda` — XGBoost 2.0 removed the
    /// `gpu_hist`/`gpu_exact` tree methods in favour of the `device` parameter.
    Hist,
}

impl std::fmt::Display for TreeMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            TreeMethod::Auto => "auto".to_owned(),
            TreeMethod::Exact => "exact".to_owned(),
            TreeMethod::Approx => "approx".to_owned(),
            TreeMethod::Hist => "hist".to_owned(),
        };
        write!(f, "{}", result)
    }
}

impl From<String> for TreeMethod {
    fn from(s: String) -> Self {
        use std::borrow::Borrow;
        Self::from(s.borrow())
    }
}

impl<'a> From<&'a str> for TreeMethod {
    fn from(s: &'a str) -> Self {
        match s {
            "auto" => TreeMethod::Auto,
            "exact" => TreeMethod::Exact,
            "approx" => TreeMethod::Approx,
            "hist" => TreeMethod::Hist,
            // Compat shim: XGBoost 2.0 removed the gpu_* tree methods (GPU
            // selection moved to the `device` parameter); map to the CPU
            // spellings rather than emitting strings XGBoost 3.x rejects —
            // but loudly, because the mapping alone lands on the CPU.
            "gpu_exact" | "gpu_hist" => {
                log::warn!(
                    "tree_method '{s}' was removed in XGBoost 2.0; mapping to the CPU '{}' method. \
                     For GPU training set BoosterParameters' device to Device::Cuda instead.",
                    if s == "gpu_exact" { "exact" } else { "hist" }
                );
                if s == "gpu_exact" { TreeMethod::Exact } else { TreeMethod::Hist }
            }
            _ => panic!("no known tree_method for {}", s),
        }
    }
}

/// Sampling method for the training instances (XGBoost `sampling_method`).
///
/// Only used when `subsample < 1.0`.
#[derive(Clone, Default, PartialEq, Eq)]
pub enum SamplingMethod {
    /// Each row has equal probability of being selected (XGBoost default).
    /// Works on any device.
    #[default]
    Uniform,

    /// Rows are selected with probability proportional to the regularized
    /// absolute gradient. Lets `subsample` go as low as ~0.1 without accuracy
    /// loss — a large training speedup — but **requires `device=cuda` with
    /// `tree_method=hist`**; the CPU updaters reject it.
    GradientBased,
}

impl std::fmt::Display for SamplingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            SamplingMethod::Uniform => "uniform",
            SamplingMethod::GradientBased => "gradient_based",
        };
        write!(f, "{}", result)
    }
}

/// Provides a modular way to construct and to modify the trees. This is an advanced parameter that is usually set
/// automatically, depending on some other parameters. However, it could be also set explicitly by a user.
#[derive(Clone)]
pub enum TreeUpdater {
    /// Non-distributed column-based construction of trees.
    GrowColMaker,

    /// Distributed tree construction with row-based data splitting based on global proposal of histogram counting.
    GrowHistMaker,

    /// Grow tree with the quantile histogram method (the `hist` tree method's updater).
    GrowQuantileHistMaker,

    /// Synchronizes trees in all distributed nodes.
    Sync,

    /// Refreshes tree’s statistics and/or leaf values based on the current data.
    /// Note that no random subsampling of data rows is performed.
    Refresh,

    /// Prunes the splits where loss < min_split_loss (or gamma).
    Prune,
}

impl std::fmt::Display for TreeUpdater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            TreeUpdater::GrowColMaker => "grow_colmaker".to_owned(),
            TreeUpdater::GrowHistMaker => "grow_histmaker".to_owned(),
            TreeUpdater::GrowQuantileHistMaker => "grow_quantile_histmaker".to_owned(),
            TreeUpdater::Sync => "sync".to_owned(),
            TreeUpdater::Refresh => "refresh".to_owned(),
            TreeUpdater::Prune => "prune".to_owned(),
        };
        write!(f, "{}", result)
    }
}

/// A type of boosting process to run.
#[derive(Clone, Default)]
pub enum ProcessType {
    /// The normal boosting process which creates new trees.
    #[default]
    Default,

    /// Starts from an existing model and only updates its trees. In each boosting iteration,
    /// a tree from the initial model is taken, a specified sequence of updater plugins is run for that tree,
    /// and a modified tree is added to the new model. The new model would have either the same or smaller number of
    /// trees, depending on the number of boosting iteratons performed.
    /// Currently, the following built-in updater plugins could be meaningfully used with this process type:
    /// 'refresh', 'prune'. With 'update', one cannot use updater plugins that create new trees.
    Update,
}

impl std::fmt::Display for ProcessType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            ProcessType::Default => "default".to_owned(),
            ProcessType::Update => "update".to_owned(),
        };
        write!(f, "{}", result)
    }
}

/// Controls the way new nodes are added to the tree.
#[derive(Clone, Default)]
pub enum GrowPolicy {
    /// Split at nodes closest to the root.
    #[default]
    Depthwise,

    /// Split at noeds with highest loss change.
    LossGuide,
}

impl std::fmt::Display for GrowPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            GrowPolicy::Depthwise => "depthwise".to_owned(),
            GrowPolicy::LossGuide => "lossguide".to_owned(),
        };
        write!(f, "{}", result)
    }
}

// Note: the `predictor` parameter (cpu_predictor/gpu_predictor) and
// `sketch_eps` were removed from XGBoost (2.0 and 1.7 respectively) and are
// silently ignored by 3.x, so this wrapper no longer exposes or emits them.
// Predictor/device selection is via the `device` parameter; sketch granularity
// is controlled by `max_bin`.

/// BoosterParameters for Tree Booster. Create using
/// [`TreeBoosterParametersBuilder`](struct.TreeBoosterParametersBuilder.html).
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
#[builder(default)]
pub struct TreeBoosterParameters {
    /// Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly
    /// get the weights of new features, and eta actually shrinks the feature weights to make the boosting process
    /// more conservative.
    ///
    /// * range: [0.0, 1.0]
    /// * default: 0.3
    eta: f32,

    /// Minimum loss reduction required to make a further partition on a leaf node of the tree.
    /// The larger, the more conservative the algorithm will be.
    ///
    /// * range: [0,∞]
    /// * default: 0
    gamma: f32,

    /// Maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting.
    /// 0 indicates no limit, limit is required for depth-wise grow policy.
    ///
    /// * range: [0,∞]
    /// * default: 6
    max_depth: u32,

    /// Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf
    /// node with the sum of instance weight less than min_child_weight, then the building process will give up
    /// further partitioning.
    /// In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node.
    /// The larger, the more conservative the algorithm will be.
    ///
    /// * range: [0,∞]
    /// * default: 1
    min_child_weight: f32,

    /// Maximum delta step we allow each tree’s weight estimation to be.
    /// If the value is set to 0, it means there is no constraint. If it is set to a positive value,
    /// it can help making the update step more conservative. Usually this parameter is not needed,
    /// but it might help in logistic regression when class is extremely imbalanced.
    /// Set it to value of 1-10 might help control the update.
    ///
    /// * range: [0,∞]
    /// * default: 0
    max_delta_step: f32,

    /// Subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half
    /// of the data instances to grow trees and this will prevent overfitting.
    ///
    /// * range: (0, 1]
    /// * default: 1.0
    subsample: f32,

    /// Subsample ratio of columns when constructing each tree.
    ///
    /// * range: (0.0, 1.0]
    /// * default: 1.0
    colsample_bytree: f32,

    /// Subsample ratio of columns for each split, in each level.
    ///
    /// * range: (0.0, 1.0]
    /// * default: 1.0
    colsample_bylevel: f32,

    /// Subsample ratio of columns for each node.
    ///
    /// * range: (0.0, 1.0]
    /// * default: 1.0
    colsample_bynode: f32,

    /// L2 regularization term on weights, increase this value will make model more conservative.
    ///
    /// * default: 1
    lambda: f32,

    /// L1 regularization term on weights, increase this value will make model more conservative.
    ///
    /// * default: 0
    alpha: f32,

    /// The tree construction algorithm used in XGBoost.
    #[builder(default = "TreeMethod::default()")]
    tree_method: TreeMethod,

    /// Control the balance of positive and negative weights, useful for unbalanced classes.
    /// A typical value to consider: sum(negative cases) / sum(positive cases).
    ///
    /// default: 1.0
    scale_pos_weight: f32,

    /// Sequence of tree updaters to run, providing a modular way to construct and to modify the trees.
    ///
    /// * default: vec![]
    updater: Vec<TreeUpdater>,

    /// This is a parameter of the ‘refresh’ updater plugin. When this flag is true, tree leafs as well as tree nodes'
    /// stats are updated. When it is false, only node stats are updated.
    ///
    /// * default: true
    refresh_leaf: bool,

    /// A type of boosting process to run.
    ///
    /// * default: ProcessType::Default
    process_type: ProcessType,

    /// Controls a way new nodes are added to the tree.  Currently supported only if tree_method is set to 'hist'.
    grow_policy: GrowPolicy,

    /// Maximum number of nodes to be added. Only relevant for the `GrowPolicy::LossGuide` grow
    /// policy.
    ///
    /// * default: 0
    max_leaves: u32,

    /// This is only used if 'hist' is specified as tree_method.
    /// Maximum number of discrete bins to bucket continuous features.
    /// Increasing this number improves the optimality of splits at the cost of higher computation time.
    ///
    /// * default: 256
    max_bin: u32,

    /// Number of trees to train in parallel for boosted random forest.
    ///
    /// * default: 1
    num_parallel_tree: u32,

    /// Sampling method used when `subsample < 1.0`. `GradientBased` permits
    /// much lower subsample ratios (~0.1) without accuracy loss but requires
    /// `device=cuda` + `tree_method=hist`.
    ///
    /// * default: SamplingMethod::Uniform
    sampling_method: SamplingMethod,

    /// Maximum number of categories for which one-hot encoded splits are used;
    /// categorical features with more categories use partition-based splits.
    /// Only relevant for columns marked categorical (see
    /// `DMatrix::set_feature_types`).
    ///
    /// * range: [1,∞], XGBoost default: 4
    /// * default: `None` (let XGBoost decide)
    max_cat_to_onehot: Option<u32>,

    /// Maximum number of categories considered per partition-based categorical
    /// split. Lower values are faster and regularize more; higher values find
    /// better splits on high-cardinality categorical features.
    ///
    /// * range: [1,∞], XGBoost default: 64
    /// * default: `None` (let XGBoost decide)
    max_cat_threshold: Option<u32>,
}

impl Default for TreeBoosterParameters {
    fn default() -> Self {
        TreeBoosterParameters {
            eta: 0.3,
            gamma: 0.0,
            max_depth: 6,
            min_child_weight: 1.0,
            max_delta_step: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            colsample_bylevel: 1.0,
            colsample_bynode: 1.0,
            lambda: 1.0,
            alpha: 0.0,
            tree_method: TreeMethod::default(),
            scale_pos_weight: 1.0,
            updater: Vec::new(),
            refresh_leaf: true,
            process_type: ProcessType::default(),
            grow_policy: GrowPolicy::default(),
            max_leaves: 0,
            max_bin: 256,
            num_parallel_tree: 1,
            sampling_method: SamplingMethod::default(),
            max_cat_to_onehot: None,
            max_cat_threshold: None,
        }
    }
}

impl TreeBoosterParameters {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = vec![
            ("booster".to_owned(), "gbtree".to_owned()),
            ("eta".to_owned(), self.eta.to_string()),
            ("gamma".to_owned(), self.gamma.to_string()),
            ("max_depth".to_owned(), self.max_depth.to_string()),
            ("min_child_weight".to_owned(), self.min_child_weight.to_string()),
            ("max_delta_step".to_owned(), self.max_delta_step.to_string()),
            ("subsample".to_owned(), self.subsample.to_string()),
            ("colsample_bytree".to_owned(), self.colsample_bytree.to_string()),
            ("colsample_bylevel".to_owned(), self.colsample_bylevel.to_string()),
            ("colsample_bynode".to_owned(), self.colsample_bynode.to_string()),
            ("lambda".to_owned(), self.lambda.to_string()),
            ("alpha".to_owned(), self.alpha.to_string()),
            ("tree_method".to_owned(), self.tree_method.to_string()),
            ("scale_pos_weight".to_owned(), self.scale_pos_weight.to_string()),
            ("refresh_leaf".to_owned(), (self.refresh_leaf as u8).to_string()),
            ("process_type".to_owned(), self.process_type.to_string()),
            ("grow_policy".to_owned(), self.grow_policy.to_string()),
            ("max_leaves".to_owned(), self.max_leaves.to_string()),
            ("max_bin".to_owned(), self.max_bin.to_string()),
            ("num_parallel_tree".to_owned(), self.num_parallel_tree.to_string()),
        ];

        // Don't pass anything to XGBoost if the user didn't specify anything.
        // This allows XGBoost to figure it out on it's own, and suppresses the
        // warning message during training.
        // See: https://github.com/davechallis/rust-xgboost/issues/7
        if !self.updater.is_empty() {
            v.push((
                "updater".to_owned(),
                self.updater
                    .iter()
                    .map(|u| u.to_string())
                    .collect::<Vec<String>>()
                    .join(","),
            ));
        }

        // Emitted only when non-default: `gradient_based` is rejected by the
        // CPU updaters, so an unconditional emit would break CPU training.
        if self.sampling_method != SamplingMethod::Uniform {
            v.push(("sampling_method".to_owned(), self.sampling_method.to_string()));
        }
        if let Some(n) = self.max_cat_to_onehot {
            v.push(("max_cat_to_onehot".to_owned(), n.to_string()));
        }
        if let Some(n) = self.max_cat_threshold {
            v.push(("max_cat_threshold".to_owned(), n.to_string()));
        }

        v
    }
}

impl TreeBoosterParametersBuilder {
    fn validate(&self) -> Result<(), String> {
        Interval::new_closed_closed(0.0, 1.0).validate(&self.eta, "eta")?;
        Interval::new_open_closed(0.0, 1.0).validate(&self.subsample, "subsample")?;
        Interval::new_open_closed(0.0, 1.0).validate(&self.colsample_bytree, "colsample_bytree")?;
        Interval::new_open_closed(0.0, 1.0).validate(&self.colsample_bylevel, "colsample_bylevel")?;
        Interval::new_open_closed(0.0, 1.0).validate(&self.colsample_bynode, "colsample_bynode")?;
        // The C++ side enforces a lower bound of 1 on both; fail here with a
        // direct message instead of an opaque configure-time CHECK.
        if let Some(Some(0)) = self.max_cat_to_onehot {
            return Err("max_cat_to_onehot must be >= 1".to_owned());
        }
        if let Some(Some(0)) = self.max_cat_threshold {
            return Err("max_cat_threshold must be >= 1".to_owned());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_params() {
        let p = TreeBoosterParameters::default();
        assert_eq!(p.eta, 0.3);
        let p = TreeBoosterParametersBuilder::default().build().unwrap();
        assert_eq!(p.eta, 0.3);
    }
}
