//! BoosterParameters for configuring learning objectives and evaluation metrics for all
//! booster types.

use std;
use std::default::Default;

use super::Interval;

/// Learning objective used when training a booster model.
#[derive(Clone, Default)]
pub enum Objective {
    /// Linear regression.
    #[default]
    RegLinear,

    /// Logistic regression.
    RegLogistic,

    /// Logistic regression for binary classification, outputs probability.
    BinaryLogistic,

    /// Logistic regression for binary classification, outputs scores before logistic transformation.
    BinaryLogisticRaw,

    // The `gpu:*` objective variants were removed: the prefix was dropped in
    // XGBoost 1.0 and 3.x rejects those names outright ("Unknown objective
    // function"). GPU training is selected via the `device` parameter instead.
    /// Poisson regression for count data, outputs mean of poisson distribution.
    CountPoisson,

    /// Cox regression for right censored survival time data (negative values are considered right
    /// censored).
    ///
    /// predictions are returned on the hazard ratio scale (i.e., as `HR = exp(marginal_prediction)`
    /// in the proportional hazard function `h(t) = h0(t) * HR`).
    SurvivalCox,

    /// Multiclass classification using the softmax objective, with given number of classes.
    MultiSoftmax(u32),

    /// Multiclass classification using the softmax objective, with given number of classes.
    ///
    /// Outputs probabilities per class.
    MultiSoftprob(u32),

    /// Ranking task which minimises pairwise loss.
    RankPairwise,

    /// Gamma regression with log-link. Output is the mean of the gamma distribution.
    RegGamma,

    /// Tweedie regression with log-link. Takes an optional **tweedie variance power** parameter
    /// which controls the variance of the Tweedie distribution.
    ///
    /// * Set closer to 2 to shift towards a gamma distribution
    /// * Set closer to 1 to shift towards a Poisson distribution
    ///
    /// *range*: (1, 2)
    ///
    /// Set to `None` to use XGBoost's default (currently `1.5`).
    RegTweedie(Option<f32>),

    /// Quantile regression with the pinball loss.
    ///
    /// Takes the list of quantiles to estimate (each in `(0, 1)`); the trained
    /// model produces one prediction column per quantile, so `predict` output
    /// has length `num_rows * alphas.len()` (row-major).
    RegQuantile(Vec<f32>),

    /// Expectile regression (XGBoost 3.3+).
    ///
    /// Takes the list of expectiles to estimate (each in `(0, 1)`); like
    /// [`RegQuantile`](Self::RegQuantile), the model produces one prediction
    /// column per expectile.
    RegExpectile(Vec<f32>),
}

impl std::fmt::Display for Objective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match self {
            Objective::RegLinear => "reg:squarederror",
            Objective::RegLogistic => "reg:logistic",
            Objective::BinaryLogistic => "binary:logistic",
            Objective::BinaryLogisticRaw => "binary:logitraw",
            Objective::CountPoisson => "count:poisson",
            Objective::SurvivalCox => "survival:cox",
            Objective::MultiSoftmax(_) => "multi:softmax", // num_class conf must also be set
            Objective::MultiSoftprob(_) => "multi:softprob", // num_class conf must also be set
            Objective::RankPairwise => "rank:pairwise",
            Objective::RegGamma => "reg:gamma",
            Objective::RegTweedie(_) => "reg:tweedie",
            Objective::RegQuantile(_) => "reg:quantileerror", // quantile_alpha must also be set
            Objective::RegExpectile(_) => "reg:expectileerror", // expectile_alpha must also be set
        };
        write!(f, "{}", result)
    }
}

/// Type of evaluation metrics to use during learning.
#[derive(Clone)]
pub enum Metrics {
    /// Automatically selects metrics based on learning objective.
    Auto,

    /// Use custom list of metrics.
    Custom(Vec<EvaluationMetric>),
}

/// Type of evaluation metric used on validation data.
#[derive(Clone)]
pub enum EvaluationMetric {
    /// Root Mean Square Error.
    RMSE,

    /// Mean Absolute Error.
    MAE,

    /// Negative log-likelihood.
    LogLoss,

    /// Binary classification error rate with default threshold of 0.5.
    /// It is calculated as #(wrong cases)/#(all cases).
    BinaryError,

    /// Binary classification error rate with custom threshold.
    /// It is calculated as #(wrong cases)/#(all cases).
    /// For the predictions, the evaluation will regard the instances with prediction value larger than
    /// the given threshold as positive instances, and the others as negative instances.
    BinaryErrorRate(f32),

    /// Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    MultiClassErrorRate,

    /// Multiclass logloss.
    MultiClassLogLoss,

    /// Area under the curve for ranking evaluation.
    AUC,

    /// Normalized Discounted Cumulative Gain.
    NDCG,

    /// NDCG with top N positions cut off.
    NDCGCut(u32),

    /// NDCG with scores of lists without any positive samples evaluated as 0 instead of 1.
    NDCGNegative,

    /// NDCG with scores of lists without any positive samples evaluated as 0 instead of 1, and top
    /// N positions cut off.
    NDCGCutNegative(u32),

    /// Mean average precision.
    MAP,

    /// MAP with top N positions cut off.
    MAPCut(u32),

    /// MAP with scores of lists without any positive samples evaluated as 0 instead of 1.
    MAPNegative,

    /// MAP with scores of lists without any positive samples evaluated as 0 instead of 1, and top
    /// N positions cut off.
    MAPCutNegative(u32),

    /// Negative log likelihood for Poisson regression.
    PoissonLogLoss,

    /// Negative log likelihood for Gamma regression.
    GammaLogLoss,

    /// Negative log likelihood for Cox proportional hazards regression.
    CoxLogLoss,

    /// Residual deviance for Gamma regression.
    GammaDeviance,

    /// Negative log likelihood for Tweedie regression, at the given value of
    /// the variance power `rho`.
    ///
    /// XGBoost requires the metric in `tweedie-nloglik@rho` form (the bare name
    /// is rejected with "must be in format tweedie-nloglik@rho"); `rho` must be
    /// in the range [1, 2) and would typically match the objective's
    /// `tweedie_variance_power`.
    TweedieLogLoss(f32),
}

impl std::fmt::Display for EvaluationMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            EvaluationMetric::RMSE => "rmse".to_owned(),
            EvaluationMetric::MAE => "mae".to_owned(),
            EvaluationMetric::LogLoss => "logloss".to_owned(),
            EvaluationMetric::BinaryError => "error".to_owned(),
            EvaluationMetric::BinaryErrorRate(t) => format!("error@{}", t),
            EvaluationMetric::MultiClassErrorRate => "merror".to_owned(),
            EvaluationMetric::MultiClassLogLoss => "mlogloss".to_owned(),
            EvaluationMetric::AUC => "auc".to_owned(),
            EvaluationMetric::NDCG => "ndcg".to_owned(),
            EvaluationMetric::NDCGCut(n) => format!("ndcg@{}", n),
            EvaluationMetric::NDCGNegative => "ndcg-".to_owned(),
            EvaluationMetric::NDCGCutNegative(n) => format!("ndcg@{}-", n),
            EvaluationMetric::MAP => "map".to_owned(),
            EvaluationMetric::MAPCut(n) => format!("map@{}", n),
            EvaluationMetric::MAPNegative => "map-".to_owned(),
            EvaluationMetric::MAPCutNegative(n) => format!("map@{}-", n),
            EvaluationMetric::PoissonLogLoss => "poisson-nloglik".to_owned(),
            EvaluationMetric::GammaLogLoss => "gamma-nloglik".to_owned(),
            EvaluationMetric::CoxLogLoss => "cox-nloglik".to_owned(),
            EvaluationMetric::GammaDeviance => "gamma-deviance".to_owned(),
            EvaluationMetric::TweedieLogLoss(rho) => format!("tweedie-nloglik@{}", rho),
        };
        write!(f, "{}", result)
    }
}

/// BoosterParameters that configure the learning objective.
///
/// See [`LearningTaskParametersBuilder`](struct.LearningTaskParametersBuilder.html), for details
/// on parameters.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
#[builder(default)]
pub struct LearningTaskParameters {
    /// Learning objective used when training.
    ///
    /// *default*: [`RegLinear`](enum.Objective.html#variant.RegLinear)
    pub(crate) objective: Objective,

    /// Initial prediction score, i.e. global bias.
    ///
    /// *default*: `None` — the parameter is not sent to XGBoost, which then
    /// estimates the intercept from the training data (`boost_from_average`,
    /// the XGBoost 2.0+/Python default). Setting an explicit value disables
    /// that estimation, matching pre-2.0 behaviour (where the default was 0.5).
    #[builder(setter(strip_option))]
    base_score: Option<f32>,

    /// Metrics to use on evaluation data sets during training.
    ///
    /// *default*: [`Auto`](enum.Metrics.html#variant.Auto) (i.e. metrics selected automatically based on objective)
    pub(crate) eval_metrics: Metrics,

    /// Random seed.
    ///
    /// *default*: 0
    seed: u64,
}

impl Default for LearningTaskParameters {
    fn default() -> Self {
        LearningTaskParameters {
            objective: Objective::default(),
            base_score: None,
            eval_metrics: Metrics::Auto,
            seed: 0,
        }
    }
}

impl LearningTaskParameters {
    pub fn objective(&self) -> &Objective {
        &self.objective
    }

    /// Set the learning objective.
    ///
    /// Runs the same objective-specific validation as the builder (e.g.
    /// quantile/expectile alpha lists non-empty and in `(0, 1)`), so invalid
    /// values fail here with a clear message instead of an opaque C++ `CHECK`
    /// at the first training update.
    pub fn set_objective<T: Into<Objective>>(&mut self, objective: T) -> Result<(), String> {
        let objective = objective.into();
        validate_objective(&objective)?;
        self.objective = objective;
        Ok(())
    }

    pub fn base_score(&self) -> Option<f32> {
        self.base_score
    }

    pub fn set_base_score<T: Into<Option<f32>>>(&mut self, base_score: T) {
        self.base_score = base_score.into();
    }

    pub fn eval_metrics(&self) -> &Metrics {
        &self.eval_metrics
    }

    pub fn set_eval_metrics<T: Into<Metrics>>(&mut self, eval_metrics: T) {
        self.eval_metrics = eval_metrics.into();
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    /// Render an alpha list as `[a,b,...]` for XGBoost's ParamArray parser.
    fn alpha_list(alphas: &[f32]) -> String {
        let mut s = String::from("[");
        for (i, a) in alphas.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&a.to_string());
        }
        s.push(']');
        s
    }

    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        let mut v = Vec::new();

        match &self.objective {
            Objective::MultiSoftmax(n) | Objective::MultiSoftprob(n) => {
                v.push(("num_class".to_owned(), n.to_string()));
            }
            Objective::RegTweedie(Some(n)) => {
                v.push(("tweedie_variance_power".to_owned(), n.to_string()));
            }
            // The alpha lists are sent in `[a, b, ...]` form: XGBoost's
            // ParamArray parser accepts a JSON-style array (see the bundled
            // src/common/param_array.cc), matching what the Python binding
            // sends for lists.
            Objective::RegQuantile(alphas) => {
                v.push(("quantile_alpha".to_owned(), Self::alpha_list(alphas)));
            }
            Objective::RegExpectile(alphas) => {
                v.push(("expectile_alpha".to_owned(), Self::alpha_list(alphas)));
            }
            _ => {}
        }

        v.push(("objective".to_owned(), self.objective.to_string()));
        // Only sent when explicitly set: passing base_score disables XGBoost
        // 3.x's automatic intercept estimation (boost_from_average).
        if let Some(base_score) = self.base_score {
            v.push(("base_score".to_owned(), base_score.to_string()));
        }
        v.push(("seed".to_owned(), self.seed.to_string()));

        if let Metrics::Custom(eval_metrics) = &self.eval_metrics {
            for metric in eval_metrics {
                v.push(("eval_metric".to_owned(), metric.to_string()));
            }
        }

        v
    }
}

impl LearningTaskParametersBuilder {
    fn validate(&self) -> Result<(), String> {
        if let Some(objective) = &self.objective {
            validate_objective(objective)?;
        }
        Ok(())
    }
}

/// Objective-specific parameter validation, shared by the builder and
/// [`LearningTaskParameters::set_objective`] so neither entry point can smuggle
/// values XGBoost only rejects (opaquely) at configure time.
fn validate_objective(objective: &Objective) -> Result<(), String> {
    match objective {
        Objective::RegTweedie(variance_power) => {
            Interval::new_closed_closed(1.0, 2.0).validate(variance_power, "tweedie_variance_power")
        }
        Objective::RegQuantile(alphas) => validate_alphas(alphas, "quantile_alpha"),
        Objective::RegExpectile(alphas) => validate_alphas(alphas, "expectile_alpha"),
        _ => Ok(()),
    }
}

/// Alpha lists must be non-empty with every value strictly inside (0, 1);
/// XGBoost rejects values outside that range at configure time, but with a
/// far less direct error message.
fn validate_alphas(alphas: &[f32], name: &str) -> Result<(), String> {
    if alphas.is_empty() {
        return Err(format!("{} must contain at least one value", name));
    }
    for a in alphas {
        if !(*a > 0.0 && *a < 1.0) {
            return Err(format!("{} values must be in (0, 1), got {}", name, a));
        }
    }
    Ok(())
}
