//! BoosterParameters for configuring linear boosters.

use std::default::Default;

/// Linear model algorithm.
#[derive(Clone, Default)]
pub enum LinearUpdate {
    /// Parallel coordinate descent algorithm based on shotgun algorithm. Uses ‘hogwild’ parallelism and
    /// therefore produces a nondeterministic solution on each run.
    #[default]
    Shotgun,

    /// Ordinary coordinate descent algorithm. Also multithreaded but still produces a deterministic solution.
    CoordDescent,
}

impl std::fmt::Display for LinearUpdate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let result = match *self {
            LinearUpdate::Shotgun => "shotgun".to_owned(),
            LinearUpdate::CoordDescent => "coord_descent".to_owned(),
        };
        write!(f, "{}", result)
    }
}

/// BoosterParameters for Linear Booster.
#[derive(Builder, Clone)]
#[builder(default)]
pub struct LinearBoosterParameters {
    /// L2 regularization term on weights, increase this value will make model more conservative.
    /// Normalised to number of training examples.
    ///
    /// * default: 0.0
    lambda: f32,

    /// L1 egularization term on weights, increase this value will make model more conservative.
    /// Normalised to number of training examples.
    ///
    /// * default: 0.0
    alpha: f32,

    /// Linear model algorithm.
    ///
    /// * default: `LinearUpdate::Shotgun`
    updater: LinearUpdate,
}

impl LinearBoosterParameters {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        vec![
            ("booster".to_owned(), "gblinear".to_owned()),
            ("lambda".to_owned(), self.lambda.to_string()),
            ("alpha".to_owned(), self.alpha.to_string()),
            ("updater".to_owned(), self.updater.to_string()),
        ]
    }
}

impl Default for LinearBoosterParameters {
    fn default() -> Self {
        LinearBoosterParameters {
            lambda: 0.0,
            alpha: 0.0,
            updater: LinearUpdate::default(),
        }
    }
}
