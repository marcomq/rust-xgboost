# Golden files for XGBoost upgrades

`tests/regression.rs` checks that an XGBoost update (a submodule bump, new prebuilt libraries, or changes
merged from another fork) leaves model scoring unchanged. It covers the calls a typical serving application makes:
`load_buffer`, `get_feature_names`, `DMatrix::from_dense` with one row and NaN for missing values,
`predict`, `save_buffer` and `dump_model_vec`.

There are two models, each in its own directory: `binary/` (`binary:logistic`, one score per row) and
`multiclass/` (`multi:softprob` with 4 classes, one probability per class and row). Both are trained on
the same synthetic data (20 features with skewed values, flags, codes, missing values and noise columns).
The files were written by the **trusted** version:

| File | Content |
|---|---|
| `<model>/model.ubj`, `<model>/model.json` | The trained model in both formats |
| `<model>/predictions.txt` | Its scores on 4000 holdout rows, plus an all-missing row and an all-zero row |
| `<model>/metrics.txt` | Holdout quality (AUC and logloss, or accuracy and mlogloss), label rates, tree count, and the model JSON fields that serving code commonly reads |
| `perf_<os>_<arch>_<profile>.txt` | Latency and throughput baseline for both models. Machine-specific, so not committed |

## What the tests check

- A model saved by the old version, loaded with the new one, gives the same scores (within 1e-6). This
  runs for batch and single-row prediction and for both UBJ and JSON models.
- The model JSON still parses and still has the fields serving code commonly reads (booster name, objective,
  `num_parallel_tree`, `num_class`).
- A model retrained on the same data scores at least as well as before: each quality metric may get worse
  by at most 0.005, and scores move by at most 0.01 on average.
- Latency and throughput stay within 1.25x of the baseline (`XGB_PERF_TOLERANCE` changes the factor):
  - p50 latency of single-row scoring (p99 is printed but not checked: it swings by more than 1.5x between
    identical runs)
  - single-row scores per second on one thread
  - single-row scores per second on every core, with one Booster per thread
  - batch rows per second
  - model load time

## Workflow

```sh
# 1. On the trusted version: record the golden files once (only redo this on purpose).
#    The performance baseline is separate, so it can be re-recorded without touching the golden files.
XGB_BLESS=1 cargo test --test regression
XGB_BLESS_PERF=1 cargo test --release --test regression -- --ignored

# 2. Apply the update, then on the same machine run
cargo test --test regression
cargo test --release --test regression -- --ignored --nocapture
```

Re-bless only after you have decided to accept a change. The commit diff of `metrics.txt` and
`predictions.txt` then documents what changed.

## CI

`.github/workflows/regression.yml` runs on Linux and macOS. On every push it runs the golden score tests.
On a pull request it also records the performance baseline from the PR's base commit, then measures the PR
on the same runner, with a 1.5x tolerance because shared runners are noisy. Both builds use the PR's test
file and golden models, so they are measured the same way.

To check a fork's changes, push them to a branch, then start the workflow manually on that branch. Its
`baseline` input sets what to compare against (default `master`).
