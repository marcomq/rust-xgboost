//! Benchmarks for prediction FFI entry points and per-round metadata overhead.
//!
//! Two questions are measured here:
//!
//! 1. `predict_entry_points`: is the deprecated `XGBoosterPredict` (wrapped by
//!    `Booster::predict`) actually slower than its replacement
//!    `XGBoosterPredictFromDMatrix` (wrapped by `Booster::predict_matrix`)?
//!    This decides whether consolidating onto the new entry point is worth it.
//!
//! 2. `per_round_overhead`: how expensive is the feature-info FFI getter that
//!    `validate_features` runs on every custom-objective boosting round, relative
//!    to the real work of a single boosting round (`update`)? This decides whether
//!    hoisting/caching that check out of the training loop is worth the complexity.
//!    Answered (M3 Pro, July 2026): ~37 ns for the getter vs ~2 ms for a round on
//!    agaricus (0.002%), with no feature names set (the common case — the C++ side
//!    then copies an empty vector). Even with names set it is microseconds. Do NOT
//!    add a cache for this.
//!
//! Run with: cargo bench --bench predict_benchmark

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use xgb::parameters::{self, learning, tree};
use xgb::{Booster, DMatrix, PredictConfig};

const TRAIN: &str = r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#;
const TEST: &str = r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.test?format=libsvm"}"#;

fn booster_params() -> parameters::BoosterParameters {
    let tree_params = tree::TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.3)
        .build()
        .unwrap();
    let learning_params = learning::LearningTaskParametersBuilder::default()
        .objective(learning::Objective::BinaryLogistic)
        .build()
        .unwrap();
    parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .build()
        .unwrap()
}

fn trained_booster(dtrain: &DMatrix, dtest: &DMatrix, rounds: i32) -> Booster {
    let mut booster = Booster::new_with_cached_dmats(&booster_params(), &[dtrain, dtest]).unwrap();
    for i in 0..rounds {
        booster.update(dtrain, i).unwrap();
    }
    booster
}

fn bench_predict_entry_points(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    // Predict on the larger training set (6513 rows) for more signal.
    let booster = trained_booster(&dtrain, &dtest, 50);
    let cfg = PredictConfig::default().as_json();

    let mut group = c.benchmark_group("predict_entry_points");
    group.bench_function("predict_legacy_XGBoosterPredict", |b| {
        b.iter(|| black_box(booster.predict(black_box(&dtrain)).unwrap()));
    });
    group.bench_function("predict_matrix_FromDMatrix", |b| {
        b.iter(|| black_box(booster.predict_matrix(black_box(&dtrain), &cfg).unwrap()));
    });
    group.finish();
}

/// Sweep the two predict entry points across row counts to check whether a
/// size-dependent crossover exists (as it did for dense DMatrix creation).
fn bench_predict_scaling(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let num_features = dtrain.num_cols(); // 127 for agaricus; must match for prediction
    let booster = trained_booster(&dtrain, &dtest, 50);
    let cfg = PredictConfig::default().as_json();

    let mut group = c.benchmark_group("predict_scaling");

    for num_rows in [1_000usize, 10_000, 100_000, 500_000] {
        // Synthetic dense matrix with the right feature count; values are
        // irrelevant for timing. Built once, outside the measured loop.
        let mut seed: u64 = 999;
        let data: Vec<f32> = (0..num_rows * num_features)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f64 / u64::MAX as f64) as f32
            })
            .collect();
        let dmat = DMatrix::from_dense(&data, num_rows).unwrap();
        let label = format!("{}rows", num_rows);

        group.bench_with_input(BenchmarkId::new("predict_legacy", &label), &dmat, |b, dmat| {
            b.iter(|| black_box(booster.predict(black_box(dmat)).unwrap()));
        });
        group.bench_with_input(BenchmarkId::new("predict_matrix", &label), &dmat, |b, dmat| {
            b.iter(|| black_box(booster.predict_matrix(black_box(dmat), &cfg).unwrap()));
        });
    }

    group.finish();
}

/// Serving path: building a DMatrix then predicting vs. inplace prediction
/// straight off a `&[f32]`. The DMatrix build is pure overhead that inplace
/// prediction skips, and it dominates latency for small batches.
fn bench_inplace_predict(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let num_features = dtrain.num_cols();
    let booster = trained_booster(&dtrain, &dtest, 50);
    let cfg = PredictConfig::default().as_json();

    let mut group = c.benchmark_group("inplace_predict");

    for num_rows in [1usize, 100, 10_000] {
        let mut seed: u64 = 4242;
        let data: Vec<f32> = (0..num_rows * num_features)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f64 / u64::MAX as f64) as f32
            })
            .collect();
        let label = format!("{}rows", num_rows);

        // DMatrix round-trip: from_dense + predict_matrix (both on the new API).
        group.bench_with_input(BenchmarkId::new("dmatrix_then_predict", &label), &data, |b, data| {
            b.iter(|| {
                let dmat = DMatrix::from_dense(black_box(data), num_rows).unwrap();
                black_box(booster.predict_matrix(&dmat, &cfg).unwrap())
            });
        });

        // Inplace: predict straight off the slice, no DMatrix.
        group.bench_with_input(BenchmarkId::new("predict_from_dense", &label), &data, |b, data| {
            b.iter(|| black_box(booster.predict_from_dense(black_box(data), num_rows).unwrap()));
        });
    }

    group.finish();
}

/// Sparse serving path: from_csr + predict vs inplace predict_from_csr.
fn bench_inplace_predict_csr(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let num_features = dtrain.num_cols();
    let booster = trained_booster(&dtrain, &dtest, 50);
    let cfg = PredictConfig::default().as_json();

    let mut group = c.benchmark_group("inplace_predict_csr");

    for num_rows in [1usize, 100, 10_000] {
        // ~40% dense synthetic CSR with the model's feature count.
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut seed: u64 = 31337;
        let mut nnz = 0u64;
        for _ in 0..num_rows {
            for j in 0..num_features {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed % 10 < 4 {
                    indices.push(j as u64);
                    values.push((seed % 1000) as f32 / 1000.0);
                    nnz += 1;
                }
            }
            indptr.push(nnz);
        }
        let label = format!("{}rows", num_rows);
        let input = (indptr, indices, values);

        group.bench_with_input(
            BenchmarkId::new("csr_then_predict", &label),
            &input,
            |b, (ip, ix, v)| {
                b.iter(|| {
                    let dmat =
                        DMatrix::from_csr(black_box(ip), black_box(ix), black_box(v), Some(num_features)).unwrap();
                    black_box(booster.predict_matrix(&dmat, &cfg).unwrap())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("predict_from_csr", &label),
            &input,
            |b, (ip, ix, v)| {
                b.iter(|| {
                    black_box(
                        booster
                            .predict_from_csr(black_box(ip), black_box(ix), black_box(v), num_features)
                            .unwrap(),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Model checkpointing: `save_buffer` (UBJ/JSON encode) vs the raw memory-snapshot
/// `XGBoosterSerializeToBuffer`. Measured via direct FFI since the latter is not
/// (yet) wrapped — if it wins decisively it's worth a public method.
fn bench_serialize(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let booster = trained_booster(&dtrain, &dtest, 200); // larger model for signal

    let mut group = c.benchmark_group("model_serialize");

    group.bench_function("save_buffer_ubj", |b| {
        b.iter(|| black_box(booster.save_buffer(true).unwrap()));
    });

    group.bench_function("serialize_to_buffer_raw", |b| {
        b.iter(|| black_box(booster.serialize_to_buffer().unwrap()));
    });

    group.finish();
}

fn bench_per_round_overhead(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let booster = trained_booster(&dtrain, &dtest, 50);

    let mut group = c.benchmark_group("per_round_overhead");

    // Upper bound for what `validate_features` costs each custom-objective round:
    // one XGBoosterGetStrFeatureInfo FFI call (this public getter also allocates
    // the names, which the internal count-only path does not, so it overestimates).
    group.bench_function("feature_info_ffi_getter", |b| {
        b.iter(|| black_box(booster.get_feature_names().unwrap()));
    });

    // For scale: the real work of a single boosting round. A fresh booster is
    // built per batch so the model doesn't grow unboundedly across iterations.
    group.bench_function("single_update_round", |b| {
        b.iter_batched(
            || Booster::new_with_cached_dmats(&booster_params(), &[&dtrain]).unwrap(),
            |mut bst| {
                bst.update(black_box(&dtrain), 0).unwrap();
                bst
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Latency-sensitive serving path: booster pinned to `nthread=1` (the
/// configuration the README recommends below ~1000 rows), small batches.
/// Single-threaded prediction is stable enough to resolve per-call FFI
/// overhead (allocations, JSON building) that OpenMP dispatch noise hides
/// in the default-thread groups above.
fn bench_serving_single_thread(c: &mut Criterion) {
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let dtest = DMatrix::load(TEST).unwrap();
    let num_features = dtrain.num_cols();
    let mut booster = trained_booster(&dtrain, &dtest, 50);
    booster.set_param("nthread", "1").unwrap();

    let mut group = c.benchmark_group("serving_nthread1");
    group.measurement_time(std::time::Duration::from_secs(6));

    for num_rows in [1usize, 16, 100] {
        let mut seed: u64 = 2024;
        let data: Vec<f32> = (0..num_rows * num_features)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed as f64 / u64::MAX as f64) as f32
            })
            .collect();

        // CSR version of the same shape, ~40% dense.
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        let mut nnz = 0u64;
        for _ in 0..num_rows {
            for j in 0..num_features {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                if seed % 10 < 4 {
                    indices.push(j as u64);
                    values.push((seed % 1000) as f32 / 1000.0);
                    nnz += 1;
                }
            }
            indptr.push(nnz);
        }

        let label = format!("{}rows", num_rows);

        group.bench_with_input(BenchmarkId::new("predict_from_dense", &label), &data, |b, data| {
            b.iter(|| black_box(booster.predict_from_dense(black_box(data), num_rows).unwrap()));
        });

        // Steady-state buffer reuse: one output Vec kept across iterations.
        group.bench_with_input(BenchmarkId::new("predict_from_dense_into", &label), &data, |b, data| {
            let mut out = Vec::new();
            b.iter(|| {
                booster
                    .predict_from_dense_into(black_box(data), num_rows, &mut out)
                    .unwrap();
                black_box(&out);
            });
        });

        let csr = (indptr, indices, values);
        group.bench_with_input(BenchmarkId::new("predict_from_csr", &label), &csr, |b, (ip, ix, v)| {
            b.iter(|| {
                black_box(
                    booster
                        .predict_from_csr(black_box(ip), black_box(ix), black_box(v), num_features)
                        .unwrap(),
                )
            });
        });

        group.bench_with_input(
            BenchmarkId::new("predict_from_csr_into", &label),
            &csr,
            |b, (ip, ix, v)| {
                let mut out = Vec::new();
                b.iter(|| {
                    booster
                        .predict_from_csr_into(black_box(ip), black_box(ix), black_box(v), num_features, &mut out)
                        .unwrap();
                    black_box(&out);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_predict_entry_points,
    bench_predict_scaling,
    bench_inplace_predict,
    bench_inplace_predict_csr,
    bench_serialize,
    bench_per_round_overhead,
    bench_serving_single_thread
);
criterion_main!(benches);
