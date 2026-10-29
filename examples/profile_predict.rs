//! Profiling harness: hammer single-row inplace prediction so a sampling
//! profiler (macOS `sample`, Instruments, samply) can attribute where the
//! per-call microseconds go inside libxgboost. Not a benchmark — run length
//! is fixed wall-clock so the profiler window is easy to aim.
//!
//! Usage: cargo run --release --example profile_predict
//! Then, in another shell:  sample <printed pid> 10 -f /tmp/xgb_profile.txt

use std::time::{Duration, Instant};

use xgb::parameters::{self, learning, tree};
use xgb::{Booster, DMatrix};

const TRAIN: &str = r#"{"uri": "xgboost-sys/xgboost/demo/data/agaricus.txt.train?format=libsvm"}"#;

fn main() {
    println!("pid: {}", std::process::id());

    // Same model shape as benches/predict_benchmark.rs: 127 features, 50 trees.
    let dtrain = DMatrix::load(TRAIN).unwrap();
    let tree_params = tree::TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.3)
        .build()
        .unwrap();
    let learning_params = learning::LearningTaskParametersBuilder::default()
        .objective(learning::Objective::BinaryLogistic)
        .build()
        .unwrap();
    let params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .build()
        .unwrap();
    let mut booster = Booster::new_with_cached_dmats(&params, &[&dtrain]).unwrap();
    for i in 0..50 {
        booster.update(&dtrain, i).unwrap();
    }
    booster.set_param("nthread", "1").unwrap();

    let num_features = dtrain.num_cols();
    let row = vec![0.5f32; num_features];
    let mut out = Vec::new();

    // Warm up (creates the proxy DMatrix, sizes the output buffer).
    booster.predict_from_dense_into(&row, 1, &mut out).unwrap();
    println!("warmed up; entering 30s single-row predict loop");

    let start = Instant::now();
    let mut calls = 0u64;
    let mut last_report = Instant::now();
    while start.elapsed() < Duration::from_secs(30) {
        for _ in 0..1000 {
            booster.predict_from_dense_into(&row, 1, &mut out).unwrap();
        }
        calls += 1000;
        if last_report.elapsed() > Duration::from_secs(5) {
            let ns_per_call = start.elapsed().as_nanos() as u64 / calls;
            println!("{} calls, {} ns/call", calls, ns_per_call);
            last_report = Instant::now();
        }
    }
    let ns_per_call = start.elapsed().as_nanos() as u64 / calls;
    println!("done: {} calls, {} ns/call, result={:?}", calls, ns_per_call, out);
}
