// Golden tests guarding XGBoost upgrades, see tests/golden/README.md.
// Run with XGB_BLESS=1 on the trusted version to record the golden files, XGB_BLESS_PERF=1 for the speed baseline.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use xgb::parameters::{learning, tree, BoosterParametersBuilder, BoosterType, TrainingParametersBuilder};
use xgb::{Booster, DMatrix};

const FEATURE_NAMES: [&str; 20] = [
    "skewed",
    "cyclic",
    "flag_a",
    "category",
    "sparse_int",
    "count",
    "ratio",
    "flag_b",
    "noise_0",
    "noise_1",
    "noise_2",
    "noise_3",
    "noise_4",
    "noise_5",
    "noise_6",
    "noise_7",
    "noise_8",
    "noise_9",
    "noise_10",
    "noise_11",
];
const NUM_FEATURES: usize = FEATURE_NAMES.len();
const NUM_CLASSES: usize = 4;
const CLASS_THRESHOLDS: [f32; NUM_CLASSES - 1] = [3.5, 4.5, 6.0];
const TRAIN_ROWS: usize = 8000;
const TEST_ROWS: usize = 4000;
const ROUNDS: u32 = 100;
const SINGLE_ROW_CHECKS: usize = 500;
const STORED_ROW_STEP: usize = 20;

const PREDICTION_TOLERANCE: f32 = 1e-6;
const MAX_QUALITY_LOSS: f64 = 0.005;
const MAX_RETRAIN_MEAN_ABS_DIFF: f64 = 0.03;
const DEFAULT_PERF_TOLERANCE: f64 = 1.25;
const LATENCY_WARMUP_CALLS: usize = 500;
const LATENCY_CALLS: usize = 10_000;
const THROUGHPUT_WINDOW: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug)]
enum Case {
    Binary,
    Multiclass,
}

impl Case {
    const ALL: [Case; 2] = [Case::Binary, Case::Multiclass];

    fn name(self) -> &'static str {
        match self {
            Case::Binary => "binary",
            Case::Multiclass => "multiclass",
        }
    }

    fn outputs_per_row(self) -> usize {
        match self {
            Case::Binary => 1,
            Case::Multiclass => NUM_CLASSES,
        }
    }

    fn objective(self) -> learning::Objective {
        match self {
            Case::Binary => learning::Objective::BinaryLogistic,
            Case::Multiclass => learning::Objective::MultiSoftprob(NUM_CLASSES as u32),
        }
    }

    fn label(self, noisy_signal: f32) -> f32 {
        match self {
            Case::Binary => (noisy_signal > CLASS_THRESHOLDS[NUM_CLASSES - 2]) as u8 as f32,
            Case::Multiclass => CLASS_THRESHOLDS.iter().filter(|&&t| noisy_signal > t).count() as f32,
        }
    }

    // (name, value, higher is better)
    fn quality(self, predictions: &[f32]) -> Vec<(&'static str, f64, bool)> {
        let (_, labels) = scoring_rows(self);
        let holdout = &predictions[..TEST_ROWS * self.outputs_per_row()];
        match self {
            Case::Binary => vec![
                ("auc", auc(holdout, &labels), true),
                ("logloss", logloss(holdout, &labels), false),
            ],
            Case::Multiclass => vec![
                ("accuracy", accuracy(holdout, &labels), true),
                ("mlogloss", mlogloss(holdout, &labels), false),
            ],
        }
    }

    fn golden_dir(self) -> PathBuf {
        golden_dir().join(self.name())
    }
}

struct Golden {
    model_ubj: Vec<u8>,
    model_json: Vec<u8>,
    predictions: Vec<f32>,
    metrics: BTreeMap<String, String>,
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn blessing() -> bool {
    std::env::var_os("XGB_BLESS").is_some()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    // Only exact IEEE operations from here on, so the data is bit-identical on every platform.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn maybe_missing(&mut self, rate: f32, value: f32) -> f32 {
        if self.unit() < rate {
            f32::NAN
        } else {
            value
        }
    }
}

// Skewed values, flags, codes, missing values and pure noise columns, labelled by thresholding a noisy signal.
fn synthetic(case: Case, num_rows: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed);
    let mut data = Vec::with_capacity(num_rows * NUM_FEATURES);
    let mut labels = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        let u = rng.unit();
        let skewed = (u * u * u * 500_000.0).round() / 100.0;
        let cyclic = (rng.next() % 24) as f32;
        let flag_a = rng.next().is_multiple_of(3) as u8 as f32;
        let category = (rng.next() % 30) as f32;
        let sparse_int = (rng.next() % 2000) as f32;
        let sparse_int = rng.maybe_missing(0.1, sparse_int);
        let count = (rng.unit() * rng.unit() * 20.0).floor();
        let ratio = rng.unit() * 8.0;
        let ratio = rng.maybe_missing(0.2, ratio);
        let flag_b = (rng.unit() < 0.15) as u8 as f32;
        let row_start = data.len();
        data.extend_from_slice(&[skewed, cyclic, flag_a, category, sparse_int, count, ratio, flag_b]);
        for _ in 8..NUM_FEATURES {
            let noise = rng.unit();
            data.push(rng.maybe_missing(0.05, noise));
        }

        let mut signal = count * 0.15;
        signal += if skewed > 1000.0 { 1.0 } else { 0.0 };
        signal += if cyclic < 5.0 { 0.8 } else { 0.0 };
        signal += if flag_a == 1.0 && flag_b == 1.0 { 1.5 } else { 0.0 };
        signal += if [7.0, 13.0, 21.0].contains(&category) { 1.0 } else { 0.0 };
        signal += if sparse_int.is_nan() || sparse_int < 30.0 { 0.7 } else { 0.0 };
        signal += if ratio > 4.0 { 1.0 } else { 0.0 };
        let noise_0 = data[row_start + 8];
        signal += if noise_0.is_nan() { 0.0 } else { noise_0 * 0.5 };
        labels.push(case.label(signal + rng.unit() * 3.0));
    }
    (data, labels)
}

// Holdout rows plus the edge cases serving code hits: an all-missing row and an all-zero row.
fn scoring_rows(case: Case) -> (Vec<f32>, Vec<f32>) {
    let (mut data, labels) = synthetic(case, TEST_ROWS, 2);
    data.extend(std::iter::repeat_n(f32::NAN, NUM_FEATURES));
    data.extend(std::iter::repeat_n(0.0, NUM_FEATURES));
    (data, labels)
}

// The golden file keeps every STORED_ROW_STEP-th holdout row and both edge-case rows.
fn stored_rows() -> impl Iterator<Item = usize> {
    (0..TEST_ROWS).step_by(STORED_ROW_STEP).chain([TEST_ROWS, TEST_ROWS + 1])
}

fn stored_scores(case: Case, all: &[f32]) -> Vec<f32> {
    let outputs = case.outputs_per_row();
    stored_rows().flat_map(|i| all[i * outputs..(i + 1) * outputs].iter().copied()).collect()
}

// Weighted by class: plain multiclass sums are always the row count, since each row's probabilities add up to 1.
fn score_sum(case: Case, scores: &[f32]) -> f64 {
    let outputs = case.outputs_per_row();
    scores
        .iter()
        .enumerate()
        .map(|(i, &s)| s as f64 * (i % outputs + 1) as f64)
        .sum()
}

fn train_model(case: Case) -> Booster {
    let (data, labels) = synthetic(case, TRAIN_ROWS, 1);
    let mut dtrain = DMatrix::from_dense(&data, TRAIN_ROWS).unwrap();
    dtrain.set_labels(&labels).unwrap();

    let tree_params = tree::TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.1)
        .tree_method(tree::TreeMethod::Hist)
        .build()
        .unwrap();
    let learning_params = learning::LearningTaskParametersBuilder::default()
        .objective(case.objective())
        .build()
        .unwrap();
    let booster_params = BoosterParametersBuilder::default()
        .booster_type(BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .build()
        .unwrap();
    let params = TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .boost_rounds(ROUNDS)
        .booster_params(booster_params)
        .build()
        .unwrap();
    let booster = Booster::train(&params).unwrap();
    booster.set_feature_names(&FEATURE_NAMES.to_vec()).unwrap();
    booster
}

fn predict_all(case: Case, booster: &Booster) -> Vec<f32> {
    let (data, _) = scoring_rows(case);
    let dmat = DMatrix::from_dense(&data, data.len() / NUM_FEATURES).unwrap();
    booster.predict(&dmat).unwrap()
}

fn auc(scores: &[f32], labels: &[f32]) -> f64 {
    let mut ranked: Vec<(f32, f32)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut rank_sum, mut positives, mut i) = (0.0, 0.0, 0);
    while i < ranked.len() {
        let mut j = i;
        while j < ranked.len() && ranked[j].0 == ranked[i].0 {
            j += 1;
        }
        let mid_rank = (i + j + 1) as f64 / 2.0;
        for &(_, label) in &ranked[i..j] {
            if label == 1.0 {
                rank_sum += mid_rank;
                positives += 1.0;
            }
        }
        i = j;
    }
    let negatives = ranked.len() as f64 - positives;
    (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)
}

fn neg_log(p: f32) -> f64 {
    -(p as f64).clamp(1e-15, 1.0).ln()
}

fn logloss(scores: &[f32], labels: &[f32]) -> f64 {
    let total: f64 = scores
        .iter()
        .zip(labels)
        .map(|(&p, &y)| if y == 1.0 { neg_log(p) } else { neg_log(1.0 - p) })
        .sum();
    total / scores.len() as f64
}

fn mlogloss(probabilities: &[f32], labels: &[f32]) -> f64 {
    let total: f64 = probabilities
        .chunks(NUM_CLASSES)
        .zip(labels)
        .map(|(row, &y)| neg_log(row[y as usize]))
        .sum();
    total / labels.len() as f64
}

fn accuracy(probabilities: &[f32], labels: &[f32]) -> f64 {
    let correct = probabilities
        .chunks(NUM_CLASSES)
        .zip(labels)
        .filter(|(row, &y)| {
            let predicted = (0..NUM_CLASSES).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            predicted == y as usize
        })
        .count();
    correct as f64 / labels.len() as f64
}

fn model_json(booster: &Booster) -> serde_json::Value {
    let text = String::from_utf8(booster.save_buffer(false).unwrap()).unwrap();
    serde_json::from_str(&text.replace("NaN", "null"))
        .expect("model JSON must stay parseable by serde_json after replacing NaN")
}

fn json_str(json: &serde_json::Value, pointer: &str) -> String {
    json.pointer(pointer)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("model JSON lacks {pointer}"))
        .to_owned()
}

const METADATA_POINTERS: [(&str, &str); 4] = [
    ("gradient_booster", "/learner/gradient_booster/name"),
    ("objective", "/learner/objective/name"),
    (
        "num_parallel_tree",
        "/learner/gradient_booster/model/gbtree_model_param/num_parallel_tree",
    ),
    ("num_class", "/learner/learner_model_param/num_class"),
];

fn read_key_values(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect()
}

fn write_key_values(values: &BTreeMap<String, String>) -> String {
    values.iter().map(|(k, v)| format!("{k}={v}\n")).collect()
}

fn bless(case: Case) {
    let booster = train_model(case);
    let predictions = predict_all(case, &booster);
    let json = model_json(&booster);
    let (_, labels) = synthetic(case, TRAIN_ROWS, 1);

    let mut metrics = BTreeMap::new();
    for (name, value, _) in case.quality(&predictions) {
        metrics.insert(name.to_owned(), value.to_string());
    }
    let classes = if case.outputs_per_row() == 1 { 1..2 } else { 0..NUM_CLASSES };
    for class in classes {
        let rate = labels.iter().filter(|&&y| y == class as f32).count() as f32 / labels.len() as f32;
        metrics.insert(format!("label_{class}_rate_train"), rate.to_string());
    }
    metrics.insert("score_sum".to_owned(), score_sum(case, &predictions).to_string());
    metrics.insert(
        "trees".to_owned(),
        booster.dump_model_vec(false).unwrap().len().to_string(),
    );
    for (key, pointer) in METADATA_POINTERS {
        metrics.insert(key.to_owned(), json_str(&json, pointer));
    }
    metrics.insert(
        "version".to_owned(),
        json.pointer("/version").map(|v| v.to_string()).unwrap_or_default(),
    );

    let dir = case.golden_dir();
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("model.ubj"), booster.save_buffer(true).unwrap()).unwrap();
    fs::write(dir.join("model.json"), booster.save_buffer(false).unwrap()).unwrap();
    let outputs = case.outputs_per_row();
    let lines: String = stored_rows()
        .map(|i| {
            let scores: Vec<String> = predictions[i * outputs..(i + 1) * outputs]
                .iter()
                .map(|p| p.to_string())
                .collect();
            format!("{i} {}\n", scores.join(" "))
        })
        .collect();
    fs::write(dir.join("predictions.txt"), lines).unwrap();
    fs::write(dir.join("metrics.txt"), write_key_values(&metrics)).unwrap();
    eprintln!("blessed golden files in {}: {metrics:?}", dir.display());
}

fn golden(case: Case) -> &'static Golden {
    static GOLDEN: [OnceLock<Golden>; Case::ALL.len()] = [const { OnceLock::new() }; Case::ALL.len()];
    GOLDEN[case as usize].get_or_init(|| {
        if blessing() {
            bless(case);
        }
        let dir = case.golden_dir();
        let read = |name: &str| {
            fs::read(dir.join(name)).unwrap_or_else(|e| {
                panic!(
                    "missing {}/{name} ({e}); record it with XGB_BLESS=1, see tests/golden/README.md",
                    case.name()
                )
            })
        };
        let text = String::from_utf8(read("predictions.txt")).unwrap();
        let mut rows = Vec::new();
        let mut predictions = Vec::new();
        for line in text.lines() {
            let mut fields = line.split_whitespace();
            rows.push(fields.next().unwrap().parse::<usize>().unwrap());
            predictions.extend(fields.map(|p| p.parse::<f32>().unwrap()));
        }
        assert_eq!(
            rows,
            stored_rows().collect::<Vec<_>>(),
            "{}: stored rows changed, re-bless with XGB_BLESS=1",
            case.name()
        );
        Golden {
            model_ubj: read("model.ubj"),
            model_json: read("model.json"),
            predictions,
            metrics: read_key_values(&String::from_utf8(read("metrics.txt")).unwrap()),
        }
    })
}

fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: number of predictions changed");
    let (worst, diff) = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| if a.is_nan() || e.is_nan() { f32::INFINITY } else { (a - e).abs() })
        .enumerate()
        .fold((0, 0.0f32), |acc, (i, d)| if d > acc.1 { (i, d) } else { acc });
    assert!(
        diff <= PREDICTION_TOLERANCE,
        "{context}: output {worst} scores {} instead of {} (diff {diff})",
        actual[worst],
        expected[worst]
    );
}

fn assert_matches_golden(case: Case, actual: &[f32], context: &str) {
    let golden = golden(case);
    assert_close(&stored_scores(case, actual), &golden.predictions, context);
    let expected: f64 = golden.metrics["score_sum"].parse().unwrap();
    let diff = (score_sum(case, actual) - expected).abs();
    assert!(
        diff <= (actual.len() * case.outputs_per_row()) as f64 * PREDICTION_TOLERANCE as f64,
        "{context}: sum of all {} scores moved by {diff}",
        actual.len()
    );
}

#[test]
fn stored_ubj_model_scores_identically() {
    for case in Case::ALL {
        let golden = golden(case);
        let booster = Booster::load_buffer(&golden.model_ubj).unwrap();
        assert_eq!(booster.get_feature_names().unwrap(), FEATURE_NAMES);
        assert_matches_golden(
            case,
            &predict_all(case, &booster),
            &format!("{case:?}: batch, UBJ model"),
        );
    }
}

#[test]
fn stored_json_model_scores_identically() {
    for case in Case::ALL {
        let golden = golden(case);
        let booster = Booster::load_buffer(&golden.model_json).unwrap();
        assert_eq!(booster.get_feature_names().unwrap(), FEATURE_NAMES);
        assert_matches_golden(
            case,
            &predict_all(case, &booster),
            &format!("{case:?}: batch, JSON model"),
        );
    }
}

#[test]
fn single_row_scoring_matches_batch() {
    for case in Case::ALL {
        let golden = golden(case);
        let booster = Booster::load_buffer(&golden.model_ubj).unwrap();
        let batch = predict_all(case, &booster);
        let (data, _) = scoring_rows(case);
        let rows: Vec<&[f32]> = data.chunks(NUM_FEATURES).collect();
        let outputs = case.outputs_per_row();
        let mut checked: Vec<usize> = (0..SINGLE_ROW_CHECKS).collect();
        checked.extend([rows.len() - 2, rows.len() - 1]);
        for i in checked {
            let dmat = DMatrix::from_dense(rows[i], 1).unwrap();
            let scores = booster.predict(&dmat).unwrap();
            let expected = &batch[i * outputs..(i + 1) * outputs];
            assert_close(&scores, expected, &format!("{case:?}: single row {i}"));
        }
    }
}

#[test]
fn model_metadata_is_unchanged() {
    for case in Case::ALL {
        let golden = golden(case);
        let booster = Booster::load_buffer(&golden.model_ubj).unwrap();
        let json = model_json(&booster);
        for (key, pointer) in METADATA_POINTERS {
            assert_eq!(json_str(&json, pointer), golden.metrics[key], "{case:?}: {key} changed");
        }
        assert_eq!(
            booster.dump_model_vec(false).unwrap().len().to_string(),
            golden.metrics["trees"]
        );

        let resaved = Booster::load_buffer(&booster.save_buffer(true).unwrap()).unwrap();
        assert_matches_golden(
            case,
            &predict_all(case, &resaved),
            &format!("{case:?}: re-saved as UBJ"),
        );
    }
}

#[test]
fn retrained_model_quality_is_unchanged() {
    for case in Case::ALL {
        let golden = golden(case);
        let predictions = predict_all(case, &train_model(case));
        for (name, value, higher_is_better) in case.quality(&predictions) {
            let before: f64 = golden.metrics[name].parse().unwrap();
            eprintln!("{case:?} retrained: {name} {value:.5} (golden {before:.5})");
            let loss = if higher_is_better { before - value } else { value - before };
            assert!(
                loss <= MAX_QUALITY_LOSS,
                "{case:?}: {name} went from {before} to {value}"
            );
        }
        let stored = stored_scores(case, &predictions);
        let mean_abs_diff = stored
            .iter()
            .zip(&golden.predictions)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / stored.len() as f64;
        eprintln!("{case:?} retrained: mean |score diff| {mean_abs_diff:.6}");
        assert!(
            mean_abs_diff <= MAX_RETRAIN_MEAN_ABS_DIFF,
            "{case:?}: retrained scores drifted by {mean_abs_diff} on average"
        );
    }
}

fn score_single_row(booster: &Booster, row: &[f32]) {
    let dmat = DMatrix::from_dense(row, 1).unwrap();
    std::hint::black_box(booster.predict(&dmat).unwrap());
}

fn latency_percentiles(booster: &Booster, rows: &[&[f32]]) -> (f64, f64) {
    for row in rows.iter().cycle().take(LATENCY_WARMUP_CALLS) {
        score_single_row(booster, row);
    }
    let mut samples: Vec<f64> = rows
        .iter()
        .cycle()
        .take(LATENCY_CALLS)
        .map(|row| {
            let start = Instant::now();
            score_single_row(booster, row);
            start.elapsed().as_nanos() as f64
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let percentile = |p: f64| samples[((samples.len() - 1) as f64 * p) as usize];
    (percentile(0.50), percentile(0.99))
}

// Single-row scores per second summed over `threads`, each with its own Booster as a server would hold.
fn single_row_throughput(model: &[u8], rows: &[&[f32]], threads: usize) -> f64 {
    let scored: usize = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    let booster = Booster::load_buffer(model).unwrap();
                    let mut next = t * 997;
                    for _ in 0..LATENCY_WARMUP_CALLS {
                        score_single_row(&booster, rows[next % rows.len()]);
                        next += 1;
                    }
                    let start = Instant::now();
                    let mut count = 0;
                    while start.elapsed() < THROUGHPUT_WINDOW {
                        score_single_row(&booster, rows[next % rows.len()]);
                        next += 1;
                        count += 1;
                    }
                    count
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).sum()
    });
    scored as f64 / THROUGHPUT_WINDOW.as_secs_f64()
}

fn best_of<const N: usize>(mut measure: impl FnMut() -> f64, better: fn(f64, f64) -> f64) -> f64 {
    (1..N).fold(measure(), |best, _| better(best, measure()))
}

fn measure_performance(case: Case, threads: usize, current: &mut BTreeMap<String, f64>) {
    let golden = golden(case);
    let booster = Booster::load_buffer(&golden.model_ubj).unwrap();
    let (data, _) = scoring_rows(case);
    let rows: Vec<&[f32]> = data.chunks(NUM_FEATURES).collect();
    let mut insert = |metric: &str, value: f64| current.insert(format!("{}.{metric}", case.name()), value);

    let (p50, p99) = (0..3)
        .map(|_| latency_percentiles(&booster, &rows))
        .fold((f64::INFINITY, f64::INFINITY), |(a, b), (p50, p99)| {
            (a.min(p50), b.min(p99))
        });
    insert("latency_p50_ns", p50);
    insert("latency_p99_ns", p99);
    insert(
        "throughput_1_thread_per_sec",
        best_of::<3>(|| single_row_throughput(&golden.model_ubj, &rows, 1), f64::max),
    );
    insert(
        &format!("throughput_{threads}_threads_per_sec"),
        best_of::<3>(|| single_row_throughput(&golden.model_ubj, &rows, threads), f64::max),
    );
    // A fresh DMatrix per call: XGBoost caches predictions per matrix.
    let batch_rows_per_sec = best_of::<10>(
        || {
            let start = Instant::now();
            let batch = DMatrix::from_dense(&data, rows.len()).unwrap();
            std::hint::black_box(booster.predict(&batch).unwrap());
            rows.len() as f64 / start.elapsed().as_secs_f64()
        },
        f64::max,
    );
    insert("throughput_batch_rows_per_sec", batch_rows_per_sec);
    let load_model_ns = best_of::<10>(
        || {
            let start = Instant::now();
            std::hint::black_box(Booster::load_buffer(&golden.model_ubj).unwrap());
            start.elapsed().as_nanos() as f64
        },
        f64::min,
    );
    insert("load_model_ns", load_model_ns);
}

#[test]
#[ignore = "timing test: run with --release -- --ignored on an idle machine"]
fn scoring_latency_and_throughput_are_not_regressed() {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    let mut current = BTreeMap::new();
    for case in Case::ALL {
        measure_performance(case, threads, &mut current);
    }

    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let path = golden_dir().join(format!(
        "perf_{}_{}_{profile}.txt",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    let current: BTreeMap<String, String> = current.into_iter().map(|(k, v)| (k, format!("{v:.2}"))).collect();
    if std::env::var_os("XGB_BLESS_PERF").is_some() {
        fs::write(&path, write_key_values(&current)).unwrap();
        eprintln!("recorded performance baseline {}: {current:?}", path.display());
        return;
    }
    let baseline = read_key_values(
        &fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing {} ({e}); record it with XGB_BLESS_PERF=1", path.display())),
    );
    let tolerance: f64 = std::env::var("XGB_PERF_TOLERANCE")
        .map(|v| v.parse().unwrap())
        .unwrap_or(DEFAULT_PERF_TOLERANCE);
    let mut regressions = Vec::new();
    for (name, value) in &current {
        let now: f64 = value.parse().unwrap();
        let Some(before) = baseline.get(name).map(|v| v.parse::<f64>().unwrap()) else {
            eprintln!("{name}: {now:.0} (no baseline)");
            continue;
        };
        eprintln!(
            "{name}: {before:.0} -> {now:.0} ({:+.1}%)",
            (now / before - 1.0) * 100.0
        );
        let slowdown = if name.ends_with("_per_sec") { before / now } else { now / before };
        // p99 is reported only: it swings by more than 1.5x between identical runs.
        if slowdown > tolerance && !name.contains("p99") {
            regressions.push(format!("{name} {before:.0} -> {now:.0}"));
        }
    }
    assert!(
        regressions.is_empty(),
        "more than {tolerance}x slower than the baseline: {regressions:?}"
    );
}
