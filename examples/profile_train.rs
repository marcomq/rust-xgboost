//! Profiling harness: train a hist booster on a synthetic dense dataset and
//! report where wall-clock time goes (construction vs per-round training), so
//! a sampling profiler (macOS `sample`, Instruments, samply) can attribute the
//! per-round milliseconds inside libxgboost. Not a criterion benchmark — the
//! M3 Pro cannot resolve small effects there; this prints raw wall-clock.
//!
//! Usage: cargo run --release --example profile_train [-- OPTIONS]
//! Then, in another shell:  sample <printed pid> 15 -f /tmp/xgb_train_sample.txt
//!
//! Options (all optional, "--key value"):
//!   --rows N       rows in the synthetic dataset      (default 100000)
//!   --cols N       feature columns                    (default 64)
//!   --rounds N     boosting rounds                    (default 50)
//!   --max-bin N    hist max_bin                       (default 256)
//!   --nthread N    xgboost nthread (0 = library default, all cores)
//!   --matrix M     dmatrix | quantile | both          (default both)
//!   --lossguide N  grow_policy=lossguide with max_leaves N (default: depthwise)
//!   --per-round    print every round's time (default: first 5 + summary)

use std::time::Instant;

use xgb::parameters::{self, learning, tree};
use xgb::{Booster, DMatrix};

struct Opts {
    rows: usize,
    cols: usize,
    rounds: usize,
    max_bin: u32,
    nthread: u32,
    matrix: String,
    max_leaves: Option<u32>,
    per_round: bool,
}

fn parse_opts() -> Opts {
    let mut o = Opts {
        rows: 100_000,
        cols: 64,
        rounds: 50,
        max_bin: 256,
        nthread: 0,
        matrix: "both".to_owned(),
        max_leaves: None,
        per_round: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "--per-round" {
            o.per_round = true;
            i += 1;
            continue;
        }
        let val = args.get(i + 1).unwrap_or_else(|| panic!("missing value for {}", key));
        match key {
            "--rows" => o.rows = val.parse().unwrap(),
            "--cols" => o.cols = val.parse().unwrap(),
            "--rounds" => o.rounds = val.parse().unwrap(),
            "--max-bin" => o.max_bin = val.parse().unwrap(),
            "--nthread" => o.nthread = val.parse().unwrap(),
            "--matrix" => o.matrix = val.clone(),
            "--lossguide" => o.max_leaves = Some(val.parse().unwrap()),
            _ => panic!("unknown option {}", key),
        }
        i += 2;
    }
    o
}

/// Deterministic xorshift so runs are comparable without a rand dependency.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Uniform in [0, 1).
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Dense row-major features plus binary labels from a noisy linear rule, so
/// the trees have real structure to find (all-noise labels make degenerate,
/// unrepresentative trees).
fn make_data(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut data = Vec::with_capacity(rows * cols);
    let mut labels = Vec::with_capacity(rows);
    let weights: Vec<f32> = (0..cols).map(|c| if c % 3 == 0 { 1.0 } else { -0.5 }).collect();
    for _ in 0..rows {
        let mut acc = 0.0f32;
        for w in &weights {
            let v = rng.next_f32();
            acc += w * v;
            data.push(v);
        }
        let noise = rng.next_f32() - 0.5;
        labels.push(if acc + noise > 0.0 { 1.0 } else { 0.0 });
    }
    (data, labels)
}

fn params(o: &Opts) -> parameters::BoosterParameters {
    let mut b = tree::TreeBoosterParametersBuilder::default();
    b.tree_method(tree::TreeMethod::Hist)
        .max_depth(6)
        .max_bin(o.max_bin)
        .eta(0.3);
    if let Some(max_leaves) = o.max_leaves {
        // max_depth 0 = unlimited: let max_leaves alone bound lossguide trees.
        b.grow_policy(tree::GrowPolicy::LossGuide)
            .max_leaves(max_leaves)
            .max_depth(0);
    }
    let tree_params = b.build().unwrap();
    let learning_params = learning::LearningTaskParametersBuilder::default()
        .objective(learning::Objective::BinaryLogistic)
        .build()
        .unwrap();
    parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .threads(if o.nthread == 0 { None } else { Some(o.nthread) })
        .build()
        .unwrap()
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Train `rounds` rounds on `dtrain`, print per-round + total time; return
/// total training ms.
fn train(label: &str, dtrain: &DMatrix, o: &Opts) -> f64 {
    let mut booster = Booster::new_with_cached_dmats(&params(o), &[dtrain]).unwrap();
    let mut round_ms = Vec::with_capacity(o.rounds);
    let start = Instant::now();
    for i in 0..o.rounds {
        let t = Instant::now();
        booster.update(dtrain, i as i32).unwrap();
        let m = ms(t.elapsed());
        if o.per_round || i < 5 {
            println!("  [{}] round {:3}: {:8.2} ms", label, i, m);
        }
        round_ms.push(m);
    }
    let total = ms(start.elapsed());
    let mut sorted = round_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    println!(
        "  [{}] {} rounds: total {:.1} ms, mean {:.2} ms/round, median {:.2}, min {:.2}, max {:.2}",
        label,
        o.rounds,
        total,
        total / o.rounds as f64,
        median,
        sorted[0],
        sorted[sorted.len() - 1]
    );
    total
}

fn main() {
    let o = parse_opts();
    println!("pid: {}", std::process::id());
    println!(
        "config: rows={} cols={} rounds={} max_bin={} nthread={} matrix={}",
        o.rows,
        o.cols,
        o.rounds,
        o.max_bin,
        if o.nthread == 0 { "default".to_owned() } else { o.nthread.to_string() },
        o.matrix
    );

    let t = Instant::now();
    let (data, labels) = make_data(o.rows, o.cols);
    println!("data generation: {:.1} ms", ms(t.elapsed()));

    if o.matrix == "dmatrix" || o.matrix == "both" {
        let t = Instant::now();
        let mut dtrain = DMatrix::from_dense(&data, o.rows).unwrap();
        dtrain.set_labels(&labels).unwrap();
        let build = ms(t.elapsed());
        println!("[dmatrix] construction: {:.1} ms", build);
        let train_ms = train("dmatrix", &dtrain, &o);
        println!("[dmatrix] construction + training: {:.1} ms", build + train_ms);
    }

    if o.matrix == "quantile" || o.matrix == "both" {
        let t = Instant::now();
        let dtrain = DMatrix::from_dense_quantile(&data, o.rows, Some(&labels), o.max_bin).unwrap();
        let build = ms(t.elapsed());
        println!("[quantile] construction: {:.1} ms", build);
        let train_ms = train("quantile", &dtrain, &o);
        println!("[quantile] construction + training: {:.1} ms", build + train_ms);
    }
}
