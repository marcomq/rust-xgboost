//! Proves the inplace-prediction serving loop performs zero Rust-side heap
//! allocations per call once warmed up.
//!
//! Criterion timings on a shared developer machine cannot resolve the
//! ~100-250ns/call that allocation removal saves (run-to-run scheduling noise
//! is +/-10-30%), so the property is asserted directly with a counting global
//! allocator instead. Only the Rust wrapper's allocations are counted; the C++
//! side of XGBoost does not go through Rust's `#[global_allocator]`.
//!
//! The counter is per-thread: the libtest harness runs tests on parallel
//! threads and prints progress from its own, and with a process-global counter
//! those allocations bleed into the measurement window (this flaked on the
//! Windows/Linux CI runners, whose debug-build timing overlapped the tests).
//! Each test also hammers the allocator from a background thread while
//! measuring, so that immunity is itself under test rather than a matter of
//! scheduling luck.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use xgb::parameters::{self, learning, tree};
use xgb::{Booster, DMatrix};

struct CountingAlloc;

thread_local! {
    /// Allocations made by the current thread. `const` init keeps first
    /// access inside the allocator hook allocation-free.
    static THREAD_ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // try_with: never panic (and so abort) inside the allocator, even for
        // allocations during thread teardown.
        let _ = THREAD_ALLOCS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|count| count.set(count.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Allocations made so far by the calling thread.
fn allocations() -> usize {
    THREAD_ALLOCS.with(Cell::get)
}

/// Run `measured` while a background thread allocates in a tight loop,
/// returning the number of allocations the *current* thread made inside it.
/// The noise thread reproduces, deterministically, the concurrent-allocation
/// conditions that made a process-global counter flake in CI.
fn count_own_allocations(measured: impl FnOnce()) -> usize {
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            while !stop.load(Ordering::Relaxed) {
                std::hint::black_box(vec![0u8; 64]);
            }
        });
        let before = allocations();
        measured();
        let after = allocations();
        stop.store(true, Ordering::Relaxed);
        after - before
    })
}

/// Deterministic dense binary-classification dataset.
fn synthetic_dense(num_rows: usize, num_cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut data = vec![0f32; num_rows * num_cols];
    let mut labels = vec![0f32; num_rows];
    let mut seed = 1u64;
    for i in 0..num_rows {
        let mut acc = 0f32;
        for v in &mut data[i * num_cols..(i + 1) * num_cols] {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            *v = (seed % 1000) as f32 / 1000.0;
            acc += *v;
        }
        labels[i] = if acc > num_cols as f32 / 2.0 { 1.0 } else { 0.0 };
    }
    (data, labels)
}

/// CSR view of the same kind of data, ~40% dense.
fn synthetic_csr(num_rows: usize, num_cols: usize) -> (Vec<u64>, Vec<u64>, Vec<f32>, Vec<f32>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut labels = vec![0f32; num_rows];
    let mut seed = 7u64;
    let mut nnz = 0u64;
    for label in labels.iter_mut() {
        let mut acc = 0f32;
        for j in 0..num_cols {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            if seed % 10 < 4 {
                let v = (seed % 1000) as f32 / 1000.0;
                indices.push(j as u64);
                values.push(v);
                acc += v;
                nnz += 1;
            }
        }
        indptr.push(nnz);
        *label = if acc > 1.0 { 1.0 } else { 0.0 };
    }
    (indptr, indices, values, labels)
}

fn trained_booster(dm: &DMatrix) -> Booster {
    let tree_params = tree::TreeBoosterParametersBuilder::default()
        .tree_method(tree::TreeMethod::Hist)
        .max_depth(4)
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
    let mut bst = Booster::new_with_cached_dmats(&params, &[dm]).unwrap();
    for i in 0..10 {
        bst.update(dm, i).unwrap();
    }
    bst
}

#[test]
fn inplace_predict_into_is_allocation_free_when_warm() {
    let (num_rows, num_cols) = (128, 16);
    let (data, labels) = synthetic_dense(num_rows, num_cols);
    let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
    dm.set_labels(&labels).unwrap();
    let mut bst = trained_booster(&dm);
    bst.set_param("nthread", "1").unwrap();

    let single_row = &data[..num_cols];
    let mut out = Vec::new();

    // Warm up: creates the proxy DMatrix and grows `out` to capacity.
    bst.predict_from_dense_into(single_row, 1, &mut out).unwrap();
    let expected = out.clone();

    let allocs = count_own_allocations(|| {
        for _ in 0..100 {
            bst.predict_from_dense_into(single_row, 1, &mut out).unwrap();
        }
    });

    assert_eq!(out, expected, "warm calls must keep producing identical predictions");
    assert_eq!(allocs, 0, "predict_from_dense_into must not allocate once warm");
}

#[test]
fn inplace_csr_predict_into_is_allocation_free_when_warm() {
    let (num_rows, num_cols) = (128, 16);
    let (indptr, indices, values, labels) = synthetic_csr(num_rows, num_cols);
    let mut dm = DMatrix::from_csr(&indptr, &indices, &values, Some(num_cols)).unwrap();
    dm.set_labels(&labels).unwrap();
    let mut bst = trained_booster(&dm);
    bst.set_param("nthread", "1").unwrap();

    // Single-row CSR slice (first row).
    let row_indptr = &indptr[..2];
    let row_nnz = indptr[1] as usize;
    let row_indices = &indices[..row_nnz];
    let row_values = &values[..row_nnz];
    let mut out = Vec::new();

    bst.predict_from_csr_into(row_indptr, row_indices, row_values, num_cols, &mut out)
        .unwrap();
    let expected = out.clone();

    let allocs = count_own_allocations(|| {
        for _ in 0..100 {
            bst.predict_from_csr_into(row_indptr, row_indices, row_values, num_cols, &mut out)
                .unwrap();
        }
    });

    assert_eq!(out, expected, "warm calls must keep producing identical predictions");
    assert_eq!(allocs, 0, "predict_from_csr_into must not allocate once warm");
}

#[test]
fn dmatrix_predict_into_is_allocation_free_when_warm() {
    let (num_rows, num_cols) = (128, 16);
    let (data, labels) = synthetic_dense(num_rows, num_cols);
    let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
    dm.set_labels(&labels).unwrap();
    let mut bst = trained_booster(&dm);
    bst.set_param("nthread", "1").unwrap();

    let mut out = Vec::new();

    // Warm up: fills XGBoost's prediction cache and grows `out` to capacity.
    bst.predict_into(&dm, &mut out).unwrap();
    let expected = out.clone();

    let allocs = count_own_allocations(|| {
        for _ in 0..100 {
            bst.predict_into(&dm, &mut out).unwrap();
        }
    });

    assert_eq!(out, expected, "warm calls must keep producing identical predictions");
    assert_eq!(allocs, 0, "predict_into must not allocate once warm");
}

#[test]
fn dmatrix_predict_margin_into_is_allocation_free_when_warm() {
    // Same shape as the predict_into test, but for the margin variant — the
    // only predict-into path previously unpinned by this suite. It shares
    // `predict_raw` with `predict_into`, differing only in the static config,
    // so this asserts nothing allocating creeps into that variant.
    let (num_rows, num_cols) = (128, 16);
    let (data, labels) = synthetic_dense(num_rows, num_cols);
    let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
    dm.set_labels(&labels).unwrap();
    let mut bst = trained_booster(&dm);
    bst.set_param("nthread", "1").unwrap();

    let mut out = Vec::new();

    bst.predict_margin_into(&dm, &mut out).unwrap();
    let expected = out.clone();

    let allocs = count_own_allocations(|| {
        for _ in 0..100 {
            bst.predict_margin_into(&dm, &mut out).unwrap();
        }
    });

    assert_eq!(out, expected, "warm calls must keep producing identical predictions");
    assert_eq!(allocs, 0, "predict_margin_into must not allocate once warm");
}

#[test]
fn custom_objective_round_allocates_only_objective_outputs() {
    // `boost()` builds the gradient/hessian array-interface JSON on the stack
    // (see `write_interface_{1d,2d}` in booster.rs), so the only per-round Rust
    // allocations left in `update_custom` are the two Vecs the objective itself
    // returns. Pin that count exactly: a regression back to heap-built JSON
    // (previously 4+ allocations per round — two `format!` Strings plus two
    // `CString`s) fails this assertion.
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

    let (num_rows, num_cols) = (128, 16);
    let (data, labels) = synthetic_dense(num_rows, num_cols);
    let mut dm = DMatrix::from_dense(&data, num_rows).unwrap();
    dm.set_labels(&labels).unwrap();
    let mut bst = trained_booster(&dm);
    bst.set_param("nthread", "1").unwrap();

    // Warm up one custom round (fills XGBoost's internal prediction caches).
    bst.update_custom(&dm, 10, logistic_obj).unwrap();

    const ROUNDS: i32 = 10;
    let allocs = count_own_allocations(|| {
        for i in 0..ROUNDS {
            bst.update_custom(&dm, 11 + i, logistic_obj).unwrap();
        }
    });

    assert_eq!(
        allocs,
        2 * ROUNDS as usize,
        "update_custom must allocate only the objective's grad/hess Vecs per round"
    );
}
