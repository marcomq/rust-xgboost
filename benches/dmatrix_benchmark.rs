//! Benchmarks for DMatrix creation methods.
//!
//! This benchmark compares different DMatrix creation strategies:
//! - Auto-tuned (default): Uses single-thread for small matrices, multi-thread for large
//! - Single-threaded: Always uses nthread=1
//! - Multi-threaded: Always uses default threading (all cores)
//!
//! Run with: cargo bench

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::ffi;
use std::hint::black_box;
use std::ptr;

// We need to access the internal XGBoost functions directly for comparison
extern crate xgboost_sys;

/// Creates a JSON-encoded array interface string for f32 data.
fn make_array_interface_f32(data: &[f32]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<f4","version":3}}"#,
        ptr, len
    )
}

/// Creates a JSON-encoded array interface string for u64 data.
fn make_array_interface_u64(data: &[u64]) -> String {
    let ptr = data.as_ptr() as usize;
    let len = data.len();
    format!(
        r#"{{"data":[{},false],"shape":[{}],"strides":null,"typestr":"<u8","version":3}}"#,
        ptr, len
    )
}

/// Generate sparse CSR data with specified dimensions and density.
fn generate_sparse_data(num_rows: usize, num_cols: usize, density: f64) -> (Vec<u64>, Vec<u64>, Vec<f32>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    // Simple deterministic pseudo-random for reproducibility
    let mut seed: u64 = 12345;
    let mut nnz = 0u64;

    for _ in 0..num_rows {
        for col in 0..num_cols {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let rand = (seed as f64) / (u64::MAX as f64);
            if rand < density {
                indices.push(col as u64);
                data.push(rand as f32);
                nnz += 1;
            }
        }
        indptr.push(nnz);
    }

    (indptr, indices, data)
}

/// Create DMatrix using single-threaded mode (nthread=1).
fn from_csr_single_thread(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
) -> Result<xgboost_sys::DMatrixHandle, i32> {
    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();
    let config = ffi::CString::new(r#"{"missing": NaN, "nthread": 1}"#).unwrap();

    let ret = unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        )
    };

    if ret == 0 { Ok(handle) } else { Err(ret) }
}

/// Create DMatrix using multi-threaded mode (default threading).
fn from_csr_multi_thread(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
) -> Result<xgboost_sys::DMatrixHandle, i32> {
    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();
    let config = ffi::CString::new(r#"{"missing": NaN}"#).unwrap();

    let ret = unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        )
    };

    if ret == 0 { Ok(handle) } else { Err(ret) }
}

/// Create DMatrix using auto-tuned mode (single-thread for small, multi-thread for large).
/// This mirrors the implementation in DMatrix::from_csr.
fn from_csr_auto_tuned(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
) -> Result<xgboost_sys::DMatrixHandle, i32> {
    const SINGLE_THREAD_THRESHOLD: usize = 5000;

    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();

    // Use single thread for small matrices to avoid thread synchronization overhead
    let config = if data.len() < SINGLE_THREAD_THRESHOLD {
        ffi::CString::new(r#"{"missing": NaN, "nthread": 1}"#).unwrap()
    } else {
        ffi::CString::new(r#"{"missing": NaN}"#).unwrap()
    };

    let ret = unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        )
    };

    if ret == 0 { Ok(handle) } else { Err(ret) }
}

/// Free a DMatrix handle.
fn free_dmatrix(handle: xgboost_sys::DMatrixHandle) {
    unsafe {
        xgboost_sys::XGDMatrixFree(handle);
    }
}

/// Generate a dense row-major matrix of the given shape.
fn generate_dense_data(num_rows: usize, num_cols: usize) -> Vec<f32> {
    let mut seed: u64 = 6789;
    (0..num_rows * num_cols)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64 / u64::MAX as f64) as f32
        })
        .collect()
}

/// Old path: legacy, hardcoded single-threaded XGDMatrixCreateFromMat.
fn from_dense_legacy_mat(data: &[f32], num_rows: usize) -> Result<xgboost_sys::DMatrixHandle, i32> {
    let mut handle = ptr::null_mut();
    let ret = unsafe {
        xgboost_sys::XGDMatrixCreateFromMat(
            data.as_ptr(),
            num_rows as xgboost_sys::bst_ulong,
            (data.len() / num_rows) as xgboost_sys::bst_ulong,
            f32::NAN,
            &mut handle,
        )
    };
    if ret == 0 { Ok(handle) } else { Err(ret) }
}

/// Legacy parallel path: XGDMatrixCreateFromMat_omp with default thread count.
fn from_dense_legacy_mat_omp(data: &[f32], num_rows: usize) -> Result<xgboost_sys::DMatrixHandle, i32> {
    let mut handle = ptr::null_mut();
    let ret = unsafe {
        xgboost_sys::XGDMatrixCreateFromMat_omp(
            data.as_ptr(),
            num_rows as xgboost_sys::bst_ulong,
            (data.len() / num_rows) as xgboost_sys::bst_ulong,
            f32::NAN,
            &mut handle,
            0, // 0 = use omp default thread count
        )
    };
    if ret == 0 { Ok(handle) } else { Err(ret) }
}

/// New path: array-interface XGDMatrixCreateFromDense with single/multi-thread
/// auto-tuning, mirroring DMatrix::from_dense.
fn from_dense_array_interface(data: &[f32], num_rows: usize) -> Result<xgboost_sys::DMatrixHandle, i32> {
    const SINGLE_THREAD_THRESHOLD: usize = 30000;
    let mut handle = ptr::null_mut();
    let num_cols = data.len() / num_rows;
    let ptr = data.as_ptr() as usize;
    let data_interface = format!(
        r#"{{"data":[{},false],"shape":[{},{}],"strides":null,"typestr":"<f4","version":3}}"#,
        ptr, num_rows, num_cols
    );
    let data_json = ffi::CString::new(data_interface).unwrap();
    let config = if data.len() < SINGLE_THREAD_THRESHOLD {
        ffi::CString::new(r#"{"missing": NaN, "nthread": 1}"#).unwrap()
    } else {
        ffi::CString::new(r#"{"missing": NaN}"#).unwrap()
    };
    let ret = unsafe { xgboost_sys::XGDMatrixCreateFromDense(data_json.as_ptr(), config.as_ptr(), &mut handle) };
    if ret == 0 { Ok(handle) } else { Err(ret) }
}

fn bench_from_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_dense");

    // (num_rows, num_cols, label) -> num_rows * num_cols elements
    let test_cases = [
        (1000, 10, "10k_elems"),   // below 30k threshold
        (5000, 10, "50k_elems"),   // above threshold
        (10000, 50, "500k_elems"), // large
        (50000, 50, "2.5M_elems"), // xlarge
    ];

    for (num_rows, num_cols, label) in test_cases {
        let data = generate_dense_data(num_rows, num_cols);

        group.bench_with_input(BenchmarkId::new("legacy_CreateFromMat", label), &data, |b, data| {
            b.iter(|| {
                let handle = from_dense_legacy_mat(black_box(data), num_rows).unwrap();
                free_dmatrix(handle);
            })
        });

        group.bench_with_input(BenchmarkId::new("legacy_CreateFromMat_omp", label), &data, |b, data| {
            b.iter(|| {
                let handle = from_dense_legacy_mat_omp(black_box(data), num_rows).unwrap();
                free_dmatrix(handle);
            })
        });

        group.bench_with_input(BenchmarkId::new("array_CreateFromDense", label), &data, |b, data| {
            b.iter(|| {
                let handle = from_dense_array_interface(black_box(data), num_rows).unwrap();
                free_dmatrix(handle);
            })
        });
    }

    group.finish();
}

fn bench_from_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_csr");

    // Test different matrix sizes
    // Format: (num_rows, density) -> results in approximately num_rows * 100 * density non-zeros
    let test_cases = [
        (500, 0.10, "small_5k_nnz"),      // ~5000 nnz (at threshold)
        (1000, 0.05, "small_5k_nnz_2"),   // ~5000 nnz (at threshold)
        (1000, 0.10, "medium_10k_nnz"),   // ~10000 nnz (above threshold)
        (5000, 0.10, "large_50k_nnz"),    // ~50000 nnz (well above threshold)
        (10000, 0.10, "xlarge_100k_nnz"), // ~100000 nnz (large matrix)
    ];

    for (num_rows, density, label) in test_cases {
        let num_cols = 100;
        let (indptr, indices, data) = generate_sparse_data(num_rows, num_cols, density);
        let nnz = data.len();

        // Single-threaded baseline
        group.bench_with_input(
            BenchmarkId::new("single_thread", label),
            &(&indptr, &indices, &data, num_cols, nnz),
            |b, (indptr, indices, data, num_cols, _)| {
                b.iter(|| {
                    let handle =
                        from_csr_single_thread(black_box(*indptr), black_box(*indices), black_box(*data), *num_cols)
                            .unwrap();
                    free_dmatrix(handle);
                })
            },
        );

        // Multi-threaded
        group.bench_with_input(
            BenchmarkId::new("multi_thread", label),
            &(&indptr, &indices, &data, num_cols, nnz),
            |b, (indptr, indices, data, num_cols, _)| {
                b.iter(|| {
                    let handle =
                        from_csr_multi_thread(black_box(*indptr), black_box(*indices), black_box(*data), *num_cols)
                            .unwrap();
                    free_dmatrix(handle);
                })
            },
        );

        // Auto-tuned (our implementation)
        group.bench_with_input(
            BenchmarkId::new("auto_tuned", label),
            &(&indptr, &indices, &data, num_cols, nnz),
            |b, (indptr, indices, data, num_cols, _)| {
                b.iter(|| {
                    let handle =
                        from_csr_auto_tuned(black_box(*indptr), black_box(*indices), black_box(*data), *num_cols)
                            .unwrap();
                    free_dmatrix(handle);
                })
            },
        );
    }

    group.finish();
}

/// Print a formatted comparison table after benchmarks complete.
/// Run this separately with: cargo run --release --example bench_table
#[cfg(feature = "print_table")]
fn print_comparison_table() {
    use std::time::Instant;

    println!();
    println!("=============================================================================");
    println!("                    DMatrix::from_csr Performance Comparison                 ");
    println!("=============================================================================");
    println!();
    println!("Threshold: 5000 non-zeros (single-thread below, multi-thread above)");
    println!();
    println!(
        "{:>12} | {:>8} | {:>12} | {:>12} | {:>12} | {:>8}",
        "Rows", "NNZ", "Single (µs)", "Multi (µs)", "Auto (µs)", "Winner"
    );
    println!(
        "{:-<12}-+-{:-<8}-+-{:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    let test_cases = [
        (500, 0.10),   // ~5000 nnz
        (1000, 0.05),  // ~5000 nnz
        (1000, 0.10),  // ~10000 nnz
        (5000, 0.10),  // ~50000 nnz
        (10000, 0.10), // ~100000 nnz
    ];

    for (num_rows, density) in test_cases {
        let num_cols = 100;
        let (indptr, indices, data) = generate_sparse_data(num_rows, num_cols, density);
        let nnz = data.len();
        let iterations = if nnz < 20000 { 100 } else { 50 };

        // Warmup
        for _ in 0..5 {
            let h = from_csr_single_thread(&indptr, &indices, &data, num_cols).unwrap();
            free_dmatrix(h);
        }

        // Single-threaded
        let start = Instant::now();
        for _ in 0..iterations {
            let h = from_csr_single_thread(&indptr, &indices, &data, num_cols).unwrap();
            free_dmatrix(h);
        }
        let single_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Multi-threaded
        let start = Instant::now();
        for _ in 0..iterations {
            let h = from_csr_multi_thread(&indptr, &indices, &data, num_cols).unwrap();
            free_dmatrix(h);
        }
        let multi_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Auto-tuned
        let start = Instant::now();
        for _ in 0..iterations {
            let h = from_csr_auto_tuned(&indptr, &indices, &data, num_cols).unwrap();
            free_dmatrix(h);
        }
        let auto_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Determine winner
        let min_time = single_us.min(multi_us).min(auto_us);
        let winner = if (auto_us - min_time).abs() < 1.0 {
            "Auto"
        } else if (single_us - min_time).abs() < 1.0 {
            "Single"
        } else {
            "Multi"
        };

        // Mark if auto picked the right strategy
        let optimal = if (nnz < 5000 && single_us <= multi_us) || (nnz >= 5000 && multi_us <= single_us) {
            "+"
        } else {
            "-"
        };

        println!(
            "{:>12} | {:>8} | {:>12.2} | {:>12.2} | {:>12.2} | {:>6} {}",
            num_rows, nnz, single_us, multi_us, auto_us, winner, optimal
        );
    }

    println!();
    println!("Legend:");
    println!("  + = Auto-tuning chose the optimal strategy for this matrix size");
    println!("  - = Auto-tuning did not choose the optimal strategy");
    println!("  S = Auto uses Single-thread, M = Auto uses Multi-thread");
    println!();
}

criterion_group!(benches, bench_from_dense, bench_from_csr);
criterion_main!(benches);

// Allow running the table printer directly
#[cfg(feature = "print_table")]
fn main() {
    print_comparison_table();
}
