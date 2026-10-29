//! Benchmark comparison table for DMatrix::from_csr
//!
//! This example runs a quick benchmark comparing different threading strategies
//! and displays results in a formatted table.
//!
//! Run with: cargo run --release --example bench_table

use std::ffi;
use std::ptr;
use std::time::Instant;

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
) -> xgboost_sys::DMatrixHandle {
    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();
    let config = ffi::CString::new(r#"{"missing": NaN, "nthread": 1}"#).unwrap();

    unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        );
    }
    handle
}

/// Create DMatrix using multi-threaded mode (default threading).
fn from_csr_multi_thread(indptr: &[u64], indices: &[u64], data: &[f32], num_cols: usize) -> xgboost_sys::DMatrixHandle {
    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();
    let config = ffi::CString::new(r#"{"missing": NaN}"#).unwrap();

    unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        );
    }
    handle
}

/// Create DMatrix using auto-tuned mode (mirrors DMatrix::from_csr implementation).
fn from_csr_auto_tuned(indptr: &[u64], indices: &[u64], data: &[f32], num_cols: usize) -> xgboost_sys::DMatrixHandle {
    const SINGLE_THREAD_THRESHOLD: usize = 30000;

    let mut handle = ptr::null_mut();

    let indptr_json = ffi::CString::new(make_array_interface_u64(indptr)).unwrap();
    let indices_json = ffi::CString::new(make_array_interface_u64(indices)).unwrap();
    let data_json = ffi::CString::new(make_array_interface_f32(data)).unwrap();

    let config = if data.len() < SINGLE_THREAD_THRESHOLD {
        ffi::CString::new(r#"{"missing": NaN, "nthread": 1}"#).unwrap()
    } else {
        ffi::CString::new(r#"{"missing": NaN}"#).unwrap()
    };

    unsafe {
        xgboost_sys::XGDMatrixCreateFromCSR(
            indptr_json.as_ptr(),
            indices_json.as_ptr(),
            data_json.as_ptr(),
            num_cols as xgboost_sys::bst_ulong,
            config.as_ptr(),
            &mut handle,
        );
    }
    handle
}

/// Free a DMatrix handle.
fn free_dmatrix(handle: xgboost_sys::DMatrixHandle) {
    unsafe {
        xgboost_sys::XGDMatrixFree(handle);
    }
}

fn benchmark_method<F>(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
    iterations: usize,
    method: F,
) -> f64
where
    F: Fn(&[u64], &[u64], &[f32], usize) -> xgboost_sys::DMatrixHandle,
{
    // Warmup
    for _ in 0..5 {
        let h = method(indptr, indices, data, num_cols);
        free_dmatrix(h);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let h = method(indptr, indices, data, num_cols);
        free_dmatrix(h);
    }
    start.elapsed().as_micros() as f64 / iterations as f64
}

fn main() {
    println!();
    println!("==============================================================================");
    println!("                    DMatrix::from_csr Performance Comparison                  ");
    println!("==============================================================================");
    println!();
    println!("Auto-tuning threshold: 30000 non-zeros");
    println!("  - Below threshold: uses single-thread (nthread=1)");
    println!("  - Above threshold: uses multi-thread (default)");
    println!();
    println!(
        "{:>8} | {:>8} | {:>14} | {:>14} | {:>14} | {:>8}",
        "Rows", "NNZ", "Single (us)", "Multi (us)", "Auto (us)", "Winner"
    );
    println!(
        "{:-<8}-+-{:-<8}-+-{:-<14}-+-{:-<14}-+-{:-<14}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    let test_cases = [
        (200, 0.10),   // ~2000 nnz (below threshold)
        (500, 0.05),   // ~2500 nnz (below threshold)
        (500, 0.10),   // ~5000 nnz (at threshold)
        (1000, 0.10),  // ~10000 nnz (above threshold)
        (2000, 0.10),  // ~20000 nnz (above threshold)
        (5000, 0.10),  // ~50000 nnz (well above threshold)
        (10000, 0.10), // ~100000 nnz (large matrix)
    ];

    for (num_rows, density) in test_cases {
        let num_cols = 100;
        let (indptr, indices, data) = generate_sparse_data(num_rows, num_cols, density);
        let nnz = data.len();
        let iterations = if nnz < 20000 { 100 } else { 50 };

        let single_us = benchmark_method(&indptr, &indices, &data, num_cols, iterations, from_csr_single_thread);

        let multi_us = benchmark_method(&indptr, &indices, &data, num_cols, iterations, from_csr_multi_thread);

        let auto_us = benchmark_method(&indptr, &indices, &data, num_cols, iterations, from_csr_auto_tuned);

        // Determine actual winner (fastest method)
        let winner = if single_us <= multi_us { "Single" } else { "Multi" };

        // What did auto-tuning choose?
        let auto_chose = if nnz < 30000 { "Single" } else { "Multi" };

        // Did auto choose the actual winner?
        let auto_status = if auto_chose == winner { "OK" } else { "MISS" };

        println!(
            "{:>8} | {:>8} | {:>14.2} | {:>14.2} | {:>14.2} | {:>6} ({})",
            num_rows, nnz, single_us, multi_us, auto_us, winner, auto_status
        );
    }

    println!();
    println!("------------------------------------------------------------------------------");
    println!("Finding optimal threshold (where multi-thread becomes faster):");
    println!();

    // Test various sizes to find crossover point
    let test_sizes = [
        (100, 0.10),  // ~1000 nnz
        (200, 0.10),  // ~2000 nnz
        (500, 0.10),  // ~5000 nnz
        (1000, 0.10), // ~10000 nnz
        (2000, 0.10), // ~20000 nnz
        (3000, 0.10), // ~30000 nnz
        (4000, 0.10), // ~40000 nnz
        (5000, 0.10), // ~50000 nnz
    ];

    println!(
        "{:>10} | {:>12} | {:>12} | {:>10}",
        "NNZ", "Single (us)", "Multi (us)", "Faster"
    );
    println!("{:-<10}-+-{:-<12}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    let mut crossover_nnz = 0;
    for (num_rows, density) in test_sizes {
        let (indptr, indices, data) = generate_sparse_data(num_rows, 100, density);
        let nnz = data.len();
        let iterations = if nnz < 20000 { 100 } else { 50 };

        let single_us = benchmark_method(&indptr, &indices, &data, 100, iterations, from_csr_single_thread);
        let multi_us = benchmark_method(&indptr, &indices, &data, 100, iterations, from_csr_multi_thread);

        let faster = if single_us <= multi_us { "Single" } else { "Multi" };
        if faster == "Multi" && crossover_nnz == 0 {
            crossover_nnz = nnz;
        }

        println!(
            "{:>10} | {:>12.2} | {:>12.2} | {:>10}",
            nnz, single_us, multi_us, faster
        );
    }

    println!();
    if crossover_nnz > 0 {
        println!(
            "Crossover point: Multi-threading becomes faster around {} non-zeros",
            crossover_nnz
        );
        println!();
        if crossover_nnz > 5000 {
            let suggested = ((crossover_nnz / 10000) * 10000).max(10000);
            println!(
                "RECOMMENDATION: Consider adjusting SINGLE_THREAD_THRESHOLD (currently 30000) to ~{}",
                suggested
            );
        }
    } else {
        println!("Single-threaded was faster for all tested sizes up to 50k non-zeros.");
        println!("Consider increasing the threshold significantly or removing multi-threading.");
    }
    println!();
}
