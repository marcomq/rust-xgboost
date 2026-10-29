//! Benchmark for deprecated XGDMatrixCreateFromCSREx API
//!
//! This benchmark ONLY works with XGBoost versions prior to 3.1 where the
//! deprecated functions still exist.
//!
//! To run this benchmark:
//! 1. Checkout the old xgboost-sys version (see BENCHMARK_VERIFICATION.md)
//! 2. Run: cargo run --release --example bench_deprecated
//!
//! This will output timing results that can be compared against bench_table.rs

extern crate xgboost_sys;

/// Generate sparse CSR data with specified dimensions and density.
/// Uses the same algorithm as bench_table.rs for reproducibility.
#[cfg(feature = "deprecated_api")]
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

#[cfg(feature = "deprecated_api")]
fn from_csr_deprecated(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
    num_rows: usize,
) -> xgboost_sys::DMatrixHandle {
    let mut handle = ptr::null_mut();

    unsafe {
        xgboost_sys::XGDMatrixCreateFromCSREx(
            indptr.as_ptr(),
            indices.as_ptr() as *const u32,
            data.as_ptr(),
            indptr.len(),
            data.len(),
            num_cols,
            &mut handle,
        );
    }
    handle
}

#[cfg(feature = "deprecated_api")]
fn free_dmatrix(handle: xgboost_sys::DMatrixHandle) {
    unsafe {
        xgboost_sys::XGDMatrixFree(handle);
    }
}

#[cfg(feature = "deprecated_api")]
fn benchmark_deprecated(
    indptr: &[u64],
    indices: &[u64],
    data: &[f32],
    num_cols: usize,
    num_rows: usize,
    iterations: usize,
) -> f64 {
    // Warmup
    for _ in 0..5 {
        let h = from_csr_deprecated(indptr, indices, data, num_cols, num_rows);
        free_dmatrix(h);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let h = from_csr_deprecated(indptr, indices, data, num_cols, num_rows);
        free_dmatrix(h);
    }
    start.elapsed().as_micros() as f64 / iterations as f64
}

#[cfg(feature = "deprecated_api")]
fn main() {
    println!();
    println!("==============================================================================");
    println!("           Deprecated XGDMatrixCreateFromCSREx Benchmark                      ");
    println!("==============================================================================");
    println!();
    println!("This benchmark uses the deprecated API that was removed in XGBoost 3.1");
    println!();

    let test_cases = [
        (100, 100, 0.10),   // ~1000 nnz
        (1000, 100, 0.10),  // ~10000 nnz
        (2000, 100, 0.10),  // ~20000 nnz
        (3000, 100, 0.10),  // ~30000 nnz
        (5000, 100, 0.10),  // ~50000 nnz
        (10000, 100, 0.10), // ~100000 nnz
    ];

    println!("{:>8} | {:>8} | {:>14}", "Rows", "NNZ", "Deprecated (us)");
    println!("{:-<8}-+-{:-<8}-+-{:-<14}", "", "", "");

    for (num_rows, num_cols, density) in test_cases {
        let (indptr, indices, data) = generate_sparse_data(num_rows, num_cols, density);
        let nnz = data.len();
        let iterations = if nnz < 20000 { 100 } else { 50 };

        let deprecated_us = benchmark_deprecated(&indptr, &indices, &data, num_cols, num_rows, iterations);

        println!("{:>8} | {:>8} | {:>14.2}", num_rows, nnz, deprecated_us);
    }

    println!();
    println!("Save these results to compare with XGBoost 3.1.3 bench_table.rs output.");
    println!();
}

#[cfg(not(feature = "deprecated_api"))]
fn main() {
    eprintln!();
    eprintln!("==============================================================================");
    eprintln!("  ERROR: This benchmark requires the deprecated XGBoost API                   ");
    eprintln!("==============================================================================");
    eprintln!();
    eprintln!("The deprecated XGDMatrixCreateFromCSREx function was removed in XGBoost 3.1.");
    eprintln!();
    eprintln!("To run this benchmark, you need to:");
    eprintln!("  1. Checkout an older version of xgboost-sys (pre-3.1)");
    eprintln!("  2. Enable the deprecated_api feature");
    eprintln!();
    eprintln!("See BENCHMARK_VERIFICATION.md for detailed instructions.");
    eprintln!();
}
