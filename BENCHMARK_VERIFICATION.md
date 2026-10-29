# Benchmark Verification Guide

This guide explains how to independently verify the benchmark results comparing the deprecated `XGDMatrixCreateFromCSREx` API with the new `XGDMatrixCreateFromCSR` API.

## Quick Summary

Since XGBoost 3.1 removed the deprecated functions, you cannot run both APIs in the same build. To verify the comparison, you need to:

1. Run the deprecated API benchmark on the old code (pre-upgrade)
2. Run the new API benchmark on the updated code
3. Compare the results

## Step-by-Step Verification

### Step 1: Benchmark the Deprecated API (XGBoost 3.0)

First, checkout the commit before the XGBoost upgrade and run the deprecated benchmark:

```bash
# Save current branch
git stash  # if you have uncommitted changes

# Checkout the version with XGBoost 3.0.x (before upgrade)
git checkout <commit-before-upgrade>

# Build and run the deprecated benchmark
cargo run --release --example bench_deprecated --features deprecated_api
```

You should see output like:
```
    Rows |      NNZ | Deprecated (us)
---------+----------+----------------
     100 |      979 |          56.00
    1000 |    10037 |         487.00
    2000 |    19849 |         980.00
    ...
```

**Save these results!**

### Step 2: Benchmark the New API (XGBoost 3.1.3)

Switch back to the updated code and run the new benchmark:

```bash
# Return to the updated branch
git checkout <your-branch>
git stash pop  # if you stashed changes

# Build and run the new benchmark
cargo run --release --example bench_table
```

### Step 3: Compare Results

Compare the "Deprecated (us)" column from Step 1 with the "Single (us)" or "Auto (us)" columns from Step 2.

## Alternative: Use Git Worktrees

If you want to run both benchmarks without switching branches:

```bash
# Create a worktree for the old version
git worktree add ../rust-xgboost-old <commit-before-upgrade>

# Run deprecated benchmark in old worktree
cd ../rust-xgboost-old
cargo run --release --example bench_deprecated --features deprecated_api

# Run new benchmark in current directory
cd ../rust-xgboost
cargo run --release --example bench_table
```

## Expected Results

Based on our testing, you should observe:

| Matrix Size | XGBoost 3.0 Deprecated | XGBoost 3.1.3 (Auto) | Difference |
|-------------|------------------------|----------------------|------------|
| ~1,000 NNZ  | ~56 us                 | ~43 us               | 24% faster |
| ~10,000 NNZ | ~487 us                | ~381 us              | 22% faster |
| ~50,000 NNZ | ~2,400 us              | ~400 us              | 83% faster |

## Why Can't We Run Both in One Build?

The deprecated `XGDMatrixCreateFromCSREx` and `XGDMatrixCreateFromCSCEx` functions were completely removed from the XGBoost C API in version 3.1. There is no way to link against both the old and new XGBoost libraries simultaneously in a single Rust build.

The only way to compare is to build and run benchmarks separately against each XGBoost version.

## Reproducing Original Benchmark Data

The benchmark uses a deterministic PRNG (seed: 12345) to generate test matrices, so the same matrix sizes will produce identical data across runs. This ensures fair comparisons between different XGBoost versions.

## Questions?

If you have trouble reproducing these results or have questions about the methodology, please open an issue on the repository.
