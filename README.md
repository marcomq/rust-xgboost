[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Macos/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/macos.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Linux/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/linux.yml)
[![Actions Status](https://github.com/agene0001/rust-xgboost/workflows/Windows/badge.svg)](https://github.com/agene0001/rust-xgboost/actions/workflows/windows.yml)


# rust-xgboost


This is mostly a fork of https://github.com/davechallis/rust-xgboost but uses 
another xgboost version and links it dynamically instead of linking it statically as in the original library.

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Creates a shared library and uses Ninja instead of makefiles as generator.

## Requirements

By default the crate builds XGBoost from the pinned submodule (`local_build` feature), so the
headers used for bindgen and the runtime library can never disagree. This requires `cmake`
(and uses `ninja` when available). Alternatively, the `use_prebuilt_xgb` feature downloads an
already compiled library: `--no-default-features --features use_prebuilt_xgb`.

On mac you need to install `libomp` (`brew install libomp`). 
On debian, you need `libclang-dev` (`apt install -y libclang-dev`)

## Documentation

* [Documentation](https://docs.rs/xgboost)

Basic usage example:

```rust
extern crate xgb;

use xgb::{parameters, DMatrix, Booster};

fn main() {
    // training matrix with 5 training examples and 3 features
    let x_train = &[1.0, 1.0, 1.0,
                    1.0, 1.0, 0.0,
                    1.0, 1.0, 1.0,
                    0.0, 0.0, 0.0,
                    1.0, 1.0, 1.0];
    let num_rows = 5;
    let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];

    // convert training data into XGBoost's matrix format
    let mut dtrain = DMatrix::from_dense(x_train, num_rows).unwrap();

    // set ground truth labels for the training matrix
    dtrain.set_labels(y_train).unwrap();

    // test matrix with 1 row
    let x_test = &[0.7, 0.9, 0.6];
    let num_rows = 1;
    let y_test = &[1.0];
    let mut dtest = DMatrix::from_dense(x_test, num_rows).unwrap();
    dtest.set_labels(y_test).unwrap();

    // configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build().unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build().unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build().unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // overall configuration for training/evaluation
    let params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)                         // dataset to train with
        .boost_rounds(2)                         // number of training iterations
        .booster_params(booster_params)          // model parameters
        .evaluation_sets(Some(evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build().unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&params).unwrap();

    println!("{:?}", bst.predict(&dtest).unwrap());
}
```

See the [examples](https://github.com/agene0001/rust-xgboost/tree/master/examples) directory for
more detailed examples of different features.

## Performance

See [docs/SERVING.md](docs/SERVING.md) for a complete guide to building and
calling this crate for maximum performance. Summary:

For latency-sensitive serving of small batches (roughly under 1000 rows):

* Pin the booster to one thread after loading: `booster.set_param("nthread", "1")`.
  Small-batch latency is dominated by OpenMP thread dispatch; on a 127-feature/50-tree
  binary model this measures ~11-20x faster for single rows.
* Predict straight off your `&[f32]`/CSR slices with `predict_from_dense` /
  `predict_from_csr` (inplace prediction) instead of building a `DMatrix` per request.
* Reuse one output buffer across requests with `predict_from_dense_into` /
  `predict_from_csr_into` (or `predict_into` when batch-scoring a `DMatrix`). The
  warm serving loop then performs zero heap allocations in the wrapper (verified
  by `tests/zero_alloc.rs`).
* `Booster` and `DMatrix` are `Send`: load a model once and move it into a worker
  thread, or keep one booster per thread in a pool. `Booster` is deliberately not
  `Sync` — concurrent prediction on one instance would race on its cached
  inplace-prediction proxy — so use per-thread instances (cheap to create with
  `Booster::load_buffer`) rather than a shared reference.

For training and large-batch throughput, two flags tune the `local_build`
C++ compilation:

```sh
XGB_BUILD_NATIVE=0  # disable native codegen (-march/-mcpu=native); ON by default
XGB_BUILD_IPO=1     # link-time optimization for libxgboost; off by default
```

Native codegen is on by default because a from-source build usually runs on the
machine that built it; disable it when deploying the locally built binary to
other machines (or older CPUs of the same family). Expect the largest gains
from native codegen on x86-64 hosts with AVX2/AVX-512.

For large training sets with the `hist` tree method, prefer
`DMatrix::from_dense_quantile` / `from_csr_quantile`, which store pre-binned data
(~1 byte per value instead of 4); per-round training speed is the same as a
regular `DMatrix` — the win is memory. The biggest training speed knobs are
`max_bin` (256 → 64 measured ~1.7x faster per round; validate accuracy) and
`eval_period` (evaluation sets cost a full prediction pass per round by
default). See docs/SERVING.md §3.

## Status

The version number tracks the bundled XGBoost version.

This is still a very early stage of development, so the API is changing as usability issues occur,
or new features are supported. This is still expected to be compatible to an earlier rust-xgboost library.

Builds against XGBoost 3.3.0.

## Use prebuilt xgboost library or build it

Xgboost is kind of complicated to compile, especially when there is GPU support involved.
It is sometimes easier to use a pre-built library, which the `use_prebuilt_xgb` feature does.

This fork builds from the pinned submodule by default (`local_build`), because the bundled headers
and the library have to be the same XGBoost version — linking 3.0.x binaries against 3.3.0 headers
fails at run time, far from the cause, with errors like `Unknown objective function: reg:expectile`.

With `use_prebuilt_xgb`, the library is downloaded from this repository's release for the crate
version. The tag is derived from `CARGO_PKG_VERSION`, so it cannot drift from the crate version: a
release that was never published fails the build loudly instead of silently falling back to
version-skewed binaries.

Every downloaded asset is verified against a SHA-256 recorded in `xgboost-sys/build.rs`. The check
runs before the bytes are written, so a truncated transfer, a body mangled in transit, or an asset
re-uploaded under a tag that was already consumed is rejected rather than left on disk as a library.
An asset with no recorded digest is accepted with a warning; set `XGB_REQUIRE_CHECKSUMS=1` to make
that an error instead, which is what CI should do.

To fetch the same assets from a mirror, serve them under the same flat `<platform>-<file>` names and
set:

```sh
XGBOOST_LIB_URL=https://your-mirror.example/xgboost-libs
```

You can also point at an existing directory of libraries with `$XGBOOST_LIB_DIR`, in which case
nothing is downloaded or verified.

If you prefer to use xgboost from homebrew, which may have GPU support, your can for example define
```
XGBOOST_LIB_DIR=${HOMEBREW_PREFIX}/opt/xgboost/lib
```

If you want to use it by yourself, you can disable the use_prebuild_xgb feature:
```
xgb = { version = "3",  default-features = false, features=["local_build"] }
```
This would require `cmake` and `ninja-build` as build dependencies.

If you want build it locally, after cloning, perform `git submodule update --init --recursive`
to install submodule dependencies.

brew commands for MacOs to compile locally:
- brew install libomp
- brew install cmake
- brew install ninja
- brew install llvm

### Running binaries outside `cargo run`

libxgboost is linked dynamically. `cargo run` and `cargo test` always work
because Cargo adds the library's directory to the loader's environment
(`PATH` / `LD_LIBRARY_PATH` / `DYLD_FALLBACK_LIBRARY_PATH`) for the child
process — but a binary started directly (e.g. `./target/release/myapp`) gets
no such help.

To support that, the build script stages the shared library
(`xgboost.dll` / `libxgboost.so` / `libxgboost.dylib`) next to the
executables in the target profile directory, for both the `local_build` and
`use_prebuilt_xgb` paths.

On **Windows** that is sufficient — the loader searches the exe's directory.

On **Linux/macOS** the loader only looks next to the exe if the binary
carries an `$ORIGIN` / `@loader_path` rpath. Cargo does not propagate linker
args from dependency build scripts, so the *binary* crate has to add it
itself. Either in the binary crate's `build.rs`:

```rust
fn main() {
    let target = std::env::var("TARGET").unwrap();
    if target.contains("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    } else if target.contains("apple") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }
}
```

or in its `.cargo/config.toml`:

```toml
[target.'cfg(target_os = "linux")']
rustflags = ["-C", "link-arg=-Wl,-rpath,$ORIGIN"]

[target.'cfg(target_os = "macos")']
rustflags = ["-C", "link-arg=-Wl,-rpath,@loader_path"]
```

When deploying, copy the staged library alongside the executable.

### Supported Platforms

Prebuilt lib and built locally:

* Mac OS
* Linux

Prebuilt lib only

* Windows 

Local windows built is possible, but steps may require manual copy of VS output files.

GPU support on windows:

How to get a .lib and .dll from pip , using a VS Developer CMD prompt:
```
python3 -m venv .venv
.venv\Scripts\activate.bat
pip install xgboost
pip show xgboost
# check Location entry
copy {Location}\xgboost.dll .
gendef xgboost.dll
lib /def:xgboost.def /machine:x64" /out:xgboost.lib
```
