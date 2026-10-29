[![Actions Status](https://github.com/marcomq/rust-xgboost/workflows/Macos/badge.svg)](https://github.com/marcomq/rust-xgboost/actions/workflows/macos.yml)
[![Actions Status](https://github.com/marcomq/rust-xgboost/workflows/Linux/badge.svg)](https://github.com/marcomq/rust-xgboost/actions/workflows/linux.yml)
[![Actions Status](https://github.com/marcomq/rust-xgboost/workflows/Windows/badge.svg)](https://github.com/marcomq/rust-xgboost/actions/workflows/windows.yml)


# rust-xgboost


This is mostly a fork of https://github.com/davechallis/rust-xgboost but uses 
another xgboost version and links it dynamically instead of linking it statically as in the original library.

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Creates a shared library and uses Ninja instead of makefiles as generator.

## Disclaimers

- **This crate does not follow semver.** The first two version numbers always match the
  XGBoost library version it is built against (e.g. `3.0.x` uses XGBoost 3.0). The last
  number is this crate's own release counter, and any release may contain breaking API
  changes. Pin an exact version (e.g. `xgb = "=3.0.7"`) if you need stability.

## Requirements

It is highly recommended to use the `use_prebuilt_xgb` feature, which is enabled by default.
It will use an already compiled xgboost library which will be downloaded as build step of this crate.
On Mac, it will use an arm64 shared library. On windows and linux, it is using x64 architecture.

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

See the [examples](https://github.com/marcomq/rust-xgboost/tree/master/examples) directory for
more detailed examples of different features.

## Status

The version number is just an indicator that xboost 3.0.0 is used.

This is still a very early stage of development, so the API is changing as usability issues occur,
or new features are supported. This is still expected to be compatible to an earlier rust-xgboost library.

Builds against XGBoost 3.0.0.

## Use prebuilt xgboost library or build it

Xgboost is kind of complicated to compile, especially when there is GPU support involved.
It is sometimes easier to use a pre-build library. Therefore, the feature flag `use_prebuilt_xgb` is enabled by default.
This is using a prebuilt shared library in xboost-sys/lib by default. You can also use a custom folder by defining `$XGBOOST_LIB_DIR`.

The library is looked up in the following order, so a build only reaches the network when it has to:

1. `$XGBOOST_LIB_DIR` - link against an existing directory, nothing is copied or downloaded
2. `target/<profile>/deps/` - the copy an earlier build of this crate already put in place, reused
   only if it still matches the pinned checksum
3. `$XGBOOST_LIB_CACHE`, defaulting to `$CARGO_HOME/xgboost-prebuilt/<tag>` - a download cache that
   survives `cargo clean` and is shared between checkouts
4. `xgboost-sys/lib/<platform>/` - the copies in this repository, present in a git checkout but not
   in the published crate
5. Download, trying `$XGBOOST_LIB_URL` first if set, then the bundled mirrors in turn

Every file is pinned by SHA-256. A download that does not match, including an error page served
during an outage, is rejected and the next source is tried. Files already on disk are re-checked on
each build, so a bad copy from an earlier build repairs itself rather than breaking every
subsequent build.

To download from your own mirror, expose the files as `<base>/<platform>/<file>` and set:

```sh
XGBOOST_LIB_URL=https://your-mirror.example/xgboost-libs
```

A base URL containing `/releases/download/` is treated as a GitHub release instead, whose asset
namespace is flat: the files are looked up as `<base>/<platform>-<file>`.

For a fully offline build, either point `$XGBOOST_LIB_DIR` at a prepared directory, or place the
files in `$XGBOOST_LIB_CACHE/<platform>/`.

On macOS the prebuilt dylib is stamped with an absolute Homebrew install name. The build rewrites
it to the copy it just verified and re-signs it ad-hoc, so the pinned library is the one loaded at
runtime rather than whatever Homebrew happens to have installed.

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
