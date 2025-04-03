# rust-xgboost

This is mostly a fork of https://github.com/davechallis/rust-xgboost but uses 
another xgboost version and links it dynamically instead of linkit it static as the original library.

Rust bindings for the [XGBoost](https://xgboost.ai) gradient boosting library.

Creates a shared library and uses Ninja instead of makefiles as generator.

## Requirements

It is highly recommended to use the `use_prebuilt_xgb` feature, which is enabled by default.
It will use an already installed xgboost library from homebrew or pip. 

brew commands for MacOs:

with use_prebuilt_lib:
- brew install xgboost

compile yourself:
- brew install libomp
- brew install cmake
- brew install ninja
- brew install llvm

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

See the [examples](https://github.com/davechallis/rust-xgboost/tree/master/examples) directory for
more detailed examples of different features.

## Status

Currently in a very early stage of development, so the API is changing as usability issues occur,
or new features are supported.

If you build it locally, after cloning, perform `git submodule update --init --recursive`
to install submodule dependencies.s

Builds against XGBoost 3.0.0.

Deactivated tests - functions probably not working correctly:

- booster::dump_model
- dmatrix::get_set_base_margin
- dmatrix::get_set_group
- dmatrix::get_set_weights

## Use prebuilt xgboost library

Xgboost is kind of complicated to compile, especially when there is GPU support involved.
It is sometimes easier to use a pre-build library.
The feature flag `use_prebuilt_xgb` is enabled by default.
This will use the version installed in `$XGBOOST_LIB_DIR`. If it isn't set, it will use `homebrew` on MacOs and the one from `python3 -m pip info xgboost` on Windows and Linux.


If you want to compile xgboost by yourself, you can disable the default feature:
```
xgb = { version = "3.0.0", default-features = false }
```

### Platforms

Tested, prebuilt and locally compiled:

* Mac OS 
* Linux

Tested, prebuilt only:

* Windows
