# 3.0.6 (Unreleased)

## Fixed
* Fixed deprecation warnings from XGBoost C API:
  - Replaced `XGBoosterBoostOneIter` with `XGBoosterTrainOneIter`
  - Replaced `XGDMatrixCreateFromCSREx` with `XGDMatrixCreateFromCSR`
  - Replaced `XGDMatrixCreateFromCSCEx` with `XGDMatrixCreateFromCSC`
  - Replaced `XGDMatrixSetUIntInfo` with `XGDMatrixSetInfoFromInterface`
  - Replaced `XGDMatrixCreateFromFile` with `XGDMatrixCreateFromURI`

## Added
* Added `BinaryError` variant to `EvaluationMetric` for default 0.5 threshold (simpler alternative to `BinaryErrorRate(0.5)`)
* Added `validate_features()` method to `Booster` for checking feature name/count consistency
* Added callback support to `TrainingParameters`:
  - New `CallbackEnv` struct with iteration info and evaluation results
  - New `TrainingCallback` type for callback functions
  - Callbacks can return `false` to stop training early

## Changed
* `Booster::update_custom()` now requires an `iteration` parameter
* Added safety documentation for `Booster::new_with_cached_dmats()` explaining DMatrix lifetime

# 0.1.4 (2019-03-05)

* `Booster::load_buffer` method added (thanks [jonathanstrong](https://github.com/jonathanstrong))
