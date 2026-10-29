## Content
This folder contains binary shared libraries of version 3.4.1 of xgboost.
The libs are not needed if feature "use_prebuilt_xgb" is disabled.

They have been uploaded here for convenience and because homebrew isn't able to download a specific version.
All libraries are CPU-only.

Source: the official XGBoost wheels on PyPI, fetched and patched by `scripts/fetch-prebuilt-libs.sh`.
- linux_amd64, linux_arm64: `xgboost-cpu` manylinux_2_28 wheels, patched to an unversioned soname and the system `libgomp.so.1`
- mac_arm64: `xgboost` macosx_12_0_arm64 wheel, install name changed to `@rpath/libxgboost.dylib` (needs `brew install libomp`)
- win_amd64: `xgboost-cpu` win_amd64 wheel; `xgboost.lib` is an import library generated from the DLL's exports

`libdmlc.a` is left over from the 3.0 builds. It is only used by the `local_build` feature, which builds its own copy.
