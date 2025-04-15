## Content
This folder contains binary shared libraries of version 3.0.0 of xgboost.
The libs are not needed if feature "use_prebuilt_xgb" is disabled.

They have been uploaded here for convenience and because homebrew isn't able to download a specific version. 
The selection was made internally for our developers - we currently don't need to add other binaries. There probably is no GPU support.

Source:
- linux_amd64: homebrew installed via curl, unknown GPU support, xgboost 3.0.0, debian 12, amd64 docker container
- mac_arm64: homebrew installed via curl, unknown GPU support, xgboost 3.0.0, Sequioa 15.3, arm64
- win_amd64: built locally without GPU support as pip dll was > 150MB, python 3.13, Visua Studio 2022

