use bindgen;
use fs_extra;

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let xgb_root = Path::new("xgboost").canonicalize().unwrap();

    let header = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("dmlc-core").join("include").display()));

    #[cfg(feature = "cuda")]
    let bindings = bindings.clang_arg("-I/usr/local/cuda/include");
    let bindings = bindings.generate().expect("Unable to generate bindings.");

    let out_path = PathBuf::from(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    #[allow(unused_mut)]
    let mut homebrew_path = "/opt/homebrew";
    #[cfg(not(target_arch = "aarch64"))]
    {
        homebrew_path = "/opt/local";
    }
    if target.contains("apple") {
        println!("cargo:rustc-link-search=native={}/opt/libomp/lib", &homebrew_path);
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        let xgboost_lib_dir = std::env::var("XGBOOST_LIB_DIR").unwrap_or_else(|_err| {
            if target.contains("apple") {
                format!("{}/opt/xgboost/lib", &homebrew_path)
            } else {
                let pip_result: String;
                let xgboost_lib: &str;
                #[cfg(target_os = "linux")]
                {
                    xgboost_lib = "libxgboost.so";
                    pip_result = String::from_utf8(
                        std::process::Command::new("sh")
                            .arg("-c")
                            .arg("python3 -m pip show xgboost")
                            .output()
                            .expect("Please install xgboost via pip or set $XGBOOST_LIB_DIR")
                            .stdout,
                    )
                    .expect("sh output not utf8")
                };
                #[cfg(not(target_os = "linux"))]
                {
                    xgboost_lib = "xgboost.dll";
                    pip_result = String::from_utf8(
                        std::process::Command::new("cmd")
                            .arg("/C")
                            .arg("python3 -m pip show xgboost")
                            .output()
                            .expect("Please install xgboost via pip or set $XGBOOST_LIB_DIR")
                            .stdout,
                    )
                    .expect("cmd output not utf8");
                }
                for line in pip_result.lines() {
                    if line.starts_with("Location: ") {
                        let base_path = &line.replace("Location: ", "");
                        let xgboost_path = format!("{}/xgboost/lib", &base_path);
                        let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", &out_dir))).unwrap();
                        let deps_path = deps_path.to_str().unwrap();
                        fs_extra::file::copy(
                            &format!("{}/{}", &xgboost_path, &xgboost_lib),
                            &format!("{}/{}", &deps_path, &xgboost_lib),
                            &fs_extra::file::CopyOptions::new().overwrite(true),
                        )
                        .unwrap();

                        #[cfg(target_os = "windows")]
                        {
                            println!("cargo:rustc-link-search=native={}", &deps_path);
                            std::process::Command::new("gendef")
                                .args(&["xgboost.dll"])
                                .status()
                                .unwrap();
                            std::process::Command::new("cmd")
                                .args(&[
                                    "lib",
                                    &format!("/def:{}/xgboost.def", &deps_path),
                                    "/machine:x64",
                                    &format!("/out:{}/xgboost.lib", &deps_path),
                                ])
                                .status()
                                .unwrap();
                        }
                        // cmd:
                        /*
                        PROBLEM: LIBPATH !!!
                        gendef xgboost.dll
                        lib /def:xgboost.def /machine:x64 /out:xgboost.lib
                         */

                        #[cfg(target_os = "linux")]
                        std::process::Command::new("cp")
                            .args(&[
                                "-r",
                                &format!("{}/xgboost/lib/.", &base_path),
                                &deps_path.to_str().unwrap(),
                            ])
                            .status()
                            .unwrap();

                        return xgboost_path;
                    }
                }
                panic!("Please set $XGBOOST_LIB_DIR or install xgboost via pip");
            }
        });
        println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
    }

    #[cfg(not(feature = "use_prebuilt_xgb"))]
    {
        // compile XGBOOST with cmake and ninja
        let xgb_root = Path::new(&out_dir).join("xgboost");

        // copy source code into OUT_DIR for compilation if it doesn't exist
        if !xgb_root.exists() {
            std::fs::create_dir(&xgb_root).unwrap();
            let xgb_lib = Path::new("xgboost").canonicalize().unwrap();
            fs_extra::dir::copy(
                xgb_lib.to_str().unwrap(),
                xgb_root.to_str().unwrap(),
                &fs_extra::dir::CopyOptions::new(),
            )
            .unwrap_or_else(|e| {
                panic!("Failed to copy {} to {}: {}", xgb_lib.display(), xgb_root.display(), e);
            });
        }
        let xgb_root = xgb_root.canonicalize().unwrap();

        // CMake
        let mut dst = cmake::Config::new(&xgb_root);
        let dst = dst.generator("Ninja");
        let dst = dst.define("CMAKE_BUILD_TYPE", "RelWithDebInfo");

        #[cfg(feature = "cuda")]
        let mut dst = dst
            .define("USE_CUDA", "ON")
            .define("BUILD_WITH_CUDA", "ON")
            .define("BUILD_WITH_CUDA_CUB", "ON");

        let dst = dst.build();

        println!("cargo:rustc-link-search=native={}", dst.display());
        println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
        println!("cargo:rustc-link-search=native={}", dst.join("lib64").display());
        println!("cargo:rustc-link-lib=static=dmlc");
    }

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else {
        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=stdc++fs");
            println!("cargo:rustc-link-lib=dylib=gomp");
        }
    }

    println!("cargo:rustc-link-lib=dylib=xgboost");

    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-link-search={}", "/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=static=cudart_static");
    }
}
