use bindgen;
use std::env;
use std::path::{Path, PathBuf};

const GITHUB_URL: &str = "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.0/xgboost-sys/lib/";

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let xgb_root = Path::new("xgboost").canonicalize().unwrap();

    let wrapper_h = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_string_lossy())
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!("-I{}", xgb_root.join("dmlc-core").join("include").display()));

    #[cfg(feature = "cuda")]
    let bindings = bindings.clang_arg("-I/usr/local/cuda/include");
    let bindings = bindings.generate().expect("Unable to generate bindings.");

    let out_path = PathBuf::from(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    if target.contains("apple") {
        println!("cargo:rustc-link-search=native={}/opt/libomp/lib", &std::env::var("HOMEBREW_PREFIX").unwrap());
    }
    
    #[cfg(feature = "use_prebuilt_xgb")]
    {
        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {

            let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            let deps_path = deps_path.to_string_lossy();
            println!("cargo:rustc-link-search=native={}", deps_path);
            if cfg!(all(target_os="macos", target_arch = "aarch64")) {
                let path = format!("{GITHUB_URL}/mac_arm64");
                if !std::fs::exists(format!("{deps_path}/libxgboost.dylib")).unwrap() {
                    web_copy(&format!("{path}/libxgboost.dylib"), &format!("{deps_path}/libxgboost.dylib")).unwrap();
                    web_copy(&format!("{path}/libdmlc.a"), &format!("{deps_path}/libdmlc.a")).unwrap();
                }
            } else if cfg!(all(target_os="linux", target_arch = "x86_64")) {
                let path = format!("{GITHUB_URL}/linux_amd64");
                if !std::fs::exists(format!("{deps_path}/libxgboost.so")).unwrap() {
                    web_copy(&format!("{path}/libxgboost.so"), &format!("{deps_path}/libxgboost.so")).unwrap();
                    web_copy(&format!("{path}/libdmlc.a"), &format!("{deps_path}/libdmlc.a")).unwrap();
                }
            } else if cfg!(all(target_os="windows", target_arch = "x86_64")) { 
                let path = format!("{GITHUB_URL}/win_amd64");
                if !std::fs::exists(format!("{deps_path}/xgboost.dll")).unwrap() {
                    web_copy(&format!("{path}/xgboost.dll"), &format!("{deps_path}/xgboost.dll")).unwrap();
                    web_copy(&format!("{path}/xgboost.lib"), &format!("{deps_path}/xgboost.lib")).unwrap();
                }
             } else {
                if let Ok(homebrew_path) = std::env::var("HOMEBREW_PREFIX") {
                    let xgboost_lib_dir = format!("{}/opt/xgboost/lib", &homebrew_path);
                    println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
                } else {
                    panic!("Please set $XGBOOST_LIB_DIR")
                }
            }
        }
    }

    #[cfg(not(feature = "use_prebuilt_xgb"))]
    {
        // compile XGBOOST with cmake and ninja

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

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn web_copy(web_src: &str, target: &str) -> Result<()>{
    dbg!(&web_src);
    let resp = reqwest::blocking::get(web_src)?;
    let body = resp.bytes()?;
    std::fs::write(target, &body)?;
    Ok(())
}