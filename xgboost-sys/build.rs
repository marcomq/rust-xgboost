use bindgen;
use std::env;
use std::path::{Path, PathBuf};

/// Repository publishing the prebuilt release assets (built from the pinned
/// submodule by `.github/workflows/release-libs.yml`, flat `<target>-<file>`
/// names).
#[cfg(feature = "use_prebuilt_xgb")]
const PREBUILT_RELEASE_REPO: &str = "https://github.com/agene0001/rust-xgboost";

/// Fallback: XGBoost 3.0.0 binaries committed in the upstream fork's repo at its
/// v3.0.1 tag. These are OLDER than the bundled headers (version skew), so they
/// are only used when explicitly opted into via `XGB_ALLOW_LEGACY_PREBUILT=1`.
#[cfg(feature = "use_prebuilt_xgb")]
const LEGACY_URL: &str = "https://github.com/marcomq/rust-xgboost/raw/refs/tags/v3.0.1/xgboost-sys/lib";

/// Release-asset base URL for this crate version.
///
/// Derived from `CARGO_PKG_VERSION` rather than hardcoded so the tag cannot
/// drift from the crate version: bumping the version in Cargo.toml
/// automatically points at the matching release, and a release that was never
/// published fails loudly in `fetch_lib` instead of silently falling back to
/// version-skewed binaries.
#[cfg(feature = "use_prebuilt_xgb")]
fn prebuilt_release_url() -> String {
    format!(
        "{PREBUILT_RELEASE_REPO}/releases/download/v{}",
        env::var("CARGO_PKG_VERSION").unwrap()
    )
}

/// Base URL the release assets are fetched from: `$XGBOOST_LIB_URL` when set,
/// otherwise this crate version's release.
///
/// A mirror is expected to serve the same flat `<target>-<file>` layout the
/// GitHub release uses, so it can stand in without any other change.
#[cfg(feature = "use_prebuilt_xgb")]
fn release_base_url() -> String {
    match env::var("XGBOOST_LIB_URL") {
        Ok(url) if !url.trim().is_empty() => url.trim_end_matches('/').to_string(),
        _ => prebuilt_release_url(),
    }
}

/// SHA-256 of each published release asset, as `(target_dir, file, digest)`.
///
/// Verification is what makes the release contract enforceable rather than
/// merely documented. `error_for_status` already stops a 404 page being written
/// out as a library, but it cannot catch a truncated transfer, a body mangled
/// by an intercepting proxy, or a release asset re-uploaded with different
/// contents under a tag that was already consumed.
///
/// Digests are specific to a crate version, because `prebuilt_release_url`
/// derives the tag from `CARGO_PKG_VERSION`: a version bump republishes the
/// assets and needs new entries here. Recompute them against the release with
///
/// ```sh
/// gh release download v$VERSION --repo agene0001/rust-xgboost --dir /tmp/assets
/// sha256sum /tmp/assets/*
/// ```
///
/// An asset with no entry is downloaded and accepted with a warning rather than
/// failing the build, so adding a platform does not require a digest up front;
/// `XGB_REQUIRE_CHECKSUMS=1` turns that warning into an error, which is what CI
/// should set.
///
/// Recorded from the v3.3.0 release assets.
#[cfg(feature = "use_prebuilt_xgb")]
const PREBUILT_SHA256: &[(&str, &str, &str)] = &[
    (
        "linux_amd64",
        "libxgboost.so",
        "7187aed5c7c5f173b2bfec5685dc8fece7bf1c9660452d0438559af87f91c48f",
    ),
    (
        "linux_amd64",
        "libdmlc.a",
        "91eb93f9929680235e8dd20a3ae150c2e2b8bdb20d27a49ec98dfa0167517a6c",
    ),
    (
        "linux_arm64",
        "libxgboost.so",
        "269d565c1835383d72df84682caa3d6b06d3e93f0629583608698c8faff2e555",
    ),
    (
        "linux_arm64",
        "libdmlc.a",
        "f127a5c7895f8048c109cc1e2526d92e5fe24d138bdd5ad54bd535e2177d8c2a",
    ),
    (
        "mac_arm64",
        "libxgboost.dylib",
        "0dbe3b5c9221cccf5eb437e70273506bff8182ce428a5569e945c54579801bdc",
    ),
    (
        "mac_arm64",
        "libdmlc.a",
        "2058df2e8450741d2e865d008f5eac0269a09d0ab9ea7e817211b17f6e0fe97f",
    ),
    (
        "win_amd64",
        "xgboost.dll",
        "8ddb8a1bcd0f9a709d6e3f56b9751f46de093b11cec8a3d4593cc09d3d28308c",
    ),
    (
        "win_amd64",
        "xgboost.lib",
        "3eeba07c32d5137c5cdc54236e910b83cd69fe89572e254b3a04f3b393f25814",
    ),
];

#[cfg(feature = "use_prebuilt_xgb")]
fn expected_sha256(target_dir: &str, file: &str) -> Option<&'static str> {
    PREBUILT_SHA256
        .iter()
        .find(|(t, f, _)| *t == target_dir && *f == file)
        .map(|(_, _, digest)| *digest)
}

#[cfg(feature = "use_prebuilt_xgb")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut acc, byte| {
            use std::fmt::Write as _;
            let _ = write!(acc, "{byte:02x}");
            acc
        })
}

/// Version of the bundled XGBoost submodule, parsed from its `version_config.h`.
fn bundled_xgboost_version(xgb_root: &Path) -> String {
    let header = xgb_root.join("include").join("xgboost").join("version_config.h");
    let text =
        std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("cannot read {}: {e}", header.display()));
    let field = |name: &str| -> u32 {
        let needle = format!("#define {name} ");
        text.lines()
            .find_map(|line| {
                let rest = line.trim().strip_prefix(&needle)?;
                rest.split_whitespace().next()?.parse::<u32>().ok()
            })
            .unwrap_or_else(|| panic!("could not parse {name} from {}", header.display()))
    };
    format!(
        "{}.{}.{}",
        field("XGBOOST_VER_MAJOR"),
        field("XGBOOST_VER_MINOR"),
        field("XGBOOST_VER_PATCH")
    )
}

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    // dunce strips Windows' \\?\ extended-length prefix, which std's
    // canonicalize produces and which breaks CMake's file(GLOB) (configure
    // fails with "No SOURCES given to target: xgboost").
    let xgb_root = dunce::canonicalize(Path::new("xgboost")).unwrap();

    // This crate's version tracks the bundled XGBoost version, and the prebuilt
    // release tag is derived from it (see `prebuilt_release_url`). Flag a
    // submodule bump that forgot the Cargo.toml bump: under `use_prebuilt_xgb`
    // it would point at the wrong release tag, and for `local_build` it makes
    // the published crate version lie about what it bundles. Warn rather than
    // panic so an intentional skew (e.g. a pre-release submodule pin, or a
    // user-supplied XGBOOST_LIB_DIR) stays buildable; the release workflow
    // enforces the invariant strictly before publishing assets.
    let crate_version = env::var("CARGO_PKG_VERSION").unwrap();
    let bundled_version = bundled_xgboost_version(&xgb_root);
    if crate_version != bundled_version {
        println!(
            "cargo:warning=version skew: xgboost-sys is {crate_version} but the bundled submodule is \
             XGBoost {bundled_version} (xgboost-sys/xgboost/include/xgboost/version_config.h). Bump \
             `version` in Cargo.toml (and the root crate's) to {bundled_version}, then publish \
             matching release assets with the `Release XGBoost binaries` workflow."
        );
    }

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
        println!(
            "cargo:rustc-link-search=native={}/opt/libomp/lib",
            &std::env::var("HOMEBREW_PREFIX").unwrap_or("/opt/homebrew".into())
        );
    }

    #[cfg(feature = "use_prebuilt_xgb")]
    {
        for var in ["XGBOOST_LIB_DIR", "XGBOOST_LIB_URL"] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {
            let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            let deps_path = deps_path.to_string_lossy();
            println!("cargo:rustc-link-search=native={}", deps_path);
            if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                if !std::fs::exists(format!("{deps_path}/libxgboost.dylib")).unwrap() {
                    fetch_lib("mac_arm64", "libxgboost.dylib", &deps_path).unwrap();
                    fetch_lib("mac_arm64", "libdmlc.a", &deps_path).unwrap();
                }
                stage_lib_next_to_exe(Path::new(&format!("{deps_path}/libxgboost.dylib")), &out_dir);
            } else if cfg!(target_os = "linux") {
                let target_dir = if cfg!(target_arch = "aarch64") {
                    "linux_arm64"
                } else {
                    "linux_amd64"
                };
                if !std::fs::exists(format!("{deps_path}/libxgboost.so")).unwrap() {
                    fetch_lib(target_dir, "libxgboost.so", &deps_path).unwrap();
                    fetch_lib(target_dir, "libdmlc.a", &deps_path).unwrap();
                }
                stage_lib_next_to_exe(Path::new(&format!("{deps_path}/libxgboost.so")), &out_dir);
            } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
                if !std::fs::exists(format!("{deps_path}/xgboost.dll")).unwrap() {
                    fetch_lib("win_amd64", "xgboost.dll", &deps_path).unwrap();
                    fetch_lib("win_amd64", "xgboost.lib", &deps_path).unwrap();
                }
                stage_lib_next_to_exe(Path::new(&format!("{deps_path}/xgboost.dll")), &out_dir);
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

    #[cfg(feature = "local_build")]
    {
        // compile XGBOOST with cmake (and ninja, when available)

        // Rebuild the C++ when the bundled sources change. Watching only
        // version_config.h (the old approach) caught submodule version bumps
        // but silently measured stale code after hand-edits to the carried
        // patches — a rerun here only triggers an incremental cmake build, so
        // watching the whole tree is cheap. dmlc-core is deliberately not
        // watched (never patched here; 0.25% of the hot path).
        println!("cargo:rerun-if-changed={}", xgb_root.join("src").display());
        println!("cargo:rerun-if-changed={}", xgb_root.join("include").display());

        // CMake
        let mut dst = cmake::Config::new(&xgb_root);
        let ninja_available = std::process::Command::new("ninja")
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false);
        let dst = if ninja_available { dst.generator("Ninja") } else { &mut dst };
        // Release (-O3, no debug info) to match upstream's distributed builds;
        // RelWithDebInfo's -O2 leaves performance on the table in the hot loops.
        let dst = dst.define("CMAKE_BUILD_TYPE", "Release");

        // XGB_BUILD_NATIVE tunes codegen for the build machine (clang spells
        // the flag -mcpu=native on aarch64 and -march=native elsewhere).
        // Default ON: local_build compiles on the machine that will run the
        // binary, so "portable across CPUs" protects nobody by default while
        // making every default build (and every benchmark that forgets the
        // env var) measure the slow path. Cross-machine deploys of a locally
        // built artifact are the exception and opt out with XGB_BUILD_NATIVE=0.
        // The distributed release assets are unaffected — release-libs.yml
        // invokes cmake directly and never runs this script.
        println!("cargo:rerun-if-env-changed=XGB_BUILD_NATIVE");
        if env::var("XGB_BUILD_NATIVE").map_or(true, |v| v != "0") {
            let flag = if target.contains("aarch64") {
                "-mcpu=native"
            } else {
                "-march=native"
            };
            dst.cflag(flag).cxxflag(flag);
        }
        // XGB_BUILD_IPO=1 enables link-time optimization (CMake IPO) for the
        // C++ build. Passed explicitly as ON/OFF rather than only when set:
        // -D values persist in CMakeCache.txt across reconfigures, so an
        // explicit OFF is required for unsetting the env var to take effect.
        println!("cargo:rerun-if-env-changed=XGB_BUILD_IPO");
        let ipo = if env::var("XGB_BUILD_IPO").is_ok_and(|v| v == "1") {
            "ON"
        } else {
            "OFF"
        };
        let dst = dst.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", ipo);

        // Hide non-API symbols in the shared library. The C API keeps its
        // explicit __attribute__((visibility("default"))) (c_api.h XGB_DLL),
        // so the Rust link is unaffected; what this buys is that cross-TU
        // calls inside libxgboost.so stop being interposable on Linux ELF
        // (no PLT indirection), plus a smaller export table everywhere.
        let dst = dst.define("HIDE_CXX_SYMBOLS", "ON");

        // Keep every build artifact inside cmake's per-OUT_DIR binary dir.
        // Upstream's default links the shared library into the SOURCE tree
        // (<xgboost>/lib) — a single location shared by every profile and
        // every concurrent cargo invocation using this checkout, so two
        // builds racing through the ninja link step fail with
        // "LNK1104: cannot open file ...xgboost.dll" (observed with a
        // background `cargo build` overlapping `cargo run --profile
        // production` right after a pin bump). With this ON the link output
        // stays under OUT_DIR; the install step still places the library in
        // <dst>/bin|lib, where the staging candidates below pick it up.
        let dst = dst.define("KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR", "ON");

        // Escape hatch for perf experiments: XGB_CMAKE_DEFINES="A=1;B=OFF"
        // passes arbitrary -D defines to the bundled build (e.g.
        // USE_OPENMP=OFF, BUILD_STATIC_LIB=ON) without editing this script.
        // Caveat: -D values persist in CMakeCache.txt, so REMOVING an entry
        // does not restore the default — set it explicitly (e.g. USE_OPENMP=ON)
        // or delete the cmake build dir under OUT_DIR.
        println!("cargo:rerun-if-env-changed=XGB_CMAKE_DEFINES");
        if let Ok(defines) = env::var("XGB_CMAKE_DEFINES") {
            for def in defines.split(';').filter(|d| !d.trim().is_empty()) {
                match def.split_once('=') {
                    Some((key, val)) => {
                        dst.define(key.trim(), val.trim());
                    }
                    None => panic!(
                        "XGB_CMAKE_DEFINES entry {def:?} is not KEY=VALUE \
                         (full value: {defines:?})"
                    ),
                }
            }
        }

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

        let lib_file = if target.contains("windows") {
            "xgboost.dll"
        } else if target.contains("apple") {
            "libxgboost.dylib"
        } else {
            "libxgboost.so"
        };
        // cmake emits the runtime library under bin/ on Windows and lib/ (or
        // lib64/ on some distros) elsewhere.
        let candidates = [
            dst.join("bin").join(lib_file),
            dst.join("lib").join(lib_file),
            dst.join("lib64").join(lib_file),
        ];
        match candidates.iter().find(|p| p.exists()) {
            Some(src) => stage_lib_next_to_exe(src, &out_dir),
            None => println!(
                "cargo:warning=built {lib_file} not found under {} — cannot stage it next to the executables",
                dst.display()
            ),
        }
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

/// The dynamic loader does not search the cmake out dir or
/// target/<profile>/deps where the library lands, so binaries run outside
/// `cargo run` (which patches PATH / LD_LIBRARY_PATH /
/// DYLD_FALLBACK_LIBRARY_PATH) fail to find it. Stage a copy in the profile
/// root, where the final executables are emitted. On Windows that alone fixes
/// it — the loader searches the exe's directory. On Linux/macOS the loader
/// only looks next to the exe if the binary carries an $ORIGIN/@loader_path
/// rpath, and Cargo does not propagate link-args from dependency build
/// scripts, so the binary crate must add that itself — see "Running binaries
/// outside cargo run" in the README.
fn stage_lib_next_to_exe(lib_src: &Path, out_dir: &str) {
    let lib_name = lib_src.file_name().unwrap();
    // OUT_DIR = target/<profile>/build/<crate>-<hash>/out → ../../.. = profile root
    let profile_dir = match dunce::canonicalize(Path::new(&format!("{}/../../..", out_dir))) {
        Ok(dir) => dir,
        Err(e) => {
            println!(
                "cargo:warning=could not resolve profile dir to stage {}: {e}",
                lib_name.display()
            );
            return;
        }
    };
    if let Err(e) = std::fs::copy(lib_src, profile_dir.join(lib_name)) {
        // Non-fatal: a running executable can hold a lock on the old copy.
        println!(
            "cargo:warning=could not stage {} into {}: {e}",
            lib_src.display(),
            profile_dir.display()
        );
    }
}

#[cfg(feature = "use_prebuilt_xgb")]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Download one prebuilt library into `deps_path` from this crate version's
/// release assets.
///
/// A missing asset is a hard error rather than a silent downgrade. The legacy
/// binaries are XGBoost 3.0.0 — older than the bundled headers — so linking
/// them mixes ABIs and every API added since 3.0 fails at run time far from the
/// cause (a missing v3.3.0 release surfaced as an "Unknown objective function:
/// reg:expectileerror" test failure, not as a build error). Set
/// `XGB_ALLOW_LEGACY_PREBUILT=1` to opt back into that fallback.
///
/// The asset is verified against [`PREBUILT_SHA256`] when its digest is on
/// record; see that table for what happens while one is missing.
#[cfg(feature = "use_prebuilt_xgb")]
fn fetch_lib(target_dir: &str, file: &str, deps_path: &str) -> Result<()> {
    let dest = format!("{deps_path}/{file}");
    let expected = expected_sha256(target_dir, file);
    if expected.is_none() {
        println!("cargo:rerun-if-env-changed=XGB_REQUIRE_CHECKSUMS");
        if env::var("XGB_REQUIRE_CHECKSUMS").is_ok_and(|v| v == "1") {
            panic!(
                "no recorded SHA-256 for {target_dir}/{file}, and XGB_REQUIRE_CHECKSUMS=1.\n\
                 Add the digest to PREBUILT_SHA256 in xgboost-sys/build.rs, or build from source \
                 with `cargo build --features local_build` (this crate's default)."
            );
        }
        println!(
            "cargo:warning=no recorded SHA-256 for {target_dir}/{file}; accepting the download \
             unverified. Record the digest in PREBUILT_SHA256 (xgboost-sys/build.rs), or set \
             XGB_REQUIRE_CHECKSUMS=1 to make this an error."
        );
    }

    let release_url = release_base_url();
    let primary = format!("{release_url}/{target_dir}-{file}");
    let primary_err = match web_copy(&primary, &dest, expected) {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };

    println!("cargo:rerun-if-env-changed=XGB_ALLOW_LEGACY_PREBUILT");
    if !env::var("XGB_ALLOW_LEGACY_PREBUILT").is_ok_and(|v| v == "1") {
        panic!(
            "prebuilt asset is unusable: {primary}\n\
             \x20 cause: {primary_err}\n\
             \n\
             Usually the release for this crate version has not been published (the tag is derived \
             from CARGO_PKG_VERSION = {version}); a checksum mismatch above instead means the asset \
             was served but did not match PREBUILT_SHA256. Fix by either:\n\
             \n\
             1. Publishing the assets: run the `Release XGBoost binaries` workflow\n\
             \x20  (`gh workflow run release-libs.yml`) — it defaults to the tag this build\n\
             \x20  expects, v{version}.\n\
             2. Building libxgboost from the pinned submodule instead:\n\
             \x20  `cargo build --features local_build` (this crate's default).\n\
             3. Supplying your own libraries via XGBOOST_LIB_DIR.\n\
             \n\
             As a last resort, XGB_ALLOW_LEGACY_PREBUILT=1 falls back to XGBoost 3.0.0 binaries, \
             which are version-skewed against the bundled {version} headers; expect missing \
             symbols and unknown-parameter errors at run time.",
            version = env::var("CARGO_PKG_VERSION").unwrap()
        );
    }

    println!(
        "cargo:warning=prebuilt asset {primary} unusable ({primary_err}); \
         XGB_ALLOW_LEGACY_PREBUILT=1 is set, so falling back to legacy XGBoost 3.0.0 binaries. \
         These are older than the bundled headers — APIs added since 3.0 will fail at run time."
    );
    // Unpinned: the legacy binaries predate PREBUILT_SHA256 and are a different
    // XGBoost version, so there is no digest of this crate's assets to hold
    // them to. The opt-in env var and the warning above are the safeguard.
    web_copy(&format!("{LEGACY_URL}/{target_dir}/{file}"), &dest, None)
}

/// Download `web_src` to `target`, rejecting it unless it hashes to `expected`.
///
/// `expected` is `None` only for a source whose contents are not pinned (the
/// legacy fallback), or while an asset's digest has not been recorded yet.
#[cfg(feature = "use_prebuilt_xgb")]
fn web_copy(web_src: &str, target: &str, expected: Option<&str>) -> Result<()> {
    dbg!(&web_src);
    // error_for_status is load-bearing: without it a 404 page would be written
    // out as the library file, and the fallback in fetch_lib would never run.
    let resp = reqwest::blocking::get(web_src)?.error_for_status()?;
    let body = resp.bytes()?;

    if let Some(want) = expected {
        let got = sha256_hex(&body);
        if !got.eq_ignore_ascii_case(want) {
            // An error rather than a panic, so a mismatch reads as "this source
            // did not work out" and fetch_lib falls through to its next option
            // exactly as it does for a 404.
            return Err(format!("checksum mismatch for {web_src}: expected {want}, got {got}").into());
        }
    }

    // Written only once the bytes are known good, so a rejected download never
    // lands on disk where a later build would find it and skip the fetch.
    std::fs::write(target, &body)?;
    Ok(())
}
