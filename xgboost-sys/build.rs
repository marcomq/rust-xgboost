use bindgen;
use std::env;
use std::path::{Path, PathBuf};

/// Tag the prebuilt libraries are pinned to. Must match the checksums in `prebuilt_artifacts`.
///
/// This tracks the libraries, not the crate version: it only moves when the binaries themselves
/// change, so a crate release that leaves them untouched keeps pointing at the same verified bytes.
#[cfg(feature = "use_prebuilt_xgb")]
const LIB_TAG: &str = "v3.4.0";

/// Base URLs tried in order when an artifact has to be downloaded. `{tag}` is replaced with
/// [`LIB_TAG`], so the tag is pinned in exactly one place. Each entry is expanded further by
/// `mirror_url`, so a mirror may lay its files out either as `<base>/<platform>/<file>` or, for
/// GitHub release assets, as `<base>/<platform>-<file>`.
#[cfg(feature = "use_prebuilt_xgb")]
const MIRRORS: &[&str] = &[
    "https://github.com/marcomq/rust-xgboost/raw/refs/tags/{tag}/xgboost-sys/lib",
    "https://github.com/marcomq/rust-xgboost/releases/download/{tag}",
];

fn main() {
    let target = env::var("TARGET").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();
    let xgb_root = Path::new("xgboost").canonicalize().unwrap();

    let wrapper_h = xgb_root.join("include").join("xgboost").join("c_api.h");
    let bindings = bindgen::Builder::default()
        .header(wrapper_h.to_string_lossy())
        // Only the C API: libc types like `_IO_FILE` break layout tests when libclang mis-parses them.
        .allowlist_function("XG.*")
        .allowlist_type("bst_.*|.*Handle")
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
        for var in ["XGBOOST_LIB_DIR", "XGBOOST_LIB_CACHE", "XGBOOST_LIB_URL"] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        if let Ok(xgboost_lib_dir) = std::env::var("XGBOOST_LIB_DIR") {
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else if let Some(platform) = prebuilt_platform() {
            let deps_path = dunce::canonicalize(Path::new(&format!("{}/../../../deps", out_dir))).unwrap();
            println!("cargo:rustc-link-search=native={}", deps_path.display());
            for artifact in prebuilt_artifacts(platform) {
                if let Err(e) = provide_artifact(platform, artifact, &deps_path) {
                    if artifact.required {
                        panic!(
                            "Could not obtain prebuilt {}/{}: {e}\n\
                             Set $XGBOOST_LIB_DIR to a directory holding a prebuilt xgboost library, \
                             $XGBOOST_LIB_CACHE to a directory holding the artifacts, or \
                             $XGBOOST_LIB_URL to a mirror base URL.",
                            platform, artifact.file
                        );
                    }
                    println!("cargo:warning=optional artifact {} unavailable: {e}", artifact.file);
                }
            }
        } else if let Ok(homebrew_path) = std::env::var("HOMEBREW_PREFIX") {
            let xgboost_lib_dir = format!("{}/opt/xgboost/lib", &homebrew_path);
            println!("cargo:rustc-link-search=native={}", xgboost_lib_dir);
        } else {
            panic!("Please set $XGBOOST_LIB_DIR")
        }
    }

    #[cfg(feature = "local_build")]
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

/// One prebuilt file, pinned by content hash so a truncated or error-page download can never be
/// mistaken for the real library.
#[cfg(feature = "use_prebuilt_xgb")]
struct Artifact {
    file: &'static str,
    sha256: &'static str,
    /// `false` for files that are fetched for convenience but not linked against, so a failure to
    /// obtain them must not break the build.
    required: bool,
}

/// Prebuilt libraries are only published for these targets.
///
/// Note: `cfg!` in a build script describes the host, matching the previous behaviour. Cross
/// compilation would need `CARGO_CFG_TARGET_OS` / `CARGO_CFG_TARGET_ARCH` instead.
#[cfg(feature = "use_prebuilt_xgb")]
fn prebuilt_platform() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("mac_arm64")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("linux_arm64")
    } else if cfg!(target_os = "linux") {
        Some("linux_amd64")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("win_amd64")
    } else {
        None
    }
}

/// Checksums are of the files committed under `xgboost-sys/lib/`, which are the same blobs the
/// mirrors serve for [`LIB_TAG`].
///
/// `libdmlc.a` is only linked by the `local_build` feature, so it is not required here.
#[cfg(feature = "use_prebuilt_xgb")]
fn prebuilt_artifacts(platform: &str) -> &'static [Artifact] {
    match platform {
        "mac_arm64" => &[
            Artifact {
                file: "libxgboost.dylib",
                sha256: "b0eaad2ca2080d6ef99e17d96cd4fe9f63e519e13a90cc4d4ad20435bacdd127",
                required: true,
            },
            Artifact {
                file: "libdmlc.a",
                sha256: "9670e7587af234cbbdf6eef8f2241ebe64074a09aecc36551ad38e18e94dd7b0",
                required: false,
            },
        ],
        "linux_amd64" => &[
            Artifact {
                file: "libxgboost.so",
                sha256: "bc3bb6544a25bbc975acdf2f20e83aa3a5197e51b4b9bfe536c1376e1a0ce665",
                required: true,
            },
            Artifact {
                file: "libdmlc.a",
                sha256: "b8c472437a5153a4549f82f85c53a6ba482dae28b180d68ad87297e394e0a999",
                required: false,
            },
        ],
        "linux_arm64" => &[
            Artifact {
                file: "libxgboost.so",
                sha256: "85f71762ef3ed451e4df2c9de962de135c6d86baf5fc9f2f836879f8df66b344",
                required: true,
            },
            Artifact {
                file: "libdmlc.a",
                sha256: "8cff76772920bf70688de75da8b3e219f28c179fc42bbbd16d8d89548746e7fe",
                required: false,
            },
        ],
        "win_amd64" => &[
            Artifact {
                file: "xgboost.dll",
                sha256: "92883916c34b11e203072545e67f9d2ba28a1952e03fdb49aec0817341d60665",
                required: true,
            },
            Artifact {
                file: "xgboost.lib",
                sha256: "d32fbf89c06053d06927fe9f65a5736e27e569cd26e1ffec437a73639fa064a9",
                required: true,
            },
        ],
        _ => &[],
    }
}

/// Place `artifact` in `deps_path`, preferring sources that need no network at all.
///
/// Order: an already-valid copy in `deps_path`, then the local directories from
/// [`local_lib_dirs`], then the mirrors. Anything that fails its checksum is discarded and the
/// next source is tried, so a bad file can never persist across builds.
#[cfg(feature = "use_prebuilt_xgb")]
fn provide_artifact(platform: &str, artifact: &Artifact, deps_path: &Path) -> Result<()> {
    let dest = deps_path.join(artifact.file);

    if dest.exists() {
        if let Ok(bytes) = std::fs::read(&dest) {
            let actual = sha256_hex(&bytes);
            // The stamp records the pinned hash and what the file must hash to after
            // `install_artifact` rewrote it. Both have to match: the second catches corruption,
            // the first catches a file left behind by a build that pinned a different version.
            if let Ok(stamp) = std::fs::read_to_string(stamp_path(&dest)) {
                let mut lines = stamp.lines();
                if lines.next() == Some(artifact.sha256) && lines.next() == Some(actual.as_str()) {
                    return Ok(());
                }
            }
            if actual == artifact.sha256 {
                // Intact, but left by a build that predates `install_artifact`. Finalize it in
                // place rather than re-fetching bytes we already have.
                install_artifact(&dest, &bytes, artifact.sha256)?;
                return Ok(());
            }
        }
        // A previous build stored a truncated file or an HTML error page. Drop it so this build
        // can heal itself instead of failing at link time forever.
        println!("cargo:warning=discarding corrupt {}, re-fetching", dest.display());
        std::fs::remove_file(&dest)?;
        let _ = std::fs::remove_file(stamp_path(&dest));
    }

    for dir in local_lib_dirs(platform) {
        let src = dir.join(artifact.file);
        if let Ok(bytes) = std::fs::read(&src) {
            if sha256_hex(&bytes) == artifact.sha256 {
                install_artifact(&dest, &bytes, artifact.sha256)?;
                return Ok(());
            }
            println!("cargo:warning=ignoring {} (checksum mismatch)", src.display());
        }
    }

    let mut errors = Vec::new();
    for base in mirror_bases() {
        let url = mirror_url(&base, platform, artifact.file);
        match download_verified(&url, artifact.sha256) {
            Ok(bytes) => {
                install_artifact(&dest, &bytes, artifact.sha256)?;
                cache_store(platform, artifact.file, &bytes);
                return Ok(());
            }
            Err(e) => errors.push(format!("  {url}: {e}")),
        }
    }
    Err(format!("all sources failed:\n{}", errors.join("\n")).into())
}

/// Put verified bytes in place and make the result loadable from `deps`.
///
/// macOS copies a dependency's own `install_name` into every binary linking it, and the prebuilt
/// dylibs carry an absolute Homebrew path. Without this rewrite the pinned library is put in place
/// and then ignored at run time in favour of whatever happens to sit at that path — which may be a
/// different, ABI-incompatible XGBoost. Since the rewrite changes the file, its post-rewrite hash
/// is stamped alongside the pinned one, so later builds can tell an intact copy from a corrupt one
/// and from one installed for a different pin.
///
/// The new name is `dest` itself rather than an `@rpath` entry, because a `-sys` crate cannot add
/// an rpath to the binaries of the packages that depend on it.
#[cfg(feature = "use_prebuilt_xgb")]
fn install_artifact(dest: &Path, bytes: &[u8], expected_sha256: &str) -> Result<()> {
    write_atomic(dest, bytes)?;
    if cfg!(target_os = "macos") && dest.extension().is_some_and(|ext| ext == "dylib") {
        let status = std::process::Command::new("install_name_tool")
            .arg("-id")
            .arg(dest)
            .arg(dest)
            .status()?;
        if !status.success() {
            return Err(format!("install_name_tool -id failed for {}", dest.display()).into());
        }
        // Rewriting the load command invalidates the signature, and arm64 macOS refuses to map an
        // incorrectly signed image, killing the process at load time. Re-sign ad-hoc.
        let status = std::process::Command::new("codesign")
            .args(["--force", "--sign", "-"])
            .arg(dest)
            .status()?;
        if !status.success() {
            return Err(format!("codesign failed for {}", dest.display()).into());
        }
    }
    let installed = sha256_hex(&std::fs::read(dest)?);
    std::fs::write(stamp_path(dest), format!("{expected_sha256}\n{installed}\n"))?;
    Ok(())
}

#[cfg(feature = "use_prebuilt_xgb")]
fn stamp_path(dest: &Path) -> PathBuf {
    let mut path = dest.as_os_str().to_os_string();
    path.push(".sha256");
    PathBuf::from(path)
}

/// Directories searched before going to the network: an explicit cache, the default cache under
/// `CARGO_HOME`, and the copies committed in this repository (absent from the published crate,
/// which excludes `lib/*`).
#[cfg(feature = "use_prebuilt_xgb")]
fn local_lib_dirs(platform: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(cache) = cache_dir() {
        dirs.push(cache.join(platform));
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        dirs.push(Path::new(&manifest).join("lib").join(platform));
    }
    dirs
}

#[cfg(feature = "use_prebuilt_xgb")]
fn cache_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("XGBOOST_LIB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    let cargo_home = std::env::var("CARGO_HOME").ok().map(PathBuf::from).or_else(|| {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .map(|h| Path::new(&h).join(".cargo"))
    })?;
    Some(cargo_home.join("xgboost-prebuilt").join(LIB_TAG))
}

/// Keep a verified copy outside `target/`, so a `cargo clean` or a second checkout does not
/// re-download 15 MB.
#[cfg(feature = "use_prebuilt_xgb")]
fn cache_store(platform: &str, file: &str, bytes: &[u8]) {
    if let Some(dir) = cache_dir().map(|d| d.join(platform)) {
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = write_atomic(&dir.join(file), bytes);
        }
    }
}

#[cfg(feature = "use_prebuilt_xgb")]
fn mirror_bases() -> Vec<String> {
    let mut bases: Vec<String> = std::env::var("XGBOOST_LIB_URL").into_iter().collect();
    bases.extend(MIRRORS.iter().map(|m| m.replace("{tag}", LIB_TAG)));
    bases
}

/// GitHub release assets live in a flat namespace, so the platform becomes a filename prefix
/// there; every other mirror uses a directory per platform.
#[cfg(feature = "use_prebuilt_xgb")]
fn mirror_url(base: &str, platform: &str, file: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.contains("/releases/download/") {
        format!("{base}/{platform}-{file}")
    } else {
        format!("{base}/{platform}/{file}")
    }
}

/// Download `url`, retrying transient failures, and return the body only if it hashes to
/// `expected_sha256`.
#[cfg(feature = "use_prebuilt_xgb")]
fn download_verified(url: &str, expected_sha256: &str) -> Result<Vec<u8>> {
    const ATTEMPTS: u32 = 3;
    /// The libraries run to tens of megabytes, far past ureq's 10 MB default body limit.
    const MAX_BODY: u64 = 256 * 1024 * 1024;

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(300)))
        .build()
        .into();

    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
        }
        // ureq treats 4xx/5xx as errors by default, so an outage page cannot reach the checksum
        // step, let alone the disk.
        let result = agent.get(url).call().map_err(|e| e.to_string()).and_then(|mut resp| {
            resp.body_mut()
                .with_config()
                .limit(MAX_BODY)
                .read_to_vec()
                .map_err(|e| e.to_string())
        });
        match result {
            Ok(body) => {
                let actual = sha256_hex(&body);
                if actual == expected_sha256 {
                    return Ok(body);
                }
                last_err = Some(format!(
                    "checksum mismatch ({} bytes, sha256 {actual}, expected {expected_sha256})",
                    body.len()
                ));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| "download failed".into()).into())
}

/// Write via a temporary file in the same directory, so an interrupted build leaves no
/// half-written library behind.
#[cfg(feature = "use_prebuilt_xgb")]
fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<()> {
    // Suffix the whole file name rather than replacing the extension: `xgboost.dll` and
    // `xgboost.lib` would otherwise share a temporary path within one build.
    let mut name = dest
        .file_name()
        .ok_or_else(|| format!("no file name in destination {}", dest.display()))?
        .to_os_string();
    name.push(format!(".tmp{}", std::process::id()));
    let tmp = dest.with_file_name(name);
    std::fs::write(&tmp, bytes)?;
    if let Err(e) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

#[cfg(feature = "use_prebuilt_xgb")]
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut out = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}
