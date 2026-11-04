#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(feature = "runtime_load")]
pub use runtime::{available, load};

#[cfg(feature = "runtime_load")]
mod runtime {
    use libloading::Library;
    use std::cell::{Cell, RefCell};
    use std::ffi::{CString, OsStr};
    use std::os::raw::{c_char, c_int};
    use std::sync::OnceLock;

    static LIB: OnceLock<Result<Library, String>> = OnceLock::new();

    // Must run before the first XGBoost call; otherwise that call settles on the default search.
    pub fn load(path: impl AsRef<OsStr>) -> Result<(), String> {
        let lib = open(path.as_ref())?;
        LIB.set(Ok(lib))
            .map_err(|_| "the xgboost library is already resolved".to_string())
    }

    // Without a prior `load`: only `$XGBOOST_LIB_PATH` if set; otherwise the platform's library
    // search, then (debug builds only) the build's library directory.
    pub fn available() -> Result<(), String> {
        library().map(|_| ()).map_err(str::to_string)
    }

    fn library() -> Result<&'static Library, &'static str> {
        LIB.get_or_init(|| {
            if let Some(path) = std::env::var_os("XGBOOST_LIB_PATH") {
                return open(&path).map_err(|e| format!("$XGBOOST_LIB_PATH: {e}"));
            }
            let name = libloading::library_filename("xgboost");
            #[cfg(debug_assertions)]
            let built = option_env!("XGBOOST_BUILD_LIB_DIR").map(|d| std::path::Path::new(d).join(&name));
            #[cfg(not(debug_assertions))]
            let built: Option<std::path::PathBuf> = None;
            open(&name).or_else(|e| built.map_or(Err(e.clone()), |p| open(p.as_os_str()).map_err(|e2| format!("{e}; {e2}"))))
        })
        .as_ref()
        .map_err(String::as_str)
    }

    // Rejects a library of another major version: the shims would call it with the wrong ABI.
    fn open(path: &OsStr) -> Result<Library, String> {
        let lib = unsafe { Library::new(path) }.map_err(|e| e.to_string())?;
        let (mut major, mut minor, mut patch): (c_int, c_int, c_int) = (0, 0, 0);
        {
            let version = unsafe { lib.get::<unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int)>(b"XGBoostVersion\0") }
                .map_err(|e| format!("{} is not an xgboost library: {e}", path.to_string_lossy()))?;
            unsafe { version(&mut major, &mut minor, &mut patch) };
        }
        let expected: c_int = env!("XGBOOST_HEADER_MAJOR").parse().expect("XGBOOST_HEADER_MAJOR");
        if major != expected {
            return Err(format!(
                "{} is xgboost {major}.{minor}.{patch}, these bindings need {expected}.x",
                path.to_string_lossy()
            ));
        }
        Ok(lib)
    }

    pub(crate) fn symbol<F: Copy>(name: &[u8]) -> Option<F> {
        unsafe { library().ok()?.get::<F>(name) }.ok().map(|s| *s)
    }

    thread_local! {
        static PENDING: Cell<bool> = const { Cell::new(false) };
        static MESSAGE: RefCell<CString> = RefCell::new(CString::default());
    }

    pub(crate) fn fail(name: &str) {
        let why = match library() {
            Err(e) => format!("xgboost library not available: {e}"),
            Ok(_) => format!("xgboost library has no symbol {name}"),
        };
        MESSAGE.with(|m| *m.borrow_mut() = CString::new(why).unwrap_or_default());
        PENDING.with(|p| p.set(true));
    }

    pub(crate) fn pending_error() -> Option<*const c_char> {
        PENDING
            .with(|p| p.replace(false))
            .then(|| MESSAGE.with(|m| m.borrow().as_ptr()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_matrix() {
        let config = r#"{"uri": "xgboost/demo/data/agaricus.txt.train?format=libsvm&indexing_mode=1"}"#;

        let mut handle = std::ptr::null_mut();
        let config = std::ffi::CString::new(config).unwrap();
        let ret_val = unsafe { XGDMatrixCreateFromURI(config.as_ptr(), &mut handle) };
        assert_eq!(ret_val, 0);

        let mut num_rows = 0;
        let ret_val = unsafe { XGDMatrixNumRow(handle, &mut num_rows) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_rows, 6513);

        let mut num_cols = 0;
        let ret_val = unsafe { XGDMatrixNumCol(handle, &mut num_cols) };
        assert_eq!(ret_val, 0);
        assert_eq!(num_cols, 126);

        let ret_val = unsafe { XGDMatrixFree(handle) };
        assert_eq!(ret_val, 0);
    }

    #[cfg(feature = "runtime_load")]
    #[test]
    fn runtime_load_rejects_bad_paths_and_late_loads() {
        assert!(load("/nonexistent/libxgboost").is_err());
        assert_eq!(available(), Ok(()));
        let built = std::path::Path::new(env!("XGBOOST_BUILD_LIB_DIR")).join(libloading::library_filename("xgboost"));
        assert!(load(built).unwrap_err().contains("already resolved"));
    }
}
