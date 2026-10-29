#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

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
}
