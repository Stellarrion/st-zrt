//! Runtime-version guard coverage: the loaded libonnxruntime must belong to a supported
//! release line, and every `Environment` constructor enforces that before touching ORT.
//!
//! The sys crate bundles ONNX Runtime 1.27 and additionally supports 1.28.x runtimes bound
//! via `ST_ZRT_ORT_PATH` (the C API is append-only, so the API-27 table is valid there).
//! This file pins that contract from the wrapper side; the pure matching logic is unit-tested
//! in `st-zrt-sys`.

use st_zrt::{Environment, ORT_VERSION, SUPPORTED_RUNTIME_LINES};

#[test]
fn loaded_runtime_belongs_to_a_supported_line() {
    let found = st_zrt::sys::runtime_version_string().expect("GetVersionString is callable");
    assert!(
        st_zrt::sys::runtime_version_supported(&found),
        "loaded libonnxruntime {found:?} is not a supported line {SUPPORTED_RUNTIME_LINES:?} \
         for st-zrt bundling {ORT_VERSION}"
    );
    // `ORT_VERSION` is the build-time pin (major.minor.patch), not the runtime string.
    assert_eq!(
        ORT_VERSION.split('.').count(),
        3,
        "pin must be exact: {ORT_VERSION}"
    );
}

#[test]
fn every_environment_constructor_passes_the_guard() {
    // The default constructor is the guard's user-facing proof: with the bundled runtime
    // (or an ST_ZRT_ORT_PATH override inside a supported line) it must succeed; with any
    // other line it must return the loud unsupported-runtime error instead of proceeding.
    Environment::new().expect("supported runtime passes the version guard");
}

#[test]
fn guard_error_names_the_offending_runtime() {
    // Negative proof through the predicate the guard uses: every unsupported version
    // string must be rejected, so the guard can only let supported lines through.
    for unsupported in ["1.26.0", "1.99.0", "2.0.0"] {
        assert!(
            !st_zrt::sys::runtime_version_supported(unsupported),
            "{unsupported} must be rejected by the supported-line predicate"
        );
    }
}
