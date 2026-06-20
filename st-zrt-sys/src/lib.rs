//! st-zrt-sys — hand-written FFI to libonnxruntime, in the `zrt` namespace.
//!
//! NO bindgen: `OrtApi`'s function-pointer table is macro-defined (bindgen can't parse
//! it and drags in legacy `Ort*` names). Instead, the `st-zrt-sys-codegen` tool
//! preprocesses `onnxruntime_c_api.h` (`gcc -E -P`), parses the fully-expanded
//! `struct OrtApi`, and emits [`mod generated`] — the exhaustive IDX_*/typed-alias/
//! accessor table with zrt names. This file keeps only the hand-written core: opaque
//! handles, the stable enums, `ApiBase`, the `Api` table primitive, the entry point,
//! and `status_to_result`.
//!
//! The only C symbol linked by name is `OrtGetApiBase` (`#[link_name]`); everything
//! else is reached through `Api` by positional index.
#![allow(non_camel_case_types, non_snake_case, dead_code, clippy::all)]

use std::ffi::c_void;
use std::os::raw::c_char;
#[cfg(feature = "custom-ops")]
use std::os::raw::c_int;

/// libonnxruntime API version we bind (1.26.0 → API version 26).
pub const API_VERSION: u32 = 26;

// ─── opaque-handle macro (the generated table invokes this) ──────────────────
macro_rules! opaque_handle {
    ($name:ident) => {
        #[repr(C)]
        pub struct $name(crate::private::Opaque);
    };
}
mod private {
    /// Incomplete-opaque marker. We only ever hold `*mut Handle` values handed to us
    /// by ORT; the interior is never constructed or read on our side.
    #[repr(C)]
    pub struct Opaque(*const u8);
}

// ─── enums (ONNX / ORT values, stable) ───────────────────────────────────────
/// ONNXTensorElementDataType.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Undefined = 0,
    Float = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    Uint32 = 12,
    Uint64 = 13,
    Bfloat16 = 16,
}

/// OrtLoggingLevel (a.k.a. LogLevel).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}
pub type LogLevel = LoggingLevel;

/// GraphOptimizationLevel.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
    DisableAll = 0,
    Basic = 1,
    Extended = 2,
    Layout = 3,
    All = 99,
}

/// OrtAllocatorType.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorType {
    Invalid = -1,
    Device = 0,
    Arena = 1,
    ReadOnly = 2,
}

/// OrtMemType.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemType {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
}

/// ONNXType (the top-level value kind).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxType {
    Unknown = 0,
    Tensor = 1,
    Sequence = 2,
    Map = 3,
    Optional = 4,
    SparseTensor = 5,
    Opaque = 6,
}

/// OrtSparseFormat.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    Undefined = 0,
    Coo = 0x1,
    Csrc = 0x2,
    BlockSparse = 0x4,
}

/// OrtSparseIndicesFormat.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseIndicesFormat {
    Coo = 0,
    CsrInner = 1,
    CsrOuter = 2,
    BlockSparse = 3,
}

/// ExecutionMode.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential = 0,
    Parallel = 1,
}

/// OrtErrorCode — the status code `GetErrorCode` returns and `CreateStatus` accepts
/// (`onnxruntime_c_api.h:257-273`). `#[repr(i32)]` makes it ABI-compatible with the
/// `c_int`-typed generated status API (`CreateStatusFn` / `GetErrorCodeFn` — both map the
/// C `OrtErrorCode` to `c_int`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtErrorCode {
    /// `ORT_OK` — success.
    Ok = 0,
    /// `ORT_FAIL` — generic failure; also the code custom-op kernels surface on error.
    Fail = 1,
    /// `ORT_INVALID_ARGUMENT`.
    InvalidArgument = 2,
    /// `ORT_NO_SUCHFILE`.
    NoSuchFile = 3,
    /// `ORT_NO_MODEL`.
    NoModel = 4,
    /// `ORT_ENGINE_ERROR`.
    EngineError = 5,
    /// `ORT_RUNTIME_EXCEPTION`.
    RuntimeException = 6,
    /// `ORT_INVALID_PROTOBUF`.
    InvalidProtobuf = 7,
    /// `ORT_MODEL_LOADED`.
    ModelLoaded = 8,
    /// `ORT_NOT_IMPLEMENTED`.
    NotImplemented = 9,
    /// `ORT_INVALID_GRAPH`.
    InvalidGraph = 10,
    /// `ORT_EP_FAIL`.
    EpFail = 11,
    /// `ORT_MODEL_LOAD_CANCELED`.
    ModelLoadCanceled = 12,
    /// `ORT_MODEL_REQUIRES_COMPILATION`.
    ModelRequiresCompilation = 13,
    /// `ORT_NOT_FOUND`.
    NotFound = 14,
}

impl OrtErrorCode {
    /// Map a raw `c_int` error code to the enum, or `None` if it is outside the known range
    /// (a future ORT version could add codes; st-zrt-local errors use negative codes).
    pub fn from_c_int(code: core::ffi::c_int) -> Option<Self> {
        Some(match code {
            0 => Self::Ok,
            1 => Self::Fail,
            2 => Self::InvalidArgument,
            3 => Self::NoSuchFile,
            4 => Self::NoModel,
            5 => Self::EngineError,
            6 => Self::RuntimeException,
            7 => Self::InvalidProtobuf,
            8 => Self::ModelLoaded,
            9 => Self::NotImplemented,
            10 => Self::InvalidGraph,
            11 => Self::EpFail,
            12 => Self::ModelLoadCanceled,
            13 => Self::ModelRequiresCompilation,
            14 => Self::NotFound,
            _ => return None,
        })
    }
}

/// OrtOpAttrType — the value kind of a custom-op attribute (CreateOpAttr / OpAttr_GetType).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpAttrType {
    Undefined = 0,
    Int = 1,
    Ints = 2,
    Float = 3,
    Floats = 4,
    String = 5,
    Strings = 6,
    Graph = 7,
    Tensor = 8,
}

// ─── custom-op authoring: the OrtCustomOp vtable (feature `custom-ops`) ───────
//
// `OrtCustomOp` is the struct a custom-op author fills in: the op's schema (name,
// input/output element types + characteristics, opset range) plus the kernel callbacks ORT
// invokes (Create/Compute/Destroy). It is passed to ORT by pointer (via
// `CustomOpDomain_Add`); its storage must outlive every session using it (a `pub static`
// built by the `custom_op!` macro satisfies this).
//
// The field *parameter types* are erased to `*const c_void` / `*mut c_void` for the ORT
// handles (`op`, `api`, `info`, `ctx`, the shape-infer context) — this is layout-neutral
// (every fn-pointer slot is one word regardless of its signature, so only field COUNT/ORDER
// must match the C `struct OrtCustomOp` in onnxruntime_c_api.h), and the `custom_op!`
// trampolines cast to the real handle types at call time. `version` must equal
// `API_VERSION` (ORT reads it to size the table).
#[cfg(feature = "custom-ops")]
#[repr(C)]
pub struct OrtCustomOp {
    pub version: u32,
    pub create_kernel: Option<
        unsafe extern "C" fn(
            op: *const c_void,
            api: *const c_void,
            info: *const c_void,
        ) -> *mut c_void,
    >,
    pub get_name: Option<unsafe extern "C" fn(op: *const c_void) -> *const c_char>,
    pub get_execution_provider_type:
        Option<unsafe extern "C" fn(op: *const c_void) -> *const c_char>,
    pub get_input_type: Option<unsafe extern "C" fn(op: *const c_void, index: usize) -> i32>,
    pub get_input_type_count: Option<unsafe extern "C" fn(op: *const c_void) -> usize>,
    pub get_output_type: Option<unsafe extern "C" fn(op: *const c_void, index: usize) -> i32>,
    pub get_output_type_count: Option<unsafe extern "C" fn(op: *const c_void) -> usize>,
    pub kernel_compute: Option<unsafe extern "C" fn(kernel: *mut c_void, ctx: *mut c_void)>,
    pub kernel_destroy: Option<unsafe extern "C" fn(kernel: *mut c_void)>,
    pub get_input_characteristic:
        Option<unsafe extern "C" fn(op: *const c_void, index: usize) -> i32>,
    pub get_output_characteristic:
        Option<unsafe extern "C" fn(op: *const c_void, index: usize) -> i32>,
    pub get_input_memory_type: Option<unsafe extern "C" fn(op: *const c_void, index: usize) -> i32>,
    pub get_variadic_input_min_arity: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub get_variadic_input_homogeneity: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub get_variadic_output_min_arity: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub get_variadic_output_homogeneity: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub create_kernel_v2: Option<
        unsafe extern "C" fn(
            op: *const c_void,
            api: *const c_void,
            info: *const c_void,
            kernel: *mut *mut c_void,
        ) -> generated::StatusPtr,
    >,
    pub kernel_compute_v2:
        Option<unsafe extern "C" fn(kernel: *mut c_void, ctx: *mut c_void) -> generated::StatusPtr>,
    pub infer_output_shape_fn:
        Option<unsafe extern "C" fn(op: *const c_void, sctx: *mut c_void) -> generated::StatusPtr>,
    pub get_start_version: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub get_end_version: Option<unsafe extern "C" fn(op: *const c_void) -> c_int>,
    pub get_may_inplace: Option<
        unsafe extern "C" fn(input_index: *mut *mut c_int, output_index: *mut *mut c_int) -> usize,
    >,
    pub release_may_inplace:
        Option<unsafe extern "C" fn(input_index: *mut c_int, output_index: *mut c_int)>,
    pub get_alias_map: Option<
        unsafe extern "C" fn(input_index: *mut *mut c_int, output_index: *mut *mut c_int) -> usize,
    >,
    pub release_alias_map:
        Option<unsafe extern "C" fn(input_index: *mut c_int, output_index: *mut c_int)>,
}

/// OrtCustomOpInputOutputCharacteristic — whether an input/output is required, optional, or
/// variadic (onnxruntime_c_api.h:7456). `#[repr(i32)]` matches the C `enum`.
#[cfg(feature = "custom-ops")]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomOpInputOutputCharacteristic {
    Required = 0,
    Optional = 1,
    Variadic = 2,
}

// Compile-time check: the `#[repr(C)]` table is exactly 208 bytes on 64-bit
// (1×u32 + 4 pad + 25 fn-ptr slots). Catches a wrong field *count*; field *order* is pinned
// by the named-field construction and checked at runtime in st-zrt-sys's test suite.
#[cfg(feature = "custom-ops")]
const _: () = assert!(std::mem::size_of::<OrtCustomOp>() == 208);

/// OrtLoggingFunction — the custom-logger callback (`CreateEnvWithCustomLogger`).
pub type LoggingFunction = Option<
    unsafe extern "C" fn(
        param: *mut c_void,
        severity: LoggingLevel,
        category: *const c_char,
        logid: *const c_char,
        status_messages: *const *const c_char,
        num_status_messages: usize,
    ),
>;

// ─── ApiBase ─────────────────────────────────────────────────────────────────
/// `OrtApiBase`. We only use `GetApi` (offset 0) and `GetVersionString` (offset 1).
#[repr(C)]
pub struct ApiBase {
    pub get_api: Option<unsafe extern "C" fn(version: u32) -> *const Api>,
    pub get_version_string: Option<unsafe extern "C" fn() -> *const c_char>,
}

impl ApiBase {
    #[inline]
    pub unsafe fn version_string(&self) -> Option<&'static std::ffi::CStr> {
        let f = self.get_version_string?;
        Some(std::ffi::CStr::from_ptr(f()))
    }
}

// ─── Api: the function-pointer table primitive ───────────────────────────────
/// `OrtApi` — a table of function pointers, indexed positionally. The struct is
/// effectively `[*const c_void; N]` (all slots pointer-sized, no padding). The
/// generated table (in [`mod generated`]) adds the typed accessors.
#[repr(C)]
pub struct Api([u8; 0]);

impl Api {
    /// Read the raw function pointer at positional index `idx`.
    #[inline]
    pub unsafe fn fn_ptr(&self, idx: usize) -> *const c_void {
        (self as *const Self as *const *const c_void)
            .add(idx)
            .read()
    }

    /// Typed accessor used by every generated wrapper: null-check the slot, then
    /// reinterpret the data pointer as the fn-pointer type `T`.
    #[inline]
    unsafe fn f<T: Copy>(&self, idx: usize) -> T {
        let p = self.fn_ptr(idx);
        assert!(
            !p.is_null(),
            "st-zrt-sys: Api[{idx}] is null — header/version mismatch"
        );
        std::mem::transmute_copy(&p)
    }
}

/// GENERATED: the exhaustive `OrtApi` table — IDX_* indices, typed fn aliases, Api
/// accessors, and the opaque handle types. zrt names; no `Ort*`; no bindgen.
/// Regenerate via `cargo run -p st-zrt-sys-codegen -- <header> <this file>`.
pub mod generated;

// ─── entry point (the one C symbol we link by name) ──────────────────────────
extern "C" {
    #[link_name = "OrtGetApiBase"]
    fn get_api_base_ffi() -> *const ApiBase;
}

/// The global `OrtApiBase`.
#[inline]
pub fn api_base() -> *const ApiBase {
    unsafe { get_api_base_ffi() }
}

/// The `Api` function-pointer table for [`API_VERSION`].
///
/// # Panics
/// Panics if the engine can't be loaded or the API version is unavailable.
#[inline]
pub fn api() -> *const Api {
    unsafe {
        let base = get_api_base_ffi();
        assert!(
            !base.is_null(),
            "st-zrt-sys: OrtGetApiBase returned null — libonnxruntime not loaded"
        );
        let get_api = (*base).get_api.expect("st-zrt-sys: GetApi missing");
        let api = get_api(API_VERSION);
        assert!(!api.is_null(), "st-zrt-sys: GetApi({API_VERSION}) returned null — version mismatch with libonnxruntime");
        api
    }
}

/// Turn a raw `OrtStatus*` into `Result`: `Ok(())` if null, else code+message.
/// On the error path the status is consumed (released).
#[inline]
pub unsafe fn status_to_result(
    api: &Api, status: generated::StatusPtr,
) -> Result<(), (i32, std::ffi::CString)> {
    if status.is_null() {
        return Ok(());
    }
    let code = api.get_error_code()(status as *const generated::StatusHandle);
    let msg_ptr = api.get_error_message()(status as *const generated::StatusHandle);
    let msg = if msg_ptr.is_null() {
        std::ffi::CString::new("(null error message)").unwrap()
    } else {
        std::ffi::CStr::from_ptr(msg_ptr).to_owned()
    };
    api.release_status()(status);
    Err((code, msg))
}

// Re-export the generated table at the crate root so callers use `st_zrt_sys::IDX_RUN`
// / `st_zrt_sys::Api` uniformly regardless of whether a symbol is hand-written or
// generated.
pub use generated::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_table_loads() {
        let base = api_base();
        assert!(!base.is_null(), "api_base() null");
        let api = api();
        assert!(!api.is_null(), "api() null — GetApi({API_VERSION}) failed");
        if let Some(vs) = unsafe { (*base).version_string() } {
            let s = vs.to_string_lossy();
            eprintln!("onnxruntime: {s}");
            assert!(s.contains("1.26"), "unexpected ort version: {s}");
        }
    }

    /// Functional index proof: call CreateSessionOptions(idx 10) + Release(idx 100)
    /// through the GENERATED table. Crashes/errors if any index/signature is wrong.
    #[test]
    fn generated_indices_functionally_validated() {
        unsafe {
            let api_ref = &*api();
            let mut opts: *mut SessionOptionsHandle = std::ptr::null_mut();
            let st = api_ref.create_session_options()(&mut opts);
            assert!(st.is_null(), "CreateSessionOptions failed");
            assert!(!opts.is_null(), "CreateSessionOptions gave null handle");
            api_ref.release_session_options()(opts);
            eprintln!("generated_indices_functionally_validated: CreateSessionOptions(10) + Release(100) OK");
        }
    }

    /// Proves the `OrtErrorCode` enum's integer values match ORT's encoding: build a status
    /// with `CreateStatus(<code>)`, read it back with `GetErrorCode`, and assert the round
    /// trip for several variants. Exercises IDX_CREATE_STATUS (0) + IDX_GET_ERROR_CODE (1).
    #[test]
    fn ort_error_code_round_trips() {
        unsafe {
            let api_ref = &*api();
            let msg = b"zrt probe\0";
            for code in [
                OrtErrorCode::InvalidArgument,
                OrtErrorCode::Fail,
                OrtErrorCode::NotFound,
            ] {
                let status = api_ref.create_status()(code as core::ffi::c_int, msg.as_ptr().cast());
                assert!(!status.is_null(), "CreateStatus gave a null status");
                let got = api_ref.get_error_code()(status as *const StatusHandle);
                assert_eq!(got, code as core::ffi::c_int, "GetErrorCode round trip");
                assert_eq!(
                    OrtErrorCode::from_c_int(got),
                    Some(code),
                    "from_c_int maps back"
                );
                api_ref.release_status()(status);
            }
            eprintln!(
                "ort_error_code_round_trips: enum values match ORT (InvalidArgument/Fail/NotFound)"
            );
        }
    }

    /// The four sub-API gateway getters (model-editor feature) must return non-null at
    /// API version 26 — proves the gateway indices are correct and that ORT populates the
    /// OrtModelEditorApi / OrtCompileApi / OrtEpApi / OrtInteropApi function tables. A
    /// wrong index crashes; a null means ORT didn't populate that sub-API.
    #[cfg(feature = "model-editor")]
    #[test]
    fn sub_api_gateways_non_null() {
        unsafe {
            let api_ref = &*api();
            assert!(
                !api_ref.get_model_editor_api()().is_null(),
                "GetModelEditorApi null"
            );
            assert!(!api_ref.get_compile_api()().is_null(), "GetCompileApi null");
            assert!(!api_ref.get_ep_api()().is_null(), "GetEpApi null");
            assert!(!api_ref.get_interop_api()().is_null(), "GetInteropApi null");
            eprintln!("sub-API gateways all non-null: model_editor/compile/ep/interop");
        }
    }

    /// Pins the field *order* of `OrtCustomOp` (the size const-assert only catches a wrong
    /// *count*; two transposed fn-ptr fields have the same size). `addr_of!` computes field
    /// addresses without reading through the dangling pointer, so this is sound.
    #[cfg(feature = "custom-ops")]
    #[test]
    fn ortcustomop_field_offsets() {
        use core::ptr::addr_of;
        let p = core::ptr::NonNull::<OrtCustomOp>::dangling().as_ptr();
        let base = p as *const u8 as usize;
        unsafe {
            assert_eq!(addr_of!((*p).version) as *const u8 as usize - base, 0);
            assert_eq!(addr_of!((*p).get_name) as *const u8 as usize - base, 16);
            assert_eq!(
                addr_of!((*p).create_kernel_v2) as *const u8 as usize - base,
                136
            );
            assert_eq!(
                addr_of!((*p).release_alias_map) as *const u8 as usize - base,
                200
            );
        }
        eprintln!(
            "OrtCustomOp field order pinned: version@0, get_name@16, create_kernel_v2@136, release_alias_map@200"
        );
    }
}
