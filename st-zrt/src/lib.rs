//! st-zrt — Stellarion's zero-overhead Rust runtime over onnxruntime.
//!
//! Scope (locked, see `DESIGN.md`): the *runtime library* only. Kernels are reused,
//! not written; a serving layer is a separate, later project. The win lives in the
//! binding/session/memory/IO/scheduling layer — zero binding tax, zero-copy tensor
//! I/O, pre-marshaled names, reused run options.
//!
//! This safe layer sits over [`st_zrt_sys`] — the exhaustive, **generated** FFI table
//! (see `st-zrt-sys/src/generated.rs`, produced by `st-zrt-sys-codegen`).

pub use st_zrt_sys as sys;
pub use sys::{
    AllocatorType, ElementType, ExecutionMode, GraphOptimizationLevel, LoggingLevel, MemType,
    OrtErrorCode, SparseFormat, SparseIndicesFormat,
};

mod allocator;
mod arena;
#[cfg(feature = "custom-ops")]
mod custom_ops;
mod element;
mod environment;
#[cfg(feature = "ep")]
mod ep;
#[cfg(feature = "ep")]
mod ep_device;
mod error;
mod initializer;
#[cfg(feature = "model-editor")]
mod interop;
mod io_binding;
mod memory;
mod metadata;
#[cfg(feature = "model-editor")]
mod model_editor;
mod prepacked;
mod run_options;
mod runtime;
#[cfg(feature = "serde")]
mod serde_support;
mod session;
mod session_options;
mod tensor;
mod threading;
mod type_info;

pub use allocator::{Allocation, Allocator, AllocatorStats, AllocatorStatsDelta};
pub use arena::{ArenaCfg, ArenaExtendStrategy};
#[cfg(feature = "custom-ops")]
pub use custom_ops::{
    CustomOp, CustomOpDomain, KernelContext, KernelInfo, Op, OpAttr, OpIoSpec, OwnedKernelInfo,
    ShapeInferContext,
};
// The generic trampolines the `#[macro_export] custom_op!` macro names via
// `$crate::__priv`. `#[doc(hidden)]` keeps them out of the public surface.
#[cfg(feature = "custom-ops")]
#[doc(hidden)]
pub use custom_ops::__priv;
pub use element::TensorElement;
pub use environment::Environment;
#[cfg(feature = "ep")]
pub use ep::{
    CannOptions, CudaArenaExtendStrategy, CudaCudnnConvAlgoSearch, CudaOptions, CudaPreset,
    CudaProviderOptions, DnnlOptions, EpProvider, MigraphxOptions, OpenvinoOptions, RocmOptions,
    TensorRtOptions,
};
#[cfg(feature = "ep")]
pub use ep_device::{get_ep_devices, EpDevice};
pub use error::{Error, Result};
pub use initializer::OwnedInitializer;
#[cfg(feature = "model-editor")]
pub use interop::{
    deinit_graphics_interop_for_ep_device, init_graphics_interop_for_ep_device,
    ExternalMemoryDescriptor, ExternalMemoryHandle, ExternalMemoryHandleType,
    ExternalResourceImporter, ExternalSemaphoreDescriptor, ExternalSemaphoreHandle,
    ExternalSemaphoreType, ExternalTensorDescriptor, GraphicsApi, GraphicsInteropConfig,
};
pub use io_binding::{IoBinding, OutputValue};
pub use memory::{MemoryInfo, MemoryInfoSnapshot};
pub use metadata::ModelMetadata;
#[cfg(feature = "model-editor")]
pub use model_editor::{
    compile_api, ep_api, interop_api, model_editor_api, Graph, Model, ModelCompilationOptions,
    Node, NodeAttr, TypeInfo, ValueInfo,
};
pub use prepacked::PrepackedWeightsContainer;
pub use run_options::RunOptions;
pub use runtime::{
    DynamicIoOptions, DynamicIoRuntime, Lane, Runtime, RuntimeMode, ShapeBucket, ShapeKey,
    StaticIoLane, StaticIoRuntime,
};
pub use session::{
    AllocatedOutputTensorIoLane, AllocatedTensorIoLane, DeviceOutputTensorIoLane, LaneBufferPolicy,
    LaneRunAllocatorStats, PreparedIoBinding, PreparedRun, RunFuture, Session, StaticTensorIoLane,
    TensorIoLane,
};
pub use session_options::{ArenaState, MemPatternState, SessionOptions};
pub use tensor::{
    AllocatedTensor, MmapTensorOptions, OwnedValue, RunInput, SparseTensor, StringTensor, Tensor,
    TensorBuffer, TensorView,
};
pub use threading::{ThreadManager, ThreadingOptions};
pub use type_info::TensorTypeAndShapeInfo;

// ─── crate-private helpers shared across modules ─────────────────────────────
/// Borrow the live `Api` function-pointer table (a process global; lives forever).
#[inline]
pub(crate) fn api() -> &'static sys::Api {
    // SAFETY: the table is a process-global returned by the engine.
    unsafe { &*sys::api() }
}

/// Turn a raw `OrtStatus*` into `Result<()>`: null ⇒ Ok; else Err (code+message),
/// with the status released.
#[inline]
pub(crate) fn check(status: sys::StatusPtr) -> Result<()> {
    unsafe { sys::status_to_result(&*sys::api(), status).map_err(Error::from) }
}

/// Copy a non-null C string into an owned UTF-8 `String`.
#[inline]
pub(crate) unsafe fn cstr_to_string(
    raw: *const std::ffi::c_char, what: &'static str,
) -> Result<String> {
    std::ffi::CStr::from_ptr(raw)
        .to_str()
        .map(str::to_owned)
        .map_err(|_| Error::new(-1, format!("zrt: {what} is not valid UTF-8")))
}

#[inline]
pub(crate) fn ensure_non_null<T>(ptr: *mut T, what: &'static str) -> Result<*mut T> {
    if ptr.is_null() {
        Err(Error::new(-1, format!("zrt: {what} pointer is null")))
    } else {
        Ok(ptr)
    }
}

#[inline]
pub(crate) fn slice_data_ptr<T>(ptr: *mut T, len: usize, what: &'static str) -> Result<*mut T> {
    if ptr.is_null() {
        if len == 0 {
            Ok(std::ptr::NonNull::<T>::dangling().as_ptr())
        } else {
            Err(Error::new(-1, format!("zrt: {what} pointer is null")))
        }
    } else {
        Ok(ptr)
    }
}

/// Byte size of one element of an ONNX tensor element type (0 for opaque/string).
pub(crate) fn element_size(e: sys::ElementType) -> usize {
    use sys::ElementType::*;
    match e {
        Float | Int32 | Uint32 => 4,
        Double | Int64 | Uint64 | Complex64 => 8,
        Complex128 => 16,
        Uint16 | Int16 | Float16 | Bfloat16 => 2,
        Uint8 | Int8 | Bool | Float8E4M3FN | Float8E4M3FNUZ | Float8E5M2 | Float8E5M2FNUZ => 1,
        Uint4 | Int4 | Float4E2M1 => 0,
        Undefined | String => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{element_size, sys::ElementType};

    #[test]
    fn element_size_covers_quantized_and_float8_metadata_types() {
        assert_eq!(element_size(ElementType::Int8), 1);
        assert_eq!(element_size(ElementType::Uint8), 1);
        assert_eq!(element_size(ElementType::Float8E4M3FN), 1);
        assert_eq!(element_size(ElementType::Float8E4M3FNUZ), 1);
        assert_eq!(element_size(ElementType::Float8E5M2), 1);
        assert_eq!(element_size(ElementType::Float8E5M2FNUZ), 1);

        // Packed 4-bit tensors are not exposed as a typed Rust slice. ORT may
        // run graphs that contain these types internally, but ZRT reports them
        // as opaque metadata until a packed storage wrapper exists.
        assert_eq!(element_size(ElementType::Int4), 0);
        assert_eq!(element_size(ElementType::Uint4), 0);
        assert_eq!(element_size(ElementType::Float4E2M1), 0);
    }
}
