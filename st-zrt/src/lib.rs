//! st-zrt — Stellarion's zero-overhead Rust runtime over onnxruntime.
//!
//! Scope (locked, see `docs/architecture.md`): the *runtime library* only. Kernels are reused,
//! not written; a serving layer is a separate, later project. The win lives in the
//! binding/session/memory/IO/scheduling layer — zero binding tax, zero-copy tensor
//! I/O, pre-marshaled names, reused run options.
//!
//! This safe layer sits over [`st_zrt_sys`] — the exhaustive, **generated** FFI table
//! (see `st-zrt-sys/src/generated.rs`, produced by `st-zrt-sys-codegen`).
//!
//! # Examples
//!
//! Load a model and run one inference on a caller-owned buffer. Marked `no_run` so it compiles
//! against the real API (a doc-rot guard) without requiring a `model.onnx` on disk:
//!
//! ```no_run
//! use st_zrt::{
//!     Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
//! };
//!
//! fn main() -> st_zrt::Result<()> {
//!     let env = Environment::new()?;
//!     let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
//!     let session = Session::new(&env, "model.onnx", opts)?;
//!     let mem = MemoryInfo::cpu()?;
//!
//!     let input_buf = vec![0.0_f32; 784];
//!     let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &mem)?;
//!
//!     let mut outputs: Vec<Option<OwnedValue>> =
//!         (0..session.output_count()).map(|_| None).collect();
//!     session.run(&[&input], &mut outputs)?;
//!
//!     let logits = outputs[0].as_ref().unwrap().as_slice::<f32>()?;
//!     println!("{:?}", &logits[..3.min(logits.len())]);
//!     Ok(())
//! }
//! ```

pub use st_zrt_sys as sys;
pub use sys::{
    AllocatorType, ElementType, ExecutionMode, ExecutionProviderDevicePolicy,
    GraphOptimizationLevel, LoggingLevel, MemType, OrtErrorCode, ProfilingEventCategory,
    SparseFormat, SparseIndicesFormat,
};

mod allocator;
mod arena;
mod compat;
#[cfg(feature = "cuda")]
mod cuda_rt;
#[cfg(feature = "custom-ops")]
mod custom_ops;
mod diagnostics;
mod element;
mod environment;
#[cfg(feature = "ep")]
mod ep;
#[cfg(feature = "model-editor")]
mod ep_authoring;
#[cfg(feature = "ep")]
mod ep_device;
mod error;
mod hardware;
mod initializer;
#[cfg(feature = "model-editor")]
mod interop;
mod io_binding;
mod lora;
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
pub mod shape_plan;
mod spsc;
mod tensor;
mod threading;
mod type_info;

pub use allocator::{
    Allocation, Allocator, AllocatorStats, AllocatorStatsDelta, KeyValuePairs, KeyValuePairsView,
};
pub use arena::{ArenaCfg, ArenaExtendStrategy};
#[cfg(feature = "ep")]
pub use compat::compatibility_for_ep_devices;
pub use compat::{
    CompiledModelCompatibility, compatibility_info_from_bytes, compatibility_info_from_path,
};
#[cfg(feature = "cuda")]
pub use cuda_rt::{
    CompletionEventRef, CudaCompletionPoller, CudaEvent, CudaStream, PinnedBuffer, device_count,
    memcpy_async_d2d, stream_wait_event,
};
#[cfg(feature = "custom-ops")]
pub use custom_ops::{
    CustomOp, CustomOpDomain, KernelContext, KernelInfo, Logger, Op, OpAttr, OpIoSpec,
    OwnedKernelInfo, ShapeInferContext,
};
pub use diagnostics::{
    available_providers, build_info, current_gpu_device_id, set_current_gpu_device_id,
};
// The generic trampolines the `#[macro_export] custom_op!` macro names via
// `$crate::__priv`. `#[doc(hidden)]` keeps them out of the public surface.
#[cfg(feature = "custom-ops")]
#[doc(hidden)]
pub use custom_ops::__priv;
pub use element::TensorElement;
pub use environment::{
    EnvCreationOptions, Environment, LanguageProjection, LogRecord, ThreadPoolCallbacksConfig,
};
#[cfg(feature = "ep")]
pub use ep::{
    CannOptions, CudaArenaExtendStrategy, CudaConfig, CudaCudnnConvAlgoSearch, CudaOptions,
    CudaProviderOptions, DeviceInputPolicy, DnnlOptions, EpProvider, MigraphxOptions,
    OpenvinoOptions, RocmOptions, TensorRtOptions,
};
#[cfg(feature = "model-editor")]
pub use ep_authoring::{
    EpAuthor, EpFactoryAuthor, EpFactoryInstance, EpGraphRef, EpGraphSupportInfoRef, EpInstance,
    KernelDef, KernelDefBuilder, OpSchema, OpSchemaTypeConstraint, OwnedEpDevice,
    OwnedHardwareDevice, ProfilingEvent, ep_factory_instance, ep_factory_vtable, ep_instance,
    ep_vtable,
};
#[cfg(feature = "ep")]
pub use ep_device::{EpAssignedNode, EpAssignedSubgraph, EpDevice, SyncStream, get_ep_devices};
pub use error::{Error, Result};
pub use hardware::{
    DeviceEpIncompatibilityDetails, HardwareDevice, hardware_device_ep_incompatibility_details,
    hardware_devices, num_hardware_devices,
};
pub use initializer::{ExternalInitializerInfo, OwnedInitializer};
#[cfg(feature = "model-editor")]
pub use interop::{
    ExternalMemoryDescriptor, ExternalMemoryHandle, ExternalMemoryHandleType,
    ExternalResourceImporter, ExternalSemaphoreDescriptor, ExternalSemaphoreHandle,
    ExternalSemaphoreType, ExternalTensorDescriptor, GraphicsApi, GraphicsInteropConfig,
    deinit_graphics_interop_for_ep_device, init_graphics_interop_for_ep_device,
};
pub use io_binding::{IoBinding, OutputValue};
pub use lora::LoraAdapter;
pub use memory::{
    DeviceMemoryType, MemoryClass, MemoryDeviceSnapshot, MemoryInfo, MemoryInfoDeviceType,
    MemoryInfoSnapshot,
};
pub use metadata::ModelMetadata;
#[cfg(feature = "model-editor")]
pub use model_editor::{
    Graph, Model, ModelCompilationOptions, Node, NodeAttr, TypeInfo, ValueInfo, compile_api,
    ep_api, interop_api, model_editor_api,
};
pub use prepacked::PrepackedWeightsContainer;
pub use run_options::{MaterializedRunOptions, RunOptions};
pub use runtime::{
    CompletionStatus, DynamicIoOptions, DynamicIoRuntime, InFlightRun, Lane, LaneEnqueueError,
    LaneHotPathAudit, OwnedDynamicIoRun, PreparedBucketId, Runtime, RuntimeMode, ServingLane,
    ServingLanePool, ServingLanePoolGuard, ShapeBucket, ShapeKey, ShapeSpec, StaticIoRuntime,
    TensorBufferAudit,
};
#[cfg(feature = "cuda")]
pub use runtime::{GpuChainEnqueueError, GpuChainedDynamicIoRun};
pub use session::{
    AllocatedOutputTensorIoLane, AllocatedTensorIoLane, DeviceOutputTensorIoLane,
    ExecutionProviderDeviceSnapshot, IoDirection, IoPlacement, LaneRunAllocatorStats,
    PreparedIoBinding, PreparedRun, RunFuture, Session, StaticTensorIoLane, TensorIoLane,
};
pub use session_options::{ArenaState, MemPatternState, SessionOptions};
pub use shape_plan::{
    CanonicalShape, ClassifyError, FallbackPolicy, OutputPolicy, ServingShapePlan,
    ServingShapePlanBuilder, ShapeId, ShapePlanError,
};
pub use spsc::{
    SendError as SpscSendError, SpscReceiver, SpscSender, TryRecvError as SpscTryRecvError,
    TrySendError as SpscTrySendError, bounded_spsc, bounded_spsc_with_spins,
};
pub use tensor::{
    AllocatedTensor, BufferSpec, BufferStorage, DeviceValue, MmapTensorOptions, OwnedValue,
    RunInput, SparseTensor, StringTensor, Tensor, TensorBuffer, TensorView,
};
pub use threading::{ThreadManager, ThreadingOptions};
pub use type_info::{
    MapTypeInfo, OptionalTypeInfo, RuntimeTypeInfo, SequenceTypeInfo, TensorTypeAndShapeInfo,
    TensorTypeAndShapeInfoView,
};

// ─── crate-private helpers shared across modules ─────────────────────────────
/// Cached reference to the live `Api` function-pointer table (a process global).
///
/// Resolved once via `sys::api()` and stored in a `OnceLock`. After initialization,
/// `api()` is a single atomic load — no `OrtGetApiBase`/`GetApi` C calls on the hot path.
static API: std::sync::OnceLock<&'static sys::Api> = std::sync::OnceLock::new();

/// Borrow the cached live `Api` function-pointer table.
#[inline]
pub(crate) fn api() -> &'static sys::Api {
    API.get_or_init(|| {
        // SAFETY: ORT documents its API table as process-global; st-zrt-sys links the shared
        // library for the process lifetime and does not expose an unload operation.
        unsafe { &*sys::api() }
    })
}

/// Turn a raw `OrtStatus*` into `Result<()>`: null ⇒ Ok; else Err (code+message),
/// with the status released. Uses the cached `Api` reference — no table re-discovery.
#[inline]
pub(crate) fn check(status: sys::StatusPtr) -> Result<()> {
    unsafe { sys::status_to_result(api(), status).map_err(Error::from) }
}

/// Copy a non-null C string into an owned UTF-8 `String`.
#[inline]
pub(crate) unsafe fn cstr_to_string(
    raw: *const std::ffi::c_char, what: &'static str,
) -> Result<String> {
    unsafe {
        std::ffi::CStr::from_ptr(raw)
            .to_str()
            .map(str::to_owned)
            .map_err(|_| Error::new(-1, format!("zrt: {what} is not valid UTF-8")))
    }
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
        Uint8 | Int8 | Bool | Float8E4M3FN | Float8E4M3FNUZ | Float8E5M2 | Float8E5M2FNUZ
        | Float8E8M0 => 1,
        Uint4 | Int4 | Uint2 | Int2 | Float4E2M1 => 0,
        Undefined | String => 0,
    }
}

pub(crate) fn packed_element_bits(e: sys::ElementType) -> Option<usize> {
    use sys::ElementType::*;
    match e {
        Uint4 | Int4 | Float4E2M1 => Some(4),
        Uint2 | Int2 => Some(2),
        _ => None,
    }
}

pub(crate) fn tensor_byte_len(elem_type: sys::ElementType, count: usize) -> Result<usize> {
    if let Some(bits) = packed_element_bits(elem_type) {
        return count
            .checked_mul(bits)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| Error::new(-1, "tensor byte length overflows usize"));
    }
    count
        .checked_mul(element_size(elem_type))
        .ok_or_else(|| Error::new(-1, "tensor byte length overflows usize"))
}

/// Test-only process-wide mutex serializing `Environment` creation in this crate's unit tests.
///
/// `CreateEnv` constructs ORT's default `LoggingManager`, and ORT enforces "only one instance of
/// LoggingManager created with InstanceType::Default can exist at any point in time". Cargo runs a
/// binary's tests on a thread pool, so concurrent default-`Environment` creations across test
/// modules race that singleton and fail intermittently with
/// `logging.cc:158 ... InstanceType::Default`. ORT builds that default-instance manager for both
/// `Environment::new` and `Environment::new_with_logger`, so every unit test that creates any
/// `Environment` acquires this lock for its whole body. (`release-check.sh` sets
/// `RUST_TEST_THREADS=1` for the same reason; this keeps plain `cargo test` reliable.)
#[cfg(test)]
pub(crate) static TEST_ENV_CREATION_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`TEST_ENV_CREATION_MUTEX`]; keep the guard for the test's whole body.
#[cfg(test)]
pub(crate) fn lock_default_env_creation() -> std::sync::MutexGuard<'static, ()> {
    TEST_ENV_CREATION_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::{element_size, sys::ElementType, tensor_byte_len};

    #[test]
    fn element_size_covers_quantized_and_float8_metadata_types() {
        assert_eq!(element_size(ElementType::Int8), 1);
        assert_eq!(element_size(ElementType::Uint8), 1);
        assert_eq!(element_size(ElementType::Float8E4M3FN), 1);
        assert_eq!(element_size(ElementType::Float8E4M3FNUZ), 1);
        assert_eq!(element_size(ElementType::Float8E5M2), 1);
        assert_eq!(element_size(ElementType::Float8E5M2FNUZ), 1);
        // FLOAT8E8M0 (ONNX 1.21 / ORT 1.27): 8-bit float, size 1.
        assert_eq!(element_size(ElementType::Float8E8M0), 1);

        // Packed sub-byte tensors are not exposed as typed logical-element slices.
        // Raw packed bytes are handled separately because one Rust scalar is not one
        // logical tensor element for these types.
        assert_eq!(element_size(ElementType::Int4), 0);
        assert_eq!(element_size(ElementType::Uint4), 0);
        assert_eq!(element_size(ElementType::Float4E2M1), 0);
        // Packed 2-bit (ONNX 1.21 / ORT 1.27): 4 values per byte.
        assert_eq!(element_size(ElementType::Uint2), 0);
        assert_eq!(element_size(ElementType::Int2), 0);
    }

    #[test]
    fn tensor_byte_len_covers_packed_sub_byte_types() {
        assert_eq!(tensor_byte_len(ElementType::Uint4, 0).unwrap(), 0);
        assert_eq!(tensor_byte_len(ElementType::Uint4, 1).unwrap(), 1);
        assert_eq!(tensor_byte_len(ElementType::Int4, 2).unwrap(), 1);
        assert_eq!(tensor_byte_len(ElementType::Float4E2M1, 3).unwrap(), 2);
        assert_eq!(tensor_byte_len(ElementType::Uint2, 4).unwrap(), 1);
        assert_eq!(tensor_byte_len(ElementType::Int2, 5).unwrap(), 2);
    }
}
