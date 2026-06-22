//! `Session` — a pre-marshaled inference session.
//!
//! Input/output names are resolved once at construction (anti-pattern O1 fix: no
//! per-run `FeedsFetchesManager` rebuild, no name marshaling on the hot path) and
//! `RunOptions` is reused (anti-pattern O4 fix). `run(&self)` is shared-reentrant —
//! ORT's `Run` is thread-safe on a session.
use crate::allocator::{Allocator, AllocatorStats, AllocatorStatsDelta};
use crate::element::TensorElement;
use crate::environment::{EnvInner, Environment};
use crate::initializer::OwnedInitializer;
use crate::io_binding::{IoBinding, OutputValue};
use crate::memory::MemoryInfo;
use crate::prepacked::{PrepackedWeightsContainer, PrepackedWeightsInner};
use crate::run_options::RunOptions;
use crate::session_options::SessionOptions;
use crate::tensor::{AllocatedTensor, OwnedValue, RunInput, TensorBuffer};
use crate::{Error, Result, api, check, sys};
use futures_util::task::AtomicWaker;
use std::cell::UnsafeCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

const STACK_IO_HANDLES: usize = 8;
const AUTO_ALIGNED_BUFFER_THRESHOLD_BYTES: usize = 1 << 20;
const AUTO_ALIGNED_BUFFER_ALIGNMENT: usize = 4096;
const AUTO_HUGEPAGE_BUFFER_THRESHOLD_BYTES: usize = 2 << 20;
const HUGEPAGE_BUFFER_ALIGNMENT: usize = 2 << 20;

/// Buffer allocation policy for tensor I/O lanes.
///
/// The default [`Self::Auto`] policy keeps tiny tensors on plain `Vec` storage and uses
/// 4096-byte aligned, prefaulted storage for tensors at or above 1 MiB. At or above 2 MiB it
/// additionally uses 2 MiB alignment and a best-effort hugepage hint before prefaulting. The
/// large-buffer policies avoid first-touch page faults and give CPU kernels aligned output
/// targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneBufferPolicy {
    /// Plain zeroed `Vec<T>` storage.
    Vec,
    /// Plain `Vec<T>` storage with pages touched during lane construction.
    Prefaulted,
    /// Explicitly aligned zeroed storage.
    Aligned { alignment: usize },
    /// Explicitly aligned storage with pages touched during lane construction.
    AlignedPrefaulted { alignment: usize },
    /// 2 MiB aligned storage with a best-effort hugepage hint.
    HugePage,
    /// 2 MiB aligned storage with a best-effort hugepage hint and prefaulting.
    HugePagePrefaulted,
    /// Explicitly aligned storage with a best-effort hugepage hint and prefaulting.
    AlignedHugePagePrefaulted { alignment: usize },
    /// Explicitly aligned storage locked in RAM with `mlock` where supported.
    AlignedMlocked { alignment: usize },
    /// Explicitly aligned storage prefaulted and locked in RAM with `mlock` where supported.
    AlignedMlockedPrefaulted { alignment: usize },
    /// 2 MiB aligned storage with a best-effort hugepage hint and `mlock` where supported.
    HugePageMlocked,
    /// 2 MiB aligned storage with a best-effort hugepage hint, prefaulting, and `mlock`.
    HugePageMlockedPrefaulted,
    /// Explicitly aligned storage with a hugepage hint, prefaulting, and `mlock`.
    AlignedHugePageMlockedPrefaulted { alignment: usize },
    /// `Vec` below 1 MiB, 4096-byte aligned + prefaulted at 1-2 MiB, and 2 MiB aligned +
    /// hugepage-hinted + prefaulted at or above 2 MiB.
    Auto,
}

impl Default for LaneBufferPolicy {
    #[inline]
    fn default() -> Self {
        Self::Auto
    }
}

/// Per-I/O cached type/shape from the model's STATIC type-info. Resolved once at
/// construction so the hot path needs no static metadata introspection. Carries the value kind
/// so sequence/map values do not fail session construction.
struct CachedIo {
    onnx_type: sys::OnnxType,
    elem_type: sys::ElementType,
    count: Option<usize>,
    dims: Vec<i64>,
    symbolic: Vec<Option<String>>,
}

pub struct Session {
    sess: *mut sys::SessionHandle,
    input_names: Vec<CString>,
    input_ptrs: Vec<*const c_char>,
    input_meta: Vec<CachedIo>,
    output_names: Vec<CString>,
    output_ptrs: Vec<*const c_char>,
    output_meta: Vec<CachedIo>,
    run_opts: RunOptions,
    /// Optional caller-owned initializers handed to ORT at session creation. Kept alive until
    /// after the ORT session is released.
    _owned_initializers: Vec<OwnedInitializer>,
    /// Optional prepacked-weight cache. Kept alive until after the ORT session is released.
    _prepacked_weights: Option<Arc<PrepackedWeightsInner>>,
    /// Keeps the Env alive for this Session's whole lifetime — an `Arc` ref cloned from the
    /// `Environment` passed to [`Self::new`]/[`Self::from_bytes`]. ORT sessions reference the
    /// Env's thread pools/allocator, so this prevents the use-after-free that releasing the
    /// Env first would cause (RESULTS.md §8). Declared last: drops after `sess`, so the ORT
    /// session is released before the Env's final ref can go away. Never read directly — its
    /// purpose is purely to hold the `Arc` ref (drop-guard); `_`-prefixed for that reason.
    _env: Arc<EnvInner>,
}

/// A prepared regular `Run` invocation: input value handles and output slots are allocated
/// once, then reused across calls. ORT still owns the output tensors; use
/// [`PreparedIoBinding`] when caller-owned output buffers are desired.
pub struct PreparedRun<'s, 'i> {
    session: &'s Session,
    input_handles: Vec<*const sys::ValueHandle>,
    output_handles: Vec<*mut sys::ValueHandle>,
    outputs: Vec<Option<OwnedValue>>,
    _inputs: PhantomData<&'i dyn RunInput>,
}

/// A bind-once, run-many IoBinding wrapper tied to the lifetimes of the bound input and
/// output values. This is the ergonomic zero-copy output path: callers allocate buffers,
/// wrap them in [`OutputValue`], prepare the binding once, then call [`Self::run`].
pub struct PreparedIoBinding<'s, 'v> {
    session: &'s Session,
    binding: IoBinding,
    _values: PhantomData<&'v ()>,
}

/// A borrowed-session, bind-once tensor I/O lane.
///
/// Each lane owns stable input and output buffers plus one IoBinding. Mutate inputs,
/// call [`Self::run`], then read outputs. No per-run allocation, copy, or name binding is
/// performed by ZRT. Use [`crate::Runtime`] when you need an owned static lane set.
pub struct TensorIoLane<'s, T: TensorElement> {
    session: &'s Session,
    // Drop before the tensor buffers whose value handles it references.
    binding: IoBinding,
    inputs: Vec<TensorBuffer<T>>,
    outputs: Vec<TensorBuffer<T>>,
}

/// A borrowed-session lane with caller-owned inputs and ORT-allocator-owned outputs.
///
/// This mirrors `BindOutputToDevice` style benchmarking while still binding concrete output
/// tensors once. It is useful when comparing against wrapper APIs that let ORT pick output
/// memory placement/alignment, or when caller-owned output buffers are not desired.
pub struct AllocatedOutputTensorIoLane<'s, T: TensorElement> {
    session: &'s Session,
    // Drop before the tensor buffers whose value handles it references.
    binding: IoBinding,
    inputs: Vec<TensorBuffer<T>>,
    outputs: Vec<AllocatedTensor<T>>,
}

/// A borrowed-session lane with caller-owned inputs and outputs bound to a memory/device target.
///
/// This uses ORT `BindOutputToDevice`, then retrieves the bound output values after each run.
/// It is useful for dynamic-shape outputs and for matching wrapper APIs that bind output by
/// memory location rather than by a pre-created concrete tensor.
pub struct DeviceOutputTensorIoLane<'s, T: TensorElement> {
    session: &'s Session,
    // Drop before the tensor buffers whose value handles it references.
    binding: IoBinding,
    inputs: Vec<TensorBuffer<T>>,
    outputs: Vec<OwnedValue>,
}

/// A borrowed-session lane whose inputs and outputs are both allocated by ORT.
///
/// This is the closest CPU comparison to wrapper APIs that keep tensors in ORT allocator
/// memory and mutate them in place between runs.
pub struct AllocatedTensorIoLane<'s, T: TensorElement> {
    session: &'s Session,
    // Drop before the tensor buffers whose value handles it references.
    binding: IoBinding,
    inputs: Vec<AllocatedTensor<T>>,
    outputs: Vec<AllocatedTensor<T>>,
}

/// A borrowed-session tensor I/O lane with compile-time input/output counts.
///
/// This is the fixed-arity sibling of [`TensorIoLane`]. It still prepares all buffers and
/// bindings once, but stores them as arrays so hot services can use a concrete lane type.
pub struct StaticTensorIoLane<'s, T: TensorElement, const INPUTS: usize, const OUTPUTS: usize> {
    session: &'s Session,
    // Drop before the tensor buffers whose value handles it references.
    binding: IoBinding,
    inputs: [TensorBuffer<T>; INPUTS],
    outputs: [TensorBuffer<T>; OUTPUTS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneRunAllocatorStats {
    pub before: AllocatorStats,
    pub after: AllocatorStats,
}

impl LaneRunAllocatorStats {
    /// Numeric allocator-counter deltas between the before/after snapshots.
    #[inline]
    pub fn delta(&self) -> AllocatorStatsDelta {
        self.before.diff(&self.after)
    }
}

pub(crate) fn lane_tensor_buffer<T>(
    shape: &[i64], mem: &MemoryInfo, policy: LaneBufferPolicy,
) -> Result<TensorBuffer<T>>
where
    T: TensorElement + Clone + Default,
{
    let bytes = lane_shape_bytes::<T>(shape)?;
    match resolve_lane_buffer_policy(policy, bytes) {
        LaneBufferPolicy::Vec => TensorBuffer::zeros(shape, mem),
        LaneBufferPolicy::Prefaulted => TensorBuffer::zeros_prefaulted(shape, mem),
        LaneBufferPolicy::Aligned { alignment } => {
            TensorBuffer::zeros_aligned(shape, alignment, mem)
        },
        LaneBufferPolicy::AlignedPrefaulted { alignment } => {
            TensorBuffer::zeros_aligned_prefaulted(shape, alignment, mem)
        },
        LaneBufferPolicy::HugePage => {
            TensorBuffer::zeros_aligned_hugepage(shape, HUGEPAGE_BUFFER_ALIGNMENT, mem)
        },
        LaneBufferPolicy::HugePagePrefaulted => {
            TensorBuffer::zeros_aligned_hugepage_prefaulted(shape, HUGEPAGE_BUFFER_ALIGNMENT, mem)
        },
        LaneBufferPolicy::AlignedHugePagePrefaulted { alignment } => {
            TensorBuffer::zeros_aligned_hugepage_prefaulted(shape, alignment, mem)
        },
        LaneBufferPolicy::AlignedMlocked { alignment } => {
            TensorBuffer::zeros_aligned_mlocked(shape, alignment, mem)
        },
        LaneBufferPolicy::AlignedMlockedPrefaulted { alignment } => {
            TensorBuffer::zeros_aligned_mlocked_prefaulted(shape, alignment, mem)
        },
        LaneBufferPolicy::HugePageMlocked => {
            TensorBuffer::zeros_aligned_hugepage_mlocked(shape, HUGEPAGE_BUFFER_ALIGNMENT, mem)
        },
        LaneBufferPolicy::HugePageMlockedPrefaulted => {
            TensorBuffer::zeros_aligned_hugepage_mlocked_prefaulted(
                shape,
                HUGEPAGE_BUFFER_ALIGNMENT,
                mem,
            )
        },
        LaneBufferPolicy::AlignedHugePageMlockedPrefaulted { alignment } => {
            TensorBuffer::zeros_aligned_hugepage_mlocked_prefaulted(shape, alignment, mem)
        },
        LaneBufferPolicy::Auto => unreachable!("auto lane buffer policy must resolve first"),
    }
}

fn resolve_lane_buffer_policy(policy: LaneBufferPolicy, bytes: usize) -> LaneBufferPolicy {
    match policy {
        LaneBufferPolicy::Auto if bytes >= AUTO_HUGEPAGE_BUFFER_THRESHOLD_BYTES => {
            LaneBufferPolicy::HugePagePrefaulted
        },
        LaneBufferPolicy::Auto if bytes >= AUTO_ALIGNED_BUFFER_THRESHOLD_BYTES => {
            LaneBufferPolicy::AlignedPrefaulted {
                alignment: AUTO_ALIGNED_BUFFER_ALIGNMENT,
            }
        },
        LaneBufferPolicy::Auto => LaneBufferPolicy::Vec,
        other => other,
    }
}

fn lane_shape_bytes<T: TensorElement>(shape: &[i64]) -> Result<usize> {
    let mut count = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(Error::new(
                -1,
                format!("zrt: lane buffers require concrete shapes, got {shape:?}"),
            ));
        }
        count = count
            .checked_mul(dim as usize)
            .ok_or_else(|| Error::new(-1, "zrt: lane buffer element count overflows usize"))?;
    }
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::new(-1, "zrt: lane buffer byte size overflows usize"))
}

impl Session {
    /// Load `model_path` (filesystem path, UTF-8) and pre-marshal its I/O names and
    /// output type/shape (cached so the hot path needs no introspection).
    pub fn new(env: &Environment, model_path: &str, opts: SessionOptions) -> Result<Self> {
        let cpath = CString::new(model_path)
            .map_err(|_| crate::Error::new(-1, "model path contains a NUL"))?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let mut sess: *mut sys::SessionHandle = ptr::null_mut();
        let create = check(unsafe {
            api().create_session()(
                env.as_ptr(),
                cpath.as_ptr(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut sess,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        create?;
        Self::from_handle(sess, env.share())
    }

    /// Load a model from an in-memory byte buffer (`CreateSessionFromArray`, idx 8) — no
    /// temp file, no filesystem. `model_data` is a serialized ONNX model (e.g. read from
    /// disk, embedded via `include_bytes!`, or received over the network).
    pub fn from_bytes(env: &Environment, model_data: &[u8], opts: SessionOptions) -> Result<Self> {
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let mut sess: *mut sys::SessionHandle = ptr::null_mut();
        let create = check(unsafe {
            api().create_session_from_array()(
                env.as_ptr(),
                model_data.as_ptr() as *const c_void,
                model_data.len(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut sess,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        create?;
        Self::from_handle(sess, env.share())
    }

    /// Load `model_path` using a shared ORT prepacked-weight container.
    ///
    /// Use the same container across compatible sessions to let ORT reuse prepacked weights.
    pub fn new_with_prepacked_weights(
        env: &Environment, model_path: &str, opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer,
    ) -> Result<Self> {
        Self::new_with_prepacked_weights_and_owned_initializers(
            env,
            model_path,
            opts,
            prepacked,
            Vec::new(),
        )
    }

    /// Load `model_path` while replacing model initializers with ZRT-owned external tensors.
    ///
    /// The provided [`OwnedInitializer`] values are moved into the returned session, so their
    /// backing memory remains valid for the whole ORT session lifetime.
    pub fn new_with_owned_initializers(
        env: &Environment, model_path: &str, opts: SessionOptions,
        initializers: Vec<OwnedInitializer>,
    ) -> Result<Self> {
        let cpath = CString::new(model_path)
            .map_err(|_| crate::Error::new(-1, "model path contains a NUL"))?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_owned_initializers(opts_handle, &initializers)?;
            let mut sess: *mut sys::SessionHandle = ptr::null_mut();
            check(unsafe {
                api().create_session()(
                    env.as_ptr(),
                    cpath.as_ptr(),
                    opts_handle as *const sys::SessionOptionsHandle,
                    &mut sess,
                )
            })?;
            Ok(sess)
        })();
        unsafe { api().release_session_options()(opts_handle) };
        let sess = create?;
        Self::from_handle_with_resources(sess, env.share(), initializers, None)
    }

    /// Load `model_path` using both a shared prepacked-weight container and owned external
    /// initializer tensors.
    pub fn new_with_prepacked_weights_and_owned_initializers(
        env: &Environment, model_path: &str, opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer, initializers: Vec<OwnedInitializer>,
    ) -> Result<Self> {
        let cpath = CString::new(model_path)
            .map_err(|_| crate::Error::new(-1, "model path contains a NUL"))?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_owned_initializers(opts_handle, &initializers)?;
            let mut sess: *mut sys::SessionHandle = ptr::null_mut();
            check(unsafe {
                api().create_session_with_prepacked_weights_container()(
                    env.as_ptr(),
                    cpath.as_ptr(),
                    opts_handle as *const sys::SessionOptionsHandle,
                    prepacked.as_mut_ptr(),
                    &mut sess,
                )
            })?;
            Ok(sess)
        })();
        unsafe { api().release_session_options()(opts_handle) };
        let sess = create?;
        Self::from_handle_with_resources(sess, env.share(), initializers, Some(prepacked.share()))
    }

    /// Load model bytes using a shared ORT prepacked-weight container.
    pub fn from_bytes_with_prepacked_weights(
        env: &Environment, model_data: &[u8], opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer,
    ) -> Result<Self> {
        Self::from_bytes_with_prepacked_weights_and_owned_initializers(
            env,
            model_data,
            opts,
            prepacked,
            Vec::new(),
        )
    }

    /// Load model bytes while replacing model initializers with ZRT-owned external tensors.
    pub fn from_bytes_with_owned_initializers(
        env: &Environment, model_data: &[u8], opts: SessionOptions,
        initializers: Vec<OwnedInitializer>,
    ) -> Result<Self> {
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_owned_initializers(opts_handle, &initializers)?;
            let mut sess: *mut sys::SessionHandle = ptr::null_mut();
            check(unsafe {
                api().create_session_from_array()(
                    env.as_ptr(),
                    model_data.as_ptr() as *const c_void,
                    model_data.len(),
                    opts_handle as *const sys::SessionOptionsHandle,
                    &mut sess,
                )
            })?;
            Ok(sess)
        })();
        unsafe { api().release_session_options()(opts_handle) };
        let sess = create?;
        Self::from_handle_with_resources(sess, env.share(), initializers, None)
    }

    /// Load model bytes using both a shared prepacked-weight container and owned external
    /// initializer tensors.
    pub fn from_bytes_with_prepacked_weights_and_owned_initializers(
        env: &Environment, model_data: &[u8], opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer, initializers: Vec<OwnedInitializer>,
    ) -> Result<Self> {
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_owned_initializers(opts_handle, &initializers)?;
            let mut sess: *mut sys::SessionHandle = ptr::null_mut();
            check(unsafe {
                api().create_session_from_array_with_prepacked_weights_container()(
                    env.as_ptr(),
                    model_data.as_ptr() as *const c_void,
                    model_data.len(),
                    opts_handle as *const sys::SessionOptionsHandle,
                    prepacked.as_mut_ptr(),
                    &mut sess,
                )
            })?;
            Ok(sess)
        })();
        unsafe { api().release_session_options()(opts_handle) };
        let sess = create?;
        Self::from_handle_with_resources(sess, env.share(), initializers, Some(prepacked.share()))
    }

    /// Finish construction from a freshly-created session handle: pre-marshal I/O names and
    /// cache output type/shape, then build the struct. Shared by [`Self::new`] and
    /// [`Self::from_bytes`].
    fn from_handle(sess: *mut sys::SessionHandle, env: Arc<EnvInner>) -> Result<Self> {
        Self::from_handle_with_resources(sess, env, Vec::new(), None)
    }

    fn from_handle_with_resources(
        sess: *mut sys::SessionHandle, env: Arc<EnvInner>,
        owned_initializers: Vec<OwnedInitializer>,
        prepacked_weights: Option<Arc<PrepackedWeightsInner>>,
    ) -> Result<Self> {
        let sess = crate::ensure_non_null(sess, "session")?;
        let result = (|| {
            let alloc = Allocator::get_default()?;
            let (input_names, input_ptrs) = collect_io_names(sess, true, &alloc)?;
            let (output_names, output_ptrs) = collect_io_names(sess, false, &alloc)?;
            let input_meta = collect_io_meta(sess, true, input_ptrs.len())?;
            let output_meta = collect_io_meta(sess, false, output_ptrs.len())?;
            Ok(Self {
                sess,
                input_names,
                input_ptrs,
                input_meta,
                output_names,
                output_ptrs,
                output_meta,
                run_opts: RunOptions::new()?,
                _owned_initializers: owned_initializers,
                _prepacked_weights: prepacked_weights,
                _env: env,
            })
        })();
        if result.is_err() {
            unsafe { api().release_session()(sess) };
        }
        result
    }

    #[cfg(feature = "model-editor")]
    fn refresh_io_metadata(&mut self) -> Result<()> {
        let alloc = Allocator::get_default()?;
        let (input_names, input_ptrs) = collect_io_names(self.sess, true, &alloc)?;
        let (output_names, output_ptrs) = collect_io_names(self.sess, false, &alloc)?;
        let input_meta = collect_io_meta(self.sess, true, input_ptrs.len())?;
        let output_meta = collect_io_meta(self.sess, false, output_ptrs.len())?;

        self.input_names = input_names;
        self.input_ptrs = input_ptrs;
        self.input_meta = input_meta;
        self.output_names = output_names;
        self.output_ptrs = output_ptrs;
        self.output_meta = output_meta;
        Ok(())
    }

    /// The model's metadata (producer, graph name/description, domain, version, custom
    /// metadata map). Owning handle (`SessionGetModelMetadata`, idx 111); released on drop.
    pub fn metadata(&self) -> Result<crate::metadata::ModelMetadata> {
        let mut meta: *mut sys::ModelMetadataHandle = ptr::null_mut();
        check(unsafe {
            api().session_get_model_metadata()(self.sess as *const sys::SessionHandle, &mut meta)
        })?;
        let meta = crate::ensure_non_null(meta, "model metadata")?;
        Ok(unsafe { crate::metadata::ModelMetadata::from_owning(meta) })
    }

    /// Profiling start timestamp in nanoseconds as reported by ORT.
    pub fn profiling_start_time_ns(&self) -> Result<u64> {
        let mut out = 0u64;
        check(unsafe {
            api().session_get_profiling_start_time_ns()(
                self.sess as *const sys::SessionHandle,
                &mut out,
            )
        })?;
        Ok(out)
    }

    /// End ORT session profiling, flush the trace, and return the generated profile file path.
    ///
    /// Profiling must have been enabled with [`SessionOptions::enable_profiling`] before session
    /// creation. ORT allocates the returned path with the supplied allocator; ZRT copies it into
    /// a Rust `String` and frees the engine buffer before returning.
    pub fn end_profiling(&self) -> Result<String> {
        let alloc = Allocator::get_default()?;
        let mut raw: *mut c_char = ptr::null_mut();
        check(unsafe { api().session_end_profiling()(self.sess, alloc.alloc, &mut raw) })?;
        if raw.is_null() {
            return Err(Error::new(-1, "zrt: ORT returned null profiling path"));
        }
        let path = unsafe { crate::cstr_to_string(raw, "profiling path") };
        let free = unsafe { alloc.free(raw as *mut c_void) };
        match (path, free) {
            (Ok(path), Ok(())) => Ok(path),
            (Err(err), _) => Err(err),
            (_, Err(err)) => Err(err),
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut sys::SessionHandle {
        self.sess
    }

    #[inline]
    pub fn input_count(&self) -> usize {
        self.input_ptrs.len()
    }
    #[inline]
    pub fn output_count(&self) -> usize {
        self.output_ptrs.len()
    }
    pub fn input_name(&self, i: usize) -> Result<&str> {
        self.input_names
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: input index {i} out of range ({} inputs)",
                        self.input_count()
                    ),
                )
            })?
            .to_str()
            .map_err(|_| Error::new(-1, format!("zrt: input name {i} is not valid UTF-8")))
    }
    pub fn output_name(&self, i: usize) -> Result<&str> {
        self.output_names
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: output index {i} out of range ({} outputs)",
                        self.output_count()
                    ),
                )
            })?
            .to_str()
            .map_err(|_| Error::new(-1, format!("zrt: output name {i} is not valid UTF-8")))
    }
    /// Cached (value kind, element type, static element count if concrete) for input `i`.
    #[inline]
    pub fn input_meta(&self, i: usize) -> Result<(sys::OnnxType, sys::ElementType, Option<usize>)> {
        let m = self.input_meta.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!(
                    "zrt: input index {i} out of range ({} inputs)",
                    self.input_count()
                ),
            )
        })?;
        Ok((m.onnx_type, m.elem_type, m.count))
    }
    /// Cached (value kind, element type, static element count if concrete) for output `i`.
    #[inline]
    pub fn output_meta(
        &self, i: usize,
    ) -> Result<(sys::OnnxType, sys::ElementType, Option<usize>)> {
        let m = self.output_meta.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!(
                    "zrt: output index {i} out of range ({} outputs)",
                    self.output_count()
                ),
            )
        })?;
        Ok((m.onnx_type, m.elem_type, m.count))
    }
    /// Cached concrete dimensions of input `i`.
    #[inline]
    pub fn input_shape(&self, i: usize) -> Result<&[i64]> {
        Ok(&self
            .input_meta
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: input index {i} out of range ({} inputs)",
                        self.input_count()
                    ),
                )
            })?
            .dims)
    }
    /// Cached concrete dimensions of output `i` (empty for non-tensor outputs).
    #[inline]
    pub fn output_shape(&self, i: usize) -> Result<&[i64]> {
        Ok(&self
            .output_meta
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: output index {i} out of range ({} outputs)",
                        self.output_count()
                    ),
                )
            })?
            .dims)
    }
    /// Cached symbolic (named) dimensions of input `i`.
    #[inline]
    pub fn input_symbolic_dims(&self, i: usize) -> Result<&[Option<String>]> {
        Ok(&self
            .input_meta
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: input index {i} out of range ({} inputs)",
                        self.input_count()
                    ),
                )
            })?
            .symbolic)
    }
    /// Cached symbolic (named) dimensions of output `i`: `Some("batch")` where the model
    /// declared a symbolic dim, `None` where it is concrete. Empty for non-tensor outputs.
    #[inline]
    pub fn output_symbolic_dims(&self, i: usize) -> Result<&[Option<String>]> {
        Ok(&self
            .output_meta
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!(
                        "zrt: output index {i} out of range ({} outputs)",
                        self.output_count()
                    ),
                )
            })?
            .symbolic)
    }

    /// Run inference with the session's default (reused) `RunOptions`. `inputs` must be in
    /// session-input order (any mix of numeric [`crate::TensorView`] and [`crate::StringTensor`]);
    /// `outputs` receives one engine-owned value per session output. `run(&self)` is
    /// thread-safe; each call uses a transient output-handle array — the per-run cost we
    /// eliminate is MB-scale tensor allocation, not this handful of pointers.
    pub fn run(&self, inputs: &[&dyn RunInput], outputs: &mut [Option<OwnedValue>]) -> Result<()> {
        self.run_impl(inputs, outputs, self.run_opts.as_ptr())
    }

    /// Prepare a regular `Run` path for repeated calls with the same input value handles.
    /// This removes the hot-path handle-array and output-slot allocations from callers that
    /// cannot bind caller-owned outputs.
    pub fn prepare_run<'s, 'i>(
        &'s self, inputs: &[&'i dyn RunInput],
    ) -> Result<PreparedRun<'s, 'i>> {
        self.check_input_count(inputs.len())?;
        Ok(PreparedRun {
            session: self,
            input_handles: inputs.iter().map(|v| v.as_value_ptr()).collect(),
            output_handles: vec![ptr::null_mut(); self.output_count()],
            outputs: (0..self.output_count()).map(|_| None).collect(),
            _inputs: PhantomData,
        })
    }

    /// Prepare an IoBinding by session I/O order. Inputs are bound to session input names
    /// and caller-owned outputs are bound to session output names once, then reused.
    pub fn prepare_io_binding<'s, 'v>(
        &'s self, inputs: &[&'v dyn RunInput], outputs: &[&'v OutputValue<'_>],
    ) -> Result<PreparedIoBinding<'s, 'v>> {
        self.check_input_count(inputs.len())?;
        self.check_output_count(outputs.len(), "output count")?;
        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, *input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output(self.output_name(i)?, output)?;
        }
        Ok(PreparedIoBinding {
            session: self,
            binding,
            _values: PhantomData,
        })
    }

    /// Prepare an IoBinding from reusable output [`TensorBuffer`]s rather than borrowed
    /// [`OutputValue`]s.
    pub fn prepare_io_binding_buffers<'s, 'v, T: TensorElement>(
        &'s self, inputs: &[&'v dyn RunInput], outputs: &[&'v TensorBuffer<T>],
    ) -> Result<PreparedIoBinding<'s, 'v>> {
        self.check_input_count(inputs.len())?;
        self.check_output_count(outputs.len(), "output count")?;
        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, *input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(self.output_name(i)?, output)?;
        }
        Ok(PreparedIoBinding {
            session: self,
            binding,
            _values: PhantomData,
        })
    }

    /// Build one borrowed-session lane with owned reusable tensor buffers.
    ///
    /// `input_shapes` and `output_shapes` are in session I/O order. This helper is for
    /// static-shape numeric models where all bound tensors share element type `T`.
    pub fn prepare_tensor_io_lane<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
    ) -> Result<TensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.prepare_tensor_io_lane_with_buffer_policy(
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one borrowed-session lane with an explicit caller-owned buffer policy.
    pub fn prepare_tensor_io_lane_with_buffer_policy<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: LaneBufferPolicy,
    ) -> Result<TensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.check_input_count(input_shapes.len())?;
        self.check_output_count(output_shapes.len(), "output shape count")?;

        let inputs: Vec<TensorBuffer<T>> = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<_>>()?;
        let outputs: Vec<TensorBuffer<T>> = output_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<_>>()?;

        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(self.output_name(i)?, output)?;
        }

        Ok(TensorIoLane {
            session: self,
            binding,
            inputs,
            outputs,
        })
    }

    /// Build one borrowed-session lane with caller-owned inputs and ORT-allocated outputs.
    ///
    /// Inputs use [`LaneBufferPolicy::Auto`]. Outputs are allocated as concrete ORT tensors
    /// and bound once, so ORT controls output allocation/alignment while the lane still has
    /// stable output handles across runs.
    pub fn prepare_allocated_output_tensor_io_lane<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]],
    ) -> Result<AllocatedOutputTensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.prepare_allocated_output_tensor_io_lane_with_buffer_policy(
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one ORT-allocated-output lane with an explicit caller-owned input policy.
    pub fn prepare_allocated_output_tensor_io_lane_with_buffer_policy<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], input_policy: LaneBufferPolicy,
    ) -> Result<AllocatedOutputTensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.check_input_count(input_shapes.len())?;
        self.check_output_count(output_shapes.len(), "output shape count")?;

        let inputs: Vec<TensorBuffer<T>> = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, input_mem, input_policy))
            .collect::<Result<_>>()?;
        let outputs: Vec<AllocatedTensor<T>> = output_shapes
            .iter()
            .map(|shape| AllocatedTensor::for_session(self, output_mem, shape))
            .collect::<Result<_>>()?;

        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_allocated(self.output_name(i)?, output)?;
        }

        Ok(AllocatedOutputTensorIoLane {
            session: self,
            binding,
            inputs,
            outputs,
        })
    }

    /// Build one borrowed-session lane with caller-owned inputs and outputs bound to a
    /// memory/device target via ORT `BindOutputToDevice`.
    pub fn prepare_device_output_tensor_io_lane<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
    ) -> Result<DeviceOutputTensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.prepare_device_output_tensor_io_lane_with_buffer_policy(
            input_mem,
            output_mem,
            input_shapes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one device-output lane with an explicit caller-owned input policy.
    pub fn prepare_device_output_tensor_io_lane_with_buffer_policy<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        input_policy: LaneBufferPolicy,
    ) -> Result<DeviceOutputTensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.check_input_count(input_shapes.len())?;

        let inputs: Vec<TensorBuffer<T>> = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, input_mem, input_policy))
            .collect::<Result<_>>()?;

        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, input)?;
        }
        for i in 0..self.output_count() {
            binding.bind_output_device(self.output_name(i)?, output_mem)?;
        }

        Ok(DeviceOutputTensorIoLane {
            session: self,
            binding,
            inputs,
            outputs: Vec::new(),
        })
    }

    /// Build one borrowed-session lane whose inputs and outputs are both ORT-allocated.
    ///
    /// Callers mutate inputs through [`AllocatedTensorIoLane::input_mut`] and read outputs
    /// through [`AllocatedTensorIoLane::output`]. This gives ORT control over both input and
    /// output allocation/alignment while preserving bind-once lane reuse.
    pub fn prepare_allocated_tensor_io_lane<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]],
    ) -> Result<AllocatedTensorIoLane<'_, T>>
    where
        T: TensorElement + Clone + Default,
    {
        self.check_input_count(input_shapes.len())?;
        self.check_output_count(output_shapes.len(), "output shape count")?;

        let inputs: Vec<AllocatedTensor<T>> = input_shapes
            .iter()
            .map(|shape| AllocatedTensor::for_session(self, input_mem, shape))
            .collect::<Result<_>>()?;
        let outputs: Vec<AllocatedTensor<T>> = output_shapes
            .iter()
            .map(|shape| AllocatedTensor::for_session(self, output_mem, shape))
            .collect::<Result<_>>()?;

        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_allocated(self.output_name(i)?, output)?;
        }

        Ok(AllocatedTensorIoLane {
            session: self,
            binding,
            inputs,
            outputs,
        })
    }

    /// Build one fixed-arity borrowed-session lane with owned reusable tensor buffers.
    ///
    /// `INPUTS` and `OUTPUTS` must match the model I/O counts. This keeps setup fallible but
    /// gives the prepared lane array-backed storage and array accessors.
    pub fn prepare_static_tensor_io_lane<T, const INPUTS: usize, const OUTPUTS: usize>(
        &self, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<StaticTensorIoLane<'_, T, INPUTS, OUTPUTS>>
    where
        T: TensorElement + Clone + Default,
    {
        self.prepare_static_tensor_io_lane_with_buffer_policy(
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one fixed-arity borrowed-session lane with an explicit caller-owned buffer policy.
    pub fn prepare_static_tensor_io_lane_with_buffer_policy<
        T,
        const INPUTS: usize,
        const OUTPUTS: usize,
    >(
        &self, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        policy: LaneBufferPolicy,
    ) -> Result<StaticTensorIoLane<'_, T, INPUTS, OUTPUTS>>
    where
        T: TensorElement + Clone + Default,
    {
        self.check_input_count(INPUTS)?;
        self.check_output_count(OUTPUTS, "output shape count")?;

        let inputs: [TensorBuffer<T>; INPUTS] = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::new(-1, "zrt: failed to build fixed input buffer array"))?;
        let outputs: [TensorBuffer<T>; OUTPUTS] = output_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::new(-1, "zrt: failed to build fixed output buffer array"))?;

        let mut binding = IoBinding::new(self)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(self.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(self.output_name(i)?, output)?;
        }

        Ok(StaticTensorIoLane {
            session: self,
            binding,
            inputs,
            outputs,
        })
    }

    /// Build a fixed set of independent borrowed-session lanes.
    ///
    /// This returns plain borrowed-session lanes and intentionally does not schedule or lock.
    /// For an owned static lane set, use [`crate::Runtime`].
    pub fn prepare_tensor_io_lanes<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize,
    ) -> Result<Vec<TensorIoLane<'_, T>>>
    where
        T: TensorElement + Clone + Default,
    {
        self.prepare_tensor_io_lanes_with_buffer_policy(
            mem,
            input_shapes,
            output_shapes,
            lanes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a fixed set of independent borrowed-session lanes with an explicit buffer policy.
    pub fn prepare_tensor_io_lanes_with_buffer_policy<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize,
        policy: LaneBufferPolicy,
    ) -> Result<Vec<TensorIoLane<'_, T>>>
    where
        T: TensorElement + Clone + Default,
    {
        (0..lanes)
            .map(|_| {
                self.prepare_tensor_io_lane_with_buffer_policy(
                    mem,
                    input_shapes,
                    output_shapes,
                    policy,
                )
            })
            .collect()
    }

    /// Run inference with a caller-provided [`RunOptions`] — per-call log level/tag/config
    /// entries, or to cancel via [`RunOptions::terminate`] (share it as `Arc<RunOptions>`
    /// with the cancelling thread). Otherwise identical to [`Self::run`].
    pub fn run_with(
        &self, inputs: &[&dyn RunInput], outputs: &mut [Option<OwnedValue>], opts: &RunOptions,
    ) -> Result<()> {
        self.run_impl(inputs, outputs, opts.as_ptr())
    }

    fn run_impl(
        &self, inputs: &[&dyn RunInput], outputs: &mut [Option<OwnedValue>],
        opts: *const sys::RunOptionsHandle,
    ) -> Result<()> {
        self.check_input_count(inputs.len())?;
        self.check_output_count(outputs.len(), "output slot count")?;

        if inputs.len() <= STACK_IO_HANDLES && outputs.len() <= STACK_IO_HANDLES {
            let mut in_handles = [ptr::null(); STACK_IO_HANDLES];
            for (dst, input) in in_handles.iter_mut().zip(inputs.iter()) {
                *dst = input.as_value_ptr();
            }
            let mut out_handles = [ptr::null_mut(); STACK_IO_HANDLES];
            self.run_raw(
                &in_handles[..inputs.len()],
                &mut out_handles[..outputs.len()],
                opts,
            )?;
            self.stamp_outputs(&out_handles[..outputs.len()], outputs)?;
        } else {
            let in_handles: Vec<*const sys::ValueHandle> =
                inputs.iter().map(|v| v.as_value_ptr()).collect();
            let mut out_handles: Vec<*mut sys::ValueHandle> =
                vec![ptr::null_mut(); self.output_count()];
            self.run_raw(&in_handles, &mut out_handles, opts)?;
            self.stamp_outputs(&out_handles, outputs)?;
        }
        Ok(())
    }

    fn run_raw(
        &self, input_handles: &[*const sys::ValueHandle],
        output_handles: &mut [*mut sys::ValueHandle], opts: *const sys::RunOptionsHandle,
    ) -> Result<()> {
        check(unsafe {
            api().run()(
                self.sess,
                opts,
                self.input_ptrs.as_ptr(),
                input_handles.as_ptr(),
                input_handles.len(),
                self.output_ptrs.as_ptr(),
                self.output_ptrs.len(),
                output_handles.as_mut_ptr(),
            )
        })
    }

    fn check_input_count(&self, got: usize) -> Result<()> {
        let expected = self.input_count();
        if got != expected {
            return Err(crate::Error::new(
                -1,
                format!("zrt: input count mismatch: expected {expected}, got {got}"),
            ));
        }
        Ok(())
    }

    fn check_output_count(&self, got: usize, what: &str) -> Result<()> {
        let expected = self.output_count();
        if got != expected {
            return Err(crate::Error::new(
                -1,
                format!("zrt: {what} mismatch: expected {expected}, got {got}"),
            ));
        }
        Ok(())
    }

    fn stamp_outputs(
        &self, handles: &[*mut sys::ValueHandle], outputs: &mut [Option<OwnedValue>],
    ) -> Result<()> {
        for i in 0..handles.len() {
            let h = handles[i];
            let m = &self.output_meta[i];
            let count = match m.count {
                Some(count) => count,
                None if m.onnx_type == sys::OnnxType::Tensor => {
                    match crate::type_info::tensor_type_and_shape(h as *const sys::ValueHandle)
                        .and_then(|shape| shape.element_count())
                    {
                        Ok(count) => count,
                        Err(err) => {
                            for &handle in &handles[i..] {
                                if !handle.is_null() {
                                    unsafe { api().release_value()(handle) };
                                }
                            }
                            return Err(err);
                        },
                    }
                },
                None => 0,
            };
            outputs[i] = Some(OwnedValue {
                value: h,
                onnx_type: m.onnx_type,
                elem_type: m.elem_type,
                count,
            });
        }
        Ok(())
    }

    /// Run with an [`crate::IoBinding`]. Inputs/outputs are taken from the binding (bound by
    /// name), bypassing the per-run name arrays and — for caller-buffer outputs — the per-run
    /// output allocation. Thread-safe like [`Self::run`]; reuses the session's `RunOptions`.
    pub fn run_binding(&self, binding: &crate::io_binding::IoBinding) -> Result<()> {
        binding.synchronize_inputs()?;
        check(unsafe {
            api().run_with_binding()(self.sess, self.run_opts.as_ptr(), binding.as_ptr())
        })?;
        binding.synchronize_outputs()
    }

    /// Run with an [`crate::IoBinding`] and a caller-provided [`RunOptions`] (per-call config
    /// or cancellation). See [`Self::run_with`] / [`Self::run_binding`].
    pub fn run_binding_with(
        &self, binding: &crate::io_binding::IoBinding, opts: &RunOptions,
    ) -> Result<()> {
        binding.synchronize_inputs()?;
        check(unsafe { api().run_with_binding()(self.sess, opts.as_ptr(), binding.as_ptr()) })?;
        binding.synchronize_outputs()
    }

    /// Run the model asynchronously (`RunAsync`, IDX 260) on an ORT worker thread. Returns a
    /// [`RunFuture`] that resolves to the outputs — pollable by any executor (no async-runtime
    /// dependency). `RunAsync` only errors synchronously if it fails to *start*.
    ///
    /// The future borrows the session and inputs (`'a`): keep them alive until it resolves
    /// (ORT's worker thread reads them until the callback fires). Dropping the future before it
    /// resolves is the caller's hazard — the session + inputs must still outlive the in-flight
    /// run (same contract as the C API). For 'static / cross-thread use, copy inputs into owned
    /// buffers first.
    pub fn run_async<'a>(&'a self, inputs: &'a [&'a dyn RunInput]) -> Result<RunFuture<'a>> {
        self.check_input_count(inputs.len())?;

        let n = self.output_count();
        // The input-handle array and the output-handle array live in the `Arc` state so they
        // outlive this call: ORT's worker thread reads the inputs and fills the outputs
        // asynchronously, after `RunAsync` returns. Both are freed when the state's last ref
        // drops (the callback's + the future's).
        let in_handles: Box<[*const sys::ValueHandle]> = inputs
            .iter()
            .map(|v| v.as_value_ptr())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let mut out_handles: Box<[*mut sys::ValueHandle]> =
            vec![ptr::null_mut(); n].into_boxed_slice();
        let in_ptr = in_handles.as_ptr();
        let out_ptr = out_handles.as_mut_ptr();

        let state = Arc::new(AsyncState {
            result: UnsafeCell::new(None),
            done: AtomicBool::new(false),
            waker: AtomicWaker::new(),
            _in_handles: in_handles,
            _out_handles: out_handles,
        });
        // Hand the state to the callback as `user_data` (one ref via `into_raw`; the future
        // keeps another). The callback recovers + drops its ref.
        let user_data = Arc::into_raw(state.clone()) as *mut c_void;

        let started = check(unsafe {
            api().run_async()(
                self.sess,
                self.run_opts.as_ptr(),
                self.input_ptrs.as_ptr(),
                in_ptr,
                inputs.len(),
                self.output_ptrs.as_ptr(),
                self.output_ptrs.len(),
                out_ptr,
                Some(run_async_callback),
                user_data,
            )
        });

        if let Err(e) = started {
            // Never started → callback never fires: recover the into_raw'd ref so the state
            // (and its arrays) is freed when the local `state` drops on return.
            unsafe {
                drop(Arc::from_raw(user_data as *const AsyncState));
            }
            return Err(e);
        }

        Ok(RunFuture {
            state,
            _borrows: std::marker::PhantomData,
        })
    }
}

impl PreparedRun<'_, '_> {
    /// Execute the prepared regular run. Previous engine-owned outputs are released before
    /// the next ORT call so the allocator can reuse memory immediately.
    pub fn run(&mut self) -> Result<&[Option<OwnedValue>]> {
        for slot in &mut self.outputs {
            *slot = None;
        }
        self.output_handles.fill(ptr::null_mut());
        self.session.run_raw(
            &self.input_handles,
            &mut self.output_handles,
            self.session.run_opts.as_ptr(),
        )?;
        let session = self.session;
        session.stamp_outputs(&self.output_handles, &mut self.outputs)?;
        Ok(&self.outputs)
    }

    /// Run this prepared call `runs` times before serving.
    ///
    /// This primes ORT's memory-pattern/cache behavior for static-shape workloads without
    /// changing the measured serving path.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Outputs from the most recent run.
    pub fn outputs(&self) -> &[Option<OwnedValue>] {
        &self.outputs
    }

    /// Output `i` from the most recent run.
    pub fn output(&self, i: usize) -> Result<Option<&OwnedValue>> {
        self.outputs
            .get(i)
            .map(Option::as_ref)
            .ok_or_else(|| Error::new(-1, format!("zrt: output index {i} out of range")))
    }
}

impl PreparedIoBinding<'_, '_> {
    /// Execute the prepared IoBinding.
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this prepared binding `runs` times before serving.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Access the underlying binding for synchronization or device-bound output reads.
    pub fn binding(&self) -> &IoBinding {
        &self.binding
    }
}

impl<T: TensorElement> TensorIoLane<'_, T> {
    /// Execute this lane's prepared binding.
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this lane `runs` times before serving.
    ///
    /// Use this after filling representative inputs and before exposing the lane to request
    /// traffic so ORT can populate memory-pattern and execution caches on the same shape.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    ///
    /// The stats calls are diagnostic and may allocate. Use this outside latency-critical
    /// measurements to understand allocator behavior around an otherwise hot-path run.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        let before = allocator.stats()?;
        self.run()?;
        let after = allocator.stats()?;
        Ok(LaneRunAllocatorStats { before, after })
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[T]> {
        self.outputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.outputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.inputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.outputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }
}

impl<T: TensorElement> AllocatedOutputTensorIoLane<'_, T> {
    /// Execute this lane's prepared binding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this lane `runs` times before serving.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        let before = allocator.stats()?;
        self.run()?;
        let after = allocator.stats()?;
        Ok(LaneRunAllocatorStats { before, after })
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated-output lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated-output lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[T]> {
        self.outputs
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated-output lane output index {i} out of range"),
                )
            })?
            .as_slice()
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.outputs
            .get_mut(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated-output lane output index {i} out of range"),
                )
            })?
            .as_mut_slice()
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.inputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: allocated-output lane input index {i} out of range"),
            )
        })
    }

    #[inline]
    pub fn output_tensor(&self, i: usize) -> Result<&AllocatedTensor<T>> {
        self.outputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: allocated-output lane output index {i} out of range"),
            )
        })
    }
}

impl<T: TensorElement> DeviceOutputTensorIoLane<'_, T> {
    /// Execute this lane and refresh the retrieved ORT output values.
    pub fn run(&mut self) -> Result<&[OwnedValue]> {
        self.outputs.clear();
        self.session.run_binding(&self.binding)?;
        self.outputs = self.binding.output_values()?;
        Ok(&self.outputs)
    }

    /// Run this lane `runs` times before serving.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        let before = allocator.stats()?;
        self.run()?;
        let after = allocator.stats()?;
        Ok(LaneRunAllocatorStats { before, after })
    }

    #[inline]
    pub fn outputs(&self) -> &[OwnedValue] {
        &self.outputs
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&OwnedValue> {
        self.outputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: device-output lane output index {i} out of range"),
            )
        })
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: device-output lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: device-output lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.inputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: device-output lane input index {i} out of range"),
            )
        })
    }
}

impl<T: TensorElement> AllocatedTensorIoLane<'_, T> {
    /// Execute this lane's prepared binding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this lane `runs` times before serving.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        let before = allocator.stats()?;
        self.run()?;
        let after = allocator.stats()?;
        Ok(LaneRunAllocatorStats { before, after })
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated tensor lane input index {i} out of range"),
                )
            })?
            .as_slice()
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated tensor lane input index {i} out of range"),
                )
            })?
            .as_mut_slice()
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[T]> {
        self.outputs
            .get(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated tensor lane output index {i} out of range"),
                )
            })?
            .as_slice()
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.outputs
            .get_mut(i)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: allocated tensor lane output index {i} out of range"),
                )
            })?
            .as_mut_slice()
    }

    #[inline]
    pub fn input_tensor(&self, i: usize) -> Result<&AllocatedTensor<T>> {
        self.inputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: allocated tensor lane input index {i} out of range"),
            )
        })
    }

    #[inline]
    pub fn output_tensor(&self, i: usize) -> Result<&AllocatedTensor<T>> {
        self.outputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: allocated tensor lane output index {i} out of range"),
            )
        })
    }
}

impl<T: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    StaticTensorIoLane<'_, T, INPUTS, OUTPUTS>
{
    /// Execute this lane's prepared binding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this fixed-arity lane `runs` times before serving.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    ///
    /// The stats calls are diagnostic and may allocate. Use this outside latency-critical
    /// measurements to understand allocator behavior around an otherwise hot-path run.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        let before = allocator.stats()?;
        self.run()?;
        let after = allocator.stats()?;
        Ok(LaneRunAllocatorStats { before, after })
    }

    #[inline]
    pub fn inputs(&self) -> &[TensorBuffer<T>; INPUTS] {
        &self.inputs
    }

    #[inline]
    pub fn inputs_mut(&mut self) -> &mut [TensorBuffer<T>; INPUTS] {
        &mut self.inputs
    }

    #[inline]
    pub fn outputs(&self) -> &[TensorBuffer<T>; OUTPUTS] {
        &self.outputs
    }

    #[inline]
    pub fn outputs_mut(&mut self) -> &mut [TensorBuffer<T>; OUTPUTS] {
        &mut self.outputs
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[T]> {
        self.outputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.outputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.inputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.outputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // run_opts (RunOptions) drops itself; release the session explicitly.
        unsafe {
            if !self.sess.is_null() {
                api().release_session()(self.sess);
            }
        }
    }
}
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

// ── model-editor session construction (feature `model-editor`) ────────────────
#[cfg(feature = "model-editor")]
impl Session {
    /// Build a session from an in-memory [`crate::Model`] built via the model-editor API
    /// (`CreateSessionFromModel`). The model is BORROWED (released when the [`crate::Model`]
    /// drops); this validates the model, runs optimizers, and prepares it for inference.
    pub fn from_model(
        env: &Environment, model: &crate::model_editor::Model, opts: SessionOptions,
    ) -> Result<Self> {
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let create = crate::model_editor::require_sub_api_fn(
            me.CreateSessionFromModel,
            "ModelEditorApi",
            "CreateSessionFromModel",
        )?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let mut sess: *mut sys::SessionHandle = ptr::null_mut();
        let create = check(unsafe {
            create(
                env.as_ptr(),
                model.as_ptr(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut sess,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        create?;
        Self::from_handle(sess, env.share())
    }

    /// The opset `since_version` registered for `domain` on this session
    /// (`SessionGetOpsetForDomain`).
    pub fn opset_for_domain(&self, domain: &str) -> Result<i32> {
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let get_opset = crate::model_editor::require_sub_api_fn(
            me.SessionGetOpsetForDomain,
            "ModelEditorApi",
            "SessionGetOpsetForDomain",
        )?;
        let cdom = CString::new(domain)?;
        let mut opset: i32 = 0;
        check(unsafe {
            get_opset(
                self.sess as *const sys::SessionHandle,
                cdom.as_ptr(),
                &mut opset,
            )
        })?;
        Ok(opset)
    }

    /// Load an existing model (bytes) as a **model-editor session** — a session you can
    /// augment with new nodes ([`Self::apply_model`]) before [`Self::finalize`] + run
    /// (`CreateModelEditorSessionFromArray`). The model is borrowed.
    pub fn from_bytes_for_editing(
        env: &Environment, model_data: &[u8], opts: SessionOptions,
    ) -> Result<Self> {
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let create = crate::model_editor::require_sub_api_fn(
            me.CreateModelEditorSessionFromArray,
            "ModelEditorApi",
            "CreateModelEditorSessionFromArray",
        )?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let mut sess: *mut sys::SessionHandle = ptr::null_mut();
        let create = check(unsafe {
            create(
                env.as_ptr(),
                model_data.as_ptr() as *const c_void,
                model_data.len(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut sess,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        create?;
        Self::from_handle(sess, env.share())
    }

    /// Apply a constructed [`crate::Model`] (e.g. extra nodes) to this model-editor session
    /// (`ApplyModelToModelEditorSession`). The model is borrowed; call before [`Self::finalize`].
    pub fn apply_model(&self, model: &crate::model_editor::Model) -> Result<()> {
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let apply = crate::model_editor::require_sub_api_fn(
            me.ApplyModelToModelEditorSession,
            "ModelEditorApi",
            "ApplyModelToModelEditorSession",
        )?;
        check(unsafe { apply(self.sess, model.as_ptr() as *mut sys::ModelHandle) })
    }

    /// Finalize a model-editor session after any [`Self::apply_model`]
    /// (`FinalizeModelEditorSession`) — validates + prepares it for inference.
    pub fn finalize(&mut self, opts: &SessionOptions) -> Result<()> {
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let finalize = crate::model_editor::require_sub_api_fn(
            me.FinalizeModelEditorSession,
            "ModelEditorApi",
            "FinalizeModelEditorSession",
        )?;
        let opts_handle = opts.build_handle()?;
        // No EP-device attach here: `finalize` has no `env`, and any queued device attach was
        // already applied in the `from_bytes_for_editing` constructor that created this session.
        let r = check(unsafe {
            finalize(
                self.sess,
                opts_handle as *const sys::SessionOptionsHandle,
                ptr::null_mut(),
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        r?;
        self.refresh_io_metadata()
    }
}

// ─── async run (RunAsync → generic Future) ────────────────────────────────────
//
// `RunAsync` (IDX 260) returns a status only if it fails to START; the result arrives on an
// ORT worker thread via a `RunAsyncCallbackFn`. We bridge that callback to a generic
// `impl Future<Output = Result<Vec<OwnedValue>>>` with no async-runtime dependency: an
// `Arc`-shared completion state carries the result + atomic waker; the `extern "C"` callback
// fills it and wakes; `poll` returns the result or registers the waker. `done` is the
// release/acquire handoff between the ORT callback thread and the polling executor.

/// Completion state shared between [`Session::run_async`]'s [`RunFuture`] and the ORT
/// worker-thread callback.
struct AsyncState {
    result: UnsafeCell<Option<Result<Vec<OwnedValue>>>>,
    done: AtomicBool,
    waker: AtomicWaker,
    /// Kept alive in the `Arc` until the run completes + the future drops: ORT's worker thread
    /// reads the input handles and fills the output array asynchronously *after* `RunAsync`
    /// returns, so both must outlive the call. The `OrtValue`s themselves are owned separately
    /// (input tensors by the caller for `'a`; output values by the `OwnedValue`s in the result).
    _in_handles: Box<[*const sys::ValueHandle]>,
    _out_handles: Box<[*mut sys::ValueHandle]>,
}
// SAFETY: `result` is written exactly once by the ORT callback before `done.store(true,
// Release)`, then taken by the single `Future::poll(&mut self)` owner after an Acquire load
// observes completion. The waker path is coordinated by `AtomicWaker`. The handle arrays are
// written before sharing and only kept alive afterward. ORT handles may move between the
// callback and polling threads.
unsafe impl Send for AsyncState {}
unsafe impl Sync for AsyncState {}

/// A pending asynchronous inference run ([`Session::run_async`]). `await` (or `poll`) for the
/// outputs. Borrows the session + inputs for `'a` (see [`Session::run_async`]).
pub struct RunFuture<'a> {
    state: Arc<AsyncState>,
    _borrows: std::marker::PhantomData<&'a ()>,
}

impl<'a> std::future::Future for RunFuture<'a> {
    type Output = Result<Vec<OwnedValue>>;
    fn poll(
        self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.state.done.load(Ordering::Acquire) {
            return std::task::Poll::Ready(self.state.take_result());
        }
        self.state.waker.register(cx.waker());
        if self.state.done.load(Ordering::Acquire) {
            std::task::Poll::Ready(self.state.take_result())
        } else {
            std::task::Poll::Pending
        }
    }
}

impl AsyncState {
    fn complete(&self, result: Result<Vec<OwnedValue>>) {
        // SAFETY: the callback is the only writer and completion happens once per RunAsync.
        unsafe { *self.result.get() = Some(result) };
        self.done.store(true, Ordering::Release);
        self.waker.wake();
    }

    fn take_result(&self) -> Result<Vec<OwnedValue>> {
        // SAFETY: `poll` has exclusive access to the future. After `done` is observed true, the
        // callback no longer writes `result`.
        unsafe { (*self.result.get()).take() }
            .unwrap_or_else(|| Err(crate::Error::new(-1, "zrt: async result already consumed")))
    }
}

/// ORT worker-thread completion trampoline for `RunAsync`. Reconstructs the `Arc<AsyncState>`
/// from `user_data`, collects the outputs (or surfaces the error status), and wakes the future.
/// Wrapped in `catch_unwind`: a panic becomes `ORT_FAIL` and is never unwound across the FFI
/// boundary. (The input/output arrays are owned by the `Arc` state — not freed here.)
#[allow(clippy::from_raw_with_void_ptr)] // legitimate: `user_data` is an opaque FFI `void*`
unsafe extern "C" fn run_async_callback(
    user_data: *mut c_void, outputs: *mut *mut sys::ValueHandle, num_outputs: usize,
    status: sys::StatusPtr,
) {
    unsafe {
        // Recover the Arc ref we passed via `into_raw`. (Null can't happen — we always pass one.)
        let state: Arc<AsyncState> = Arc::from_raw(user_data as *const AsyncState);

        let result: Result<Vec<OwnedValue>> =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Consume the status: null ⇒ success; non-null ⇒ `check` frees it and yields the Err.
                if !status.is_null() {
                    return Err(match check(status) {
                        Err(e) => e,
                        Ok(()) => crate::Error::new(
                            sys::OrtErrorCode::Fail as i32,
                            "RunAsync returned a non-null but Ok status",
                        ),
                    });
                }
                if outputs.is_null() {
                    return Ok(Vec::new());
                }
                let handles = std::slice::from_raw_parts(outputs, num_outputs);
                OwnedValue::collect_from_raw(handles)
            }))
            .unwrap_or_else(|_| {
                Err(crate::Error::new(
                    sys::OrtErrorCode::Fail as i32,
                    "panic in RunAsync callback",
                ))
            });

        state.complete(result);

        // The input/output arrays are owned by the `Arc` state — freed when its last ref drops
        // (the callback's here + the future's), not here. `state` drops its ref at end of scope.
    }
}

fn add_owned_initializers(
    opts: *mut sys::SessionOptionsHandle, initializers: &[OwnedInitializer],
) -> Result<()> {
    for init in initializers {
        check(unsafe { api().add_initializer()(opts, init.name_ptr(), init.value_ptr()) })?;
    }
    Ok(())
}

fn build_session_options_for_env(
    env: &Environment, opts: &SessionOptions,
) -> Result<*mut sys::SessionOptionsHandle> {
    let opts_handle = opts.build_handle()?;
    let result = (|| {
        apply_ep_device_attach_or_release(env, opts_handle, opts)?;
        if env.has_global_thread_pool() && opts.use_global_thread_pool {
            check(unsafe { api().disable_per_session_threads()(opts_handle) })?;
        }
        Ok(opts_handle)
    })();
    if result.is_err() {
        unsafe { api().release_session_options()(opts_handle) };
    }
    result
}

fn apply_ep_device_attach_or_release(
    env: &Environment, opts_handle: *mut sys::SessionOptionsHandle, opts: &SessionOptions,
) -> Result<()> {
    #[cfg(feature = "ep")]
    if let Err(err) =
        crate::ep_device::apply_device_attach(env, opts_handle, &opts.ep_device_attach)
    {
        unsafe { api().release_session_options()(opts_handle) };
        return Err(err);
    }
    let _ = (env, opts_handle, opts);
    Ok(())
}

/// Fetch input or output names, freeing each engine-allocated string immediately and
/// caching a stable `CString` + a raw pointer to it.
fn collect_io_names(
    sess: *mut sys::SessionHandle, is_input: bool, alloc: &Allocator,
) -> Result<(Vec<CString>, Vec<*const c_char>)> {
    let api = api();
    let mut count: usize = 0;
    check(unsafe {
        if is_input {
            api.session_get_input_count()(sess as *const sys::SessionHandle, &mut count)
        } else {
            api.session_get_output_count()(sess as *const sys::SessionHandle, &mut count)
        }
    })?;

    let mut names = Vec::with_capacity(count);
    for i in 0..count {
        let mut raw: *mut c_char = ptr::null_mut();
        check(unsafe {
            if is_input {
                api.session_get_input_name()(
                    sess as *const sys::SessionHandle,
                    i,
                    alloc.alloc,
                    &mut raw,
                )
            } else {
                api.session_get_output_name()(
                    sess as *const sys::SessionHandle,
                    i,
                    alloc.alloc,
                    &mut raw,
                )
            }
        })?;
        if raw.is_null() {
            return Err(Error::new(-1, "zrt: session I/O name pointer is null"));
        }
        let c = unsafe { CStr::from_ptr(raw).to_owned() };
        unsafe { alloc.free(raw as *mut c_void) }?;
        names.push(c);
    }
    // Pointers into the CStrings — stable: the Vec is never reallocated after this.
    let ptrs = names.iter().map(|c| c.as_ptr()).collect();
    Ok((names, ptrs))
}

/// Resolve each input/output value kind and — for tensors — element type, dims, symbolic
/// dims, and static element count when concrete from the model's STATIC type-info.
/// The cast-to-tensor-info result is a NON-OWNING borrow of the TypeInfo (released with it).
fn collect_io_meta(
    sess: *mut sys::SessionHandle, is_input: bool, count: usize,
) -> Result<Vec<CachedIo>> {
    let api = api();
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut type_info: *mut sys::TypeInfoHandle = ptr::null_mut();
        let meta = (|| -> Result<CachedIo> {
            check(unsafe {
                if is_input {
                    api.session_get_input_type_info()(
                        sess as *const sys::SessionHandle,
                        i,
                        &mut type_info,
                    )
                } else {
                    api.session_get_output_type_info()(
                        sess as *const sys::SessionHandle,
                        i,
                        &mut type_info,
                    )
                }
            })?;
            let mut onnx_type = sys::OnnxType::Unknown;
            check(unsafe {
                api.get_onnx_type_from_type_info()(
                    type_info as *const sys::TypeInfoHandle,
                    &mut onnx_type,
                )
            })?;
            if onnx_type == sys::OnnxType::Tensor {
                let mut tensor_info: *const sys::TensorTypeAndShapeInfoHandle = ptr::null();
                check(unsafe {
                    api.cast_type_info_to_tensor_info()(
                        type_info as *const sys::TypeInfoHandle,
                        &mut tensor_info,
                    )
                })?;
                let mut etype = sys::ElementType::Undefined;
                check(unsafe { api.get_tensor_element_type()(tensor_info, &mut etype) })?;
                let mut rank: usize = 0;
                check(unsafe { api.get_dimensions_count()(tensor_info, &mut rank) })?;
                let mut dims = vec![0i64; rank];
                check(unsafe { api.get_dimensions()(tensor_info, dims.as_mut_ptr(), rank) })?;
                let mut sptrs: Vec<*const c_char> = vec![ptr::null(); rank];
                check(unsafe {
                    api.get_symbolic_dimensions()(tensor_info, sptrs.as_mut_ptr(), rank)
                })?;
                let symbolic = sptrs
                    .iter()
                    .map(|&p| {
                        if p.is_null() {
                            Ok(None)
                        } else {
                            unsafe { crate::cstr_to_string(p, "symbolic dimension") }.map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let count = crate::type_info::checked_element_count(&dims).ok();
                Ok(CachedIo {
                    onnx_type,
                    elem_type: etype,
                    count,
                    dims,
                    symbolic,
                })
            } else {
                // Sequence / map / optional output: no tensor element type or shape.
                Ok(CachedIo {
                    onnx_type,
                    elem_type: sys::ElementType::Undefined,
                    count: Some(0),
                    dims: Vec::new(),
                    symbolic: Vec::new(),
                })
            }
        })();
        if !type_info.is_null() {
            unsafe { api.release_type_info()(type_info) };
        }
        out.push(meta?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    // Session-level dynamic-output behavior is covered by the HF probe and smoke tests.
}
