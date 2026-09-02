//! `Session` — a pre-marshaled inference session.
//!
//! Input/output names are resolved once at construction (anti-pattern O1 fix: no
//! per-run `FeedsFetchesManager` rebuild, no name marshaling on the hot path) and
//! one frozen `MaterializedRunOptions` handle is reused (anti-pattern O4 fix). `run(&self)` is shared-reentrant —
//! ORT's `Run` is thread-safe on a session.
use crate::allocator::{Allocator, AllocatorStats, AllocatorStatsDelta};
use crate::element::TensorElement;
use crate::environment::{EnvInner, Environment};
use crate::initializer::OwnedInitializer;
use crate::io_binding::{IoBinding, OutputValue};
use crate::memory::{MemoryInfo, MemoryInfoSnapshot};
use crate::prepacked::{PrepackedWeightsContainer, PrepackedWeightsInner};
use crate::run_options::{MaterializedRunOptions, RunOptions};
use crate::session_options::SessionOptions;
use crate::tensor::{
    AllocatedTensor, BufferSpec, OwnedValue, RunInput, TensorBuffer, tensor_memory_info,
};
use crate::{Error, Result, api, check, sys};
use futures_util::task::AtomicWaker;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

#[cfg(feature = "cuda")]
type CudaStreamGuards = Vec<Arc<crate::CudaStream>>;
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
struct CudaStreamGuard;
#[cfg(not(feature = "cuda"))]
type CudaStreamGuards = Vec<CudaStreamGuard>;

#[cfg(feature = "cuda")]
fn cuda_stream_guards(options: &SessionOptions) -> CudaStreamGuards {
    options.cuda_stream_guards()
}
#[cfg(not(feature = "cuda"))]
fn cuda_stream_guards(_options: &SessionOptions) -> CudaStreamGuards {
    Vec::new()
}

const STACK_IO_HANDLES: usize = 8;
static NEXT_CAPTURED_GRAPH_ID: AtomicI32 = AtomicI32::new(1);

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

const GRAPH_LEASE_RELEASING: usize = 1 << (usize::BITS - 1);
const GRAPH_LEASE_ACTIVE_MASK: usize = GRAPH_LEASE_RELEASING - 1;

#[derive(Default)]
struct CapturedGraphLeaseState {
    /// High bit = release in progress; remaining bits = active runs.
    state: AtomicUsize,
    wait_lock: Mutex<()>,
    wait_cv: Condvar,
}

/// Cached per-graph lease. Setup performs the session HashMap lookup once; run acquisition is one
/// compare-exchange loop with no Mutex or HashMap on the normal path.
#[derive(Clone)]
pub(crate) struct CapturedGraphLease(Arc<CapturedGraphLeaseState>);

impl CapturedGraphLease {
    fn new() -> Self {
        Self(Arc::new(CapturedGraphLeaseState::default()))
    }

    pub(crate) fn begin_run(&self) -> CapturedGraphRunGuard {
        loop {
            let current = self.0.state.load(Ordering::Acquire);
            if current & GRAPH_LEASE_RELEASING != 0 {
                let mut wait = self.0.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
                while self.0.state.load(Ordering::Acquire) & GRAPH_LEASE_RELEASING != 0 {
                    wait = self.0.wait_cv.wait(wait).unwrap_or_else(|e| e.into_inner());
                }
                continue;
            }
            assert!(
                current < GRAPH_LEASE_ACTIVE_MASK,
                "zrt: captured-graph active run counter overflow"
            );
            if self
                .0
                .state
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return CapturedGraphRunGuard {
                    lease: self.clone(),
                };
            }
        }
    }

    fn begin_release(&self) {
        loop {
            let current = self.0.state.load(Ordering::Acquire);
            if current & GRAPH_LEASE_RELEASING != 0 {
                let mut wait = self.0.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
                while self.0.state.load(Ordering::Acquire) & GRAPH_LEASE_RELEASING != 0 {
                    wait = self.0.wait_cv.wait(wait).unwrap_or_else(|e| e.into_inner());
                }
                continue;
            }
            if self
                .0
                .state
                .compare_exchange_weak(
                    current,
                    current | GRAPH_LEASE_RELEASING,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                break;
            }
        }
        let mut wait = self.0.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
        while self.0.state.load(Ordering::Acquire) & GRAPH_LEASE_ACTIVE_MASK != 0 {
            wait = self.0.wait_cv.wait(wait).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn end_release(&self) {
        let _wait = self.0.wait_lock.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert_eq!(
            self.0.state.load(Ordering::Relaxed) & GRAPH_LEASE_ACTIVE_MASK,
            0
        );
        self.0.state.store(0, Ordering::Release);
        self.0.wait_cv.notify_all();
    }
}

/// A cheap-clone handle to one initialized ONNX Runtime session.
///
/// Clones share the native session, cached metadata, default run options, graph leases, and all
/// lifetime guards. The native handle is released only after the last `Session` (or session-owned
/// resource such as an allocator or I/O binding) drops.
#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
}

pub(crate) struct SessionInner {
    sess: *mut sys::SessionHandle,
    input_names: Vec<CString>,
    input_ptrs: Vec<*const c_char>,
    input_meta: Vec<CachedIo>,
    output_names: Vec<CString>,
    output_ptrs: Vec<*const c_char>,
    output_meta: Vec<CachedIo>,
    run_opts: MaterializedRunOptions,
    captured_graph_leases: Mutex<HashMap<i32, CapturedGraphLease>>,
    /// Optional caller-owned initializers handed to ORT at session creation. Kept alive until
    /// after the ORT session is released.
    _owned_initializers: Vec<OwnedInitializer>,
    /// Optional prepacked-weight cache. Kept alive until after the ORT session is released.
    _prepacked_weights: Option<Arc<PrepackedWeightsInner>>,
    /// Owned CUDA streams referenced by this session's provider configuration. Native session
    /// release occurs before these guards drop.
    _cuda_streams: CudaStreamGuards,
    /// Keeps the Env alive for this Session's whole lifetime. The explicit `SessionInner::drop`
    /// releases the native session before Rust drops this guard.
    _env: Arc<EnvInner>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("sess", &self.inner.sess)
            .field("inputs", &self.inner.input_names.len())
            .field("outputs", &self.inner.output_names.len())
            .field("run_opts", &self.inner.run_opts)
            .field("strong_count", &Arc::strong_count(&self.inner))
            .finish_non_exhaustive()
    }
}

pub(crate) struct CapturedGraphRunGuard {
    lease: CapturedGraphLease,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoDirection {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionProviderDeviceSnapshot {
    pub ep_name: String,
    pub ep_vendor: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IoPlacement {
    pub direction: IoDirection,
    pub index: usize,
    pub name: String,
    pub onnx_type: sys::OnnxType,
    pub element_type: sys::ElementType,
    pub shape: Vec<i64>,
    pub symbolic_dims: Vec<Option<String>>,
    pub memory_info: MemoryInfoSnapshot,
    pub ep_device: Option<ExecutionProviderDeviceSnapshot>,
}

impl LaneRunAllocatorStats {
    /// Numeric allocator-counter deltas between the before/after snapshots.
    #[inline]
    pub fn delta(&self) -> AllocatorStatsDelta {
        self.before.diff(&self.after)
    }
}

fn run_with_allocator_stats(
    allocator: &Allocator, run: impl FnOnce() -> Result<()>,
) -> Result<LaneRunAllocatorStats> {
    let before = allocator.stats()?;
    run()?;
    let after = allocator.stats()?;
    Ok(LaneRunAllocatorStats { before, after })
}

fn prime_runs(mut run: impl FnMut() -> Result<()>, runs: usize) -> Result<()> {
    for _ in 0..runs {
        run()?;
    }
    Ok(())
}

pub(crate) fn lane_tensor_buffer<T>(
    shape: &[i64], mem: &MemoryInfo, spec: BufferSpec,
) -> Result<TensorBuffer<T>>
where
    T: TensorElement + Clone + Default,
{
    TensorBuffer::zeros_with(shape, mem, spec)
}

fn ep_device_snapshot_from_ptr(
    ptr: *const sys::EpDeviceHandle,
) -> Result<Option<ExecutionProviderDeviceSnapshot>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let raw_name = unsafe { api().ep_device__ep_name()(ptr) };
    let ep_name = if raw_name.is_null() {
        String::new()
    } else {
        unsafe { crate::cstr_to_string(raw_name, "execution provider device name")? }
    };
    let raw_vendor = unsafe { api().ep_device__ep_vendor()(ptr) };
    let ep_vendor = if raw_vendor.is_null() {
        String::new()
    } else {
        unsafe { crate::cstr_to_string(raw_vendor, "execution provider device vendor")? }
    };
    Ok(Some(ExecutionProviderDeviceSnapshot { ep_name, ep_vendor }))
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
        Self::from_handle(sess, env.share(), cuda_stream_guards(&opts))
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
        Self::from_handle(sess, env.share(), cuda_stream_guards(&opts))
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
        Self::from_handle_with_resources(
            sess,
            env.share(),
            initializers,
            None,
            cuda_stream_guards(&opts),
        )
    }

    /// Load `model_path`, replacing initializers the model marks as **external-data** with the
    /// provided in-memory tensors (`AddExternalInitializers`). Each entry's name, shape, and
    /// element type must match an external initializer already in the graph; ORT verifies the
    /// match and copies the provided data in (the backing buffers need not outlive session
    /// creation). To override *normal* (non-external) initializers, use
    /// [`Self::new_with_owned_initializers`] instead.
    pub fn new_with_external_initializers(
        env: &Environment, model_path: &str, opts: SessionOptions,
        initializers: Vec<OwnedInitializer>,
    ) -> Result<Self> {
        let cpath = CString::new(model_path)
            .map_err(|_| crate::Error::new(-1, "model path contains a NUL"))?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_external_initializers_batch(opts_handle, &initializers)?;
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
        Self::from_handle_with_resources(
            sess,
            env.share(),
            initializers,
            None,
            cuda_stream_guards(&opts),
        )
    }

    /// Load `model_path` — a model whose initializers are stored in **external data files** —
    /// supplying those files's contents from memory (`AddExternalInitializersFromFilesInMemory`).
    /// Each entry is `(external_file_name, file_bytes)`; the name must match the model's
    /// external-data location. The buffers are consumed during session creation and not retained.
    pub fn new_with_external_initializer_files(
        env: &Environment, model_path: &str, opts: SessionOptions, files: Vec<(String, Vec<u8>)>,
    ) -> Result<Self> {
        let cpath = CString::new(model_path)
            .map_err(|_| crate::Error::new(-1, "model path contains a NUL"))?;
        let cfiles: Vec<(CString, Vec<u8>)> = files
            .into_iter()
            .map(|(name, bytes)| {
                let cname = CString::new(name).map_err(|_| {
                    crate::Error::new(-1, "external initializer file name contains a NUL")
                })?;
                Ok((cname, bytes))
            })
            .collect::<Result<Vec<_>>>()?;
        let opts_handle = build_session_options_for_env(env, &opts)?;
        let create = (|| -> Result<*mut sys::SessionHandle> {
            add_external_initializer_files_in_memory(opts_handle, &cfiles)?;
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
        Self::from_handle(sess, env.share(), cuda_stream_guards(&opts))
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
        Self::from_handle_with_resources(
            sess,
            env.share(),
            initializers,
            Some(prepacked.share()),
            cuda_stream_guards(&opts),
        )
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
        Self::from_handle_with_resources(
            sess,
            env.share(),
            initializers,
            None,
            cuda_stream_guards(&opts),
        )
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
        Self::from_handle_with_resources(
            sess,
            env.share(),
            initializers,
            Some(prepacked.share()),
            cuda_stream_guards(&opts),
        )
    }

    /// Finish construction from a freshly-created session handle: pre-marshal I/O names and
    /// cache output type/shape, then build the struct. Shared by [`Self::new`] and
    /// [`Self::from_bytes`].
    fn from_handle(
        sess: *mut sys::SessionHandle, env: Arc<EnvInner>, cuda_streams: CudaStreamGuards,
    ) -> Result<Self> {
        Self::from_handle_with_resources(sess, env, Vec::new(), None, cuda_streams)
    }

    fn from_handle_with_resources(
        sess: *mut sys::SessionHandle, env: Arc<EnvInner>,
        owned_initializers: Vec<OwnedInitializer>,
        prepacked_weights: Option<Arc<PrepackedWeightsInner>>, cuda_streams: CudaStreamGuards,
    ) -> Result<Self> {
        let sess = crate::ensure_non_null(sess, "session")?;
        // Resolve every fallible setup value while all lifetime guards remain in this outer frame.
        // On error, release the native session before returning and dropping those guards.
        let setup = (|| {
            let alloc = Allocator::get_default()?;
            let (input_names, input_ptrs) = collect_io_names(sess, true, &alloc)?;
            let (output_names, output_ptrs) = collect_io_names(sess, false, &alloc)?;
            let input_meta = collect_io_meta(sess, true, input_ptrs.len())?;
            let output_meta = collect_io_meta(sess, false, output_ptrs.len())?;
            let run_opts = RunOptions::new().freeze()?;
            Ok((
                input_names,
                input_ptrs,
                input_meta,
                output_names,
                output_ptrs,
                output_meta,
                run_opts,
            ))
        })();
        let (input_names, input_ptrs, input_meta, output_names, output_ptrs, output_meta, run_opts) =
            match setup {
                Ok(values) => values,
                Err(error) => {
                    unsafe { api().release_session()(sess) };
                    return Err(error);
                },
            };
        Ok(Self {
            inner: Arc::new(SessionInner {
                sess,
                input_names,
                input_ptrs,
                input_meta,
                output_names,
                output_ptrs,
                output_meta,
                run_opts,
                captured_graph_leases: Mutex::new(HashMap::new()),
                _owned_initializers: owned_initializers,
                _prepacked_weights: prepacked_weights,
                _cuda_streams: cuda_streams,
                _env: env,
            }),
        })
    }

    #[cfg(feature = "model-editor")]
    fn refresh_io_metadata(&mut self) -> Result<()> {
        let inner = Arc::get_mut(&mut self.inner).ok_or_else(|| {
            Error::local("cannot mutate model-editor session metadata while Session clones or session-owned resources exist")
        })?;
        let alloc = Allocator::get_default()?;
        let (input_names, input_ptrs) = collect_io_names(inner.sess, true, &alloc)?;
        let (output_names, output_ptrs) = collect_io_names(inner.sess, false, &alloc)?;
        let input_meta = collect_io_meta(inner.sess, true, input_ptrs.len())?;
        let output_meta = collect_io_meta(inner.sess, false, output_ptrs.len())?;

        inner.input_names = input_names;
        inner.input_ptrs = input_ptrs;
        inner.input_meta = input_meta;
        inner.output_names = output_names;
        inner.output_ptrs = output_ptrs;
        inner.output_meta = output_meta;
        Ok(())
    }

    /// The model's metadata (producer, graph name/description, domain, version, custom
    /// metadata map). Owning handle (`SessionGetModelMetadata`, idx 111); released on drop.
    pub fn metadata(&self) -> Result<crate::metadata::ModelMetadata> {
        let mut meta: *mut sys::ModelMetadataHandle = ptr::null_mut();
        check(unsafe {
            api().session_get_model_metadata()(
                self.inner.sess as *const sys::SessionHandle,
                &mut meta,
            )
        })?;
        let meta = crate::ensure_non_null(meta, "model metadata")?;
        Ok(unsafe { crate::metadata::ModelMetadata::from_owning(meta) })
    }

    /// The EP→subgraph assignment for this session (`Session_GetEpGraphAssignmentInfo`): which
    /// execution provider runs which portion of the graph. Empty when no EP is assigned (e.g. a
    /// pure-CPU session). Each subgraph borrows this session — use it to inspect node placement
    /// (the basis for counting Memcpy/transfer nodes between EPs).
    ///
    /// **Requires** the session be created with the config entry
    /// `session.record_ep_graph_assignment_info = "1"` (via
    /// [`SessionOptions::with_config_entry`]); otherwise this returns an ORT error.
    #[cfg(feature = "ep")]
    pub fn ep_graph_assignment_info(
        &self,
    ) -> Result<Vec<crate::ep_device::EpAssignedSubgraph<'_>>> {
        let mut subgraphs: *const *const sys::EpAssignedSubgraphHandle = ptr::null();
        let mut num: usize = 0;
        check(unsafe {
            api().session__get_ep_graph_assignment_info()(
                self.inner.sess as *const sys::SessionHandle,
                &mut subgraphs as *mut _ as *const *const *const sys::EpAssignedSubgraphHandle,
                &mut num,
            )
        })?;
        if subgraphs.is_null() || num == 0 {
            return Ok(Vec::new());
        }
        (0..num)
            .map(|i| {
                // SAFETY: the engine owns the array for the session's lifetime.
                let p = unsafe { *subgraphs.add(i) };
                Ok(unsafe { crate::ep_device::EpAssignedSubgraph::from_borrowed(p) })
            })
            .collect()
    }

    /// Profiling start timestamp in nanoseconds as reported by ORT.
    pub fn profiling_start_time_ns(&self) -> Result<u64> {
        let mut out = 0u64;
        check(unsafe {
            api().session_get_profiling_start_time_ns()(
                self.inner.sess as *const sys::SessionHandle,
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
        check(unsafe { api().session_end_profiling()(self.inner.sess, alloc.alloc, &mut raw) })?;
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

    /// Ask ORT to release a previously captured graph for the given annotation id
    /// (ORT 1.27 `SessionReleaseCapturedGraph`).
    ///
    /// Release support is execution-provider-specific. ORT 1.27 exposes this C API, but the legacy
    /// CUDA EP currently inherits the base no-op implementation: captured CUDA graphs remain tied to
    /// the session lifetime and `gpu_graph_id` values must not be reused after this call returns.
    /// Treat this as a best-effort provider hook, not as proof that CUDA graph memory was reclaimed.
    ///
    /// Releasing an id that was never captured is EP-specific; for providers that do not implement
    /// graph release, ORT may report success without doing work.
    /// For graph-backed ZRT lanes, this waits for tracked in-flight replays of the same annotation
    /// id to finish and prevents new tracked replays from starting until the release returns.
    pub fn release_captured_graph(&self, annotation_id: i32) -> Result<()> {
        let lease = self.captured_graph_lease(annotation_id);
        lease.begin_release();
        let result = check(unsafe {
            api().session_release_captured_graph()(self.inner.sess, annotation_id)
        });
        lease.end_release();
        result
    }

    pub(crate) fn allocate_captured_graph_id(&self) -> Result<i32> {
        NEXT_CAPTURED_GRAPH_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| {
                Error::new(
                    -1,
                    "zrt: gpu_graph_id exhausted (i32 overflow); restart the process or disable cuda_graph",
                )
            })
    }

    pub(crate) fn captured_graph_lease(&self, annotation_id: i32) -> CapturedGraphLease {
        self.inner
            .captured_graph_leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(annotation_id)
            .or_insert_with(CapturedGraphLease::new)
            .clone()
    }

    /// The number of overridable initializers in the session's graph
    /// (`SessionGetOverridableInitializerCount`).
    pub fn overridable_initializer_count(&self) -> Result<usize> {
        let mut n: usize = 0;
        check(unsafe {
            api().session_get_overridable_initializer_count()(
                self.inner.sess as *const sys::SessionHandle,
                &mut n,
            )
        })?;
        Ok(n)
    }

    /// The name of the overridable initializer at `index`
    /// (`SessionGetOverridableInitializerName`). Engine-allocated (default allocator); copied
    /// into an owned `String`.
    pub fn overridable_initializer_name(&self, index: usize) -> Result<String> {
        let alloc = Allocator::get_default()?;
        let mut raw: *mut c_char = ptr::null_mut();
        check(unsafe {
            api().session_get_overridable_initializer_name()(
                self.inner.sess as *const sys::SessionHandle,
                index,
                alloc.alloc,
                &mut raw,
            )
        })?;
        if raw.is_null() {
            return Err(Error::new(
                -1,
                "zrt: overridable initializer name pointer is null",
            ));
        }
        let name = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        let _ = unsafe { alloc.free(raw as *mut c_void) };
        Ok(name)
    }

    /// The type info of the overridable initializer at `index`
    /// (`SessionGetOverridableInitializerTypeInfo`). Owning [`crate::RuntimeTypeInfo`]; released on drop.
    pub fn overridable_initializer_type_info(
        &self, index: usize,
    ) -> Result<crate::RuntimeTypeInfo> {
        let mut info: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe {
            api().session_get_overridable_initializer_type_info()(
                self.inner.sess as *const sys::SessionHandle,
                index,
                &mut info,
            )
        })?;
        let info = crate::ensure_non_null(info, "overridable initializer type info")?;
        Ok(unsafe { crate::RuntimeTypeInfo::from_owning(info) })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut sys::SessionHandle {
        self.inner.sess
    }

    #[inline]
    pub(crate) fn share_inner(&self) -> Arc<SessionInner> {
        self.inner.clone()
    }

    #[inline]
    pub(crate) fn shares_inner(&self, inner: &Arc<SessionInner>) -> bool {
        Arc::ptr_eq(&self.inner, inner)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn uses_cuda_stream(&self, stream: &Arc<crate::CudaStream>) -> bool {
        self.inner
            ._cuda_streams
            .iter()
            .any(|existing| Arc::ptr_eq(existing, stream))
    }

    #[inline]
    pub fn input_count(&self) -> usize {
        self.inner.input_ptrs.len()
    }
    #[inline]
    pub fn output_count(&self) -> usize {
        self.inner.output_ptrs.len()
    }
    pub fn input_name(&self, i: usize) -> Result<&str> {
        self.inner
            .input_names
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
        self.inner
            .output_names
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

    /// Update execution-provider options on the live session at run time
    /// (`SetEpDynamicOptions`, idx 284). `kv` is a key/value list of provider-specific runtime
    /// knobs. Applies to sessions whose EP supports dynamic option updates (e.g. some device EPs);
    /// CPU/unsupported EPs return an ORT error.
    pub fn set_ep_dynamic_options(&self, kv: &[(&str, &str)]) -> Result<()> {
        let keys: Vec<CString> = kv
            .iter()
            .map(|(k, _)| CString::new(*k))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::new(-1, "dynamic option key contains a NUL"))?;
        let vals: Vec<CString> = kv
            .iter()
            .map(|(_, v)| CString::new(*v))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::new(-1, "dynamic option value contains a NUL"))?;
        let k_ptrs: Vec<*const c_char> = keys.iter().map(|c| c.as_ptr()).collect();
        let v_ptrs: Vec<*const c_char> = vals.iter().map(|c| c.as_ptr()).collect();
        check(unsafe {
            api().set_ep_dynamic_options()(
                self.as_ptr(),
                k_ptrs.as_ptr(),
                v_ptrs.as_ptr(),
                kv.len(),
            )
        })
    }

    /// Cached (value kind, element type, static element count if concrete) for input `i`.
    #[inline]
    pub fn input_meta(&self, i: usize) -> Result<(sys::OnnxType, sys::ElementType, Option<usize>)> {
        let m = self.inner.input_meta.get(i).ok_or_else(|| {
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
        let m = self.inner.output_meta.get(i).ok_or_else(|| {
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
            .inner
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
            .inner
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
            .inner
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
            .inner
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

    /// ORT's planned memory descriptor for input `i`.
    pub fn input_memory_info(&self, i: usize) -> Result<MemoryInfoSnapshot> {
        if i >= self.input_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: input index {i} out of range ({} inputs)",
                    self.input_count()
                ),
            ));
        }
        Ok(self.input_memory_infos()?.remove(i))
    }

    /// ORT's planned memory descriptor for output `i`.
    pub fn output_memory_info(&self, i: usize) -> Result<MemoryInfoSnapshot> {
        if i >= self.output_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: output index {i} out of range ({} outputs)",
                    self.output_count()
                ),
            ));
        }
        Ok(self.output_memory_infos()?.remove(i))
    }

    /// ORT's assigned EP device for input `i`, if ORT reports one.
    pub fn input_ep_device(&self, i: usize) -> Result<Option<ExecutionProviderDeviceSnapshot>> {
        if i >= self.input_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: input index {i} out of range ({} inputs)",
                    self.input_count()
                ),
            ));
        }
        Ok(self.input_ep_devices()?.remove(i))
    }

    /// ORT's assigned EP device for output `i`, if ORT reports one.
    pub fn output_ep_device(&self, i: usize) -> Result<Option<ExecutionProviderDeviceSnapshot>> {
        if i >= self.output_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: output index {i} out of range ({} outputs)",
                    self.output_count()
                ),
            ));
        }
        Ok(self.output_ep_devices()?.remove(i))
    }

    /// Snapshot of model I/O metadata plus ORT's memory/EP placement decisions.
    ///
    /// Diagnostic/setup API; may allocate. Do not call from the measured serving loop.
    pub fn io_placement(&self) -> Result<Vec<IoPlacement>> {
        let input_memory = self.input_memory_infos()?;
        let output_memory = self.output_memory_infos()?;
        let input_ep = self.input_ep_devices()?;
        let output_ep = self.output_ep_devices()?;
        let mut out = Vec::with_capacity(self.input_count() + self.output_count());

        for i in 0..self.input_count() {
            let meta = &self.inner.input_meta[i];
            out.push(IoPlacement {
                direction: IoDirection::Input,
                index: i,
                name: self.input_name(i)?.to_owned(),
                onnx_type: meta.onnx_type,
                element_type: meta.elem_type,
                shape: meta.dims.clone(),
                symbolic_dims: meta.symbolic.clone(),
                memory_info: input_memory[i].clone(),
                ep_device: input_ep[i].clone(),
            });
        }
        for i in 0..self.output_count() {
            let meta = &self.inner.output_meta[i];
            out.push(IoPlacement {
                direction: IoDirection::Output,
                index: i,
                name: self.output_name(i)?.to_owned(),
                onnx_type: meta.onnx_type,
                element_type: meta.elem_type,
                shape: meta.dims.clone(),
                symbolic_dims: meta.symbolic.clone(),
                memory_info: output_memory[i].clone(),
                ep_device: output_ep[i].clone(),
            });
        }
        Ok(out)
    }

    fn input_memory_infos(&self) -> Result<Vec<MemoryInfoSnapshot>> {
        let mut ptrs = vec![ptr::null(); self.input_count()];
        check(unsafe {
            api().session_get_memory_info_for_inputs()(
                self.inner.sess as *const sys::SessionHandle,
                ptrs.as_mut_ptr() as *const *const sys::MemoryInfoHandle,
                ptrs.len(),
            )
        })?;
        ptrs.into_iter()
            .map(crate::memory::snapshot_from_ptr)
            .collect()
    }

    fn output_memory_infos(&self) -> Result<Vec<MemoryInfoSnapshot>> {
        let mut ptrs = vec![ptr::null(); self.output_count()];
        check(unsafe {
            api().session_get_memory_info_for_outputs()(
                self.inner.sess as *const sys::SessionHandle,
                ptrs.as_mut_ptr() as *const *const sys::MemoryInfoHandle,
                ptrs.len(),
            )
        })?;
        ptrs.into_iter()
            .map(crate::memory::snapshot_from_ptr)
            .collect()
    }

    fn input_ep_devices(&self) -> Result<Vec<Option<ExecutionProviderDeviceSnapshot>>> {
        let mut ptrs = vec![ptr::null(); self.input_count()];
        check(unsafe {
            api().session_get_ep_device_for_inputs()(
                self.inner.sess as *const sys::SessionHandle,
                ptrs.as_mut_ptr() as *const *const sys::EpDeviceHandle,
                ptrs.len(),
            )
        })?;
        ptrs.into_iter().map(ep_device_snapshot_from_ptr).collect()
    }

    fn output_ep_devices(&self) -> Result<Vec<Option<ExecutionProviderDeviceSnapshot>>> {
        let mut ptrs = vec![ptr::null(); self.output_count()];
        check(unsafe {
            api().session_get_ep_device_for_outputs()(
                self.inner.sess as *const sys::SessionHandle,
                ptrs.as_mut_ptr() as *const *const sys::EpDeviceHandle,
                ptrs.len(),
            )
        })?;
        ptrs.into_iter().map(ep_device_snapshot_from_ptr).collect()
    }

    /// Run inference with the session's default (reused) `MaterializedRunOptions`. `inputs` must be in
    /// session-input order (any mix of numeric [`crate::TensorView`] and [`crate::StringTensor`]);
    /// `outputs` receives one engine-owned value per session output. `run(&self)` is
    /// thread-safe; each call uses a transient output-handle array — the per-run cost we
    /// eliminate is MB-scale tensor allocation, not this handful of pointers.
    pub fn run(&self, inputs: &[&dyn RunInput], outputs: &mut [Option<OwnedValue>]) -> Result<()> {
        self.run_impl(inputs, outputs, self.inner.run_opts.as_ptr())
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

    /// Fixed-arity variant of [`Self::prepare_io_binding_buffers`].
    ///
    /// This is the direct caller-owned-output migration path for code that currently uses
    /// [`Self::run`] but already has stable input and output tensors. For pooled serving lanes,
    /// prefer [`Self::prepare_tensor_io_lane`] or [`crate::StaticIoRuntime`].
    pub fn prepare_io_binding_buffer_array<
        's,
        'v,
        T: TensorElement,
        const INPUTS: usize,
        const OUTPUTS: usize,
    >(
        &'s self, inputs: [&'v dyn RunInput; INPUTS], outputs: [&'v TensorBuffer<T>; OUTPUTS],
    ) -> Result<PreparedIoBinding<'s, 'v>> {
        self.check_input_count(INPUTS)?;
        self.check_output_count(OUTPUTS, "output count")?;
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
            BufferSpec::AUTO,
        )
    }

    /// Build one borrowed-session lane with an explicit caller-owned buffer policy.
    pub fn prepare_tensor_io_lane_with_buffer_policy<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: BufferSpec,
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
    /// Inputs use [`BufferSpec::AUTO`]. Outputs are allocated as concrete ORT tensors
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
            BufferSpec::AUTO,
        )
    }

    /// Build one ORT-allocated-output lane with an explicit caller-owned input policy.
    pub fn prepare_allocated_output_tensor_io_lane_with_buffer_policy<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], input_policy: BufferSpec,
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
            BufferSpec::AUTO,
        )
    }

    /// Build one device-output lane with an explicit caller-owned input policy.
    pub fn prepare_device_output_tensor_io_lane_with_buffer_policy<T>(
        &self, input_mem: &MemoryInfo, output_mem: &MemoryInfo, input_shapes: &[&[i64]],
        input_policy: BufferSpec,
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
            BufferSpec::AUTO,
        )
    }

    /// Build one fixed-arity borrowed-session lane with an explicit caller-owned buffer policy.
    pub fn prepare_static_tensor_io_lane_with_buffer_policy<
        T,
        const INPUTS: usize,
        const OUTPUTS: usize,
    >(
        &self, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        policy: BufferSpec,
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
            BufferSpec::AUTO,
        )
    }

    /// Build a fixed set of independent borrowed-session lanes with an explicit buffer policy.
    pub fn prepare_tensor_io_lanes_with_buffer_policy<T>(
        &self, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize,
        policy: BufferSpec,
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

    /// Run inference with a caller-provided [`MaterializedRunOptions`] — per-call log level/tag/config
    /// entries, or to cancel via [`MaterializedRunOptions::terminate`] (share it as `Arc<MaterializedRunOptions>`
    /// with the cancelling thread). Otherwise identical to [`Self::run`].
    pub fn run_with(
        &self, inputs: &[&dyn RunInput], outputs: &mut [Option<OwnedValue>],
        opts: &MaterializedRunOptions,
    ) -> Result<()> {
        self.check_run_options(opts)?;
        self.run_impl(inputs, outputs, opts.as_ptr())
    }

    /// Fixed-capacity regular run for callers with compile-time I/O arity.
    ///
    /// This avoids the `Vec` fallback in [`Self::run`] even for models with more than the
    /// internal small-stack threshold. It still returns ORT-owned outputs; use IoBinding or
    /// lane APIs when caller-owned output buffers are required.
    pub fn run_array<const INPUTS: usize, const OUTPUTS: usize>(
        &self, inputs: [&dyn RunInput; INPUTS], outputs: &mut [Option<OwnedValue>; OUTPUTS],
    ) -> Result<()> {
        self.run_array_with(inputs, outputs, &self.inner.run_opts)
    }

    /// Fixed-capacity regular run with caller-provided run options.
    pub fn run_array_with<const INPUTS: usize, const OUTPUTS: usize>(
        &self, inputs: [&dyn RunInput; INPUTS], outputs: &mut [Option<OwnedValue>; OUTPUTS],
        opts: &MaterializedRunOptions,
    ) -> Result<()> {
        self.check_run_options(opts)?;
        self.check_input_count(INPUTS)?;
        self.check_output_count(OUTPUTS, "output slot count")?;
        let in_handles: [*const sys::ValueHandle; INPUTS] =
            std::array::from_fn(|i| inputs[i].as_value_ptr());
        let mut out_handles: [*mut sys::ValueHandle; OUTPUTS] =
            std::array::from_fn(|_| ptr::null_mut());
        self.run_raw(&in_handles, &mut out_handles, opts.as_ptr())?;
        self.stamp_outputs(&out_handles, outputs)
    }

    /// Copy an engine-owned value into an existing reusable tensor buffer via ORT `CopyTensors`.
    ///
    /// **Device tensors:** errors if `src` or `dst` is device-resident (the host-memcpy fallback is
    /// host-only); use `copy_value_to_tensor_buffer_on_stream` for device-resident tensors.
    pub fn copy_value_to_tensor_buffer<T: TensorElement>(
        &self, src: &OwnedValue, dst: &mut TensorBuffer<T>,
    ) -> Result<()> {
        self.copy_tensor_handles(
            &[src.value as *const sys::ValueHandle],
            &[dst.as_value_ptr()],
        )
    }

    /// Copy an engine-owned value into an existing ORT-allocated tensor via ORT `CopyTensors`.
    ///
    /// **Device tensors:** errors if `src` or `dst` is device-resident (the host-memcpy fallback is
    /// host-only); use an `_on_stream` copy for device-resident tensors.
    pub fn copy_value_to_allocated_tensor<T: TensorElement>(
        &self, src: &OwnedValue, dst: &mut AllocatedTensor<T>,
    ) -> Result<()> {
        self.copy_tensor_handles(
            &[src.value as *const sys::ValueHandle],
            &[dst.as_value_ptr()],
        )
    }

    /// Copy any ORT tensor input into an existing reusable tensor buffer via ORT `CopyTensors`.
    ///
    /// **Device tensors:** errors if `src` or `dst` is device-resident (the host-memcpy fallback is
    /// host-only); use `copy_input_to_tensor_buffer_on_stream` for device-resident tensors.
    pub fn copy_input_to_tensor_buffer<T: TensorElement>(
        &self, src: &dyn RunInput, dst: &mut TensorBuffer<T>,
    ) -> Result<()> {
        self.copy_tensor_handles(&[src.as_value_ptr()], &[dst.as_value_ptr()])
    }

    /// Copy any ORT tensor input into an existing ORT-allocated tensor via ORT `CopyTensors`.
    ///
    /// **Device tensors:** errors if `src` or `dst` is device-resident (the host-memcpy fallback is
    /// host-only); use an `_on_stream` copy for device-resident tensors.
    pub fn copy_input_to_allocated_tensor<T: TensorElement>(
        &self, src: &dyn RunInput, dst: &mut AllocatedTensor<T>,
    ) -> Result<()> {
        self.copy_tensor_handles(&[src.as_value_ptr()], &[dst.as_value_ptr()])
    }

    /// Copy an engine-owned value into a reusable tensor buffer **on a sync stream** via ORT
    /// `CopyTensors` (async, device-side). This is the cuda-graph input-refresh primitive: `dst` is
    /// the device-resident buffer the captured graph bakes, `src` is fresh host/device data, and the
    /// copy is sequenced on `stream` (typically the same stream the graph replays on) so the replay
    /// reads the new data. Unlike the synchronous [`Self::copy_value_to_tensor_buffer`], an on-stream
    /// copy must use the stream — there is no host fallback (a silent host `memcpy` would defeat the
    /// async sequencing). (feature `ep` — needs a [`crate::SyncStream`].)
    ///
    /// # Safety
    ///
    /// The copy may remain in flight after return. `src`, `dst`, and `stream` must remain alive and
    /// must not be mutated, read, or reused until the caller fences the stream/provider work.
    #[cfg(feature = "ep")]
    pub unsafe fn copy_value_to_tensor_buffer_on_stream<T: TensorElement>(
        &self, src: &OwnedValue, dst: &mut TensorBuffer<T>, stream: &crate::ep_device::SyncStream,
    ) -> Result<()> {
        self.copy_tensor_handles_on_stream(
            &[src.value as *const sys::ValueHandle],
            &[dst.as_value_ptr()],
            stream,
        )
    }

    /// Copy any ORT tensor input into a reusable tensor buffer **on a sync stream**
    /// (`CopyTensors` on `stream`). See [`Self::copy_value_to_tensor_buffer_on_stream`]. (feature `ep`.)
    ///
    /// # Safety
    ///
    /// The copy may remain in flight after return. `src`, `dst`, and `stream` must remain alive and
    /// must not be mutated, read, or reused until the caller fences the stream/provider work.
    #[cfg(feature = "ep")]
    pub unsafe fn copy_input_to_tensor_buffer_on_stream<T: TensorElement>(
        &self, src: &dyn RunInput, dst: &mut TensorBuffer<T>, stream: &crate::ep_device::SyncStream,
    ) -> Result<()> {
        self.copy_tensor_handles_on_stream(&[src.as_value_ptr()], &[dst.as_value_ptr()], stream)
    }

    /// `CopyTensors` sequenced on a sync stream — no host fallback (an on-stream copy must run on
    /// the stream; a synchronous host `memcpy` would break the async ordering the cuda-graph replay
    /// depends on). (feature `ep`.)
    #[cfg(feature = "ep")]
    fn copy_tensor_handles_on_stream(
        &self, src: &[*const sys::ValueHandle], dst: &[*const sys::ValueHandle],
        stream: &crate::ep_device::SyncStream,
    ) -> Result<()> {
        self.check_sync_stream(stream)?;
        if src.len() != dst.len() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: on-stream CopyTensors source/destination count mismatch: {} vs {}",
                    src.len(),
                    dst.len()
                ),
            ));
        }
        let mut dst_mut: Vec<*mut sys::ValueHandle> = dst
            .iter()
            .map(|&value| value as *mut sys::ValueHandle)
            .collect();
        check(unsafe {
            api().copy_tensors()(
                self.inner._env.as_ptr(),
                src.as_ptr(),
                dst_mut.as_mut_ptr(),
                stream.as_ptr(),
                src.len(),
            )
        })
    }

    fn copy_tensor_handles(
        &self, src: &[*const sys::ValueHandle], dst: &[*const sys::ValueHandle],
    ) -> Result<()> {
        if src.len() != dst.len() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: CopyTensors source/destination count mismatch: {} vs {}",
                    src.len(),
                    dst.len()
                ),
            ));
        }
        let mut dst_mut: Vec<*mut sys::ValueHandle> = dst
            .iter()
            .map(|&value| value as *mut sys::ValueHandle)
            .collect();
        let copy = check(unsafe {
            api().copy_tensors()(
                self.inner._env.as_ptr(),
                src.as_ptr(),
                dst_mut.as_mut_ptr(),
                ptr::null_mut(),
                src.len(),
            )
        });
        match copy {
            Ok(()) => Ok(()),
            Err(err) => match self.try_host_copy_tensor_handles(src, dst) {
                Ok(true) => Ok(()),
                Ok(false) => Err(err),
                Err(fallback_err) => Err(fallback_err),
            },
        }
    }

    fn try_host_copy_tensor_handles(
        &self, src: &[*const sys::ValueHandle], dst: &[*const sys::ValueHandle],
    ) -> Result<bool> {
        for (&src_value, &dst_value) in src.iter().zip(dst) {
            let src_info = tensor_memory_info(src_value)?;
            let dst_info = tensor_memory_info(dst_value)?;
            if !src_info.is_host_accessible() || !dst_info.is_host_accessible() {
                return Ok(false);
            }

            let mut src_bytes = 0usize;
            check(unsafe { api().get_tensor_size_in_bytes()(src_value, &mut src_bytes) })?;
            let mut dst_bytes = 0usize;
            check(unsafe { api().get_tensor_size_in_bytes()(dst_value, &mut dst_bytes) })?;
            if src_bytes != dst_bytes {
                return Err(Error::new(
                    -1,
                    format!(
                        "zrt: CopyTensors host fallback byte-size mismatch: source {src_bytes}, destination {dst_bytes}"
                    ),
                ));
            }

            let mut src_data: *const c_void = ptr::null();
            check(unsafe {
                api().get_tensor_data()(
                    src_value,
                    &mut src_data as *mut *const c_void as *const *const c_void,
                )
            })?;
            let mut dst_data: *mut c_void = ptr::null_mut();
            check(unsafe {
                api().get_tensor_mutable_data()(dst_value as *mut sys::ValueHandle, &mut dst_data)
            })?;
            let src_data =
                crate::slice_data_ptr(src_data as *mut u8, src_bytes, "CopyTensors source data")?;
            let dst_data = crate::slice_data_ptr(
                dst_data as *mut u8,
                dst_bytes,
                "CopyTensors destination data",
            )?;
            unsafe {
                ptr::copy_nonoverlapping(src_data as *const u8, dst_data, src_bytes);
            }
        }
        Ok(true)
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
                self.inner.sess,
                opts,
                self.inner.input_ptrs.as_ptr(),
                input_handles.as_ptr(),
                input_handles.len(),
                self.inner.output_ptrs.as_ptr(),
                self.inner.output_ptrs.len(),
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

    #[inline]
    fn check_binding_session(&self, binding: &IoBinding) -> Result<()> {
        if binding.belongs_to(self) {
            Ok(())
        } else {
            Err(Error::local("IoBinding belongs to a different Session"))
        }
    }

    #[inline]
    fn check_run_options(&self, opts: &MaterializedRunOptions) -> Result<()> {
        if opts.shares_environment(&self.inner._env) {
            Ok(())
        } else {
            Err(Error::local(
                "MaterializedRunOptions sync stream belongs to a different Environment",
            ))
        }
    }

    #[cfg(feature = "ep")]
    pub(crate) fn check_sync_stream(&self, stream: &crate::SyncStream) -> Result<()> {
        if stream.shares_env_guard(&self.inner._env) {
            Ok(())
        } else {
            Err(Error::local(
                "SyncStream belongs to a different Environment",
            ))
        }
    }

    fn stamp_outputs(
        &self, handles: &[*mut sys::ValueHandle], outputs: &mut [Option<OwnedValue>],
    ) -> Result<()> {
        for i in 0..handles.len() {
            let h = handles[i];
            let m = &self.inner.output_meta[i];
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
                memory_class: std::sync::OnceLock::new(),
            });
        }
        Ok(())
    }

    /// Run with an [`crate::IoBinding`]. Inputs/outputs are taken from the binding (bound by
    /// name), bypassing the per-run name arrays and — for caller-buffer outputs — the per-run
    /// output allocation. Thread-safe like [`Self::run`]; reuses the session's `MaterializedRunOptions`.
    pub fn run_binding(&self, binding: &crate::io_binding::IoBinding) -> Result<()> {
        self.check_binding_session(binding)?;
        binding.synchronize_inputs()?;
        check(unsafe {
            api().run_with_binding()(
                self.inner.sess,
                self.inner.run_opts.as_ptr(),
                binding.as_ptr(),
            )
        })?;
        binding.synchronize_outputs()
    }

    /// Run with an [`crate::IoBinding`] without calling ORT's bound-input/output synchronization
    /// helpers before and after the run.
    ///
    /// This is useful for fully host-resident lanes, or for advanced callers that synchronize
    /// device streams externally. Prefer [`Self::run_binding`] unless the binding's memory
    /// placement and synchronization contract are known up front.
    ///
    /// # Safety
    /// The caller must ensure all bound buffers remain alive and unmodified until every provider
    /// operation using them has completed, and must perform any synchronization required before
    /// reading outputs or releasing provider-visible resources.
    pub unsafe fn run_binding_unsynchronized(
        &self, binding: &crate::io_binding::IoBinding,
    ) -> Result<()> {
        self.check_binding_session(binding)?;
        check(unsafe {
            api().run_with_binding()(
                self.inner.sess,
                self.inner.run_opts.as_ptr(),
                binding.as_ptr(),
            )
        })
    }

    /// Run with an [`crate::IoBinding`] and a caller-provided [`MaterializedRunOptions`] (per-call config
    /// or cancellation). See [`Self::run_with`] / [`Self::run_binding`].
    pub fn run_binding_with(
        &self, binding: &crate::io_binding::IoBinding, opts: &MaterializedRunOptions,
    ) -> Result<()> {
        self.check_binding_session(binding)?;
        self.check_run_options(opts)?;
        binding.synchronize_inputs()?;
        check(unsafe {
            api().run_with_binding()(self.inner.sess, opts.as_ptr(), binding.as_ptr())
        })?;
        binding.synchronize_outputs()
    }

    /// Run with a caller-provided [`MaterializedRunOptions`] without ORT bound-input/output synchronization.
    ///
    /// See [`Self::run_binding_unsynchronized`] for the synchronization contract.
    ///
    /// # Safety
    /// The caller must uphold the binding lifetime and provider synchronization contract described
    /// by [`Self::run_binding_unsynchronized`].
    pub unsafe fn run_binding_unsynchronized_with(
        &self, binding: &crate::io_binding::IoBinding, opts: &MaterializedRunOptions,
    ) -> Result<()> {
        self.check_binding_session(binding)?;
        self.check_run_options(opts)?;
        check(unsafe { api().run_with_binding()(self.inner.sess, opts.as_ptr(), binding.as_ptr()) })
    }

    /// Run the model asynchronously (`RunAsync`, IDX 260) on an ORT worker thread. Returns a
    /// [`RunFuture`] that resolves to the outputs — pollable by any executor (no async-runtime
    /// dependency). `RunAsync` only errors synchronously if it fails to *start*.
    ///
    /// **Borrow hazard:** the future borrows the session and inputs (`'a`), and ORT's worker thread
    /// keeps reading those inputs until the run's callback fires. Keep the session and every input's
    /// backing memory alive and unmutated until the future resolves — dropping or mutating any of
    /// them first is undefined behavior. The `'a` borrow only enforces this while the `RunFuture<'a>`
    /// exists; dropping the future early is still the caller's hazard (the in-flight run continues).
    /// For 'static / cross-thread use, or to eliminate the hazard entirely, use
    /// [`Self::run_async_owned_inputs`], which moves the inputs into the run state.
    pub fn run_async<'a>(&'a self, inputs: &'a [&'a dyn RunInput]) -> Result<RunFuture<'a>> {
        let in_handles: Vec<*const sys::ValueHandle> =
            inputs.iter().map(|v| v.as_value_ptr()).collect();
        self.run_async_owned(in_handles, None)
    }

    /// Asynchronous run that takes **owned** input values (`RunAsync`, IDX 260) — the no-borrow-hazard
    /// variant of [`Self::run_async`].
    ///
    /// The inputs are moved into the run state, so the returned [`RunFuture`] borrows **only `&self`**
    /// (the session). Unlike [`Self::run_async`], there is no caller hazard: the input `OrtValue`s
    /// cannot be dropped before the run completes because the state owns them for its lifetime. Use
    /// this for 'static / cross-thread use, or anywhere you would otherwise have to carefully keep
    /// borrowed inputs alive across the future.
    ///
    /// The input count must match the session's input count.
    pub fn run_async_owned_inputs(&self, inputs: Vec<OwnedValue>) -> Result<RunFuture<'_>> {
        let in_handles: Vec<*const sys::ValueHandle> = inputs
            .iter()
            .map(|v| v.value as *const sys::ValueHandle)
            .collect();
        let owned: Box<[OwnedValue]> = inputs.into_boxed_slice();
        self.run_async_owned(in_handles, Some(owned))
    }

    /// Async-run core shared by [`Self::run_async`] (borrowed inputs),
    /// [`Self::run_async_owned_inputs`] (owned inputs), and the serving lanes
    /// ([`crate::ServingLane::run_async`], [`crate::Lane::run_async`]). Takes pre-extracted input
    /// value handles by value — they are moved into the run state, so no borrowed input slice has to
    /// outlive the call.
    ///
    /// When `owned_inputs` is `Some`, those owning `OrtValue`s are held in the state for the run's
    /// lifetime (the owned-input path). When `None`, the input values are caller-owned for `'a` (the
    /// borrowed/lane paths) — the caller MUST keep them alive until the future resolves.
    pub(crate) fn run_async_owned(
        &self, in_handles: Vec<*const sys::ValueHandle>, owned_inputs: Option<Box<[OwnedValue]>>,
    ) -> Result<RunFuture<'_>> {
        self.check_input_count(in_handles.len())?;

        let n = self.output_count();
        // The input-handle array and the output-handle array live in the `Arc` state so they
        // outlive this call: ORT's worker thread reads the inputs and fills the outputs
        // asynchronously, after `RunAsync` returns. Both are freed when the state's last ref
        // drops (the callback's + the future's).
        let input_count = in_handles.len();
        let in_handles: Box<[*const sys::ValueHandle]> = in_handles.into_boxed_slice();
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
            _owned_inputs: owned_inputs,
        });
        // Hand the state to the callback as `user_data` (one ref via `into_raw`; the future
        // keeps another). The callback recovers + drops its ref.
        let user_data = Arc::into_raw(state.clone()) as *mut c_void;

        let started = check(unsafe {
            api().run_async()(
                self.inner.sess,
                self.inner.run_opts.as_ptr(),
                self.inner.input_ptrs.as_ptr(),
                in_ptr,
                input_count,
                self.inner.output_ptrs.as_ptr(),
                self.inner.output_ptrs.len(),
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
            self.session.inner.run_opts.as_ptr(),
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
        prime_runs(|| self.run().map(|_| ()), runs)
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
        prime_runs(|| self.run(), runs)
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
        prime_runs(|| self.run(), runs)
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    ///
    /// The stats calls are diagnostic and may allocate. Use this outside latency-critical
    /// measurements to understand allocator behavior around an otherwise hot-path run.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        run_with_allocator_stats(allocator, || self.run())
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
        prime_runs(|| self.run(), runs)
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        run_with_allocator_stats(allocator, || self.run())
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
        prime_runs(|| self.run().map(|_| ()), runs)
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        run_with_allocator_stats(allocator, || self.run().map(|_| ()))
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
        prime_runs(|| self.run(), runs)
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        run_with_allocator_stats(allocator, || self.run())
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
        prime_runs(|| self.run(), runs)
    }

    /// Execute this lane while taking ORT allocator stat snapshots before and after.
    ///
    /// The stats calls are diagnostic and may allocate. Use this outside latency-critical
    /// measurements to understand allocator behavior around an otherwise hot-path run.
    pub fn run_with_allocator_stats(
        &mut self, allocator: &Allocator,
    ) -> Result<LaneRunAllocatorStats> {
        run_with_allocator_stats(allocator, || self.run())
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

impl Drop for SessionInner {
    fn drop(&mut self) {
        // Release the native session while initializers, prepacked weights, and the Env guard are
        // still alive. Rust drops those fields only after this method returns.
        unsafe {
            if !self.sess.is_null() {
                api().release_session()(self.sess);
            }
        }
    }
}
unsafe impl Send for SessionInner {}
unsafe impl Sync for SessionInner {}

impl Drop for CapturedGraphRunGuard {
    fn drop(&mut self) {
        let mut previous = self.lease.0.state.load(Ordering::Acquire);
        loop {
            if previous & GRAPH_LEASE_ACTIVE_MASK == 0 {
                eprintln!(
                    "st-zrt: captured-graph run guard found zero active-run accounting; preserving lease state"
                );
                return;
            }
            match self.lease.0.state.compare_exchange_weak(
                previous,
                previous - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => previous = observed,
            }
        }
        if previous & GRAPH_LEASE_RELEASING != 0 && previous & GRAPH_LEASE_ACTIVE_MASK == 1 {
            let _wait = self
                .lease
                .0
                .wait_lock
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.lease.0.wait_cv.notify_all();
        }
    }
}

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
        Self::from_handle(sess, env.share(), cuda_stream_guards(&opts))
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
                self.inner.sess as *const sys::SessionHandle,
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
        Self::from_handle(sess, env.share(), cuda_stream_guards(&opts))
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
        check(unsafe { apply(self.inner.sess, model.as_ptr() as *mut sys::ModelHandle) })
    }

    /// Finalize a model-editor session after any [`Self::apply_model`]
    /// (`FinalizeModelEditorSession`) — validates + prepares it for inference.
    ///
    /// Finalization mutates the native session and replaces cached I/O metadata, so this requires
    /// unique ownership of the shared session inner. Drop all `Session` clones, session allocators,
    /// allocated tensors, and I/O bindings first; otherwise this returns an error before calling ORT.
    pub fn finalize(&mut self, opts: &SessionOptions) -> Result<()> {
        if Arc::get_mut(&mut self.inner).is_none() {
            return Err(Error::local(
                "cannot finalize a model-editor session while Session clones or session-owned resources exist",
            ));
        }
        let me = crate::model_editor::model_editor_api()
            .ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))?;
        let finalize = crate::model_editor::require_sub_api_fn(
            me.FinalizeModelEditorSession,
            "ModelEditorApi",
            "FinalizeModelEditorSession",
        )?;
        #[cfg(feature = "cuda")]
        {
            let new_guards = cuda_stream_guards(opts);
            let inner = Arc::get_mut(&mut self.inner).expect("unique ownership checked above");
            for stream in new_guards {
                if !inner
                    ._cuda_streams
                    .iter()
                    .any(|existing| Arc::ptr_eq(existing, &stream))
                {
                    inner._cuda_streams.push(stream);
                }
            }
        }
        let opts_handle = opts.build_handle_for_session()?;
        // No EP-device attach here: `finalize` has no `env`, and any queued device attach was
        // already applied in the `from_bytes_for_editing` constructor that created this session.
        let r = check(unsafe {
            finalize(
                self.inner.sess,
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
    /// returns, so both must outlive the call. The input `OrtValue`s are owned separately by the
    /// caller for `'a` on the borrowed path; the owned-input path (`run_async_owned_inputs`) moves
    /// them into `_owned_inputs` so they live exactly as long as this state.
    _in_handles: Box<[*const sys::ValueHandle]>,
    _out_handles: Box<[*mut sys::ValueHandle]>,
    /// Owned input values, held for the run's lifetime on the owned-input path (`None` on the
    /// borrowed/lane paths, where inputs are caller-owned for `'a`).
    _owned_inputs: Option<Box<[OwnedValue]>>,
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

impl std::fmt::Debug for RunFuture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunFuture")
            .field("done", &self.state.done.load(Ordering::Acquire))
            .field("strong_count", &Arc::strong_count(&self.state))
            .finish()
    }
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

/// Batch-attach caller-owned initializer Ort values to a session-options handle
/// (`AddExternalInitializers`, the batch equivalent of the per-item `AddInitializer`). The
/// values must outlive the session — the caller keeps them alive via `_owned_initializers`.
fn add_external_initializers_batch(
    opts: *mut sys::SessionOptionsHandle, initializers: &[OwnedInitializer],
) -> Result<()> {
    if initializers.is_empty() {
        return Ok(());
    }
    let names: Vec<*const c_char> = initializers.iter().map(|i| i.name_ptr()).collect();
    let values: Vec<*const sys::ValueHandle> = initializers.iter().map(|i| i.value_ptr()).collect();
    check(unsafe {
        api().add_external_initializers()(opts, names.as_ptr(), values.as_ptr(), initializers.len())
    })
}

/// Provide the content of external-data initializer files from memory
/// (`AddExternalInitializersFromFilesInMemory`). Each entry is `(file_name, file_bytes)` where
/// `file_name` matches the external-data location the model references. ORT reads the buffers
/// during session creation and does not retain them afterwards.
fn add_external_initializer_files_in_memory(
    opts: *mut sys::SessionOptionsHandle, files: &[(CString, Vec<u8>)],
) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    let names: Vec<*const c_char> = files.iter().map(|(n, _)| n.as_ptr()).collect();
    // ORT's signature takes mutable buffer pointers (legacy char**); it reads, never mutates.
    let mut bufs: Vec<*mut c_char> = files
        .iter()
        .map(|(_, b)| b.as_ptr() as *mut c_char)
        .collect();
    let lengths: Vec<usize> = files.iter().map(|(_, b)| b.len()).collect();
    check(unsafe {
        api().add_external_initializers_from_files_in_memory()(
            opts,
            names.as_ptr(),
            bufs.as_mut_ptr(),
            lengths.as_ptr(),
            files.len(),
        )
    })
}

fn build_session_options_for_env(
    env: &Environment, opts: &SessionOptions,
) -> Result<*mut sys::SessionOptionsHandle> {
    #[cfg(feature = "ep")]
    opts.validate_cuda_stream_guards()?;
    let opts_handle = opts.build_handle_for_session()?;
    let result = (|| {
        apply_ep_device_attach(env, opts_handle, opts)?;
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

fn apply_ep_device_attach(
    env: &Environment, opts_handle: *mut sys::SessionOptionsHandle, opts: &SessionOptions,
) -> Result<()> {
    // Apply-only: never release `opts_handle` here. The caller
    // (`build_session_options_for_env`) is the sole owner and releases on any error.
    #[cfg(feature = "ep")]
    {
        crate::ep_device::apply_device_attach(env, opts_handle, &opts.ep_device_attach)?;
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
    use super::{
        CapturedGraphLease, CapturedGraphRunGuard, GRAPH_LEASE_ACTIVE_MASK, GRAPH_LEASE_RELEASING,
    };
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    #[test]
    fn graph_lease_release_waits_for_all_runs_and_blocks_new_runs() {
        let lease = CapturedGraphLease::new();
        let run_a = lease.begin_run();
        let run_b = lease.begin_run();
        let release_started = Arc::new(Barrier::new(2));
        let release_finished = Arc::new(Barrier::new(2));
        let worker_lease = lease.clone();
        let started = Arc::clone(&release_started);
        let finished = Arc::clone(&release_finished);
        let worker = std::thread::spawn(move || {
            started.wait();
            worker_lease.begin_release();
            finished.wait();
            worker_lease.end_release();
        });
        release_started.wait();
        while lease.0.state.load(Ordering::Acquire) & GRAPH_LEASE_RELEASING == 0 {
            std::hint::spin_loop();
        }
        assert_eq!(
            lease.0.state.load(Ordering::Acquire) & GRAPH_LEASE_ACTIVE_MASK,
            2
        );
        drop(run_a);
        drop(run_b);
        release_finished.wait();
        worker.join().expect("release worker");
        drop(lease.begin_run());
    }

    #[test]
    fn graph_leases_for_different_ids_do_not_couple() {
        let blocked = CapturedGraphLease::new();
        let independent = CapturedGraphLease::new();
        let active = blocked.begin_run();
        let release_lease = blocked.clone();
        let worker = std::thread::spawn(move || {
            release_lease.begin_release();
            release_lease.end_release();
        });
        while blocked.0.state.load(Ordering::Acquire) & GRAPH_LEASE_RELEASING == 0 {
            std::hint::spin_loop();
        }
        let independent_run = independent.begin_run();
        assert_eq!(
            independent.0.state.load(Ordering::Acquire) & GRAPH_LEASE_ACTIVE_MASK,
            1
        );
        drop(independent_run);
        drop(active);
        worker.join().expect("release worker");
    }

    #[test]
    fn corrupted_graph_run_guard_drop_does_not_underflow_or_panic() {
        let lease = CapturedGraphLease::new();
        drop(CapturedGraphRunGuard {
            lease: lease.clone(),
        });
        assert_eq!(lease.0.state.load(Ordering::Acquire), 0);
    }

    #[test]
    fn graph_lease_contended_runs_drain_without_underflow() {
        let lease = CapturedGraphLease::new();
        let mut workers = Vec::new();
        for _ in 0..8 {
            let lease = lease.clone();
            workers.push(std::thread::spawn(move || {
                for _ in 0..1_000 {
                    let guard = lease.begin_run();
                    std::hint::black_box(&guard);
                }
            }));
        }
        for worker in workers {
            worker.join().expect("run worker");
        }
        assert_eq!(lease.0.state.load(Ordering::Acquire), 0);
        lease.begin_release();
        std::thread::sleep(Duration::from_millis(1));
        lease.end_release();
        assert_eq!(lease.0.state.load(Ordering::Acquire), 0);
    }
}
