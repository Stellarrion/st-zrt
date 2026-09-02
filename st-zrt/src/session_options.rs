//! `SessionOptions` — pure-value session configuration.
//!
//! This is a plain config struct: builder methods only set fields (infallible — no FFI).
//! The ORT `SessionOptions` handle is materialized once, inside [`crate::Session::new`],
//! via [`SessionOptions::build_handle`]. This is the foundation for a future auto
//! thread-policy (the config can carry a policy before any handle exists).
use crate::environment::LogRecord;
use crate::{Result, api, check, sys};
use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::Arc;

#[cfg(feature = "serde")]
fn null_opaque_ptr() -> *mut c_void {
    std::ptr::null_mut()
}

/// State of the CPU memory arena (BFCArena). Disabling it avoids the arena's global
/// mutex + page-fault-dominated allocation for large tensors (anti-pattern E1/E2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ArenaState {
    /// ORT default (arena enabled).
    #[default]
    Default,
    /// Explicitly enable the CPU arena.
    Enabled,
    /// Arena disabled — use the OS allocator.
    Disabled,
}

/// State of the memory-pattern optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemPatternState {
    /// ORT default (memory pattern enabled where ORT can use it).
    #[default]
    Default,
    /// Explicitly enable memory-pattern optimization.
    Enabled,
    /// Disable memory-pattern optimization.
    Disabled,
}

/// EP-selection delegate — a raw C callback ORT invokes to pick which [`crate::EpDevice`]s run a
/// graph (feature `ep`, `SessionOptionsSetEpSelectionPolicyDelegate`). Receives the candidate
/// devices, EP metadata/options, and writes back the chosen device list. Expert/raw: the caller
/// provides a correctly-typed C function pointer and an opaque `state` it will receive.
#[cfg(feature = "ep")]
pub type EpSelectionDelegate = Option<
    unsafe extern "C" fn(
        *const *const sys::EpDeviceHandle,
        usize,
        *const sys::KeyValuePairsHandle,
        *const sys::KeyValuePairsHandle,
        *mut *const sys::EpDeviceHandle,
        usize,
        *mut usize,
        *mut c_void,
    ) -> sys::StatusPtr,
>;

/// Pure-value session configuration. Cloning is cheap (no handles). Consumed by
/// [`crate::Session::new`].
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SessionOptions {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::graph_opt"))]
    pub(crate) opt_level: sys::GraphOptimizationLevel,
    pub(crate) intra_threads: Option<i32>,
    pub(crate) inter_threads: Option<i32>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) execution_mode: Option<sys::ExecutionMode>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    pub(crate) log_id: Option<CString>,
    pub(crate) log_severity: Option<i32>,
    pub(crate) log_verbosity: Option<i32>,
    /// Optional per-session user logging callback (`SetUserLoggingFunction`). Applied in
    /// `build_handle`; the closure is leaked (one `Arc` ref via `Arc::into_raw`) because ORT retains
    /// the pointer for the session's lifetime and there is no unregister call. Not serializable.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) user_logger: Option<Arc<crate::environment::LoggerSlot>>,
    pub(crate) cpu_mem_arena: ArenaState,
    pub(crate) mem_pattern: MemPatternState,
    pub(crate) use_global_thread_pool: bool,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) profiling_prefix: Option<CString>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::kv_i64_pairs"))]
    pub(crate) free_dimension_overrides: Vec<(CString, i64)>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::kv_i64_pairs"))]
    pub(crate) free_dimension_overrides_by_name: Vec<(CString, i64)>,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::kv_pairs"))]
    pub(crate) config_entries: Vec<(CString, CString)>,
    /// Automatic execution-provider selection policy (feature `ep`). Applied in `build_handle`.
    #[cfg(feature = "ep")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) ep_selection_policy: Option<sys::ExecutionProviderDevicePolicy>,
    /// Queued execution-provider appends (feature `ep`). Applied in `build_handle`.
    #[cfg(feature = "ep")]
    pub(crate) ep_configs: Vec<crate::ep::EpConfig>,
    /// Queued MIGraphX config (feature `ep`) — a flat-struct EP with its own builder. Applied
    /// in `build_handle`.
    #[cfg(feature = "ep")]
    pub(crate) migraphx: Vec<crate::ep::MigraphxOptions>,
    /// Queued deprecated OpenVINO v1 config (feature `ep`) — the other flat-struct EP. Applied
    /// in `build_handle`. Prefer `ep_configs` with [`crate::ep::EpProvider::OpenVinoV2`].
    #[cfg(feature = "ep")]
    pub(crate) openvino: Vec<crate::ep::OpenvinoOptions>,
    /// Queued EP-device attaches (feature `ep`) — discovered via [`crate::get_ep_devices`];
    /// applied in the session constructors via [`crate::ep_device::apply_device_attach`] (the V2
    /// attach call needs the environment, which `build_handle` doesn't take). Discovered devices
    /// retain their originating environment and are identity-checked at construction. Not
    /// serializable — skipped under `serde`.
    #[cfg(feature = "ep")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) ep_device_attach: Vec<crate::ep_device::EpDeviceAttach>,
    /// Queued custom-op domains (feature `custom-ops`). Applied in `build_handle`. These are
    /// borrowed pointers — the referenced `CustomOpDomain`s must outlive every session built
    /// from these options (an ORT invariant). **Not serializable** (runtime handles) — skipped
    /// under `serde`.
    #[cfg(feature = "custom-ops")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_op_domains: Vec<*mut sys::CustomOpDomainHandle>,
    /// Enable ORT's built-in contrib ops (`EnableOrtCustomOps`, feature `custom-ops`).
    #[cfg(feature = "custom-ops")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) enable_ort_custom_ops: bool,
    /// Shared-library custom-op paths to register via the v2 call (feature `custom-ops`).
    #[cfg(feature = "custom-ops")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_op_libraries: Vec<CString>,
    /// Same as `custom_op_libraries` but via the legacy v1 call (returns a dlopen handle).
    #[cfg(feature = "custom-ops")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_op_libraries_v1: Vec<CString>,
    /// In-process registration-function symbols to call (feature `custom-ops`).
    #[cfg(feature = "custom-ops")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_op_functions: Vec<CString>,
    /// Optimized-model output path (`SetOptimizedModelFilePath`). Applied in `build_handle`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) optimized_model_file_path: Option<CString>,
    /// Deterministic-compute toggle (`SetDeterministicCompute`). Applied in `build_handle`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) deterministic_compute: Option<bool>,
    /// Load-cancellation flag (`SessionOptionsSetLoadCancellationFlag`). Applied in `build_handle`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) load_cancellation_flag: Option<bool>,
    /// Optional custom thread-creation callback trio (`SessionOptionsSetCustomCreateThreadFn` +
    /// `…JoinThreadFn` + `…ThreadCreationOptions`). Expert/raw — C function pointers and an
    /// opaque state passed to the create callback. Applied together in `build_handle`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_create_thread_fn: Option<sys::CustomCreateThreadFnHandle>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) custom_join_thread_fn: Option<sys::CustomJoinThreadFnHandle>,
    #[cfg_attr(feature = "serde", serde(skip, default = "null_opaque_ptr"))]
    pub(crate) custom_thread_creation_options: *mut c_void,
    /// Optional EP-selection delegate + opaque state (feature `ep`,
    /// `SessionOptionsSetEpSelectionPolicyDelegate`). Expert/raw. Applied in `build_handle`.
    #[cfg(feature = "ep")]
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) ep_selection_delegate: Option<EpSelectionDelegate>,
    #[cfg(feature = "ep")]
    #[cfg_attr(feature = "serde", serde(skip, default = "null_opaque_ptr"))]
    pub(crate) ep_selection_delegate_state: *mut c_void,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            opt_level: sys::GraphOptimizationLevel::All,
            intra_threads: None,
            inter_threads: None,
            execution_mode: None,
            log_id: None,
            log_severity: None,
            log_verbosity: None,
            user_logger: None,
            cpu_mem_arena: ArenaState::Default,
            mem_pattern: MemPatternState::Default,
            use_global_thread_pool: true,
            profiling_prefix: None,
            free_dimension_overrides: Vec::new(),
            free_dimension_overrides_by_name: Vec::new(),
            config_entries: Vec::new(),
            #[cfg(feature = "ep")]
            ep_selection_policy: None,
            #[cfg(feature = "ep")]
            ep_configs: Vec::new(),
            #[cfg(feature = "ep")]
            migraphx: Vec::new(),
            #[cfg(feature = "ep")]
            openvino: Vec::new(),
            #[cfg(feature = "ep")]
            ep_device_attach: Vec::new(),
            #[cfg(feature = "custom-ops")]
            custom_op_domains: Vec::new(),
            #[cfg(feature = "custom-ops")]
            enable_ort_custom_ops: false,
            #[cfg(feature = "custom-ops")]
            custom_op_libraries: Vec::new(),
            #[cfg(feature = "custom-ops")]
            custom_op_libraries_v1: Vec::new(),
            #[cfg(feature = "custom-ops")]
            custom_op_functions: Vec::new(),
            optimized_model_file_path: None,
            deterministic_compute: None,
            load_cancellation_flag: None,
            custom_create_thread_fn: None,
            custom_join_thread_fn: None,
            custom_thread_creation_options: ptr::null_mut(),
            #[cfg(feature = "ep")]
            ep_selection_delegate: None,
            #[cfg(feature = "ep")]
            ep_selection_delegate_state: ptr::null_mut(),
        }
    }
}

impl SessionOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Graph optimization level (ORT default is `All`; set explicitly here).
    #[inline]
    pub fn with_opt_level(mut self, level: sys::GraphOptimizationLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Intra-op thread count (parallelism within a node).
    #[inline]
    pub fn with_intra_threads(mut self, n: i32) -> Self {
        self.intra_threads = Some(n);
        self
    }

    /// Inter-op thread count (parallelism across nodes, parallel execution mode).
    #[inline]
    pub fn with_inter_threads(mut self, n: i32) -> Self {
        self.inter_threads = Some(n);
        self
    }

    /// Graph execution mode.
    ///
    /// `Sequential` is ORT's default and usually best for single-chain graphs. `Parallel`
    /// enables inter-op scheduling and can help graphs with independent branches when paired
    /// with [`Self::with_inter_threads`].
    #[inline]
    pub fn with_execution_mode(mut self, mode: sys::ExecutionMode) -> Self {
        self.execution_mode = Some(mode);
        self
    }

    /// Use ORT's sequential graph execution mode.
    #[inline]
    pub fn with_sequential_execution(self) -> Self {
        self.with_execution_mode(sys::ExecutionMode::Sequential)
    }

    /// Use ORT's parallel graph execution mode.
    #[inline]
    pub fn with_parallel_execution(self) -> Self {
        self.with_execution_mode(sys::ExecutionMode::Parallel)
    }

    /// Set the ORT session log id (`SetSessionLogId`).
    pub fn with_log_id(mut self, id: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.log_id = Some(CString::new(id)?);
        Ok(self)
    }

    /// Set the ORT session log severity (`SetSessionLogSeverityLevel`).
    ///
    /// `Verbose` is useful when diagnosing execution-provider placement and inserted Memcpy
    /// nodes during session creation.
    #[inline]
    pub fn with_log_severity(mut self, level: sys::LoggingLevel) -> Self {
        self.log_severity = Some(level as i32);
        self
    }

    /// Set the ORT session log verbosity (`SetSessionLogVerbosityLevel`).
    #[inline]
    pub fn with_log_verbosity(mut self, level: i32) -> Self {
        self.log_verbosity = Some(level);
        self
    }

    /// Use the environment's global thread pool when the environment was created with one.
    ///
    /// This is the default. ZRT applies ORT's `DisablePerSessionThreads` during session
    /// construction when a global pool is present.
    #[inline]
    pub fn use_global_thread_pool(mut self) -> Self {
        self.use_global_thread_pool = true;
        self
    }

    /// Opt this session out of an environment-level global thread pool.
    #[inline]
    pub fn use_per_session_threads(mut self) -> Self {
        self.use_global_thread_pool = false;
        self
    }

    /// Set the CPU memory arena state explicitly.
    #[inline]
    pub fn with_cpu_mem_arena(mut self, state: ArenaState) -> Self {
        self.cpu_mem_arena = state;
        self
    }

    /// Explicitly enable the CPU memory arena.
    #[inline]
    pub fn enable_cpu_mem_arena(mut self) -> Self {
        self.cpu_mem_arena = ArenaState::Enabled;
        self
    }

    /// Disable the CPU memory arena.
    #[inline]
    pub fn disable_cpu_mem_arena(mut self) -> Self {
        self.cpu_mem_arena = ArenaState::Disabled;
        self
    }

    /// Set the memory-pattern optimization state explicitly.
    #[inline]
    pub fn with_mem_pattern(mut self, state: MemPatternState) -> Self {
        self.mem_pattern = state;
        self
    }

    /// Explicitly enable the memory-pattern optimization.
    #[inline]
    pub fn enable_mem_pattern(mut self) -> Self {
        self.mem_pattern = MemPatternState::Enabled;
        self
    }

    /// Disable the memory-pattern optimization.
    #[inline]
    pub fn disable_mem_pattern(mut self) -> Self {
        self.mem_pattern = MemPatternState::Disabled;
        self
    }

    /// Enable ORT session profiling with a file prefix. Call [`crate::Session::end_profiling`]
    /// to flush profiling data and retrieve the generated profile file path.
    pub fn enable_profiling(
        mut self, profile_file_prefix: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.profiling_prefix = Some(CString::new(profile_file_prefix)?);
        Ok(self)
    }

    /// Explicitly disable ORT session profiling.
    #[inline]
    pub fn disable_profiling(mut self) -> Self {
        self.profiling_prefix = None;
        self
    }

    /// Override a free dimension by denotation before the session is created.
    ///
    /// Use this when a model marks a dynamic dimension with an ONNX denotation such as
    /// `"DATA_BATCH"`. ORT can then compile a more static plan for that dimension.
    pub fn with_free_dimension_override(
        mut self, dimension_denotation: &str, value: i64,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.free_dimension_overrides
            .push((CString::new(dimension_denotation)?, value));
        Ok(self)
    }

    /// Override a free dimension by symbolic name before the session is created.
    ///
    /// This is the common batching path for models whose first input dimension is named
    /// `"batch"` or similar.
    pub fn with_free_dimension_override_by_name(
        mut self, dimension_name: &str, value: i64,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.free_dimension_overrides_by_name
            .push((CString::new(dimension_name)?, value));
        Ok(self)
    }

    /// Enable or disable spinning for ORT's intra-op worker threads for this session.
    pub fn with_intra_op_spinning(
        self, enable: bool,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.with_config_entry("session.intra_op.allow_spinning", bool_config_value(enable))
    }

    /// Enable or disable spinning for ORT's inter-op worker threads for this session.
    pub fn with_inter_op_spinning(
        self, enable: bool,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.with_config_entry("session.inter_op.allow_spinning", bool_config_value(enable))
    }

    /// Append a session config entry (`AddSessionConfigEntry`). Returns an error if
    /// `key` or `value` contains a NUL byte.
    pub fn with_config_entry(
        mut self, key: &str, value: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.config_entries
            .push((CString::new(key)?, CString::new(value)?));
        Ok(self)
    }

    /// Whether a session-config entry for `key` is present on the materialized options
    /// (`HasSessionConfigEntry`). Builds a transient options handle to query the engine, then
    /// releases it.
    pub fn has_config_entry(&self, key: &str) -> Result<bool> {
        let opts = self.build_handle()?;
        let ckey =
            CString::new(key).map_err(|_| crate::Error::new(-1, "config key contains a NUL"))?;
        let mut present: core::ffi::c_int = 0;
        let r =
            check(unsafe { api().has_session_config_entry()(opts, ckey.as_ptr(), &mut present) });
        unsafe { api().release_session_options()(opts) };
        r?;
        Ok(present != 0)
    }

    /// Read a session-config entry set with [`Self::with_config_entry`]
    /// (`GetSessionConfigEntry`). Returns `None` if `key` was not set. Uses the two-call
    /// buffer-size dance: the first call learns the required length (ORT returns an error for a
    /// null buffer but still writes the size), the second fills it.
    pub fn config_entry(&self, key: &str) -> Result<Option<String>> {
        let opts = self.build_handle()?;
        let ckey =
            CString::new(key).map_err(|_| crate::Error::new(-1, "config key contains a NUL"))?;
        let result = read_config_entry(opts, ckey.as_ptr());
        unsafe { api().release_session_options()(opts) };
        result
    }

    /// Snapshot every session-config entry as an owning [`crate::KeyValuePairs`]
    /// (`GetSessionOptionsConfigEntries`). Empty when no entries are set.
    pub fn config_entries(&self) -> Result<crate::KeyValuePairs> {
        let opts = self.build_handle()?;
        let mut kvps: *mut sys::KeyValuePairsHandle = ptr::null_mut();
        let r = check(unsafe { api().get_session_options_config_entries()(opts, &mut kvps) });
        unsafe { api().release_session_options()(opts) };
        r?;
        if kvps.is_null() {
            // ORT returns null when there are no entries; surface an empty container.
            return crate::KeyValuePairs::new();
        }
        // SAFETY: `kvps` is a freshly-allocated owning handle from ORT.
        Ok(unsafe { crate::KeyValuePairs::from_handle(kvps) })
    }

    /// Set ORT's automatic execution-provider device-selection policy.
    #[cfg(feature = "ep")]
    #[inline]
    pub fn with_ep_selection_policy(mut self, policy: sys::ExecutionProviderDevicePolicy) -> Self {
        self.ep_selection_policy = Some(policy);
        self
    }

    /// Materialize an ORT `SessionOptions` handle from this config. The caller owns
    /// and must release the returned handle (`CreateSession` copies the options).
    /// Route this session's ORT log messages to `logger` (`SetUserLoggingFunction`). The closure
    /// runs on whatever thread ORT logs from, so it must be `Send + Sync`.
    ///
    /// **Leak:** when the session-options handle is built, the closure is leaked (one `Arc` ref via
    /// `Arc::into_raw`). ORT keeps the callback pointer for the session's whole lifetime and
    /// provides no unregister call, so it cannot be safely reclaimed. Call this once per options
    /// and avoid rebuilding options repeatedly with a logger attached.
    pub fn with_user_logging_function<L>(&mut self, logger: L) -> Result<&mut Self>
    where
        L: Fn(LogRecord) + Send + Sync + 'static,
    {
        self.user_logger = Some(crate::environment::LoggerSlot::new(logger));
        Ok(self)
    }

    /// Enable ORT's built-in contrib ops (`EnableOrtCustomOps`, feature `custom-ops`).
    #[cfg(feature = "custom-ops")]
    pub fn with_enable_ort_custom_ops(mut self) -> Self {
        self.enable_ort_custom_ops = true;
        self
    }

    /// Register a shared-library custom-op provider at `path` (`RegisterCustomOpsLibrary_V2`,
    /// feature `custom-ops`). Preferred over the v1 call — ORT manages the library handle.
    #[cfg(feature = "custom-ops")]
    pub fn with_register_custom_ops_library(
        mut self, path: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.custom_op_libraries.push(CString::new(path)?);
        Ok(self)
    }

    /// Register a shared-library custom-op provider via the legacy v1 call
    /// (`RegisterCustomOpsLibrary`, feature `custom-ops`). The returned dlopen handle is retained
    /// for the session's lifetime (the lib must stay loaded); prefer
    /// [`Self::with_register_custom_ops_library`] (v2 manages the handle).
    #[cfg(feature = "custom-ops")]
    pub fn with_register_custom_ops_library_v1(
        mut self, path: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.custom_op_libraries_v1.push(CString::new(path)?);
        Ok(self)
    }

    /// Call an in-process registration function named `func_name` to register custom ops
    /// (`RegisterCustomOpsUsingFunction`, feature `custom-ops`). The symbol must exist in the
    /// process image.
    #[cfg(feature = "custom-ops")]
    pub fn with_register_custom_ops_using_function(
        mut self, func_name: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.custom_op_functions.push(CString::new(func_name)?);
        Ok(self)
    }

    /// Write the optimized model graph to `path` after optimization
    /// (`SetOptimizedModelFilePath`). Useful for inspecting ORT's rewritten graph.
    pub fn with_optimized_model_file_path(
        mut self, path: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.optimized_model_file_path = Some(CString::new(path)?);
        Ok(self)
    }

    /// Toggle deterministic compute (`SetDeterministicCompute`) — at a perf cost, makes run
    /// outputs bit-identical across runs of the same inputs.
    pub fn with_deterministic_compute(mut self, value: bool) -> Self {
        self.deterministic_compute = Some(value);
        self
    }

    /// Set the load-cancellation flag (`SessionOptionsSetLoadCancellationFlag`). When true, a
    /// `RunOptions::terminate` request can abort model loading.
    pub fn with_load_cancellation_flag(mut self, value: bool) -> Self {
        self.load_cancellation_flag = Some(value);
        self
    }

    /// The configured execution mode, queried from a freshly materialized options handle
    /// (`GetSessionExecutionMode`). Reflects [`SessionOptions::with_execution_mode`] (ORT's
    /// default is `Sequential`).
    pub fn execution_mode(&self) -> Result<sys::ExecutionMode> {
        let opts = self.build_handle()?;
        let mut mode = sys::ExecutionMode::Sequential;
        let r = check(unsafe { api().get_session_execution_mode()(opts, &mut mode) });
        unsafe { api().release_session_options()(opts) };
        r?;
        Ok(mode)
    }

    /// Whether the memory-pattern optimization is enabled, queried from a freshly materialized
    /// options handle (`GetMemPatternEnabled`).
    pub fn mem_pattern_enabled(&self) -> Result<bool> {
        let opts = self.build_handle()?;
        let mut v: core::ffi::c_int = 0;
        let r = check(unsafe { api().get_mem_pattern_enabled()(opts, &mut v) });
        unsafe { api().release_session_options()(opts) };
        r?;
        Ok(v != 0)
    }

    /// Fork a materialized ORT session-options handle from this config (`CloneSessionOptions`).
    /// The returned handle is **owned by the caller** and must be released with
    /// `api().release_session_options()`. Advanced interop only — most callers should build a
    /// [`crate::Session`] directly.
    pub fn clone_ort_handle(&self) -> Result<*mut sys::SessionOptionsHandle> {
        let src = self.build_handle()?;
        let mut dst: *mut sys::SessionOptionsHandle = ptr::null_mut();
        let r = check(unsafe { api().clone_session_options()(src, &mut dst) });
        unsafe { api().release_session_options()(src) };
        r?;
        crate::ensure_non_null(dst, "cloned session options")
    }

    /// Install a custom thread-creation/join pair for this session's intra-op thread pool
    /// (`SessionOptionsSetCustomCreateThreadFn` + `…CustomJoinThreadFn` +
    /// `…CustomThreadCreationOptions`). Expert/raw — `create`/`join` are C function pointers and
    /// `creation_options` is an opaque state passed to `create`; the caller is responsible for
    /// their lifetime and thread-safety. Pass `None`/null to clear.
    ///
    /// # Safety
    /// The callbacks must be sound C-thread functions (the `create` callback spawns a thread that
    /// invokes the ORT worker function, `join` reaps it) and `creation_options` must outlive every
    /// session built from these options.
    pub unsafe fn with_custom_thread_handlers(
        mut self, create: sys::CustomCreateThreadFnHandle, join: sys::CustomJoinThreadFnHandle,
        creation_options: *mut c_void,
    ) -> Self {
        self.custom_create_thread_fn = Some(create);
        self.custom_join_thread_fn = Some(join);
        self.custom_thread_creation_options = creation_options;
        self
    }

    /// Install a custom EP-selection delegate (feature `ep`,
    /// `SessionOptionsSetEpSelectionPolicyDelegate`). Expert/raw — see `EpSelectionDelegate`.
    ///
    /// # Safety
    /// `delegate` must be a sound C callback and `state` must outlive every session built from
    /// these options (ORT stores both and invokes the delegate from ORT-internal threads).
    #[cfg(feature = "ep")]
    pub unsafe fn with_ep_selection_policy_delegate(
        mut self, delegate: EpSelectionDelegate, state: *mut c_void,
    ) -> Self {
        self.ep_selection_delegate = Some(delegate);
        self.ep_selection_delegate_state = state;
        self
    }

    /// Materialize an ORT session-options handle from this config, **without** attaching the user
    /// logging callback. Use for transient query/probe handles ([`Self::has_config_entry`],
    /// [`Self::config_entry`], [`Self::config_entries`], [`Self::execution_mode`],
    /// [`Self::mem_pattern_enabled`], [`Self::clone_ort_handle`], EP-option probing): these are
    /// built and released immediately, so attaching the logger would leak one unreclaimable `Arc`
    /// ref per call. Real construction must use [`Self::build_handle_for_session`].
    pub(crate) fn build_handle(&self) -> Result<*mut sys::SessionOptionsHandle> {
        self.build_handle_inner(false)
    }

    /// Like [`build_handle`](Self::build_handle) but **also attaches the user logging callback**.
    /// Use only for the true construction path — the handle fed to `CreateSession`,
    /// `FinalizeModelEditorSession`, or `CreateModelCompilationOptionsFromSessionOptions`. ORT
    /// retains the logging-callback pointer for the session's whole lifetime and exposes no
    /// unregister call, so the `Arc` ref is intentionally leaked there exactly once. Transient
    /// query handles route through [`build_handle`](Self::build_handle) to avoid the leak.
    pub(crate) fn build_handle_for_session(&self) -> Result<*mut sys::SessionOptionsHandle> {
        #[cfg(feature = "ep")]
        self.validate_cuda_stream_guards()?;
        self.build_handle_inner(true)
    }

    fn build_handle_inner(&self, attach_logger: bool) -> Result<*mut sys::SessionOptionsHandle> {
        let api = api();
        let mut opts: *mut sys::SessionOptionsHandle = ptr::null_mut();
        check(unsafe { api.create_session_options()(&mut opts) })?;
        let opts = crate::ensure_non_null(opts, "session options")?;
        let result = (|| {
            check(unsafe { api.set_session_graph_optimization_level()(opts, self.opt_level) })?;
            if let Some(n) = self.intra_threads {
                check(unsafe { api.set_intra_op_num_threads()(opts, n) })?;
            }
            if let Some(n) = self.inter_threads {
                check(unsafe { api.set_inter_op_num_threads()(opts, n) })?;
            }
            if let Some(mode) = self.execution_mode {
                check(unsafe { api.set_session_execution_mode()(opts, mode) })?;
            }
            if let Some(log_id) = &self.log_id {
                check(unsafe { api.set_session_log_id()(opts, log_id.as_ptr()) })?;
            }
            if let Some(level) = self.log_severity {
                check(unsafe { api.set_session_log_severity_level()(opts, level) })?;
            }
            if let Some(level) = self.log_verbosity {
                check(unsafe { api.set_session_log_verbosity_level()(opts, level) })?;
            }
            match self.cpu_mem_arena {
                ArenaState::Default => {},
                ArenaState::Enabled => check(unsafe { api.enable_cpu_mem_arena()(opts) })?,
                ArenaState::Disabled => check(unsafe { api.disable_cpu_mem_arena()(opts) })?,
            }
            match self.mem_pattern {
                MemPatternState::Default => {},
                MemPatternState::Enabled => check(unsafe { api.enable_mem_pattern()(opts) })?,
                MemPatternState::Disabled => check(unsafe { api.disable_mem_pattern()(opts) })?,
            }
            if let Some(prefix) = &self.profiling_prefix {
                check(unsafe { api.enable_profiling()(opts, prefix.as_ptr()) })?;
            } else {
                check(unsafe { api.disable_profiling()(opts) })?;
            }
            for (denotation, value) in &self.free_dimension_overrides {
                check(unsafe {
                    api.add_free_dimension_override()(opts, denotation.as_ptr(), *value)
                })?;
            }
            for (name, value) in &self.free_dimension_overrides_by_name {
                check(unsafe {
                    api.add_free_dimension_override_by_name()(opts, name.as_ptr(), *value)
                })?;
            }
            for (k, v) in &self.config_entries {
                check(unsafe { api.add_session_config_entry()(opts, k.as_ptr(), v.as_ptr()) })?;
            }
            #[cfg(feature = "ep")]
            if let Some(policy) = self.ep_selection_policy {
                check(unsafe { api.session_options_set_ep_selection_policy()(opts, policy) })?;
            }
            #[cfg(feature = "ep")]
            for cfg in &self.ep_configs {
                crate::ep::apply(opts, cfg)?;
            }
            #[cfg(feature = "ep")]
            for m in &self.migraphx {
                m.append_raw(opts)?;
            }
            #[cfg(feature = "ep")]
            for o in &self.openvino {
                o.append_raw(opts)?;
            }
            #[cfg(feature = "custom-ops")]
            for domain in &self.custom_op_domains {
                check(unsafe { api.add_custom_op_domain()(opts, *domain) })?;
            }
            #[cfg(feature = "custom-ops")]
            if self.enable_ort_custom_ops {
                check(unsafe { api.enable_ort_custom_ops()(opts) })?;
            }
            #[cfg(feature = "custom-ops")]
            for path in &self.custom_op_libraries {
                check(unsafe { api.register_custom_ops_library_v2()(opts, path.as_ptr()) })?;
            }
            #[cfg(feature = "custom-ops")]
            for path in &self.custom_op_libraries_v1 {
                let mut handle: *mut c_void = ptr::null_mut();
                check(unsafe {
                    api.register_custom_ops_library()(opts, path.as_ptr(), &mut handle)
                })?;
                // V1 returns a dlopen handle the caller would dlclose; we intentionally retain it
                // for the session's lifetime (the lib must stay loaded). v2 manages this internally.
                let _ = handle;
            }
            #[cfg(feature = "custom-ops")]
            for name in &self.custom_op_functions {
                check(unsafe { api.register_custom_ops_using_function()(opts, name.as_ptr()) })?;
            }
            if attach_logger {
                if let Some(slot) = &self.user_logger {
                    // Leak one Arc ref (`Arc::into_raw`): ORT retains this pointer for the session's
                    // whole lifetime and provides no unregister call, so it cannot be reclaimed.
                    // Gated behind `attach_logger` so the transient `build_handle` query path does
                    // not leak a ref on every materialization — only the construction path
                    // (`build_handle_for_session`) pays this leak, exactly once per session.
                    let param = Arc::into_raw(Arc::clone(slot)) as *mut c_void;
                    check(unsafe {
                        api.set_user_logging_function()(
                            opts,
                            crate::environment::logging_function(),
                            param,
                        )
                    })?;
                }
            }
            if let Some(path) = &self.optimized_model_file_path {
                check(unsafe { api.set_optimized_model_file_path()(opts, path.as_ptr()) })?;
            }
            if let Some(value) = self.deterministic_compute {
                check(unsafe { api.set_deterministic_compute()(opts, value) })?;
            }
            if let Some(cancel) = self.load_cancellation_flag {
                check(unsafe { api.session_options_set_load_cancellation_flag()(opts, cancel) })?;
            }
            if let Some(create) = self.custom_create_thread_fn {
                check(unsafe { api.session_options_set_custom_create_thread_fn()(opts, create) })?;
            }
            if self.custom_create_thread_fn.is_some() {
                // Apply the matching creation-options state (null clears it — a no-op default).
                check(unsafe {
                    api.session_options_set_custom_thread_creation_options()(
                        opts,
                        self.custom_thread_creation_options,
                    )
                })?;
            }
            if let Some(join) = self.custom_join_thread_fn {
                check(unsafe { api.session_options_set_custom_join_thread_fn()(opts, join) })?;
            }
            #[cfg(feature = "ep")]
            if let Some(delegate) = self.ep_selection_delegate {
                check(unsafe {
                    api.session_options_set_ep_selection_policy_delegate()(
                        opts,
                        delegate,
                        self.ep_selection_delegate_state,
                    )
                })?;
            }
            Ok(opts)
        })();
        if result.is_err() {
            unsafe { api.release_session_options()(opts) };
        }
        result
    }
}

#[inline]
fn bool_config_value(enabled: bool) -> &'static str {
    if enabled { "1" } else { "0" }
}

/// Two-call buffer dance for `GetSessionConfigEntry`: first call with a null buffer learns the
/// required length (ORT returns an error but still writes `*size`); the second call fills a
/// freshly allocated buffer of that size. `None` when the key is absent (`size` stays 0).
fn read_config_entry(
    opts: *const sys::SessionOptionsHandle, key: *const core::ffi::c_char,
) -> Result<Option<String>> {
    let mut size: usize = 0;
    // Expected to error (null buffer); we only care that ORT wrote the required length.
    let _ =
        check(unsafe { api().get_session_config_entry()(opts, key, ptr::null_mut(), &mut size) });
    if size == 0 {
        return Ok(None);
    }
    let mut buf: Vec<u8> = vec![0u8; size];
    let mut filled: usize = size;
    check(unsafe {
        api().get_session_config_entry()(
            opts,
            key,
            buf.as_mut_ptr() as *mut core::ffi::c_char,
            &mut filled,
        )
    })?;
    // The buffer is NUL-terminated; cut at the first NUL (robust to whatever ORT wrote to `filled`).
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    buf.truncate(nul);
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| crate::Error::new(-1, "zrt: session config entry is not valid UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advanced_options_build_handle() {
        let opts = SessionOptions::new()
            .with_opt_level(sys::GraphOptimizationLevel::All)
            .with_intra_threads(1)
            .with_inter_threads(1)
            .with_parallel_execution()
            .with_log_id("advanced-options")
            .expect("log id")
            .with_log_severity(sys::LoggingLevel::Verbose)
            .with_log_verbosity(1)
            .with_free_dimension_override("DATA_BATCH", 4)
            .expect("free dim denotation")
            .with_free_dimension_override_by_name("batch", 4)
            .expect("free dim name")
            .with_intra_op_spinning(false)
            .expect("intra spin")
            .with_inter_op_spinning(false)
            .expect("inter spin");

        let h = opts.build_handle().expect("advanced options handle");
        unsafe {
            api().release_session_options()(h);
        }
    }

    /// P1-5: the transient query path (`build_handle`) must NOT leak a logger `Arc` ref on each
    /// materialization — only the construction path (`build_handle_for_session`) may attach the
    /// logger. Watch the `Arc` strong count: queries are ref-neutral; one construction = +1 (the
    /// intentional, unreclaimable leak ORT requires).
    #[test]
    fn build_handle_query_path_does_not_leak_logger_arc() {
        let mut opts = SessionOptions::new();
        let slot = crate::environment::LoggerSlot::new(|_r| {});
        opts.user_logger = Some(Arc::clone(&slot));
        let baseline = Arc::strong_count(&slot);

        // Repeated transient materializations (the query path) must be Arc-ref-neutral.
        for _ in 0..4 {
            let h = opts.build_handle().expect("query handle");
            unsafe {
                api().release_session_options()(h);
            }
        }
        assert_eq!(
            Arc::strong_count(&slot),
            baseline,
            "build_handle (query path) leaked a logger Arc ref"
        );

        // The construction path attaches the logger exactly once per call (intended, unreclaimable).
        let h = opts.build_handle_for_session().expect("session handle");
        unsafe {
            api().release_session_options()(h);
        }
        assert_eq!(
            Arc::strong_count(&slot),
            baseline + 1,
            "build_handle_for_session should attach the logger exactly once"
        );
    }

    #[cfg(feature = "ep")]
    #[test]
    fn ep_selection_policy_reaches_ffi() {
        let opts = SessionOptions::new()
            .with_ep_selection_policy(sys::ExecutionProviderDevicePolicy::Default);

        let h = opts.build_handle().expect("ep selection policy handle");
        unsafe {
            api().release_session_options()(h);
        }
    }

    #[cfg(feature = "custom-ops")]
    #[test]
    fn session_options_enable_ort_custom_ops_reaches_ffi() {
        // EnableOrtCustomOps reaches the FFI. The stock ORT release is not bundled with
        // onnxruntime-extensions, so it returns a clean build-flag error; a build with the
        // extensions would accept it. We assert the documented error to prove the call is sound.
        let opts = SessionOptions::new().with_enable_ort_custom_ops();
        match opts.build_handle() {
            Ok(h) => unsafe { api().release_session_options()(h) },
            Err(e) => assert!(
                e.to_string().contains("onnxruntime-extensions"),
                "expected the extensions-not-enabled error, got: {e}"
            ),
        }
    }

    #[cfg(feature = "custom-ops")]
    #[test]
    fn session_options_custom_op_registration_paths_reach_ffi() {
        // A bogus library path (v1 + v2) and function symbol must surface as clean build-time
        // errors — proving RegisterCustomOpsLibrary(_V2) + RegisterCustomOpsUsingFunction were
        // reached (no UB).
        let opts = SessionOptions::new()
            .with_register_custom_ops_library("/nonexistent/libcustom.so")
            .expect("v2 path")
            .with_register_custom_ops_library_v1("/nonexistent/libcustom.so")
            .expect("v1 path")
            .with_register_custom_ops_using_function("NoSuchRegisterFn")
            .expect("func name");
        assert!(
            opts.build_handle().is_err(),
            "bogus custom-op registration must error, not succeed silently"
        );
    }

    #[test]
    fn session_options_t2_12_knobs_round_trip() {
        // The misc SessionOptions knobs: deterministic compute, load-cancellation flag,
        // optimized-model path, plus the execution-mode/mem-pattern getters and handle clone.
        let opts = SessionOptions::new()
            .with_deterministic_compute(true)
            .with_load_cancellation_flag(false)
            .with_optimized_model_file_path("/tmp/zrt-optimized.onnx")
            .expect("optimized path")
            .with_execution_mode(sys::ExecutionMode::Parallel);

        let h = opts.build_handle().expect("t2.12 options handle");
        // Getters query a freshly materialized handle.
        assert_eq!(
            opts.execution_mode().expect("execution mode"),
            sys::ExecutionMode::Parallel,
            "execution_mode getter must reflect the setter"
        );
        // mem_pattern defaults to enabled when not explicitly disabled.
        assert!(
            opts.mem_pattern_enabled().expect("mem pattern enabled"),
            "mem pattern is enabled by default"
        );
        // clone_ort_handle forks the materialized handle via CloneSessionOptions.
        let h2 = opts.clone_ort_handle().expect("cloned handle");
        unsafe {
            api().release_session_options()(h);
            api().release_session_options()(h2);
        }
    }

    #[test]
    fn session_options_config_entry_round_trip() {
        // GetSessionConfigEntry (buffer-size dance) + GetSessionOptionsConfigEntries (returns a
        // KeyValuePairs). Set an entry, read it back as a string and via the pairs snapshot; an
        // unset key returns None.
        let opts = SessionOptions::new()
            .with_config_entry("zrt.test.key", "abc-123")
            .expect("config entry");
        assert_eq!(
            opts.config_entry("zrt.test.key").expect("read entry"),
            Some("abc-123".to_string()),
            "set config entry must round-trip"
        );
        assert_eq!(
            opts.config_entry("zrt.absent.key").expect("read absent"),
            None,
            "an unset config entry must read as None"
        );
        let kv = opts.config_entries().expect("config entries snapshot");
        assert_eq!(
            kv.get("zrt.test.key").expect("kv get"),
            Some("abc-123".to_string()),
            "the pairs snapshot must contain the entry"
        );
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn session_options_round_trip() {
        let opts = SessionOptions::new()
            .with_opt_level(sys::GraphOptimizationLevel::Extended)
            .with_intra_threads(4)
            .with_inter_threads(2)
            .with_parallel_execution()
            .with_log_id("serde-session")
            .expect("log id")
            .with_log_severity(sys::LoggingLevel::Warning)
            .with_log_verbosity(2)
            .disable_cpu_mem_arena()
            .disable_mem_pattern()
            .with_free_dimension_override("DATA_BATCH", 4)
            .expect("free dim denotation")
            .with_free_dimension_override_by_name("batch", 4)
            .expect("free dim name")
            .with_intra_op_spinning(false)
            .expect("intra spinning")
            .with_inter_op_spinning(false)
            .expect("inter spinning")
            .with_config_entry("session.run", "1")
            .expect("config entry");

        let json = serde_json::to_string(&opts).expect("serialize");
        eprintln!("SessionOptions JSON: {json}");
        assert!(
            json.contains("\"opt_level\":2"),
            "opt_level discriminant (Extended=2) present: {json}"
        );
        assert!(
            json.contains("\"session.run\""),
            "config key present: {json}"
        );
        assert!(json.contains("\"serde-session\""), "log id present: {json}");
        assert!(
            json.contains("\"log_severity\":2"),
            "log severity present: {json}"
        );
        assert!(
            json.contains("\"log_verbosity\":2"),
            "log verbosity present: {json}"
        );

        let back: SessionOptions = serde_json::from_str(&json).expect("deserialize");
        assert!(back.custom_thread_creation_options.is_null());
        #[cfg(feature = "ep")]
        assert!(back.ep_selection_delegate_state.is_null());
        assert_eq!(back.opt_level, sys::GraphOptimizationLevel::Extended);
        assert_eq!(back.intra_threads, Some(4));
        assert_eq!(back.inter_threads, Some(2));
        assert_eq!(back.execution_mode, None);
        assert_eq!(
            back.log_id.as_ref().and_then(|id| id.to_str().ok()),
            Some("serde-session")
        );
        assert_eq!(back.log_severity, Some(sys::LoggingLevel::Warning as i32));
        assert_eq!(back.log_verbosity, Some(2));
        assert_eq!(back.cpu_mem_arena, ArenaState::Disabled);
        assert_eq!(back.mem_pattern, MemPatternState::Disabled);
        assert_eq!(
            back.free_dimension_overrides
                .iter()
                .filter(|(k, _)| k.to_str() == Ok("DATA_BATCH"))
                .count(),
            1
        );
        assert_eq!(
            back.free_dimension_overrides_by_name
                .iter()
                .filter(|(k, _)| k.to_str() == Ok("batch"))
                .count(),
            1
        );
        assert_eq!(
            back.config_entries
                .iter()
                .filter(|(k, _)| k.to_str() == Ok("session.run"))
                .count(),
            1
        );

        // The deserialized config must still materialize a live ORT handle.
        let h = back
            .build_handle()
            .expect("build handle from deserialized config");
        unsafe {
            api().release_session_options()(h);
        }

        let enabled = SessionOptions::new()
            .with_cpu_mem_arena(ArenaState::Enabled)
            .with_mem_pattern(MemPatternState::Enabled);
        assert_eq!(enabled.cpu_mem_arena, ArenaState::Enabled);
        assert_eq!(enabled.mem_pattern, MemPatternState::Enabled);
        let h = enabled.build_handle().expect("build enabled handle");
        unsafe {
            api().release_session_options()(h);
        }
    }
}
