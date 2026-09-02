//! The ORT environment (logging + global state). One per process is typical.
//!
//! The raw `OrtEnv` handle is wrapped in an [`Arc`] so it can be shared cheaply: a
//! [`crate::Session`] takes its own `Arc` clone at construction, which **keeps the Env
//! alive for the Session's whole lifetime**. This matters because ORT sessions reference
//! the Env's thread pools/allocator — releasing the Env while a Session still lives is a
//! use-after-free (the root cause of the historical ">4MB segfault").
//! Because the Session owns an `Arc` ref, that UAF can no longer occur regardless of how a
//! caller scopes its `Environment`.
use crate::{Error, Result, api, check, sys};
use std::ffi::{CString, c_char, c_void};
use std::ptr;
use std::sync::Arc;

// ─── Custom logger ────────────────────────────────────────────────────────────
//
// ORT's default logger writes to stderr. The custom-logger path routes those messages into a
// caller-supplied closure instead. The hard part is the C↔Rust callback boundary: ORT holds a
// raw `OrtLoggingFunction` + `void* param` and invokes them from arbitrary threads for the Env's
// whole lifetime. We bridge that with a `log_trampoline` whose `param` is a stable pointer to a
// boxed `dyn Fn(LogRecord) + Send + Sync`, owned by `EnvInner` so it drops exactly when the last
// Env/Session ref goes away (no ORT call can follow it).

/// A single log record ORT delivered to a user-supplied logger. The string fields are copied into
/// owned UTF-8 `String`s (`to_string_lossy`) because ORT may free the source C strings once the
/// callback returns.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// Severity of this message (`OrtLoggingLevel`).
    pub severity: sys::LoggingLevel,
    /// Log category (EP / subsystem name). Empty if ORT passed none.
    pub category: String,
    /// The environment's log id.
    pub log_id: String,
    /// Source location (`ORT_FILE`) if ORT provided one.
    pub code_location: String,
    /// The log message body.
    pub message: String,
}

/// The correct `OrtLoggingFunction` signature (`onnxruntime_c_api.h:448`):
/// `void(*)(void* param, OrtLoggingLevel, const char* category, const char* logid,
///  const char* code_location, const char* message)` — six single `const char*`/pointer args.
///
/// NOTE: the generated `sys::LoggingFunction` alias (`st-zrt-sys/src/lib.rs`) mislabels the last
/// two parameters as `status_messages: *const *const c_char` / `num_status_messages: usize`. Those
/// occupy the same ABI slots (two eight-byte values) as the real `code_location`/`message`
/// pointers, so the calls are binary-compatible — but trusting the mislabeled type to read them
/// would be wrong. We define the correct signature here and `transmute` into the alias at the
/// single call site (asserted pointer-sized).
type LoggingCallback = unsafe extern "C" fn(
    param: *mut c_void,
    severity: sys::LoggingLevel,
    category: *const c_char,
    logid: *const c_char,
    code_location: *const c_char,
    message: *const c_char,
);

/// Reinterpret the correct trampoline as the (mislabeled) generated fn-pointer type ORT expects.
pub(crate) fn logging_function() -> sys::LoggingFunction {
    // SAFETY: `LoggingCallback` and `sys::LoggingFunction` are both a nullable `extern "C"`
    // function pointer over six eight-byte parameters — identical layout. The compile-time assert
    // below pins that.
    const _: () = assert!(
        std::mem::size_of::<LoggingCallback>() == std::mem::size_of::<sys::LoggingFunction>()
    );
    unsafe { std::mem::transmute::<LoggingCallback, sys::LoggingFunction>(log_trampoline) }
}

/// The trampoline ORT calls. `param` is the `*mut LoggerSlot` we registered.
unsafe extern "C" fn log_trampoline(
    param: *mut c_void, severity: sys::LoggingLevel, category: *const c_char, logid: *const c_char,
    code_location: *const c_char, message: *const c_char,
) {
    if param.is_null() {
        return;
    }
    // SAFETY: `param` is the `*mut LoggerSlot` `UserLogger` registered via `Box::into_raw`. It stays
    // valid for as long as the `EnvInner` that owns the `UserLogger` is alive — and ORT only invokes
    // this callback while the Env is alive (sessions hold an `Arc<EnvInner>` ref). The closure is
    // `Send + Sync`, so concurrent calls from ORT's threads are sound.
    let slot: &LoggerSlot = unsafe { &*(param as *const LoggerSlot) };
    let record = LogRecord {
        severity,
        category: cstr_lossy(category),
        log_id: cstr_lossy(logid),
        code_location: cstr_lossy(code_location),
        message: cstr_lossy(message),
    };
    // A panic unwinding across the FFI boundary into C is UB; swallow it.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (slot.f)(record)));
}

fn cstr_lossy(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// A `Sized` shell around the boxed closure so its address is a *thin* pointer — required to
/// round-trip through ORT's `void* param` (a trait-object fat pointer would lose its vtable).
pub(crate) struct LoggerSlot {
    f: Box<dyn Fn(LogRecord) + Send + Sync>,
}

impl LoggerSlot {
    pub(crate) fn new(f: impl Fn(LogRecord) + Send + Sync + 'static) -> Arc<Self> {
        Arc::new(Self { f: Box::new(f) })
    }
}

/// Owns the boxed user-logger closure so it outlives every ORT call into it. Held by `EnvInner`;
/// `Drop` reconstructs the `Box` and frees it once the last Env/Session ref is gone.
pub(crate) struct UserLogger {
    raw: *mut LoggerSlot,
}

impl UserLogger {
    fn new(f: impl Fn(LogRecord) + Send + Sync + 'static) -> Self {
        let boxed: Box<LoggerSlot> = Box::new(LoggerSlot { f: Box::new(f) });
        Self {
            raw: Box::into_raw(boxed),
        }
    }

    /// The raw callback-state pointer to hand ORT as its `param`.
    fn param(&self) -> *mut c_void {
        self.raw as *mut c_void
    }
}

impl Drop for UserLogger {
    fn drop(&mut self) {
        // SAFETY: runs only when the Env drops (last `Arc<EnvInner>` ref). After that ORT holds no
        // reference that can re-enter the trampoline.
        if !self.raw.is_null() {
            unsafe { drop(Box::from_raw(self.raw)) };
        }
    }
}
// The boxed closure is `Send + Sync`; the raw pointer is only dereferenced inside the trampoline
// under the EnvInner-lifetime invariant documented above.
unsafe impl Send for UserLogger {}
unsafe impl Sync for UserLogger {}

/// Calling-language classification for ORT telemetry (`OrtLanguageProjection`). The default is
/// `C`, which also classifies any language not in this list.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageProjection {
    C = 0,
    CPlusPlus = 1,
    CSharp = 2,
    Python = 3,
    Java = 4,
    WinML = 5,
    NodeJs = 6,
}

/// Thread-pool work-enqueue callback (runs on the submitting thread). Returns callback-specific
/// data later passed to the start/stop/abandon callbacks (may be null).
pub type ThreadPoolWorkEnqueueFn =
    Option<unsafe extern "C" fn(user_context: *mut c_void) -> *mut c_void>;
/// Thread-pool work-start callback (runs on a worker thread, or the submitting thread if the
/// queue is full and work runs synchronously).
pub type ThreadPoolWorkStartFn =
    Option<unsafe extern "C" fn(user_context: *mut c_void, enqueue_data: *mut c_void)>;
/// Thread-pool work-stop callback — always called when work finishes, success or failure.
pub type ThreadPoolWorkStopFn =
    Option<unsafe extern "C" fn(user_context: *mut c_void, enqueue_data: *mut c_void)>;
/// Thread-pool work-abandon callback — enqueued work revoked/rejected without execution.
pub type ThreadPoolWorkAbandonFn =
    Option<unsafe extern "C" fn(user_context: *mut c_void, enqueue_data: *mut c_void)>;

/// Mirror of ORT's versioned `OrtThreadPoolCallbacksConfig` (`#[repr(C)]`, since v1.25) — a bundle
/// of thread-pool instrumentation callbacks + a user context, passed to
/// [`Environment::set_per_session_thread_pool_callbacks`]. All callbacks are optional. The
/// function-pointer fields use `Option<extern "C" fn>`, which Rust lays out identically to a
/// nullable C function pointer.
#[repr(C)]
pub struct ThreadPoolCallbacksConfig {
    /// Must be `ORT_API_VERSION` (= `sys::API_VERSION`); set by [`Self::new`].
    pub version: u32,
    /// Called when work is enqueued (may be null).
    pub on_enqueue: ThreadPoolWorkEnqueueFn,
    /// Called when work starts (may be null).
    pub on_start_work: ThreadPoolWorkStartFn,
    /// Called when work completes (may be null).
    pub on_stop_work: ThreadPoolWorkStopFn,
    /// Called when work is abandoned (may be null).
    pub on_abandon: ThreadPoolWorkAbandonFn,
    /// User-provided context passed to every callback (may be null). Must remain valid and be
    /// thread-safe for the lifetime of every session created from the environment.
    pub user_context: *mut c_void,
}

impl ThreadPoolCallbacksConfig {
    /// An all-null config (no instrumentation). `version` is set to `sys::API_VERSION`.
    pub fn new() -> Self {
        Self {
            version: sys::API_VERSION,
            on_enqueue: None,
            on_start_work: None,
            on_stop_work: None,
            on_abandon: None,
            user_context: ptr::null_mut(),
        }
    }

    /// Attach the work-enqueue callback.
    #[inline]
    pub fn with_on_enqueue(mut self, f: ThreadPoolWorkEnqueueFn) -> Self {
        self.on_enqueue = f;
        self
    }
    /// Attach the work-start callback.
    #[inline]
    pub fn with_on_start_work(mut self, f: ThreadPoolWorkStartFn) -> Self {
        self.on_start_work = f;
        self
    }
    /// Attach the work-stop callback.
    #[inline]
    pub fn with_on_stop_work(mut self, f: ThreadPoolWorkStopFn) -> Self {
        self.on_stop_work = f;
        self
    }
    /// Attach the work-abandon callback.
    #[inline]
    pub fn with_on_abandon(mut self, f: ThreadPoolWorkAbandonFn) -> Self {
        self.on_abandon = f;
        self
    }
    /// Set the user context shared by all callbacks.
    #[inline]
    pub fn with_user_context(mut self, ctx: *mut c_void) -> Self {
        self.user_context = ctx;
        self
    }
}

impl Default for ThreadPoolCallbacksConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Owning inner: holds the raw `OrtEnv` handle and releases it (`ReleaseEnv`) when the last
/// `Arc` reference drops. Kept `pub(crate)` so [`crate::Session`] can hold an `Arc<EnvInner>`.
pub(crate) struct EnvInner {
    env: *mut sys::EnvHandle,
    threading: Option<crate::threading::ThreadingOptions>,
    /// The boxed custom-logger callback, if this Env was created with one. Kept alive for the
    /// Env's lifetime so ORT's trampoline invocations stay valid.
    _user_logger: Option<UserLogger>,
}

impl EnvInner {
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::EnvHandle {
        self.env as *const sys::EnvHandle
    }
}

/// The ORT environment (logging + global state). Cheap to clone (one `Arc` refcount); the
/// underlying `OrtEnv` is shared and freed only when the last clone AND every `Session`
/// derived from it are dropped.
#[derive(Clone)]
pub struct Environment(Arc<EnvInner>);

impl std::fmt::Debug for Environment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Environment")
            .field("env", &(self.0).env)
            .field("strong_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

/// Fail loudly when the loaded libonnxruntime is not a supported release line.
///
/// `GetApi(API_VERSION)` succeeds against any newer runtime (the ORT C API is append-only),
/// so a too-new or wrong-major library can otherwise load and then misbehave far away from
/// the cause. Checked at the top of every `Environment` constructor — one
/// `GetVersionString` C call, never on a session hot path. Supported lines:
/// [`sys::SUPPORTED_RUNTIME_LINES`].
/// Apply library defaults to the process environment before the first ORT interaction.
///
/// Telemetry: the GPU `libonnxruntime` package ships with POSIX telemetry compiled in.
/// `st-zrt` sets `ORT_DISABLE_TELEMETRY=1` before initialization **unless the variable is
/// already present** — an explicitly exported value always wins, so users who want
/// telemetry can export `ORT_DISABLE_TELEMETRY=0` (or any other value) and keep it.
fn apply_env_creation_defaults() {
    if std::env::var_os("ORT_DISABLE_TELEMETRY").is_none() {
        // SAFETY: process-environment mutation. The value is a static literal (no
        // allocation), every concurrent writer would write the identical bytes, and the
        // write happens strictly before the first ORT environment is created in this
        // call's caller — the point ORT reads the variable.
        unsafe { std::env::set_var("ORT_DISABLE_TELEMETRY", "1") };
    }
}

/// Shared pre-`CreateEnv` checks: environment defaults (telemetry opt-out) followed by
/// the supported-runtime-line guard.
fn env_creation_checks() -> Result<()> {
    apply_env_creation_defaults();
    verify_runtime_version()
}

fn verify_runtime_version() -> Result<()> {
    if let Some(found) = sys::runtime_version_string() {
        if !sys::runtime_version_supported(&found) {
            return Err(Error::new(
                -1,
                format!(
                    "loaded libonnxruntime {found} is not supported by st-zrt (supported \
                     release lines: {:?}; this build bundles ONNX Runtime {}). Point the \
                     dynamic loader at a supported libonnxruntime, or upgrade st-zrt.",
                    sys::SUPPORTED_RUNTIME_LINES,
                    sys::ORT_VERSION
                ),
            ));
        }
    }
    Ok(())
}

impl Environment {
    /// Create with `Warning` log level and the log id `"zrt"`.
    pub fn new() -> Result<Self> {
        Self::new_with_level(sys::LoggingLevel::Warning, "zrt")
    }

    /// Create with a custom log level and log id.
    pub fn new_with_level(level: sys::LoggingLevel, logid: &str) -> Result<Self> {
        env_creation_checks()?;
        let cid = CString::new(logid)
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        // cid is copied internally by CreateEnv; safe for it to be a local.
        check(unsafe { api().create_env()(level, cid.as_ptr(), &mut env) })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Self(Arc::new(EnvInner {
            env,
            threading: None,
            _user_logger: None,
        })))
    }

    /// Create an environment with ORT global thread pools.
    ///
    /// Sessions created from this environment automatically disable per-session threads unless
    /// their [`crate::SessionOptions`] opt out with
    /// [`crate::SessionOptions::use_per_session_threads`].
    pub fn new_with_global_thread_pools(
        level: sys::LoggingLevel, logid: &str, threading: crate::threading::ThreadingOptions,
    ) -> Result<Self> {
        env_creation_checks()?;
        let cid = CString::new(logid)
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        check(unsafe {
            api().create_env_with_global_thread_pools()(
                level,
                cid.as_ptr(),
                threading.as_ptr(),
                &mut env,
            )
        })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Self(Arc::new(EnvInner {
            env,
            threading: Some(threading),
            _user_logger: None,
        })))
    }

    /// Create an Env whose log messages are routed to `logger` (`CreateEnvWithCustomLogger`).
    ///
    /// The closure runs on whatever thread ORT logs from, so it must be `Send + Sync`. It is kept
    /// alive for the Env's whole lifetime and dropped only when the Env and every [`crate::Session`]
    /// derived from it are dropped. A panic inside `logger` is caught and swallowed (unwinding into
    /// C across the callback boundary is undefined).
    ///
    /// **ORT caveat:** the logging function is *process-global* — the first `Env` created in the
    /// process installs it, and later Envs cannot override it. If a default (stderr) Env is created
    /// before this one, `logger` will not receive messages. Create the custom-logger Env first.
    pub fn new_with_logger<L>(level: sys::LoggingLevel, logid: &str, logger: L) -> Result<Self>
    where
        L: Fn(LogRecord) + Send + Sync + 'static,
    {
        env_creation_checks()?;
        let ul = UserLogger::new(logger);
        let cid = CString::new(logid)
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        check(unsafe {
            api().create_env_with_custom_logger()(
                logging_function(),
                ul.param(),
                level,
                cid.as_ptr(),
                &mut env,
            )
        })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Self(Arc::new(EnvInner {
            env,
            threading: None,
            _user_logger: Some(ul),
        })))
    }

    /// Like [`Self::new_with_logger`] but also configures ORT global thread pools
    /// (`CreateEnvWithCustomLoggerAndGlobalThreadPools`).
    pub fn new_with_logger_and_global_thread_pools<L>(
        level: sys::LoggingLevel, logid: &str, logger: L,
        threading: crate::threading::ThreadingOptions,
    ) -> Result<Self>
    where
        L: Fn(LogRecord) + Send + Sync + 'static,
    {
        env_creation_checks()?;
        let ul = UserLogger::new(logger);
        let cid = CString::new(logid)
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        check(unsafe {
            api().create_env_with_custom_logger_and_global_thread_pools()(
                logging_function(),
                ul.param(),
                level,
                cid.as_ptr(),
                threading.as_ptr(),
                &mut env,
            )
        })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Self(Arc::new(EnvInner {
            env,
            threading: Some(threading),
            _user_logger: Some(ul),
        })))
    }

    /// Build an Env from the ORT 1.24+ unified options struct (`CreateEnvWithOptions`). The single
    /// path that also accepts environment-level config entries; see [`EnvCreationOptions`].
    pub fn new_with_options(opts: EnvCreationOptions) -> Result<Self> {
        opts.create_env()
    }

    /// Change this Env's logging severity level (`UpdateEnvWithCustomLogLevel`). Messages less
    /// severe than `level` are no longer emitted. Applies to subsequent sessions/runs.
    pub fn set_log_level(&self, level: sys::LoggingLevel) -> Result<()> {
        check(unsafe { api().update_env_with_custom_log_level()((self.0).env, level) })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::EnvHandle {
        (self.0).env as *const sys::EnvHandle
    }

    /// An `Arc` clone of the Env's inner — taken by [`crate::Session`] at construction so the
    /// Env outlives the Session. Cheap (one refcount bump); the underlying handle is shared.
    #[inline]
    pub(crate) fn share(&self) -> Arc<EnvInner> {
        self.0.clone()
    }

    #[inline]
    pub(crate) fn has_global_thread_pool(&self) -> bool {
        self.0.threading.is_some()
    }

    /// Register a custom arena allocator with this environment (`CreateAndRegisterAllocator`).
    /// Sessions created AFTER this call can use the registered allocator. The combination of a
    /// [`crate::MemoryInfo`] + [`crate::ArenaCfg`] is the E1 lever — e.g. tuning the CPU arena
    /// or plugging in a device allocator. Advanced; the default allocator already covers CPU v0.1.
    pub fn register_allocator(
        &self, mem_info: &crate::memory::MemoryInfo, arena_cfg: &crate::arena::ArenaCfg,
    ) -> Result<()> {
        check(unsafe {
            api().create_and_register_allocator()(
                (self.0).env,
                mem_info.info as *const sys::MemoryInfoHandle,
                arena_cfg.as_ptr(),
            )
        })
    }

    /// Enable ORT telemetry events for this environment (`EnableTelemetryEvents`).
    pub fn enable_telemetry_events(&self) -> Result<()> {
        check(unsafe { api().enable_telemetry_events()((self.0).env) })
    }

    /// Disable ORT telemetry events for this environment (`DisableTelemetryEvents`).
    pub fn disable_telemetry_events(&self) -> Result<()> {
        check(unsafe { api().disable_telemetry_events()((self.0).env) })
    }

    /// Classify the calling language for ORT telemetry (`SetLanguageProjection`). The default is
    /// `C`; setting the real projection makes per-language telemetry accurate. No-op for telemetry
    /// disabled builds.
    pub fn set_language_projection(&self, projection: LanguageProjection) -> Result<()> {
        check(unsafe {
            api().set_language_projection()((self.0).env, projection as core::ffi::c_int)
        })
    }

    /// Install per-session thread-pool instrumentation callbacks
    /// (`SetPerSessionThreadPoolCallbacks`, since v1.25). `config` bundles optional enqueue /
    /// start-work / stop-work / abandon callbacks + a user context; all are optional. The
    /// callbacks fire from ORT-internal threads, so any shared state they touch must be thread-safe,
    /// and `config.user_context` must outlive every session created from this environment.
    pub fn set_per_session_thread_pool_callbacks(
        &self, config: &ThreadPoolCallbacksConfig,
    ) -> Result<()> {
        check(unsafe {
            api().set_per_session_thread_pool_callbacks()(
                (self.0).env,
                config as *const ThreadPoolCallbacksConfig
                    as *const sys::ThreadPoolCallbacksConfigHandle,
            )
        })
    }

    /// Register an already-created allocator handle with this environment (`RegisterAllocator`,
    /// idx 176). Distinct from [`Self::register_allocator`] (the v1 *create*-and-register). The
    /// allocator must outlive the environment; mutates the env's allocator registry.
    pub fn register_existing_allocator(&self, allocator: &crate::Allocator) -> Result<()> {
        check(unsafe { api().register_allocator()((self.0).env, allocator.alloc_handle()) })
    }

    /// Unregister the allocator backing `mem` from this environment (`UnregisterAllocator`).
    pub fn unregister_allocator(&self, mem: &crate::MemoryInfo) -> Result<()> {
        check(unsafe {
            api().unregister_allocator()((self.0).env, mem.info as *const sys::MemoryInfoHandle)
        })
    }

    /// Create and register an allocator of `provider_type` (e.g. `"CUDAExecutionProvider"`)
    /// for `mem`, tuned by `arena_cfg` and `provider_options` (`CreateAndRegisterAllocatorV2`).
    /// The allocator is owned and released by the environment.
    pub fn create_and_register_allocator_v2(
        &self, provider_type: &str, mem: &crate::MemoryInfo, arena_cfg: &crate::ArenaCfg,
        provider_options: &[(&str, &str)],
    ) -> Result<()> {
        let cprov = CString::new(provider_type)
            .map_err(|_| Error::new(-1, "provider type contains a NUL"))?;
        let keys: Vec<CString> = provider_options
            .iter()
            .map(|(k, _)| CString::new(*k))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::new(-1, "provider option key contains a NUL"))?;
        let vals: Vec<CString> = provider_options
            .iter()
            .map(|(_, v)| CString::new(*v))
            .collect::<std::result::Result<_, _>>()
            .map_err(|_| Error::new(-1, "provider option value contains a NUL"))?;
        let k_ptrs: Vec<*const c_char> = keys.iter().map(|c| c.as_ptr()).collect();
        let v_ptrs: Vec<*const c_char> = vals.iter().map(|c| c.as_ptr()).collect();
        check(unsafe {
            api().create_and_register_allocator_v2()(
                (self.0).env,
                cprov.as_ptr(),
                mem.info as *const sys::MemoryInfoHandle,
                arena_cfg.as_ptr(),
                k_ptrs.as_ptr(),
                v_ptrs.as_ptr(),
                provider_options.len(),
            )
        })
    }

    /// Create a shared (refcounted) allocator for `device`'s `mem_type`
    /// (`CreateSharedAllocator`, feature `ep`, since v1.25). Returns an owning [`crate::Allocator`]
    /// ref; dropping it decrements the shared refcount. `options` tune the allocator.
    #[cfg(feature = "ep")]
    pub fn create_shared_allocator(
        &self, device: &crate::EpDevice, mem_type: crate::DeviceMemoryType,
        allocator_type: sys::AllocatorType, options: &crate::KeyValuePairs,
    ) -> Result<crate::Allocator> {
        if !device.shares_environment(self) {
            return Err(Error::new(
                -1,
                "execution-provider device belongs to a different Environment",
            ));
        }
        let mut alloc: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe {
            api().create_shared_allocator()(
                (self.0).env,
                device.as_ptr(),
                mem_type as core::ffi::c_int,
                allocator_type,
                options.raw_ptr(),
                &mut alloc,
            )
        })?;
        let alloc = crate::ensure_non_null(alloc, "shared allocator")?;
        Ok(crate::Allocator::from_handle_owned(alloc, self.share()))
    }

    /// Get a new ref to the shared allocator for `mem` (`GetSharedAllocator`, feature `ep`).
    /// Returns an owning [`crate::Allocator`]; dropping it decrements the shared refcount. Errors
    /// if no shared allocator is registered for `mem`.
    #[cfg(feature = "ep")]
    pub fn get_shared_allocator(&self, mem: &crate::MemoryInfo) -> Result<crate::Allocator> {
        let mut alloc: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe {
            api().get_shared_allocator()(
                (self.0).env,
                mem.info as *const sys::MemoryInfoHandle,
                &mut alloc,
            )
        })?;
        let alloc = crate::ensure_non_null(alloc, "shared allocator")?;
        Ok(crate::Allocator::from_handle_owned(alloc, self.share()))
    }

    /// Release (tear down) the shared allocator for `device`'s `mem_type`
    /// (`ReleaseSharedAllocator`, feature `ep`). Distinct from dropping a single ref: this removes
    /// the shared allocator from the environment.
    #[cfg(feature = "ep")]
    pub fn release_shared_allocator(
        &self, device: &crate::EpDevice, mem_type: crate::DeviceMemoryType,
    ) -> Result<()> {
        if !device.shares_environment(self) {
            return Err(Error::new(
                -1,
                "execution-provider device belongs to a different Environment",
            ));
        }
        check(unsafe {
            api().release_shared_allocator()(
                (self.0).env,
                device.as_ptr(),
                mem_type as core::ffi::c_int,
            )
        })
    }

    /// Load and register an execution-provider shared library at `path` under `registration_name`
    /// (`RegisterExecutionProviderLibrary`). The EP becomes discoverable for device attach. The
    /// library must stay loaded for the environment's lifetime.
    pub fn register_execution_provider_library(
        &self, registration_name: &str, path: &str,
    ) -> Result<()> {
        let cname = CString::new(registration_name)
            .map_err(|_| Error::new(-1, "registration name contains a NUL"))?;
        let cpath =
            CString::new(path).map_err(|_| Error::new(-1, "library path contains a NUL"))?;
        check(unsafe {
            api().register_execution_provider_library()(
                (self.0).env,
                cname.as_ptr(),
                cpath.as_ptr(),
            )
        })
    }

    /// Unregister a previously-registered execution-provider library by `registration_name`
    /// (`UnregisterExecutionProviderLibrary`).
    ///
    /// # Safety
    ///
    /// No device, stream, allocator, session, or provider object created by this registration may
    /// remain alive. ORT may unload the provider library and invalidate all of those handles.
    pub unsafe fn unregister_execution_provider_library(
        &self, registration_name: &str,
    ) -> Result<()> {
        let cname = CString::new(registration_name)
            .map_err(|_| Error::new(-1, "registration name contains a NUL"))?;
        check(unsafe {
            api().unregister_execution_provider_library()((self.0).env, cname.as_ptr())
        })
    }
}

impl Drop for EnvInner {
    fn drop(&mut self) {
        // Last Arc ref gone (no Environment clone and no Session holding it) — release the Env.
        unsafe { api().release_env()(self.env) }
    }
}

// OrtEnv is immutable, thread-safe, and shared via Arc; safe to move across threads.
unsafe impl Send for EnvInner {}
unsafe impl Sync for EnvInner {}

// ─── EnvCreationOptions (ORT 1.24+ unified creation) ──────────────────────────
//
// `OrtEnvCreationOptions` is a plain value struct (not opaque) the caller stack-allocates and
// passes to `CreateEnvWithOptions`. We mirror its `#[repr(C)]` layout field-for-field (verified
// against onnxruntime_c_api.h:1212). The `version` field must be `ORT_API_VERSION`.

/// The C `OrtEnvCreationOptions` value struct, passed by pointer to `CreateEnvWithOptions`. All
/// pointers are read/copied synchronously by ORT during creation.
#[repr(C)]
struct RawEnvCreationOptions {
    /// Must be `ORT_API_VERSION` (= [`sys::API_VERSION`], pinned by the build to 1.27).
    version: u32,
    logging_severity_level: sys::LoggingLevel,
    log_id: *const c_char,
    custom_logging_function: sys::LoggingFunction,
    custom_logging_param: *mut c_void,
    threading_options: *const sys::ThreadingOptionsHandle,
    config_entries: *const sys::KeyValuePairsHandle,
}

/// Builder for [`Environment::new_with_options`] — the ORT 1.24+ unified Env-creation path
/// (`CreateEnvWithOptions`). Subsumes the separate custom-logger and global-thread-pool
/// constructors, and is the only path that accepts environment-level config entries.
pub struct EnvCreationOptions {
    level: sys::LoggingLevel,
    logid: String,
    logger: Option<UserLogger>,
    threading: Option<crate::threading::ThreadingOptions>,
    config_entries: Option<crate::allocator::KeyValuePairs>,
}

impl EnvCreationOptions {
    /// Start a builder for an Env logging at `level` with log id `logid`.
    pub fn new(level: sys::LoggingLevel, logid: &str) -> Self {
        Self {
            level,
            logid: logid.to_string(),
            logger: None,
            threading: None,
            config_entries: None,
        }
    }

    /// Route ORT log messages to `logger` (same lifetime/`Send+Sync` rules as
    /// [`Environment::new_with_logger`]).
    pub fn with_logger<L: Fn(LogRecord) + Send + Sync + 'static>(mut self, logger: L) -> Self {
        self.logger = Some(UserLogger::new(logger));
        self
    }

    /// Configure ORT global thread pools (sessions share them unless they opt out).
    pub fn with_thread_pools(mut self, threading: crate::threading::ThreadingOptions) -> Self {
        self.threading = Some(threading);
        self
    }

    /// Provide environment-level config entries (`ep_factory.<ep>.<key>` etc.). ORT copies the
    /// pairs at creation, so this `KeyValuePairs` need not outlive the Env.
    pub fn with_config_entries(mut self, kvps: crate::allocator::KeyValuePairs) -> Self {
        self.config_entries = Some(kvps);
        self
    }

    /// Materialize the Env (`CreateEnvWithOptions`). Consumes the builder; the logger (if any) is
    /// moved into the Env so it stays alive for the Env's lifetime.
    pub fn create_env(self) -> Result<Environment> {
        env_creation_checks()?;
        let cid = CString::new(self.logid.as_str())
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let (log_fn, log_param) = match &self.logger {
            Some(ul) => (logging_function(), ul.param()),
            None => (None, ptr::null_mut()),
        };
        let threading_ptr = self
            .threading
            .as_ref()
            .map(crate::threading::ThreadingOptions::as_ptr)
            .unwrap_or(ptr::null());
        let cfg_ptr = self
            .config_entries
            .as_ref()
            .map(crate::allocator::KeyValuePairs::raw_ptr)
            .unwrap_or(ptr::null());
        let raw = RawEnvCreationOptions {
            version: sys::API_VERSION,
            logging_severity_level: self.level,
            log_id: cid.as_ptr(),
            custom_logging_function: log_fn,
            custom_logging_param: log_param,
            threading_options: threading_ptr,
            config_entries: cfg_ptr,
        };
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        check(unsafe {
            api().create_env_with_options()(
                &raw as *const RawEnvCreationOptions as *const sys::EnvCreationOptionsHandle,
                &mut env,
            )
        })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Environment(Arc::new(EnvInner {
            env,
            threading: self.threading,
            _user_logger: self.logger,
        })))
    }
}
