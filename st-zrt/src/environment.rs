//! The ORT environment (logging + global state). One per process is typical.
//!
//! The raw `OrtEnv` handle is wrapped in an [`Arc`] so it can be shared cheaply: a
//! [`crate::Session`] takes its own `Arc` clone at construction, which **keeps the Env
//! alive for the Session's whole lifetime**. This matters because ORT sessions reference
//! the Env's thread pools/allocator — releasing the Env while a Session still lives is a
//! use-after-free (the root cause of the historical ">4MB segfault"; see RESULTS.md §8).
//! Because the Session owns an `Arc` ref, that UAF can no longer occur regardless of how a
//! caller scopes its `Environment`.
use crate::{Error, Result, api, check, sys};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

/// Owning inner: holds the raw `OrtEnv` handle and releases it (`ReleaseEnv`) when the last
/// `Arc` reference drops. Kept `pub(crate)` so [`crate::Session`] can hold an `Arc<EnvInner>`.
pub(crate) struct EnvInner {
    env: *mut sys::EnvHandle,
    threading: Option<crate::threading::ThreadingOptions>,
}

/// The ORT environment (logging + global state). Cheap to clone (one `Arc` refcount); the
/// underlying `OrtEnv` is shared and freed only when the last clone AND every `Session`
/// derived from it are dropped.
#[derive(Clone)]
pub struct Environment(Arc<EnvInner>);

impl Environment {
    /// Create with `Warning` log level and the log id `"zrt"`.
    pub fn new() -> Result<Self> {
        Self::new_with_level(sys::LoggingLevel::Warning, "zrt")
    }

    /// Create with a custom log level and log id.
    pub fn new_with_level(level: sys::LoggingLevel, logid: &str) -> Result<Self> {
        let cid = CString::new(logid)
            .map_err(|_| Error::new(-1, "environment log id contains a NUL byte"))?;
        let mut env: *mut sys::EnvHandle = ptr::null_mut();
        // cid is copied internally by CreateEnv; safe for it to be a local.
        check(unsafe { api().create_env()(level, cid.as_ptr(), &mut env) })?;
        let env = crate::ensure_non_null(env, "environment")?;
        Ok(Self(Arc::new(EnvInner {
            env,
            threading: None,
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
        })))
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
