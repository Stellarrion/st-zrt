//! `RunOptions` — per-run knobs: log severity/verbosity, run tag, config entries, and
//! terminate (cancel an in-flight run). A session owns a default one (used by
//! [`crate::Session::run`] / [`crate::Session::run_binding`]); for per-call config or
//! cancellation, build a `RunOptions` and pass it to
//! [`crate::Session::run_with`] / [`crate::Session::run_binding_with`].
use crate::{api, check, sys, Error, Result};
use std::ffi::CString;
use std::ptr;

pub struct RunOptions {
    pub(crate) opts: *mut sys::RunOptionsHandle,
}

impl RunOptions {
    /// Create a default run-options handle (all knobs unset). Released on drop.
    pub fn new() -> Result<Self> {
        let mut opts: *mut sys::RunOptionsHandle = ptr::null_mut();
        check(unsafe { api().create_run_options()(&mut opts) })?;
        let opts = crate::ensure_non_null(opts, "run options")?;
        Ok(Self { opts })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::RunOptionsHandle {
        self.opts as *const sys::RunOptionsHandle
    }

    /// Set the per-run log severity level (`RunOptionsSetRunLogSeverityLevel`).
    pub fn set_log_severity(&mut self, level: sys::LoggingLevel) -> Result<&mut Self> {
        check(unsafe {
            api().run_options_set_run_log_severity_level()(self.opts, level as core::ffi::c_int)
        })?;
        Ok(self)
    }

    /// Set the per-run log verbosity level (`RunOptionsSetRunLogVerbosityLevel`).
    pub fn set_log_verbosity(&mut self, level: i32) -> Result<&mut Self> {
        check(unsafe { api().run_options_set_run_log_verbosity_level()(self.opts, level) })?;
        Ok(self)
    }

    /// Set the per-run tag (a label that appears in logs). `RunOptionsSetRunTag`.
    pub fn set_run_tag(&mut self, tag: &str) -> Result<&mut Self> {
        let ctag = CString::new(tag).map_err(|_| Error::new(-1, "run tag contains a NUL"))?;
        check(unsafe { api().run_options_set_run_tag()(self.opts, ctag.as_ptr()) })?;
        Ok(self)
    }

    /// Add a key/value config entry to this run (`AddRunConfigEntry`). These are the
    /// `ortrun.*` / EP runtime knobs documented in `onnxruntime_run_options_config_keys.h`.
    pub fn add_config_entry(&mut self, key: &str, value: &str) -> Result<&mut Self> {
        let ckey = CString::new(key).map_err(|_| Error::new(-1, "config key contains a NUL"))?;
        let cval =
            CString::new(value).map_err(|_| Error::new(-1, "config value contains a NUL"))?;
        check(unsafe { api().add_run_config_entry()(self.opts, ckey.as_ptr(), cval.as_ptr()) })?;
        Ok(self)
    }

    /// Request termination of the run using this handle (`RunOptionsSetTerminate`). Safe to
    /// call from another thread while a run is in flight — ORT's terminate is thread-safe,
    /// so this takes `&self` and works through `Arc<RunOptions>`. The in-flight run returns
    /// an `Err` (ORT error code for cancel).
    pub fn terminate(&self) -> Result<()> {
        check(unsafe { api().run_options_set_terminate()(self.opts) })
    }
}

impl Drop for RunOptions {
    fn drop(&mut self) {
        if !self.opts.is_null() {
            unsafe { api().release_run_options()(self.opts) }
        }
    }
}
unsafe impl Send for RunOptions {}
unsafe impl Sync for RunOptions {}
