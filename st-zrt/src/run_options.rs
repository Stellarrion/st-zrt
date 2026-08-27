//! Typed per-run configuration and its materialized ONNX Runtime handle.
//!
//! [`RunOptions`] is a cloneable pure value: constructing and composing it performs no FFI.
//! [`RunOptions::freeze`] validates strings and creates a reusable [`MaterializedRunOptions`].
//! Sessions and serving lanes materialize once during setup rather than mutating ORT handles on the
//! hot path.

use crate::{Error, Result, api, check, sys};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

/// Pure-value configuration for one reusable ORT run-options handle.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RunOptions {
    log_severity: Option<i32>,
    log_verbosity: Option<i32>,
    run_tag: Option<String>,
    profiling_prefix: Option<String>,
    config_entries: Vec<(String, String)>,
    #[cfg(feature = "ep")]
    #[cfg_attr(feature = "serde", serde(skip))]
    sync_stream: Option<Arc<crate::SyncStream>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    active_lora_adapters: Vec<Arc<crate::LoraAdapter>>,
}

impl std::fmt::Debug for RunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunOptions")
            .field("log_severity", &self.log_severity)
            .field("log_verbosity", &self.log_verbosity)
            .field("run_tag", &self.run_tag)
            .field("profiling_prefix", &self.profiling_prefix)
            .field("config_entries", &self.config_entries)
            .field("sync_stream", &{
                #[cfg(feature = "ep")]
                {
                    self.sync_stream.is_some()
                }
                #[cfg(not(feature = "ep"))]
                {
                    false
                }
            })
            .field("active_lora_adapters", &self.active_lora_adapters.len())
            .finish()
    }
}

impl RunOptions {
    /// An empty, pure-value configuration. This performs no ORT call and cannot fail.
    #[inline]
    pub const fn new() -> Self {
        Self {
            log_severity: None,
            log_verbosity: None,
            run_tag: None,
            profiling_prefix: None,
            config_entries: Vec::new(),
            #[cfg(feature = "ep")]
            sync_stream: None,
            active_lora_adapters: Vec::new(),
        }
    }

    /// Configure a CUDA-graph replay that retains ORT's normal end-of-run EP synchronization.
    #[inline]
    pub fn graph_replay(graph_id: i32) -> Self {
        Self::new().with_gpu_graph_id(graph_id)
    }

    /// Configure an enqueued CUDA-graph replay. The caller must fence provider work before reading
    /// outputs, reusing buffers, or releasing a graph lease.
    #[inline]
    pub fn enqueued(graph_id: i32) -> Self {
        Self::graph_replay(graph_id).with_disable_ep_sync(true)
    }

    #[inline]
    pub fn with_log_severity(mut self, level: sys::LoggingLevel) -> Self {
        self.log_severity = Some(level as i32);
        self
    }

    #[inline]
    pub fn with_log_verbosity(mut self, level: i32) -> Self {
        self.log_verbosity = Some(level);
        self
    }

    #[inline]
    pub fn with_run_tag(mut self, tag: impl Into<String>) -> Self {
        self.run_tag = Some(tag.into());
        self
    }

    #[inline]
    pub fn with_profiling(mut self, profile_file_prefix: impl Into<String>) -> Self {
        self.profiling_prefix = Some(profile_file_prefix.into());
        self
    }

    /// Typed CUDA-graph annotation. Replaces an earlier value for the same key.
    #[inline]
    pub fn with_gpu_graph_id(mut self, graph_id: i32) -> Self {
        self.upsert_config("gpu_graph_id".to_owned(), graph_id.to_string());
        self
    }

    /// Typed `disable_synchronize_execution_providers` configuration.
    #[inline]
    pub fn with_disable_ep_sync(mut self, disabled: bool) -> Self {
        self.upsert_config(
            "disable_synchronize_execution_providers".to_owned(),
            if disabled { "1" } else { "0" }.to_owned(),
        );
        self
    }

    /// Add an arbitrary ORT run-config key/value pair. A repeated key replaces its earlier value.
    pub fn with_config(mut self, key: &str, value: &str) -> Result<Self> {
        validate_cstring(key, "run config key")?;
        validate_cstring(value, "run config value")?;
        self.upsert_config(key.to_owned(), value.to_owned());
        Ok(self)
    }

    /// Attach an owned ORT synchronization stream. Both the pure config and frozen handle retain an
    /// `Arc`, so the stream and its environment cannot disappear while ORT references them.
    #[cfg(feature = "ep")]
    #[inline]
    pub fn with_sync_stream(mut self, stream: &Arc<crate::SyncStream>) -> Self {
        self.sync_stream = Some(Arc::clone(stream));
        self
    }

    /// Keep a LoRA adapter alive for every run using the materialized options.
    #[inline]
    pub fn with_lora_adapter(mut self, adapter: &Arc<crate::LoraAdapter>) -> Self {
        self.active_lora_adapters.push(Arc::clone(adapter));
        self
    }

    #[inline]
    pub fn log_severity(&self) -> Result<Option<sys::LoggingLevel>> {
        self.log_severity.map(logging_level_from_i32).transpose()
    }

    #[inline]
    pub const fn log_verbosity(&self) -> Option<i32> {
        self.log_verbosity
    }

    #[inline]
    pub fn run_tag(&self) -> Option<&str> {
        self.run_tag.as_deref()
    }

    #[inline]
    pub fn profiling_prefix(&self) -> Option<&str> {
        self.profiling_prefix.as_deref()
    }

    #[inline]
    pub fn config_entry(&self, key: &str) -> Option<&str> {
        self.config_entries
            .iter()
            .find_map(|(candidate, value)| (candidate == key).then_some(value.as_str()))
    }

    #[inline]
    pub fn graph_id(&self) -> Option<i32> {
        self.config_entry("gpu_graph_id")?.parse().ok()
    }

    /// Validate strings and create the frozen ORT handle once.
    pub fn freeze(self) -> Result<MaterializedRunOptions> {
        let materialized = MaterializedRunOptions::new(
            #[cfg(feature = "ep")]
            self.sync_stream,
            self.active_lora_adapters,
        )?;

        if let Some(level) = self.log_severity {
            logging_level_from_i32(level)?;
            check(unsafe {
                api().run_options_set_run_log_severity_level()(materialized.opts, level)
            })?;
        }
        if let Some(level) = self.log_verbosity {
            check(unsafe {
                api().run_options_set_run_log_verbosity_level()(materialized.opts, level)
            })?;
        }
        if let Some(tag) = self.run_tag {
            let tag = validate_cstring(&tag, "run tag")?;
            check(unsafe { api().run_options_set_run_tag()(materialized.opts, tag.as_ptr()) })?;
        }
        if let Some(prefix) = self.profiling_prefix {
            let prefix = validate_cstring(&prefix, "profile file prefix")?;
            check(unsafe {
                api().run_options_enable_profiling()(materialized.opts, prefix.as_ptr())
            })?;
        }
        for (key, value) in self.config_entries {
            let key = validate_cstring(&key, "run config key")?;
            let value = validate_cstring(&value, "run config value")?;
            check(unsafe {
                api().add_run_config_entry()(materialized.opts, key.as_ptr(), value.as_ptr())
            })?;
        }
        #[cfg(feature = "ep")]
        if let Some(stream) = &materialized.sync_stream {
            unsafe { api().run_options_set_sync_stream()(materialized.opts, stream.as_ptr()) };
        }
        for adapter in &materialized.active_lora_adapters {
            check(unsafe {
                api().run_options_add_active_lora_adapter()(materialized.opts, adapter.as_ptr())
            })?;
        }
        Ok(materialized)
    }

    fn upsert_config(&mut self, key: String, value: String) {
        if let Some((_, existing)) = self
            .config_entries
            .iter_mut()
            .find(|(candidate, _)| candidate == &key)
        {
            *existing = value;
        } else {
            self.config_entries.push((key, value));
        }
    }
}

/// A frozen, reusable `OrtRunOptions` handle.
///
/// The handle owns `Arc` guards for every LoRA adapter borrowed by ORT. Configuration cannot be
/// changed after materialization; only the thread-safe cancellation bit may be set or reset.
pub struct MaterializedRunOptions {
    pub(crate) opts: *mut sys::RunOptionsHandle,
    // `Drop` releases `opts` before these guards are dropped.
    #[cfg(feature = "ep")]
    sync_stream: Option<Arc<crate::SyncStream>>,
    active_lora_adapters: Vec<Arc<crate::LoraAdapter>>,
}

impl MaterializedRunOptions {
    fn new(
        #[cfg(feature = "ep")] sync_stream: Option<Arc<crate::SyncStream>>,
        active_lora_adapters: Vec<Arc<crate::LoraAdapter>>,
    ) -> Result<Self> {
        let mut opts: *mut sys::RunOptionsHandle = ptr::null_mut();
        check(unsafe { api().create_run_options()(&mut opts) })?;
        let opts = crate::ensure_non_null(opts, "run options")?;
        Ok(Self {
            opts,
            #[cfg(feature = "ep")]
            sync_stream,
            active_lora_adapters,
        })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::RunOptionsHandle {
        self.opts as *const sys::RunOptionsHandle
    }

    pub(crate) fn shares_environment(&self, _env: &Arc<crate::environment::EnvInner>) -> bool {
        #[cfg(feature = "ep")]
        if let Some(stream) = &self.sync_stream {
            return stream.shares_env_guard(_env);
        }
        true
    }

    pub fn log_severity(&self) -> Result<sys::LoggingLevel> {
        let mut value = 0;
        check(unsafe { api().run_options_get_run_log_severity_level()(self.opts, &mut value) })?;
        logging_level_from_i32(value)
    }

    pub fn log_verbosity(&self) -> Result<i32> {
        let mut value = 0;
        check(unsafe { api().run_options_get_run_log_verbosity_level()(self.opts, &mut value) })?;
        Ok(value)
    }

    pub fn run_tag(&self) -> Result<String> {
        let mut value: *const core::ffi::c_char = ptr::null();
        check(unsafe {
            api().run_options_get_run_tag()(
                self.opts,
                &mut value as *mut _ as *const *const core::ffi::c_char,
            )
        })?;
        if value.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(value, "run tag") }
        }
    }

    pub fn config_entry(&self, key: &str) -> Result<Option<String>> {
        let key = validate_cstring(key, "run config key")?;
        let value = unsafe { api().get_run_config_entry()(self.opts, key.as_ptr()) };
        if value.is_null() {
            Ok(None)
        } else {
            unsafe { crate::cstr_to_string(value, "run config entry") }.map(Some)
        }
    }

    /// Request termination. ORT permits this from another thread while a run is in flight.
    pub fn terminate(&self) -> Result<()> {
        check(unsafe { api().run_options_set_terminate()(self.opts) })
    }

    /// Clear a previous termination request before reusing this handle.
    pub fn unset_terminate(&self) -> Result<()> {
        check(unsafe { api().run_options_unset_terminate()(self.opts) })
    }
}

impl std::fmt::Debug for MaterializedRunOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaterializedRunOptions")
            .field("opts", &self.opts)
            .field("sync_stream", &{
                #[cfg(feature = "ep")]
                {
                    self.sync_stream.is_some()
                }
                #[cfg(not(feature = "ep"))]
                {
                    false
                }
            })
            .field("active_lora_adapters", &self.active_lora_adapters.len())
            .finish()
    }
}

impl Drop for MaterializedRunOptions {
    fn drop(&mut self) {
        if !self.opts.is_null() {
            unsafe { api().release_run_options()(self.opts) }
        }
    }
}

// ORT documents one RunOptions handle as reusable by concurrent runs; terminate is thread-safe.
unsafe impl Send for MaterializedRunOptions {}
unsafe impl Sync for MaterializedRunOptions {}

fn validate_cstring(value: &str, what: &str) -> Result<CString> {
    CString::new(value).map_err(|_| Error::new(-1, format!("{what} contains a NUL")))
}

/// Map an `OrtLoggingLevel` discriminant (`#[repr(i32)]`, 0–4) to the enum, rejecting unknowns.
fn logging_level_from_i32(value: i32) -> Result<sys::LoggingLevel> {
    Ok(match value {
        0 => sys::LoggingLevel::Verbose,
        1 => sys::LoggingLevel::Info,
        2 => sys::LoggingLevel::Warning,
        3 => sys::LoggingLevel::Error,
        4 => sys::LoggingLevel::Fatal,
        other => {
            return Err(Error::new(
                -1,
                format!("zrt: unknown OrtLoggingLevel value {other}"),
            ));
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_presets_compose_and_upsert_without_ffi() {
        let plain = RunOptions::new();
        assert_eq!(plain.graph_id(), None);
        assert_eq!(
            plain.config_entry("disable_synchronize_execution_providers"),
            None
        );

        let graph = RunOptions::graph_replay(7).with_gpu_graph_id(11);
        assert_eq!(graph.graph_id(), Some(11));
        assert_eq!(graph.config_entries.len(), 1, "typed graph id must upsert");

        let enqueued = RunOptions::enqueued(7);
        assert_eq!(enqueued.graph_id(), Some(7));
        assert_eq!(
            enqueued.config_entry("disable_synchronize_execution_providers"),
            Some("1")
        );
    }

    #[test]
    fn raw_config_rejects_nuls_before_materialization() {
        let error = RunOptions::new()
            .with_config("bad\0key", "value")
            .expect_err("NUL key must fail");
        assert!(error.to_string().contains("run config key contains a NUL"));
    }

    #[test]
    fn deferred_string_validation_fails_during_freeze() {
        let error = RunOptions::new()
            .with_run_tag("bad\0tag")
            .freeze()
            .expect_err("NUL tag must fail while freezing");
        assert!(error.to_string().contains("run tag contains a NUL"));
    }

    #[cfg(feature = "ep")]
    #[test]
    fn frozen_options_own_stream_and_enforce_environment_identity() {
        let _envs = crate::lock_default_env_creation();
        let env_a = crate::Environment::new().expect("env a");
        let env_b = crate::Environment::new().expect("env b");
        let stream = crate::SyncStream::null_for_test(&env_a);
        let weak = Arc::downgrade(&stream);
        let config = RunOptions::new().with_sync_stream(&stream);
        assert_eq!(Arc::strong_count(&stream), 2);
        let cloned = config.clone();
        assert_eq!(Arc::strong_count(&stream), 3);
        drop(cloned);
        let materialized =
            MaterializedRunOptions::new(Some(config.sync_stream.unwrap()), Vec::new())
                .expect("materialize without attaching test-null stream");
        assert!(materialized.shares_environment(&env_a.share()));
        assert!(!materialized.shares_environment(&env_b.share()));
        drop(stream);
        assert!(
            weak.upgrade().is_some(),
            "frozen options must retain stream"
        );
        drop(materialized);
        assert!(weak.upgrade().is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn pure_config_serde_round_trip() {
        let config = RunOptions::enqueued(23)
            .with_log_severity(sys::LoggingLevel::Error)
            .with_log_verbosity(4)
            .with_run_tag("serde-run")
            .with_profiling("serde-profile")
            .with_config("custom.key", "custom-value")
            .expect("config");
        let json = serde_json::to_string(&config).expect("serialize");
        let back: RunOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            back.log_severity().expect("severity"),
            Some(sys::LoggingLevel::Error)
        );
        assert_eq!(back.log_verbosity(), Some(4));
        assert_eq!(back.run_tag(), Some("serde-run"));
        assert_eq!(back.profiling_prefix(), Some("serde-profile"));
        assert_eq!(back.graph_id(), Some(23));
        assert_eq!(back.config_entry("custom.key"), Some("custom-value"));
    }
    #[cfg(all(feature = "serde", feature = "ep"))]
    #[test]
    fn serde_deliberately_drops_live_sync_stream() {
        let _envs = crate::lock_default_env_creation();
        let env = crate::Environment::new().expect("env");
        let stream = crate::SyncStream::null_for_test(&env);
        let config = RunOptions::new().with_sync_stream(&stream);
        let json = serde_json::to_string(&config).expect("serialize");
        let back: RunOptions = serde_json::from_str(&json).expect("deserialize");
        assert!(back.sync_stream.is_none());
    }
}
