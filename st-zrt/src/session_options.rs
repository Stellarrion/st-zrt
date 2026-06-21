//! `SessionOptions` — pure-value session configuration.
//!
//! This is a plain config struct: builder methods only set fields (infallible — no FFI).
//! The ORT `SessionOptions` handle is materialized once, inside [`crate::Session::new`],
//! via [`SessionOptions::build_handle`]. This is the foundation for a future auto
//! thread-policy (the config can carry a policy before any handle exists).
use crate::{api, check, sys, Result};
use std::ffi::CString;
use std::ptr;

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
    /// attach call needs the environment, which `build_handle` doesn't take). The device
    /// pointers are borrowed from the environment. Not serializable — skipped under `serde`.
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
            cpu_mem_arena: ArenaState::Default,
            mem_pattern: MemPatternState::Default,
            use_global_thread_pool: true,
            profiling_prefix: None,
            free_dimension_overrides: Vec::new(),
            free_dimension_overrides_by_name: Vec::new(),
            config_entries: Vec::new(),
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

    /// Materialize an ORT `SessionOptions` handle from this config. The caller owns
    /// and must release the returned handle (`CreateSession` copies the options).
    pub(crate) fn build_handle(&self) -> Result<*mut sys::SessionOptionsHandle> {
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
    if enabled {
        "1"
    } else {
        "0"
    }
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
