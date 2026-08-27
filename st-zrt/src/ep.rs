//! Execution-provider option builders (feature `ep`).
//!
//! GPU/accelerator EPs (CUDA, TensorRT, ROCm, CANN, DNNL) are configured via key/value
//! "provider options". Each [`*Options`] wraps the ORT options handle (create / update /
//! as_string / release); queue one on a [`crate::SessionOptions`] via
//! [`crate::SessionOptions::with_execution_provider`] and it is appended at session
//! creation. The options structs are pure config — creating/updating them does NOT load
//! the EP — so the lifecycle is exercisable on any host; a GPU/accelerator is needed only
//! to actually *run* a session with the EP appended.
//!
//! (OpenVINO V2 and VitisAI are also wrapped — they take key/value options directly at append
//! time, with no options handle. MIGraphX and the **deprecated** OpenVINO v1 are wrapped via flat
//! `#[repr(C)]` config structs ([`MigraphxOptions`] / [`OpenvinoOptions`]); prefer OpenVINO V2
//! over v1.)
use crate::allocator::Allocator;
use crate::session_options::SessionOptions;
use crate::{Result, api, check, sys};
use std::ffi::{CString, c_char, c_void};
use std::ptr;
use std::sync::Arc;

/// A supported execution provider. The options-struct path (CUDA/TRT/ROCm/CANN/DNNL).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EpProvider {
    Cuda,
    TensorRt,
    Rocm,
    Cann,
    Dnnl,
    /// OpenVINO — the modern V2 key/value append path (`SessionOptionsAppendExecutionProvider_OpenVINO_V2`).
    OpenVinoV2,
    /// VitisAI — key/value append path (`SessionOptionsAppendExecutionProvider_VitisAI`).
    VitisAi,
}

/// CUDA device arena growth strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CudaArenaExtendStrategy {
    NextPowerOfTwo,
    SameAsRequested,
}

impl CudaArenaExtendStrategy {
    #[inline]
    fn as_ort_value(self) -> &'static str {
        match self {
            Self::NextPowerOfTwo => "kNextPowerOfTwo",
            Self::SameAsRequested => "kSameAsRequested",
        }
    }
}

/// cuDNN convolution algorithm search mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CudaCudnnConvAlgoSearch {
    Exhaustive,
    Heuristic,
    Default,
}

impl CudaCudnnConvAlgoSearch {
    #[inline]
    fn as_ort_value(self) -> &'static str {
        match self {
            Self::Exhaustive => "EXHAUSTIVE",
            Self::Heuristic => "HEURISTIC",
            Self::Default => "DEFAULT",
        }
    }
}

/// Placement and sequencing policy for CUDA execution-provider inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DeviceInputPolicy {
    /// CUDA EP copies inputs on its default stream (`do_copy_in_default_stream=1`).
    DefaultStream,
    /// CUDA EP uses one unified provider-level stream (`use_ep_level_unified_stream=1`).
    UnifiedStream,
    /// CUDA EP uses the owned caller stream retained by this configuration.
    UserStream,
}

/// Typed CUDA execution-provider configuration.
///
/// Unknown future string options remain available through [`Self::with_raw`]. A user-stream policy
/// can only be built with `CudaConfig::graph_replay` or `CudaConfig::with_user_stream`, both of which retain
/// an `Arc<CudaStream>`; session construction transfers that guard into `SessionInner`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CudaConfig {
    device_id: i32,
    arena: CudaArenaExtendStrategy,
    cudnn_search: CudaCudnnConvAlgoSearch,
    tf32: bool,
    cuda_graph: bool,
    graph_strict_mode: bool,
    device_inputs: DeviceInputPolicy,
    mem_limit: Option<usize>,
    cudnn_max_workspace: bool,
    prefer_nhwc: bool,
    extra: Vec<(String, String)>,
    #[cfg(feature = "cuda")]
    #[cfg_attr(feature = "serde", serde(skip))]
    stream: Option<Arc<crate::CudaStream>>,
}

impl CudaConfig {
    /// Latency/throughput preset when memory is not the primary constraint.
    pub fn performance(device_id: i32) -> Self {
        Self {
            device_id,
            arena: CudaArenaExtendStrategy::NextPowerOfTwo,
            cudnn_search: CudaCudnnConvAlgoSearch::Exhaustive,
            tf32: true,
            cuda_graph: false,
            graph_strict_mode: false,
            device_inputs: DeviceInputPolicy::DefaultStream,
            mem_limit: None,
            cudnn_max_workspace: false,
            prefer_nhwc: false,
            extra: Vec::new(),
            #[cfg(feature = "cuda")]
            stream: None,
        }
    }

    /// Bounded-memory preset with conservative arena growth and heuristic cuDNN search.
    pub fn low_memory(device_id: i32, gpu_mem_limit: usize) -> Self {
        Self {
            arena: CudaArenaExtendStrategy::SameAsRequested,
            cudnn_search: CudaCudnnConvAlgoSearch::Heuristic,
            mem_limit: Some(gpu_mem_limit),
            ..Self::performance(device_id)
        }
    }

    /// Static-shape CUDA-graph replay on an owned caller stream.
    #[cfg(feature = "cuda")]
    pub fn graph_replay(device_id: i32, stream: &Arc<crate::CudaStream>) -> Result<Self> {
        if stream.device_id() != device_id {
            return Err(crate::Error::new(
                -1,
                format!(
                    "CUDA stream belongs to device {}, but config targets device {device_id}",
                    stream.device_id()
                ),
            ));
        }
        Ok(Self {
            cuda_graph: true,
            graph_strict_mode: true,
            device_inputs: DeviceInputPolicy::UserStream,
            stream: Some(Arc::clone(stream)),
            ..Self::performance(device_id)
        })
    }

    #[inline]
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    #[inline]
    pub const fn device_input_policy(&self) -> DeviceInputPolicy {
        self.device_inputs
    }

    #[inline]
    pub const fn cuda_graph_enabled(&self) -> bool {
        self.cuda_graph
    }

    pub fn with_raw(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if is_reserved_cuda_key(&key) {
            return Err(crate::Error::new(
                -1,
                format!("CUDA option {key:?} is typed and cannot be overridden through with_raw"),
            ));
        }
        upsert_string_entry(&mut self.extra, key, value.into());
        Ok(self)
    }

    #[inline]
    pub fn with_mem_limit(mut self, bytes: usize) -> Self {
        self.mem_limit = Some(bytes);
        self
    }

    #[inline]
    pub fn with_cuda_graph(mut self, enabled: bool) -> Self {
        self.cuda_graph = enabled;
        self
    }

    #[inline]
    pub fn with_graph_strict_mode(mut self, enabled: bool) -> Self {
        self.graph_strict_mode = enabled;
        self
    }

    #[inline]
    pub fn with_tf32(mut self, enabled: bool) -> Self {
        self.tf32 = enabled;
        self
    }

    #[inline]
    pub fn with_cudnn_max_workspace(mut self, enabled: bool) -> Self {
        self.cudnn_max_workspace = enabled;
        self
    }

    #[inline]
    pub fn with_prefer_nhwc(mut self, enabled: bool) -> Self {
        self.prefer_nhwc = enabled;
        self
    }

    #[inline]
    pub fn with_device_input_policy(mut self, policy: DeviceInputPolicy) -> Result<Self> {
        if policy == DeviceInputPolicy::UserStream {
            #[cfg(not(feature = "cuda"))]
            return Err(crate::Error::new(
                -1,
                "DeviceInputPolicy::UserStream requires the `cuda` feature and an owned stream",
            ));
            #[cfg(feature = "cuda")]
            if self.stream.is_none() {
                return Err(crate::Error::new(
                    -1,
                    "DeviceInputPolicy::UserStream requires with_user_stream",
                ));
            }
        }
        self.device_inputs = policy;
        #[cfg(feature = "cuda")]
        if policy != DeviceInputPolicy::UserStream {
            self.stream = None;
        }
        Ok(self)
    }

    #[cfg(feature = "cuda")]
    pub fn with_user_stream(mut self, stream: &Arc<crate::CudaStream>) -> Result<Self> {
        if stream.device_id() != self.device_id {
            return Err(crate::Error::new(
                -1,
                format!(
                    "CUDA stream belongs to device {}, but config targets device {}",
                    stream.device_id(),
                    self.device_id
                ),
            ));
        }
        self.device_inputs = DeviceInputPolicy::UserStream;
        self.stream = Some(Arc::clone(stream));
        Ok(self)
    }

    fn provider_options(&self) -> CudaProviderOptions {
        let mut options = CudaProviderOptions::new()
            .device_id(self.device_id)
            .arena_extend_strategy(self.arena)
            .cudnn_conv_algo_search(self.cudnn_search)
            .enable_cuda_graph(self.cuda_graph)
            .enable_skip_layer_norm_strict_mode(self.graph_strict_mode)
            .use_tf32(self.tf32)
            .cudnn_conv_use_max_workspace(self.cudnn_max_workspace)
            .prefer_nhwc(self.prefer_nhwc);
        options = match self.device_inputs {
            DeviceInputPolicy::DefaultStream => options
                .do_copy_in_default_stream(true)
                .use_ep_level_unified_stream(false),
            DeviceInputPolicy::UnifiedStream => options
                .do_copy_in_default_stream(true)
                .use_ep_level_unified_stream(true),
            DeviceInputPolicy::UserStream => options
                .do_copy_in_default_stream(true)
                .use_ep_level_unified_stream(false),
        };
        if let Some(limit) = self.mem_limit {
            options = options.gpu_mem_limit(limit);
        }
        for (key, value) in &self.extra {
            options = options.with_raw_unchecked(key.clone(), value.clone());
        }
        options
    }

    #[cfg(feature = "cuda")]
    fn stream(&self) -> Option<&Arc<crate::CudaStream>> {
        self.stream.as_ref()
    }
}

impl Default for CudaConfig {
    fn default() -> Self {
        Self::performance(0)
    }
}

impl From<&CudaConfig> for CudaProviderOptions {
    fn from(config: &CudaConfig) -> Self {
        config.provider_options()
    }
}

impl From<CudaConfig> for CudaProviderOptions {
    fn from(config: CudaConfig) -> Self {
        config.provider_options()
    }
}

fn is_reserved_cuda_key(key: &str) -> bool {
    matches!(
        key,
        "device_id"
            | "arena_extend_strategy"
            | "cudnn_conv_algo_search"
            | "use_tf32"
            | "enable_cuda_graph"
            | "enable_skip_layer_norm_strict_mode"
            | "do_copy_in_default_stream"
            | "use_ep_level_unified_stream"
            | "gpu_mem_limit"
            | "cudnn_conv_use_max_workspace"
            | "prefer_nhwc"
            | "user_compute_stream"
            | "gpu_external_alloc"
            | "gpu_external_free"
            | "gpu_external_empty_cache"
    )
}

fn is_unsafe_cuda_pointer_key(key: &str) -> bool {
    matches!(
        key,
        "user_compute_stream"
            | "gpu_external_alloc"
            | "gpu_external_free"
            | "gpu_external_empty_cache"
    )
}

fn upsert_string_entry(entries: &mut Vec<(String, String)>, key: String, value: String) {
    if let Some((_, existing)) = entries.iter_mut().find(|(candidate, _)| candidate == &key) {
        *existing = value;
    } else {
        entries.push((key, value));
    }
}

/// Pure-value CUDA execution-provider configuration.
///
/// This covers ORT CUDA provider string options. Unknown future non-pointer options can be supplied
/// with [`Self::with_raw`]. Process-local stream and allocator callback pointers are rejected;
/// configure owned streams through [`CudaConfig`] or use [`CudaOptions::update_with_value`] only in
/// an explicitly unsafe low-level integration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CudaProviderOptions {
    entries: Vec<(String, String)>,
}

impl CudaProviderOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a raw CUDA provider key/value option.
    ///
    /// Use this for non-pointer ORT options added after this wrapper. Process-local pointer options
    /// are rejected.
    pub fn with_raw(mut self, key: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if is_unsafe_cuda_pointer_key(&key) {
            return Err(crate::Error::new(
                -1,
                "pointer-valued CUDA options require an explicitly unsafe low-level API",
            ));
        }
        upsert_string_entry(&mut self.entries, key, value.into());
        Ok(self)
    }

    fn with_raw_unchecked(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        upsert_string_entry(&mut self.entries, key.into(), value.into());
        self
    }

    pub fn device_id(self, device_id: i32) -> Self {
        self.with_raw_unchecked("device_id", device_id.to_string())
    }

    pub fn do_copy_in_default_stream(self, enabled: bool) -> Self {
        self.with_bool("do_copy_in_default_stream", enabled)
    }

    pub fn use_ep_level_unified_stream(self, enabled: bool) -> Self {
        self.with_bool("use_ep_level_unified_stream", enabled)
    }

    pub fn gpu_mem_limit(self, bytes: usize) -> Self {
        self.with_raw_unchecked("gpu_mem_limit", bytes.to_string())
    }

    pub fn arena_extend_strategy(self, strategy: CudaArenaExtendStrategy) -> Self {
        self.with_raw_unchecked("arena_extend_strategy", strategy.as_ort_value())
    }

    pub fn cudnn_conv_algo_search(self, search: CudaCudnnConvAlgoSearch) -> Self {
        self.with_raw_unchecked("cudnn_conv_algo_search", search.as_ort_value())
    }

    pub fn cudnn_conv_use_max_workspace(self, enabled: bool) -> Self {
        self.with_bool("cudnn_conv_use_max_workspace", enabled)
    }

    pub fn cudnn_conv1d_pad_to_nc1d(self, enabled: bool) -> Self {
        self.with_bool("cudnn_conv1d_pad_to_nc1d", enabled)
    }

    pub fn enable_cuda_graph(self, enabled: bool) -> Self {
        self.with_bool("enable_cuda_graph", enabled)
    }

    pub fn enable_skip_layer_norm_strict_mode(self, enabled: bool) -> Self {
        self.with_bool("enable_skip_layer_norm_strict_mode", enabled)
    }

    pub fn use_tf32(self, enabled: bool) -> Self {
        self.with_bool("use_tf32", enabled)
    }

    pub fn prefer_nhwc(self, enabled: bool) -> Self {
        self.with_bool("prefer_nhwc", enabled)
    }

    pub fn tunable_op_enable(self, enabled: bool) -> Self {
        self.with_bool("tunable_op_enable", enabled)
    }

    pub fn tunable_op_tuning_enable(self, enabled: bool) -> Self {
        self.with_bool("tunable_op_tuning_enable", enabled)
    }

    pub fn tunable_op_max_tuning_duration_ms(self, duration_ms: i32) -> Self {
        self.with_raw_unchecked("tunable_op_max_tuning_duration_ms", duration_ms.to_string())
    }

    #[inline]
    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    fn with_bool(self, key: &'static str, enabled: bool) -> Self {
        self.with_raw_unchecked(key, if enabled { "1" } else { "0" })
    }

    fn entry_refs(entries: &[(String, String)]) -> Vec<(&str, &str)> {
        entries
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect()
    }
}

/// A queued EP append: which provider + its key/value options. Pure data (no handles) so
/// [`SessionOptions`] stays cloneable and free of EP types when the feature is off.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct EpConfig {
    provider: EpProvider,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::kv_pairs"))]
    entries: Vec<(CString, CString)>,
    #[cfg_attr(feature = "serde", serde(default))]
    requires_owned_cuda_stream: bool,
    #[cfg(feature = "cuda")]
    #[cfg_attr(feature = "serde", serde(skip))]
    cuda_stream: Option<Arc<crate::CudaStream>>,
}

/// Generate an EP options type for one provider:
/// - `new(entries)` — create the handle + apply key/value pairs
/// - `as_string()` — serialize back to `"k=v;…"`
/// - RAII release on drop
/// - `append_raw(opts)` — register on a session-options handle
macro_rules! ep_options {
    ($Type:ident, $handle:ty, $create:ident, $update:ident, $as_string:ident, $release:ident, $append:ident) => {
        /// Provider options for an execution provider. Built from key/value pairs;
        /// released on drop.
        pub struct $Type(*mut $handle);

        impl $Type {
            /// Create the options handle and apply `entries` (key/value pairs, e.g.
            /// `("device_id", "0")`). Errors if a key/value contains a NUL byte.
            pub fn new(entries: &[(&str, &str)]) -> Result<Self> {
                let api = api();
                let mut h: *mut $handle = ptr::null_mut();
                check(unsafe { api.$create()(&mut h) })?;
                let h = crate::ensure_non_null(h, "execution provider options")?;
                let me = Self(h);
                let cstrs: Vec<(CString, CString)> = entries
                    .iter()
                    .map(|(k, v)| Ok((CString::new(*k)?, CString::new(*v)?)))
                    .collect::<std::result::Result<_, std::ffi::NulError>>()
                    .map_err(|_| {
                        crate::Error::new(-1, "ep option key/value contains a NUL byte")
                    })?;
                let keys: Vec<*const c_char> = cstrs.iter().map(|(k, _)| k.as_ptr()).collect();
                let vals: Vec<*const c_char> = cstrs.iter().map(|(_, v)| v.as_ptr()).collect();
                check(unsafe { api.$update()(me.0, keys.as_ptr(), vals.as_ptr(), entries.len()) })?;
                Ok(me)
            }

            /// Serialize the options to a string (e.g. `"device_id=0;…"`). The buffer is
            /// engine-allocated and freed via the default allocator.
            pub fn as_string(&self) -> Result<String> {
                let api = api();
                let alloc = Allocator::get_default()?;
                let mut raw: *mut c_char = ptr::null_mut();
                check(unsafe {
                    api.$as_string()(self.0 as *const $handle, alloc.alloc, &mut raw)
                })?;
                if raw.is_null() {
                    return Ok(String::new());
                }
                let s = unsafe { crate::cstr_to_string(raw, "execution provider options") };
                let free = unsafe { alloc.free(raw as *mut c_void) };
                free?;
                s
            }

            /// Append this EP to a session-options handle (`SessionOptionsAppend…`).
            pub(crate) fn append_raw(&self, opts: *mut sys::SessionOptionsHandle) -> Result<()> {
                check(unsafe { api().$append()(opts, self.0 as *const $handle) })
            }
        }

        impl Drop for $Type {
            fn drop(&mut self) {
                unsafe { api().$release()(self.0) }
            }
        }
        unsafe impl Send for $Type {}
        unsafe impl Sync for $Type {}
    };
}

ep_options!(
    CudaOptions,
    sys::CUDAProviderOptionsV2Handle,
    create_cuda_provider_options,
    update_cuda_provider_options,
    get_cuda_provider_options_as_string,
    release_cuda_provider_options,
    session_options_append_execution_provider_cuda_v2
);

impl CudaOptions {
    /// Create a live ORT CUDA options handle from the raw string escape hatch.
    pub fn from_config(config: &CudaProviderOptions) -> Result<Self> {
        if config
            .entries()
            .iter()
            .any(|(key, _)| is_unsafe_cuda_pointer_key(key))
        {
            return Err(crate::Error::new(
                -1,
                "pointer-valued CUDA options require update_with_value in an unsafe low-level path",
            ));
        }
        let refs = CudaProviderOptions::entry_refs(config.entries());
        Self::new(&refs)
    }

    /// Update a pointer-valued CUDA provider option on this live ORT options handle.
    ///
    /// # Safety
    ///
    /// `value` must point to an object valid for the lifetime ORT requires for `key`.
    pub unsafe fn update_with_value(&mut self, key: &str, value: *mut c_void) -> Result<&mut Self> {
        unsafe {
            let key = CString::new(key)
                .map_err(|_| crate::Error::new(-1, "ep option key contains a NUL byte"))?;
            check(api().update_cuda_provider_options_with_value()(
                self.0,
                key.as_ptr(),
                value,
            ))?;
            Ok(self)
        }
    }
}
ep_options!(
    TensorRtOptions,
    sys::TensorRTProviderOptionsV2Handle,
    create_tensor_rt_provider_options,
    update_tensor_rt_provider_options,
    get_tensor_rt_provider_options_as_string,
    release_tensor_rt_provider_options,
    session_options_append_execution_provider__tensor_rt_v2
);
ep_options!(
    RocmOptions,
    sys::ROCMProviderOptionsHandle,
    create_rocm_provider_options,
    update_rocm_provider_options,
    get_rocm_provider_options_as_string,
    release_rocm_provider_options,
    session_options_append_execution_provider_rocm
);
ep_options!(
    CannOptions,
    sys::CANNProviderOptionsHandle,
    create_cann_provider_options,
    update_cann_provider_options,
    get_cann_provider_options_as_string,
    release_cann_provider_options,
    session_options_append_execution_provider_cann
);
ep_options!(
    DnnlOptions,
    sys::DnnlProviderOptionsHandle,
    create_dnnl_provider_options,
    update_dnnl_provider_options,
    get_dnnl_provider_options_as_string,
    release_dnnl_provider_options,
    session_options_append_execution_provider__dnnl
);

/// Append a key/value-direct EP (OpenVINO V2, VitisAI) — no options handle; the key/value
/// pairs are passed straight to the `SessionOptionsAppend…` call.
fn append_kv(
    f: unsafe extern "C" fn(
        *mut sys::SessionOptionsHandle,
        *const *const c_char,
        *const *const c_char,
        usize,
    ) -> sys::StatusPtr,
    opts: *mut sys::SessionOptionsHandle, entries: &[(CString, CString)],
) -> Result<()> {
    let keys: Vec<*const c_char> = entries.iter().map(|(k, _)| k.as_ptr()).collect();
    let vals: Vec<*const c_char> = entries.iter().map(|(_, v)| v.as_ptr()).collect();
    check(unsafe { f(opts, keys.as_ptr(), vals.as_ptr(), entries.len()) })
}

/// Apply a queued EP config to a built session-options handle (called from
/// `SessionOptions::build_handle`). The options-handle providers create + append + release
/// (ORT copies the config during append); the key/value providers append the pairs directly.
pub(crate) fn apply(opts: *mut sys::SessionOptionsHandle, cfg: &EpConfig) -> Result<()> {
    let entries: Vec<(&str, &str)> = cfg
        .entries
        .iter()
        .map(|(k, v)| Ok((k.to_str()?, v.to_str()?)))
        .collect::<std::result::Result<_, std::str::Utf8Error>>()
        .map_err(|_| crate::Error::new(-1, "ep option entry is not UTF-8"))?;
    match cfg.provider {
        EpProvider::Cuda => {
            #[allow(unused_mut)]
            let mut options = CudaOptions::new(&entries)?;
            #[cfg(feature = "cuda")]
            if let Some(stream) = &cfg.cuda_stream {
                // The EpConfig and resulting SessionInner retain this Arc through native teardown.
                unsafe { options.update_with_value("user_compute_stream", stream.as_ptr())? };
            }
            options.append_raw(opts)
        },
        EpProvider::TensorRt => TensorRtOptions::new(&entries)?.append_raw(opts),
        EpProvider::Rocm => RocmOptions::new(&entries)?.append_raw(opts),
        EpProvider::Cann => CannOptions::new(&entries)?.append_raw(opts),
        EpProvider::Dnnl => DnnlOptions::new(&entries)?.append_raw(opts),
        // Key/value-direct appends (no options handle). On a CPU host these return
        // "EP not available" — the call still proves the index/signature is right.
        EpProvider::OpenVinoV2 => append_kv(
            unsafe { api().session_options_append_execution_provider__open_vino_v2() },
            opts,
            &cfg.entries,
        ),
        EpProvider::VitisAi => append_kv(
            unsafe { api().session_options_append_execution_provider__vitis_ai() },
            opts,
            &cfg.entries,
        ),
    }
}

// ─── MIGraphX (flat C-struct EP) ─────────────────────────────────────────────
//
// MIGraphX has no key/value append and no options handle: the caller fills a flat
// `OrtMIGraphXProviderOptions` C struct and passes it by pointer to the append. Layout is
// verified against the real header via a C probe (sizeof=88; the offsets below are pinned by
// `migraphx_struct_layout`). The `const char*` path fields are borrowed — `MigraphxOptions`
// owns the strings and builds this struct transiently for the append.

/// `OrtMIGraphXProviderOptions` (`onnxruntime_c_api.h:840`). `#[repr(C)]`; layout verified.
/// Private — reach it through the [`MigraphxOptions`] builder.
#[repr(C)]
struct MigraphxProviderOptionsRaw {
    device_id: i32,
    fp16_enable: i32,
    fp8_enable: i32,
    int8_enable: i32,
    use_native_calibration_table: i32,
    int8_calibration_table_name: *const c_char,
    save_compiled_model: i32,
    save_model_path: *const c_char,
    load_compiled_model: i32,
    load_model_path: *const c_char,
    exhaustive_tune: bool,
    mem_limit: usize,
    arena_extend_strategy: i32,
}
const _: () = assert!(std::mem::size_of::<MigraphxProviderOptionsRaw>() == 88);

/// Safe builder for the MIGraphX execution-provider config (`SessionOptionsAppendExecutionProvider_MIGraphX`).
/// Owns the path strings. Defaults match the header: `mem_limit` = `usize::MAX` (use all
/// available memory), `arena_extend_strategy` = 0 (`kNextPowerOfTwo`), all precision flags off.
/// Build one, then queue it on [`crate::SessionOptions`] via
/// [`SessionOptions::with_migraphx`].
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MigraphxOptions {
    device_id: i32,
    fp16_enable: i32,
    fp8_enable: i32,
    int8_enable: i32,
    use_native_calibration_table: i32,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    int8_calibration_table_name: Option<CString>,
    save_compiled_model: i32,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    save_model_path: Option<CString>,
    load_compiled_model: i32,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    load_model_path: Option<CString>,
    exhaustive_tune: bool,
    mem_limit: usize,
    arena_extend_strategy: i32,
}

impl Default for MigraphxOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            fp16_enable: 0,
            fp8_enable: 0,
            int8_enable: 0,
            use_native_calibration_table: 0,
            int8_calibration_table_name: None,
            save_compiled_model: 0,
            save_model_path: None,
            load_compiled_model: 0,
            load_model_path: None,
            exhaustive_tune: false,
            mem_limit: usize::MAX,    // header default: SIZE_MAX
            arena_extend_strategy: 0, // 0 = kNextPowerOfTwo
        }
    }
}

impl MigraphxOptions {
    /// Defaults (see the type docs).
    pub fn new() -> Self {
        Self::default()
    }
    /// hip device id.
    pub fn device_id(mut self, id: i32) -> Self {
        self.device_id = id;
        self
    }
    /// FP16 precision.
    pub fn fp16(mut self, on: bool) -> Self {
        self.fp16_enable = on as i32;
        self
    }
    /// FP8 precision.
    pub fn fp8(mut self, on: bool) -> Self {
        self.fp8_enable = on as i32;
        self
    }
    /// INT8 precision.
    pub fn int8(mut self, on: bool) -> Self {
        self.int8_enable = on as i32;
        self
    }
    /// Use the native INT8 calibration table at `path`. Errors if `path` contains a NUL.
    pub fn int8_calibration_table(
        mut self, path: &str,
    ) -> std::result::Result<Self, std::ffi::NulError> {
        self.int8_calibration_table_name = Some(CString::new(path)?);
        Ok(self)
    }
    /// Save the compiled model to `path`. Errors if `path` contains a NUL.
    pub fn save_model_path(mut self, path: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.save_model_path = Some(CString::new(path)?);
        self.save_compiled_model = 1;
        Ok(self)
    }
    /// Load a compiled model from `path`. Errors if `path` contains a NUL.
    pub fn load_model_path(mut self, path: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.load_model_path = Some(CString::new(path)?);
        self.load_compiled_model = 1;
        Ok(self)
    }
    /// Tuned compile.
    pub fn exhaustive_tune(mut self, on: bool) -> Self {
        self.exhaustive_tune = on;
        self
    }
    /// Memory limit in bytes (`usize::MAX` = use all available memory).
    pub fn mem_limit(mut self, bytes: usize) -> Self {
        self.mem_limit = bytes;
        self
    }
    /// Arena-extend strategy: `0` = `kNextPowerOfTwo`, `1` = `kSameAsRequested`.
    pub fn arena_extend_strategy(mut self, strategy: i32) -> Self {
        self.arena_extend_strategy = strategy;
        self
    }

    /// Build the transient C struct and append MIGraphX to a session-options handle. On a
    /// CPU/non-MIGraphX host this returns "EP not available".
    pub(crate) fn append_raw(&self, opts: *mut sys::SessionOptionsHandle) -> Result<()> {
        let raw = MigraphxProviderOptionsRaw {
            device_id: self.device_id,
            fp16_enable: self.fp16_enable,
            fp8_enable: self.fp8_enable,
            int8_enable: self.int8_enable,
            use_native_calibration_table: self.use_native_calibration_table,
            int8_calibration_table_name: self
                .int8_calibration_table_name
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
            save_compiled_model: self.save_compiled_model,
            save_model_path: self
                .save_model_path
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
            load_compiled_model: self.load_compiled_model,
            load_model_path: self
                .load_model_path
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
            exhaustive_tune: self.exhaustive_tune,
            mem_limit: self.mem_limit,
            arena_extend_strategy: self.arena_extend_strategy,
        };
        check(unsafe {
            api().session_options_append_execution_provider_mi_graph_x()(
                opts,
                &raw as *const MigraphxProviderOptionsRaw
                    as *const sys::MIGraphXProviderOptionsHandle,
            )
        })
    }
}

// ─── OpenVINO v1 (flat C-struct EP, deprecated) ──────────────────────────────
//
// Like MIGraphX, the v1 OpenVINO append takes a flat C struct by pointer — no key/value
// options, no handle. **Deprecated upstream** (the V2 key/value path supersedes it; see
// [`EpProvider::OpenVinoV2`]); wrapped for completeness/legacy configs. Layout is verified
// against the real header via a C probe (sizeof=56; offsets pinned by `openvino_struct_layout`).
// The `const char*` fields are borrowed for the append — [`OpenvinoOptions`] owns the strings
// and builds this struct transiently.

/// `OrtOpenVINOProviderOptions` (`onnxruntime_c_api.h:879`). `#[repr(C)]`; layout verified
/// (sizeof=56). Private — reach it through the [`OpenvinoOptions`] builder.
#[repr(C)]
struct OpenvinoProviderOptionsRaw {
    device_type: *const c_char,
    enable_npu_fast_compile: u8,
    device_id: *const c_char,
    num_of_threads: usize,
    cache_dir: *const c_char,
    context: *mut c_void,
    enable_opencl_throttling: u8,
    enable_dynamic_shapes: u8,
}
const _: () = assert!(std::mem::size_of::<OpenvinoProviderOptionsRaw>() == 56);

/// Safe builder for the **deprecated** OpenVINO v1 execution-provider config
/// (`SessionOptionsAppendExecutionProvider_OpenVINO`). Prefer
/// [`SessionOptions::with_execution_provider`] with [`EpProvider::OpenVinoV2`]; this flat-struct
/// path is kept only for legacy configs. Owns the device/path strings; builds the C struct
/// transiently for the append. Defaults match the header (all fields zero/null: `device_type`
/// null, `num_of_threads` 0 = ORT default, every flag off). Build one, then queue it on a
/// [`crate::SessionOptions`] via [`SessionOptions::with_openvino`].
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OpenvinoOptions {
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    device_type: Option<CString>,
    enable_npu_fast_compile: u8,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    device_id: Option<CString>,
    num_of_threads: usize,
    #[cfg_attr(feature = "serde", serde(with = "crate::serde_support::opt_cstr"))]
    cache_dir: Option<CString>,
    // `context` (void*) is an advanced OpenCL interop handle, left null — not exposed.
    enable_opencl_throttling: u8,
    enable_dynamic_shapes: u8,
}

impl OpenvinoOptions {
    /// Defaults (see the type docs).
    pub fn new() -> Self {
        Self::default()
    }
    /// Device type. Valid settings: `"CPU_FP32"`, `"CPU_FP16"`, `"GPU_FP32"`, `"GPU_FP16"`.
    /// Errors if `ty` contains a NUL.
    pub fn device_type(mut self, ty: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.device_type = Some(CString::new(ty)?);
        Ok(self)
    }
    /// Enable NPU fast compile.
    pub fn enable_npu_fast_compile(mut self, on: bool) -> Self {
        self.enable_npu_fast_compile = on as u8;
        self
    }
    /// Device id (e.g. `"0"`). Errors if `id` contains a NUL.
    pub fn device_id(mut self, id: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.device_id = Some(CString::new(id)?);
        Ok(self)
    }
    /// Number of threads (`0` = ORT default).
    pub fn num_of_threads(mut self, n: usize) -> Self {
        self.num_of_threads = n;
        self
    }
    /// Model-compile cache directory. Errors if `path` contains a NUL.
    pub fn cache_dir(mut self, path: &str) -> std::result::Result<Self, std::ffi::NulError> {
        self.cache_dir = Some(CString::new(path)?);
        Ok(self)
    }
    /// Enable OpenCL throttling.
    pub fn enable_opencl_throttling(mut self, on: bool) -> Self {
        self.enable_opencl_throttling = on as u8;
        self
    }
    /// Enable dynamic shapes.
    pub fn enable_dynamic_shapes(mut self, on: bool) -> Self {
        self.enable_dynamic_shapes = on as u8;
        self
    }

    /// Build the transient C struct and append OpenVINO v1 to a session-options handle. On a
    /// CPU/non-OpenVINO host this returns "EP not available".
    pub(crate) fn append_raw(&self, opts: *mut sys::SessionOptionsHandle) -> Result<()> {
        let raw = OpenvinoProviderOptionsRaw {
            device_type: self
                .device_type
                .as_ref()
                .map_or(ptr::null(), |s| s.as_ptr()),
            enable_npu_fast_compile: self.enable_npu_fast_compile,
            device_id: self.device_id.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            num_of_threads: self.num_of_threads,
            cache_dir: self.cache_dir.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            context: ptr::null_mut(),
            enable_opencl_throttling: self.enable_opencl_throttling,
            enable_dynamic_shapes: self.enable_dynamic_shapes,
        };
        check(unsafe {
            api().session_options_append_execution_provider__open_vino()(
                opts,
                &raw as *const OpenvinoProviderOptionsRaw
                    as *const sys::OpenVINOProviderOptionsHandle,
            )
        })
    }
}

impl SessionOptions {
    /// Queue an execution provider with the given key/value options; appended at session
    /// creation (`Session::new`). A GPU/accelerator is required only to *run* the session.
    pub fn with_execution_provider(
        mut self, provider: EpProvider, entries: &[(&str, &str)],
    ) -> Result<Self> {
        if provider == EpProvider::Cuda
            && entries
                .iter()
                .any(|(key, _)| is_unsafe_cuda_pointer_key(key))
        {
            return Err(crate::Error::new(
                -1,
                "pointer-valued CUDA options require an explicitly unsafe low-level API",
            ));
        }
        let kv: Vec<(CString, CString)> = entries
            .iter()
            .map(|(k, v)| Ok((CString::new(*k)?, CString::new(*v)?)))
            .collect::<std::result::Result<_, std::ffi::NulError>>()
            .map_err(|_| crate::Error::new(-1, "ep option key/value contains a NUL byte"))?;
        self.ep_configs.push(EpConfig {
            provider,
            entries: kv,
            requires_owned_cuda_stream: false,
            #[cfg(feature = "cuda")]
            cuda_stream: None,
        });
        Ok(self)
    }

    /// Queue a typed CUDA execution-provider configuration.
    ///
    /// User-stream configurations retain an `Arc<CudaStream>` in both `SessionOptions` and the
    /// resulting `SessionInner`, so ORT cannot outlive the stream pointer it stores.
    pub fn with_cuda(mut self, config: CudaConfig) -> Result<Self> {
        for (key, _) in &config.extra {
            if is_reserved_cuda_key(key) {
                return Err(crate::Error::new(
                    -1,
                    format!("serialized CUDA config illegally overrides typed option {key:?}"),
                ));
            }
        }
        if config.device_inputs == DeviceInputPolicy::UserStream {
            #[cfg(not(feature = "cuda"))]
            return Err(crate::Error::new(
                -1,
                "CUDA user-stream configuration requires the `cuda` feature",
            ));
            #[cfg(feature = "cuda")]
            match config.stream() {
                None => {
                    return Err(crate::Error::new(
                        -1,
                        "CUDA user-stream configuration is missing its owned stream",
                    ));
                },
                Some(stream) if stream.device_id() != config.device_id => {
                    return Err(crate::Error::new(
                        -1,
                        "CUDA user stream belongs to a different device",
                    ));
                },
                Some(_) => {},
            }
        }
        let options = config.provider_options();
        let entries: Vec<(CString, CString)> = options
            .entries
            .into_iter()
            .map(|(key, value)| Ok((CString::new(key)?, CString::new(value)?)))
            .collect::<std::result::Result<_, std::ffi::NulError>>()
            .map_err(|_| crate::Error::new(-1, "cuda ep option key/value contains a NUL byte"))?;
        self.ep_configs.push(EpConfig {
            provider: EpProvider::Cuda,
            entries,
            requires_owned_cuda_stream: config.device_inputs == DeviceInputPolicy::UserStream,
            #[cfg(feature = "cuda")]
            cuda_stream: config.stream,
        });
        Ok(self)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_stream_guards(&self) -> Vec<Arc<crate::CudaStream>> {
        self.ep_configs
            .iter()
            .filter_map(|config| config.cuda_stream.as_ref().map(Arc::clone))
            .collect()
    }

    pub(crate) fn validate_cuda_stream_guards(&self) -> Result<()> {
        for config in &self.ep_configs {
            if config.provider == EpProvider::Cuda {
                for (key, _) in &config.entries {
                    let key = key
                        .to_str()
                        .map_err(|_| crate::Error::new(-1, "CUDA option key is not valid UTF-8"))?;
                    if is_unsafe_cuda_pointer_key(key) {
                        return Err(crate::Error::new(
                            -1,
                            "serialized CUDA configuration contains an unsafe pointer option",
                        ));
                    }
                }
            }
            if config.requires_owned_cuda_stream {
                #[cfg(not(feature = "cuda"))]
                return Err(crate::Error::new(
                    -1,
                    "serialized CUDA user-stream configuration requires reattaching an owned stream",
                ));
                #[cfg(feature = "cuda")]
                if config.cuda_stream.is_none() {
                    return Err(crate::Error::new(
                        -1,
                        "serialized CUDA user-stream configuration requires reattaching an owned stream",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Queue the MIGraphX execution provider (AMD ROCm graph EP). MIGraphX takes a flat config
    /// struct, not key/value options, so it has its own builder — see [`MigraphxOptions`]. A
    /// MIGraphX-capable GPU is required only to *run* the session.
    pub fn with_migraphx(mut self, options: &MigraphxOptions) -> Self {
        self.migraphx.push(options.clone());
        self
    }

    /// Queue the **deprecated** OpenVINO v1 execution provider (flat config struct). Prefer
    /// [`Self::with_execution_provider`] with [`EpProvider::OpenVinoV2`]; this is for legacy
    /// configs. An OpenVINO-capable device is required only to *run* the session.
    pub fn with_openvino(mut self, options: &OpenvinoOptions) -> Self {
        self.openvino.push(options.clone());
        self
    }

    /// Queue one or more discovered [`crate::EpDevice`]s (all from the same EP) for attach at
    /// session creation (`SessionOptionsAppendExecutionProvider_V2`). Obtain devices via
    /// [`crate::get_ep_devices`]; `options` are optional key/value config. The devices are
    /// retained together with their originating [`crate::Environment`] guard. Session creation rejects
    /// using these options with a different environment before entering ORT. A capable device is
    /// required only to *run*.
    pub fn append_execution_provider_device(
        mut self, devices: &[&crate::EpDevice], options: &[(&str, &str)],
    ) -> Result<Self> {
        let opts: Vec<(CString, CString)> = options
            .iter()
            .map(|(k, v)| Ok((CString::new(*k)?, CString::new(*v)?)))
            .collect::<std::result::Result<_, std::ffi::NulError>>()
            .map_err(|_| crate::Error::new(-1, "ep device option key/value contains a NUL byte"))?;
        let first = devices.first().ok_or_else(|| {
            crate::Error::new(-1, "at least one execution-provider device is required")
        })?;
        let env = first.env_guard();
        if devices
            .iter()
            .any(|device| !Arc::ptr_eq(&env, &device.env_guard()))
        {
            return Err(crate::Error::new(
                -1,
                "execution-provider devices must belong to the same Environment",
            ));
        }
        let ep_name = first.ep_name()?;
        for device in devices.iter().skip(1) {
            if device.ep_name()? != ep_name {
                return Err(crate::Error::new(
                    -1,
                    "execution-provider devices must belong to the same execution provider",
                ));
            }
        }
        self.ep_device_attach
            .push(crate::ep_device::EpDeviceAttach {
                devices: devices.iter().map(|d| d.as_ptr()).collect(),
                options: opts,
                env: Some(env),
            });
        Ok(self)
    }

    /// Queue an **author-created** [`crate::OwnedEpDevice`] for attach at session creation
    /// (`SessionOptionsAppendExecutionProvider_V2`) — the in-process counterpart of
    /// [`Self::append_execution_provider_device`]. That method borrows devices the engine
    /// *discovered* via [`crate::get_ep_devices`]; this one takes a device you *authored* yourself
    /// via [`crate::OwnedEpDevice::new`] (feature `model-editor`).
    ///
    /// `options` are optional key/value config forwarded to the V2 call.
    ///
    /// # Safety
    ///
    /// `ep_device`, its [`crate::OwnedHardwareDevice`], and the leaked
    /// [`crate::EpFactoryInstance`] backing it must outlive every session built from these options.
    /// The authoring handles are not shared owners, so this lifetime cannot currently be encoded.
    /// ORT calls the factory during session initialization and teardown through retained pointers.
    #[cfg(feature = "model-editor")]
    pub unsafe fn append_ep_device(
        mut self, ep_device: &crate::OwnedEpDevice, options: &[(&str, &str)],
    ) -> Result<Self> {
        let opts: Vec<(CString, CString)> = options
            .iter()
            .map(|(k, v)| Ok((CString::new(*k)?, CString::new(*v)?)))
            .collect::<std::result::Result<_, std::ffi::NulError>>()
            .map_err(|_| crate::Error::new(-1, "ep device option key/value contains a NUL byte"))?;
        self.ep_device_attach
            .push(crate::ep_device::EpDeviceAttach {
                devices: vec![ep_device.as_ptr()],
                options: opts,
                env: None,
            });
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the provider-options lifecycle (create + update + as_string + release)
    /// via DNNL. The options are pure config, so this needs no EP/GPU installed; if the
    /// host can't create them, we skip — the FFI call still proves the index/signature is
    /// right (a wrong index crashes, it doesn't error cleanly).
    #[test]
    fn dnnl_options_lifecycle() {
        let opts = match DnnlOptions::new(&[("num_threads", "4")]) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("dnnl provider options unavailable on this host — skipping ({e})");
                return;
            },
        };
        let s = opts.as_string().expect("as_string");
        eprintln!("dnnl options: {s}");
        assert!(
            s.contains("num_threads"),
            "as_string should echo the configured key: {s}"
        );
    }

    #[test]
    fn cuda_provider_options_and_typed_config_cover_ort_keys() {
        let options = CudaProviderOptions::new()
            .device_id(2)
            .do_copy_in_default_stream(false)
            .use_ep_level_unified_stream(true)
            .gpu_mem_limit(1024)
            .arena_extend_strategy(CudaArenaExtendStrategy::SameAsRequested)
            .cudnn_conv_algo_search(CudaCudnnConvAlgoSearch::Default)
            .cudnn_conv_use_max_workspace(true)
            .cudnn_conv1d_pad_to_nc1d(true)
            .enable_cuda_graph(true)
            .enable_skip_layer_norm_strict_mode(true)
            .use_tf32(false)
            .prefer_nhwc(true)
            .tunable_op_enable(true)
            .tunable_op_tuning_enable(true)
            .tunable_op_max_tuning_duration_ms(25)
            .with_raw("future_cuda_option", "x")
            .expect("future string option");

        let entries = options.entries();
        let has = |key: &str, value: &str| entries.iter().any(|(k, v)| k == key && v == value);
        assert!(has("device_id", "2"));
        assert!(has("do_copy_in_default_stream", "0"));
        assert!(has("use_ep_level_unified_stream", "1"));
        assert!(has("gpu_mem_limit", "1024"));
        assert!(has("future_cuda_option", "x"));

        let low_mem = CudaProviderOptions::from(CudaConfig::low_memory(2, 1024));
        assert!(
            low_mem
                .entries()
                .iter()
                .any(|(key, value)| key == "arena_extend_strategy" && value == "kSameAsRequested")
        );
        let unified = CudaProviderOptions::from(
            CudaConfig::performance(0)
                .with_device_input_policy(DeviceInputPolicy::UnifiedStream)
                .expect("unified"),
        );
        assert!(
            unified
                .entries()
                .iter()
                .any(|(key, value)| key == "do_copy_in_default_stream" && value == "1")
        );
        assert!(
            unified
                .entries()
                .iter()
                .any(|(key, value)| key == "use_ep_level_unified_stream" && value == "1")
        );
        assert!(
            CudaConfig::performance(0)
                .with_raw("user_compute_stream", "1234")
                .is_err()
        );
        assert!(
            CudaProviderOptions::new()
                .with_raw("user_compute_stream", "1234")
                .is_err()
        );
        assert!(
            SessionOptions::new()
                .with_execution_provider(EpProvider::Cuda, &[("user_compute_stream", "1234")],)
                .is_err()
        );
    }

    /// OpenVINO V2 + VitisAI take key/value options directly at append time (no options
    /// handle). On a CPU host the append returns "EP not available" — which proves the FFI
    /// index/signature is right (a wrong index crashes; this errors cleanly).
    #[test]
    fn openvino_v2_and_vitisai_append_reach_ffi() {
        let h = SessionOptions::default()
            .build_handle()
            .expect("opts handle");
        for provider in [EpProvider::OpenVinoV2, EpProvider::VitisAi] {
            let cfg = EpConfig {
                provider,
                entries: Vec::new(),
                requires_owned_cuda_stream: false,
                #[cfg(feature = "cuda")]
                cuda_stream: None,
            };
            let res = apply(h, &cfg);
            eprintln!("{provider:?} apply -> {res:?}");
            assert!(
                res.is_err(),
                "{provider:?} append should error (EP not available on CPU), got Ok"
            );
        }
        unsafe {
            api().release_session_options()(h);
        }
    }

    /// Pins `MigraphxProviderOptionsRaw` field offsets to the values a C probe returned
    /// against the real header (sizeof=88; device_id@0, int8_cal_table_name@24,
    /// save_model_path@40, load_model_path@56, exhaustive_tune@64, mem_limit@72,
    /// arena_extend_strategy@80). Catches a transcription/field-order error — the size
    /// const-assert only catches a wrong count. (`addr_of!` never reads through the dangling
    /// pointer, so this is sound.)
    #[test]
    fn migraphx_struct_layout() {
        use core::ptr::{NonNull, addr_of};
        let p = NonNull::<MigraphxProviderOptionsRaw>::dangling().as_ptr();
        let base = p as usize;
        macro_rules! off {
            ($f:ident) => {
                unsafe { (addr_of!((*p).$f) as usize) - base }
            };
        }
        assert_eq!(off!(device_id), 0);
        assert_eq!(off!(int8_calibration_table_name), 24);
        assert_eq!(off!(save_model_path), 40);
        assert_eq!(off!(load_model_path), 56);
        assert_eq!(off!(exhaustive_tune), 64);
        assert_eq!(off!(mem_limit), 72);
        assert_eq!(off!(arena_extend_strategy), 80);
    }

    /// MIGraphX has no options handle — the flat C struct is passed by pointer to the append.
    /// On a CPU host the append returns "EP not available", proving the index/signature is
    /// right (a wrong index crashes; this errors cleanly). Also exercises the builder.
    #[test]
    fn migraphx_append_reaches_ffi() {
        let h = SessionOptions::default()
            .build_handle()
            .expect("opts handle");
        let opts = MigraphxOptions::new()
            .device_id(0)
            .fp16(true)
            .mem_limit(1 << 30);
        let res = opts.append_raw(h);
        eprintln!("migraphx append -> {res:?}");
        assert!(
            res.is_err(),
            "MIGraphX append should error (EP not available on CPU), got Ok"
        );
        unsafe {
            api().release_session_options()(h);
        }
    }

    /// Pins `OpenvinoProviderOptionsRaw` field offsets to the C-probe values (sizeof=56;
    /// device_type@0, enable_npu_fast_compile@8, device_id@16, num_of_threads@24,
    /// cache_dir@32, context@40, enable_opencl_throttling@48, enable_dynamic_shapes@49).
    /// Catches a field-order/typing error — the size const-assert only catches a wrong count.
    /// (`addr_of!` never reads through the dangling pointer, so this is sound.)
    #[test]
    fn openvino_struct_layout() {
        use core::ptr::{NonNull, addr_of};
        let p = NonNull::<OpenvinoProviderOptionsRaw>::dangling().as_ptr();
        let base = p as usize;
        macro_rules! off {
            ($f:ident) => {
                unsafe { (addr_of!((*p).$f) as usize) - base }
            };
        }
        assert_eq!(off!(device_type), 0);
        assert_eq!(off!(enable_npu_fast_compile), 8);
        assert_eq!(off!(device_id), 16);
        assert_eq!(off!(num_of_threads), 24);
        assert_eq!(off!(cache_dir), 32);
        assert_eq!(off!(context), 40);
        assert_eq!(off!(enable_opencl_throttling), 48);
        assert_eq!(off!(enable_dynamic_shapes), 49);
    }

    /// OpenVINO v1 has no options handle — the flat C struct is passed by pointer to the append.
    /// On a CPU host the append returns "EP not available", proving the index/signature is right
    /// (a wrong index crashes; this errors cleanly). Also exercises the builder.
    #[test]
    fn openvino_append_reaches_ffi() {
        let h = SessionOptions::default()
            .build_handle()
            .expect("opts handle");
        let opts = OpenvinoOptions::new()
            .device_type("GPU_FP16")
            .expect("device_type")
            .device_id("0")
            .expect("device_id")
            .num_of_threads(4);
        let res = opts.append_raw(h);
        eprintln!("openvino v1 append -> {res:?}");
        assert!(
            res.is_err(),
            "OpenVINO v1 append should error (EP not available on CPU), got Ok"
        );
        unsafe {
            api().release_session_options()(h);
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn ep_provider_round_trip() {
        for p in [
            EpProvider::Cuda,
            EpProvider::TensorRt,
            EpProvider::Rocm,
            EpProvider::Cann,
            EpProvider::Dnnl,
            EpProvider::OpenVinoV2,
            EpProvider::VitisAi,
        ] {
            let json = serde_json::to_string(&p).expect("serialize");
            let back: EpProvider = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(p, back, "{p:?} did not round-trip via {json}");
        }
    }

    #[test]
    fn ep_config_round_trip() {
        let cfg = EpConfig {
            provider: EpProvider::Cuda,
            entries: vec![
                (
                    CString::new("device_id").unwrap(),
                    CString::new("0").unwrap(),
                ),
                (
                    CString::new("arena_extend_strategy").unwrap(),
                    CString::new("kSameAsRequested").unwrap(),
                ),
            ],
            requires_owned_cuda_stream: false,
            #[cfg(feature = "cuda")]
            cuda_stream: None,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        eprintln!("EpConfig JSON: {json}");
        assert!(json.contains("\"device_id\""), "cuda kv present: {json}");
        let back: EpConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.provider, EpProvider::Cuda);
        assert_eq!(back.entries.len(), 2);
        assert_eq!(back.entries[0].0.to_str().unwrap(), "device_id");
        assert_eq!(back.entries[1].1.to_str().unwrap(), "kSameAsRequested");
    }

    #[test]
    fn serialized_user_stream_marker_requires_reattachment() {
        let cfg = EpConfig {
            provider: EpProvider::Cuda,
            entries: vec![(
                CString::new("device_id").unwrap(),
                CString::new("0").unwrap(),
            )],
            requires_owned_cuda_stream: true,
            #[cfg(feature = "cuda")]
            cuda_stream: None,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: EpConfig = serde_json::from_str(&json).expect("deserialize");
        let mut options = SessionOptions::new();
        options.ep_configs.push(back);
        let error = options
            .validate_cuda_stream_guards()
            .expect_err("missing owned stream must fail");
        assert!(error.message.contains("requires reattaching"));
    }

    #[test]
    fn deserialized_raw_cuda_pointer_option_is_rejected() {
        let json = r#"{"entries":[["user_compute_stream","1234"]]}"#;
        let config: CudaProviderOptions = serde_json::from_str(json).expect("deserialize config");
        let error = CudaOptions::from_config(&config)
            .err()
            .expect("safe materialization must reject raw pointers");
        assert!(error.message.contains("pointer-valued CUDA options"));
    }

    #[test]
    fn flat_ep_options_round_trip() {
        let m = MigraphxOptions::new()
            .device_id(1)
            .fp16(true)
            .mem_limit(1 << 30)
            .save_model_path("/tmp/m")
            .expect("path");
        let json = serde_json::to_string(&m).expect("serialize");
        eprintln!("MigraphxOptions JSON: {json}");
        assert!(json.contains("\"/tmp/m\""), "path present: {json}");
        let back: MigraphxOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.device_id, 1);
        assert_ne!(back.fp16_enable, 0);
        assert_eq!(back.mem_limit, 1 << 30);
        assert_eq!(
            back.save_model_path.as_ref().unwrap().to_str().unwrap(),
            "/tmp/m"
        );

        let o = OpenvinoOptions::new()
            .device_type("GPU_FP16")
            .expect("dt")
            .device_id("0")
            .expect("id")
            .num_of_threads(4);
        let json = serde_json::to_string(&o).expect("serialize");
        eprintln!("OpenvinoOptions JSON: {json}");
        let back: OpenvinoOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            back.device_type.as_ref().unwrap().to_str().unwrap(),
            "GPU_FP16"
        );
        assert_eq!(back.num_of_threads, 4);
        assert_eq!(back.device_id.as_ref().unwrap().to_str().unwrap(), "0");
    }
}
