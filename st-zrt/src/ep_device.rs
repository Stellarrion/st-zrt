//! EP device discovery + attach (feature `ep`) — the modern `OrtEpDevice` surface.
//!
//! [`get_ep_devices`] enumerates the execution-provider devices registered with an
//! [`crate::Environment`] (`GetEpDevices`, since v1.22); [`EpDevice`] exposes their name/vendor.
//! A discovered device is attached to a session by queueing it on [`crate::SessionOptions`] via
//! [`crate::SessionOptions::append_execution_provider_device`] (`SessionOptionsAppendExecutionProvider_V2`).
//!
//! The devices are **engine-owned** — [`EpDevice`] never releases the native handle, but retains an
//! `Arc` guard for the `Environment` that makes it valid. Because the V2 attach call needs the
//! environment, it is applied at session-creation (the constructors call
//! [`apply_device_attach`]), not in `SessionOptions::build_handle`.
//!
//! The EP-**authoring** surface (the `OrtEpApi` table — `KernelDefBuilder`, `OpSchema`,
//! `CreateEpDevice`, `EpGraphSupportInfo`, profiling events; ~67 fns) is for implementing a
//! custom EP in C++/Rust. It is niche and untestable on a CPU host, so it is left at the
//! [`crate::ep_api`] gateway.
use crate::{Result, api, check, sys};
use std::ffi::{CString, c_char};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

/// Enumerate the EP devices registered with `env` (`GetEpDevices`). Each returned [`EpDevice`]
/// retains `env` internally; the engine-owned device handle is never released. Returns an empty vec if
/// no EP has registered devices (e.g. a CPU-only host).
pub fn get_ep_devices(env: &crate::Environment) -> Result<Vec<EpDevice>> {
    let mut devices: *const *const sys::EpDeviceHandle = ptr::null();
    let mut num: usize = 0;
    check(unsafe {
        api().get_ep_devices()(
            env.as_ptr(),
            &mut devices as *mut _ as *const *const *const sys::EpDeviceHandle,
            &mut num,
        )
    })?;
    if devices.is_null() || num == 0 {
        return Ok(Vec::new());
    }
    (0..num)
        .map(|i| {
            let ptr = unsafe { *devices.add(i) };
            let ptr = crate::ensure_non_null(ptr as *mut sys::EpDeviceHandle, "EP device")?;
            Ok(EpDevice {
                ptr,
                env: env.share(),
            })
        })
        .collect()
}

/// An engine-owned `OrtEpDevice` discovered via [`get_ep_devices`].
///
/// The device itself is never released, but this handle retains the originating environment whose
/// lifetime makes the pointer valid. Cloning it is therefore safe and cheap.
#[derive(Clone)]
pub struct EpDevice {
    ptr: *const sys::EpDeviceHandle,
    env: Arc<crate::environment::EnvInner>,
}

impl EpDevice {
    /// The EP name (e.g. `"CUDAExecutionProvider"`).
    pub fn ep_name(&self) -> Result<String> {
        cstr_to_string(unsafe { api().ep_device__ep_name()(self.ptr) })
    }
    /// The EP vendor (e.g. `"NVIDIA"`).
    pub fn ep_vendor(&self) -> Result<String> {
        cstr_to_string(unsafe { api().ep_device__ep_vendor()(self.ptr) })
    }
    /// The underlying hardware device (`EpDevice_Device`). Borrowed from this `EpDevice` (and
    /// transitively the `Environment`); never released.
    pub fn device(&self) -> Result<crate::hardware::HardwareDevice> {
        let ptr = unsafe { api().ep_device__device()(self.ptr) };
        let ptr = crate::ensure_non_null(ptr as *mut sys::HardwareDeviceHandle, "hardware device")?;
        // SAFETY: the engine owns the handle and the cloned environment guard preserves it.
        Ok(unsafe { crate::hardware::HardwareDevice::from_borrowed(ptr, self.env_guard()) })
    }
    /// EP metadata key/value pairs (`EpDevice_EpMetadata`). Borrowed; empty when the EP sets none.
    pub fn ep_metadata(&self) -> crate::allocator::KeyValuePairsView<'_> {
        let p = unsafe { api().ep_device__ep_metadata()(self.ptr) };
        // SAFETY: engine-owned handle, valid for the borrow.
        unsafe { crate::allocator::KeyValuePairsView::from_borrowed(p) }
    }
    /// EP options key/value pairs (`EpDevice_EpOptions`). Borrowed; empty when the EP sets none.
    pub fn ep_options(&self) -> crate::allocator::KeyValuePairsView<'_> {
        let p = unsafe { api().ep_device__ep_options()(self.ptr) };
        // SAFETY: engine-owned handle, valid for the borrow.
        unsafe { crate::allocator::KeyValuePairsView::from_borrowed(p) }
    }
    /// Memory info for a device-memory kind (`EpDevice_MemoryInfo`). The handle is borrowed, so a
    /// plain-data [`crate::MemoryInfoSnapshot`] is copied out (never released).
    pub fn memory_info(
        &self, memory_type: crate::DeviceMemoryType,
    ) -> Result<crate::MemoryInfoSnapshot> {
        let p = unsafe { api().ep_device__memory_info()(self.ptr, memory_type as i32) };
        crate::memory::snapshot_from_ptr(p)
    }
    pub(crate) fn as_ptr(&self) -> *const sys::EpDeviceHandle {
        self.ptr
    }

    pub(crate) fn env_guard(&self) -> Arc<crate::environment::EnvInner> {
        Arc::clone(&self.env)
    }

    pub(crate) fn shares_environment(&self, env: &crate::Environment) -> bool {
        Arc::ptr_eq(&self.env, &env.share())
    }

    pub(crate) fn shares_device_environment(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.env, &other.env)
    }
}

/// Borrowed `OrtEpAssignedNode` — one graph node assigned to an EP, from a session's
/// [`crate::Session::ep_graph_assignment_info`]. Borrowed from the session; never released.
pub struct EpAssignedNode<'a> {
    ptr: *const sys::EpAssignedNodeHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> EpAssignedNode<'a> {
    /// # Safety
    /// `ptr` must remain valid for `'a` and must not be released by the caller.
    pub(crate) unsafe fn from_borrowed(ptr: *const sys::EpAssignedNodeHandle) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }
    /// The node's name (`EpAssignedNode_GetName`). Empty if the engine sets none.
    pub fn name(&self) -> Result<String> {
        let mut p: *const c_char = ptr::null();
        check(unsafe {
            api().ep_assigned_node__get_name()(self.ptr, &mut p as *mut _ as *const *const c_char)
        })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "assigned node name") }
        }
    }
    /// The node's operator domain (`EpAssignedNode_GetDomain`).
    pub fn domain(&self) -> Result<String> {
        let mut p: *const c_char = ptr::null();
        check(unsafe {
            api().ep_assigned_node__get_domain()(self.ptr, &mut p as *mut _ as *const *const c_char)
        })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "assigned node domain") }
        }
    }
    /// The node's operator type (`EpAssignedNode_GetOperatorType`), e.g. `"Conv"`.
    pub fn operator_type(&self) -> Result<String> {
        let mut p: *const c_char = ptr::null();
        check(unsafe {
            api().ep_assigned_node__get_operator_type()(
                self.ptr,
                &mut p as *mut _ as *const *const c_char,
            )
        })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "assigned node operator type") }
        }
    }
}

/// Borrowed `OrtEpAssignedSubgraph` — a subgraph assigned to one EP, from a session's
/// [`crate::Session::ep_graph_assignment_info`]. Borrowed from the session; never released.
pub struct EpAssignedSubgraph<'a> {
    ptr: *const sys::EpAssignedSubgraphHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> EpAssignedSubgraph<'a> {
    /// # Safety
    /// `ptr` must remain valid for `'a` and must not be released by the caller.
    pub(crate) unsafe fn from_borrowed(ptr: *const sys::EpAssignedSubgraphHandle) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }
    /// The EP this subgraph was assigned to (`EpAssignedSubgraph_GetEpName`).
    pub fn ep_name(&self) -> Result<String> {
        let mut p: *const c_char = ptr::null();
        check(unsafe {
            api().ep_assigned_subgraph__get_ep_name()(
                self.ptr,
                &mut p as *mut _ as *const *const c_char,
            )
        })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "assigned subgraph EP name") }
        }
    }
    /// The nodes assigned to this EP within the subgraph (`EpAssignedSubgraph_GetNodes`). Each
    /// node borrows this subgraph (and the session).
    pub fn nodes(&self) -> Result<Vec<EpAssignedNode<'a>>> {
        let mut nodes: *const *const sys::EpAssignedNodeHandle = ptr::null();
        let mut num: usize = 0;
        check(unsafe {
            api().ep_assigned_subgraph__get_nodes()(
                self.ptr,
                &mut nodes as *mut _ as *const *const *const sys::EpAssignedNodeHandle,
                &mut num,
            )
        })?;
        if nodes.is_null() || num == 0 {
            return Ok(Vec::new());
        }
        (0..num)
            .map(|i| {
                // SAFETY: the engine owns the array for the session's lifetime; each entry is a
                // borrowed node handle valid for `'a`.
                let p = unsafe { *nodes.add(i) };
                Ok(unsafe { EpAssignedNode::from_borrowed(p) })
            })
            .collect()
    }
}

/// An owned `OrtSyncStream` for an [`EpDevice`] — an EP-specific synchronization primitive (e.g.
/// a CUDA stream) used for asynchronous copies/runs (`CreateSyncStreamForEpDevice`).
///
/// The stream retains the environment that owns the discovered device, so both the stream and the
/// device pointer remain valid until the final `Arc<SyncStream>` drops. Released through
/// `ReleaseSyncStream` before the environment guard is released.
pub struct SyncStream {
    raw: *mut sys::SyncStreamHandle,
    _env: Arc<crate::environment::EnvInner>,
}

impl SyncStream {
    #[cfg(test)]
    pub(crate) fn null_for_test(env: &crate::Environment) -> Arc<Self> {
        Arc::new(Self {
            raw: ptr::null_mut(),
            _env: env.share(),
        })
    }

    /// Create an owned sync stream for `device` (`CreateSyncStreamForEpDevice`).
    ///
    /// `options` are EP-specific stream-creation key/value pairs (empty means pass none). Returns an
    /// ORT error if the EP does not support sync streams.
    pub fn for_ep_device(device: &EpDevice, options: &[(&str, &str)]) -> Result<Arc<Self>> {
        let opts = if options.is_empty() {
            None
        } else {
            let mut kvps = crate::allocator::KeyValuePairs::new()?;
            for (key, value) in options {
                kvps.add(key, value)?;
            }
            Some(kvps)
        };
        let opts_ptr = opts
            .as_ref()
            .map(crate::allocator::KeyValuePairs::raw_ptr)
            .unwrap_or(ptr::null());
        let mut raw: *mut sys::SyncStreamHandle = ptr::null_mut();
        check(unsafe {
            api().create_sync_stream_for_ep_device()(device.as_ptr(), opts_ptr, &mut raw)
        })?;
        let raw = crate::ensure_non_null(raw, "sync stream")?;
        Ok(Arc::new(Self {
            raw,
            _env: device.env_guard(),
        }))
    }

    /// The opaque, EP-specific native stream handle (`SyncStream_GetHandle`), e.g. a `cudaStream_t`.
    /// Do not interpret or free it; it is owned by the [`EpDevice`]'s EP.
    pub fn native_handle(&self) -> *mut std::ffi::c_void {
        unsafe { api().sync_stream__get_handle()(self.raw) }
    }

    /// The raw `OrtSyncStream*` for ORT APIs that take one directly.
    pub(crate) fn as_ptr(&self) -> *mut sys::SyncStreamHandle {
        self.raw
    }

    pub(crate) fn shares_env_guard(&self, env: &Arc<crate::environment::EnvInner>) -> bool {
        Arc::ptr_eq(&self._env, env)
    }
}

impl Drop for SyncStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { api().release_sync_stream()(self.raw) }
        }
    }
}

// ORT stream handles are used from arbitrary worker threads; immutable access is shared and final
// release occurs only after the last Arc guard disappears.
unsafe impl Send for SyncStream {}
unsafe impl Sync for SyncStream {}

fn cstr_to_string(p: *const c_char) -> Result<String> {
    if p.is_null() {
        Ok(String::new())
    } else {
        unsafe { crate::cstr_to_string(p, "execution provider device string") }
    }
}

/// A queued EP-device attach — one or more (same-EP) [`EpDevice`]s + optional key/value
/// options. Applied to a session-options handle by [`apply_device_attach`] at session creation.
/// Discovered device pointers retain and identify their originating `Environment`.
#[derive(Clone)]
pub(crate) struct EpDeviceAttach {
    pub(crate) devices: Vec<*const sys::EpDeviceHandle>,
    pub(crate) options: Vec<(CString, CString)>,
    /// Present for discovered devices; retains and identifies their originating environment.
    pub(crate) env: Option<Arc<crate::environment::EnvInner>>,
}

/// Apply queued EP-device attaches to a built session-options handle
/// (`SessionOptionsAppendExecutionProvider_V2`). Called from the session constructors (which
/// have the `env` the V2 call requires).
pub(crate) fn apply_device_attach(
    env: &crate::Environment, opts: *mut sys::SessionOptionsHandle, attaches: &[EpDeviceAttach],
) -> Result<()> {
    let f = unsafe { api().session_options_append_execution_provider_v2() };
    for attach in attaches {
        if let Some(device_env) = &attach.env {
            if !Arc::ptr_eq(device_env, &env.share()) {
                return Err(crate::Error::new(
                    -1,
                    "execution-provider devices belong to a different Environment",
                ));
            }
        }
        let keys: Vec<*const c_char> = attach.options.iter().map(|(k, _)| k.as_ptr()).collect();
        let vals: Vec<*const c_char> = attach.options.iter().map(|(_, v)| v.as_ptr()).collect();
        check(unsafe {
            f(
                opts,
                env.as_ptr() as *mut sys::EnvHandle,
                attach.devices.as_ptr(),
                attach.devices.len(),
                keys.as_ptr(),
                vals.as_ptr(),
                attach.options.len(),
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Take the crate-wide default-`LoggingManager` creation lock (see
    /// `crate::TEST_ENV_CREATION_MUTEX`); keep it for the test's whole body.
    fn serialized_env() -> std::sync::MutexGuard<'static, ()> {
        crate::lock_default_env_creation()
    }

    #[test]
    fn ep_device_retains_its_originating_environment() {
        let _envs = serialized_env();
        let env = crate::Environment::new().expect("env");
        let guard = env.share();
        let before = Arc::strong_count(&guard);
        let device = EpDevice {
            ptr: ptr::null(),
            env: env.share(),
        };
        assert_eq!(Arc::strong_count(&guard), before + 1);
        drop(env);
        assert!(
            Arc::strong_count(&guard) >= 2,
            "device must retain EnvInner"
        );
        drop(device);
        assert_eq!(Arc::strong_count(&guard), 1);
    }

    #[test]
    fn mixed_environment_devices_are_rejected_while_building_options() {
        let _envs = serialized_env();
        let env_a = crate::Environment::new().expect("env a");
        let env_b = crate::Environment::new().expect("env b");
        let a = EpDevice {
            ptr: ptr::null(),
            env: env_a.share(),
        };
        let b = EpDevice {
            ptr: ptr::null(),
            env: env_b.share(),
        };
        let error =
            match crate::SessionOptions::new().append_execution_provider_device(&[&a, &b], &[]) {
                Ok(_) => panic!("mixed environments must fail"),
                Err(error) => error,
            };
        assert_eq!(
            error.message,
            "execution-provider devices must belong to the same Environment"
        );
    }

    #[test]
    fn sync_stream_environment_identity_is_retained() {
        let _envs = serialized_env();
        let env_a = crate::Environment::new().expect("env a");
        let env_b = crate::Environment::new().expect("env b");
        let stream = SyncStream {
            raw: ptr::null_mut(),
            _env: env_a.share(),
        };
        assert!(stream.shares_env_guard(&env_a.share()));
        assert!(!stream.shares_env_guard(&env_b.share()));
    }

    #[test]
    fn discovered_device_attach_rejects_a_different_environment_before_ffi() {
        let _envs = serialized_env();
        let device_env = crate::Environment::new().expect("device env");
        let session_env = crate::Environment::new().expect("session env");
        let attach = EpDeviceAttach {
            devices: vec![ptr::null()],
            options: Vec::new(),
            env: Some(device_env.share()),
        };
        let error = apply_device_attach(&session_env, ptr::null_mut(), &[attach])
            .expect_err("cross-environment device attach must fail");
        assert_eq!(
            error.message,
            "execution-provider devices belong to a different Environment"
        );
    }

    /// Enumerate the registered EP devices (printing name/vendor), and — if any are present —
    /// exercise the V2 attach path on a real session-options handle. On a CPU-only host
    /// `get_ep_devices` returns none and the attach is skipped; on a GPU host the CUDA device
    /// is discovered + attached.
    #[test]
    fn enumerate_and_attach_ep_devices() {
        let _envs = serialized_env();
        let env = crate::Environment::new().expect("env");
        let devices = get_ep_devices(&env).expect("get_ep_devices");
        eprintln!("discovered {} EP device(s):", devices.len());
        for d in &devices {
            eprintln!(
                "  - {} ({})",
                d.ep_name().expect("ep name"),
                d.ep_vendor().expect("ep vendor")
            );
        }
        if devices.is_empty() {
            eprintln!("no EP devices registered (CPU-only host) — attach skipped");
            return;
        }
        // Queue the first device + apply the V2 attach on a real handle (proves the FFI path).
        let opts = crate::SessionOptions::new()
            .append_execution_provider_device(&[&devices[0]], &[])
            .expect("queue device attach");
        let h = opts.build_handle().expect("opts handle");
        let r = apply_device_attach(&env, h, &opts.ep_device_attach);
        eprintln!(
            "apply_device_attach({}) -> {r:?}",
            devices[0].ep_name().expect("ep name")
        );
        // Reaching here + releasing cleanly proves the V2 append reached the FFI without crashing.
        unsafe {
            crate::api().release_session_options()(h);
        }
    }

    /// `CreateSyncStreamForEpDevice` reaches the FFI and returns a clean result (never panics). The
    /// CPU EP typically does NOT support sync streams, so construction usually errors
    /// (NOT_IMPLEMENTED) — the error path is the common CPU outcome. If the EP does support streams,
    /// the happy path exercises `native_handle`, owned `RunOptions::with_sync_stream` materialization, and clean `Drop`.
    #[test]
    fn sync_stream_construction_is_clean() {
        let _envs = serialized_env();
        let env = crate::Environment::new().expect("env");
        let devices = get_ep_devices(&env).expect("get_ep_devices");
        if devices.is_empty() {
            eprintln!("sync_stream: no EP devices registered — skipping");
            return;
        }
        let device = &devices[0];
        let name = device.ep_name().unwrap_or_else(|_| "<unknown>".into());
        match SyncStream::for_ep_device(device, &[]) {
            Err(e) => {
                eprintln!("sync_stream: {name} does not support sync streams ({e})");
            },
            Ok(s) => {
                eprintln!(
                    "sync_stream: {name} supports streams; native_handle non-null = {}",
                    !s.native_handle().is_null()
                );
                let opts = crate::RunOptions::new()
                    .with_sync_stream(&s)
                    .freeze()
                    .expect("run opts");
                drop(s); // The materialized options retain the final stream Arc.
                drop(opts); // ReleaseRunOptions, then ReleaseSyncStream, then the Env guard.
                eprintln!("sync_stream: owned by MaterializedRunOptions + dropped cleanly");
            },
        }
    }

    /// Exercise the EP-device + hardware-device introspection accessors (`EpDevice_Device`,
    /// `EpDevice_EpMetadata`/`EpOptions`, `EpDevice_MemoryInfo`, and the `HardwareDevice_*`
    /// family). On a CPU-only host discovery is empty and we skip; on the RTX 4090 the CUDA
    /// device is discovered and every accessor is read against real data.
    #[test]
    fn ep_device_introspection_accessors() {
        let _envs = serialized_env();
        let env = crate::Environment::new().expect("env");
        let devices = get_ep_devices(&env).expect("get_ep_devices");
        if devices.is_empty() {
            eprintln!("introspection: no EP devices registered — skipping");
            return;
        }
        let d = &devices[0];
        let hw = d.device().expect("hardware device");
        eprintln!(
            "ep_device device: ty={} vendor_id={:#06x} vendor={} device_id={}",
            hw.ty(),
            hw.vendor_id(),
            hw.vendor().unwrap_or_default(),
            hw.device_id(),
        );
        // Borrowed KVP views tolerate an absent key (return None).
        assert_eq!(d.ep_metadata().get("absent").unwrap(), None);
        assert_eq!(d.ep_options().get("absent").unwrap(), None);
        match d.memory_info(crate::DeviceMemoryType::Default) {
            Ok(s) => eprintln!(
                "memory_info(Default): name={} device_id={}",
                s.name, s.device_id
            ),
            Err(e) => eprintln!("memory_info(Default): {e}"),
        }
        // The hardware_devices() enumeration reads the same accessors over every device.
        for hw in crate::hardware::hardware_devices(&env).expect("hardware_devices") {
            eprintln!(
                "hw: ty={} vendor_id={:#06x} vendor={} device_id={}",
                hw.ty(),
                hw.vendor_id(),
                hw.vendor().unwrap_or_default(),
                hw.device_id(),
            );
        }
    }
}
