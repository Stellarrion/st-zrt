//! EP device discovery + attach (feature `ep`) — the modern `OrtEpDevice` surface.
//!
//! [`get_ep_devices`] enumerates the execution-provider devices registered with an
//! [`crate::Environment`] (`GetEpDevices`, since v1.22); [`EpDevice`] exposes their name/vendor.
//! A discovered device is attached to a session by queueing it on [`crate::SessionOptions`] via
//! [`crate::SessionOptions::append_execution_provider_device`] (`SessionOptionsAppendExecutionProvider_V2`).
//!
//! The devices are **borrowed from ORT** — the `GetEpDevices` array is engine-owned, valid while
//! the `Environment` lives; [`EpDevice`] does not release. Because the V2 attach call needs the
//! environment, it is applied at session-creation (the constructors call
//! [`apply_device_attach`]), not in `SessionOptions::build_handle`.
//!
//! The EP-**authoring** surface (the `OrtEpApi` table — `KernelDefBuilder`, `OpSchema`,
//! `CreateEpDevice`, `EpGraphSupportInfo`, profiling events; ~67 fns) is for implementing a
//! custom EP in C++/Rust. It is niche and untestable on a CPU host, so it is left at the
//! [`crate::ep_api`] gateway.
use crate::{api, check, sys, Result};
use std::ffi::{c_char, CString};
use std::ptr;

/// Enumerate the EP devices registered with `env` (`GetEpDevices`). The returned [`EpDevice`]s
/// are borrowed from ORT — valid while `env` is alive; do not release. Returns an empty vec if
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
    let out = (0..num)
        .map(|i| EpDevice {
            ptr: unsafe { *devices.add(i) },
        })
        .collect();
    Ok(out)
}

/// A borrowed `OrtEpDevice` — an execution-provider device discovered via [`get_ep_devices`].
/// Engine-owned; never released. Valid while the [`crate::Environment`] it came from is alive.
#[derive(Clone, Copy)]
pub struct EpDevice {
    ptr: *const sys::EpDeviceHandle,
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
    pub(crate) fn as_ptr(&self) -> *const sys::EpDeviceHandle {
        self.ptr
    }
}

fn cstr_to_string(p: *const c_char) -> Result<String> {
    if p.is_null() {
        Ok(String::new())
    } else {
        unsafe { crate::cstr_to_string(p, "execution provider device string") }
    }
}

/// A queued EP-device attach — one or more (same-EP) [`EpDevice`]s + optional key/value
/// options. Applied to a session-options handle by [`apply_device_attach`] at session creation.
/// The device pointers are borrowed from the `Environment` (must outlive the session).
/// A queued EP-device attach — one or more (same-EP) [`EpDevice`]s + optional key/value
/// options. Applied to a session-options handle by [`apply_device_attach`] at session creation.
/// The device pointers are borrowed from the `Environment` (must outlive the session).
#[derive(Clone)]
pub(crate) struct EpDeviceAttach {
    pub(crate) devices: Vec<*const sys::EpDeviceHandle>,
    pub(crate) options: Vec<(CString, CString)>,
}

/// Apply queued EP-device attaches to a built session-options handle
/// (`SessionOptionsAppendExecutionProvider_V2`). Called from the session constructors (which
/// have the `env` the V2 call requires).
pub(crate) fn apply_device_attach(
    env: &crate::Environment, opts: *mut sys::SessionOptionsHandle, attaches: &[EpDeviceAttach],
) -> Result<()> {
    let f = unsafe { api().session_options_append_execution_provider_v2() };
    for attach in attaches {
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

    /// Enumerate the registered EP devices (printing name/vendor), and — if any are present —
    /// exercise the V2 attach path on a real session-options handle. On a CPU-only host
    /// `get_ep_devices` returns none and the attach is skipped; on a GPU host the CUDA device
    /// is discovered + attached.
    #[test]
    fn enumerate_and_attach_ep_devices() {
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
}
