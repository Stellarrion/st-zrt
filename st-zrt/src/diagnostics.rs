//! Engine diagnostics: the execution providers compiled into this ORT build, the build
//! version string, and the current GPU device id.
use crate::{Result, api, check};
use std::ptr;

/// The names of the execution providers compiled into this ORT build
/// (`GetAvailableProviders`), e.g. `["CPUExecutionProvider"]` on a CPU build. The provider-name
/// array is engine-allocated and freed via `ReleaseAvailableProviders`.
pub fn available_providers() -> Result<Vec<String>> {
    let mut providers: *mut *mut core::ffi::c_char = ptr::null_mut();
    let mut len: core::ffi::c_int = 0;
    check(unsafe { api().get_available_providers()(&mut providers, &mut len) })?;
    if providers.is_null() || len <= 0 {
        return Ok(Vec::new());
    }
    let result = (|| -> Result<Vec<String>> {
        let n = len as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = unsafe { *providers.add(i) };
            if s.is_null() {
                out.push(String::new());
            } else {
                out.push(unsafe { crate::cstr_to_string(s, "available provider name") }?);
            }
        }
        Ok(out)
    })();
    // Best-effort free of the engine-allocated array.
    unsafe { api().release_available_providers()(providers, len) };
    result
}

/// The ORT build/version string (`GetBuildInfoString`). Engine-owned static string; not freed.
pub fn build_info() -> Result<String> {
    let p = unsafe { api().get_build_info_string()() };
    if p.is_null() {
        Ok(String::new())
    } else {
        unsafe { crate::cstr_to_string(p, "build info") }
    }
}

/// The current GPU device id ORT will use (`GetCurrentGpuDeviceId`). Returns an error on a
/// build/host without a CUDA EP (no GPU).
pub fn current_gpu_device_id() -> Result<i32> {
    let mut id: core::ffi::c_int = 0;
    check(unsafe { api().get_current_gpu_device_id()(&mut id) })?;
    Ok(id)
}

/// Set the GPU device id ORT should use (`SetCurrentGpuDeviceId`). Returns an error on a
/// build/host without a CUDA EP.
pub fn set_current_gpu_device_id(device_id: i32) -> Result<()> {
    check(unsafe { api().set_current_gpu_device_id()(device_id) })
}
