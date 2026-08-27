//! Model compatibility pre-flight (ORT 1.27): probe whether a model will run on a given
//! execution provider **before** loading it.
//!
//! [`compatibility_info_from_path`] / [`compatibility_info_from_bytes`] fetch an opaque
//! compatibility-info string the engine allocates for a (model, EP type) pair. With the `ep`
//! feature, [`compatibility_for_ep_devices`] turns that string + a set of same-EP devices into
//! a [`CompiledModelCompatibility`] verdict. This lets a serving layer reject an incompatible
//! model/EP combination at admission time instead of at session creation.
use crate::{Allocator, Error, Result, api, check};
use std::ffi::CString;
use std::ptr;

/// Verdict returned by `compatibility_for_ep_devices` (`OrtCompiledModelCompatibility`).
///
/// Values mirror the ORT C enum exactly (`onnxruntime_c_api.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum CompiledModelCompatibility {
    /// The EP does not apply to this model (e.g. a mismatched EP type was queried).
    EpNotApplicable = 0,
    /// The model is supported and runs optimally on this EP.
    EpSupportedOptimal = 1,
    /// Supported, but a recompilation (e.g. a TensorRT cache rebuild) is preferable.
    EpSupportedPreferRecompilation = 2,
    /// The EP cannot run this model.
    EpUnsupported = 3,
}

impl CompiledModelCompatibility {
    #[cfg(feature = "ep")]
    fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => Self::EpNotApplicable,
            1 => Self::EpSupportedOptimal,
            2 => Self::EpSupportedPreferRecompilation,
            3 => Self::EpUnsupported,
            other => {
                return Err(Error::new(
                    -1,
                    format!("zrt: unknown CompiledModelCompatibility value {other}"),
                ));
            },
        })
    }

    /// True when the model can run on this EP — optimally or with a recompilation.
    #[inline]
    pub fn is_supported(self) -> bool {
        matches!(
            self,
            Self::EpSupportedOptimal | Self::EpSupportedPreferRecompilation
        )
    }
}

/// Fetch the compatibility-info string for a model on disk and an EP type
/// (`GetCompatibilityInfoFromModel`). `ep_type` is an EP registration name such as
/// `"CUDAExecutionProvider"` or `"CPUExecutionProvider"` (non-empty).
///
/// Returns `Ok(None)` when the model carries no compatibility info for `ep_type` — per the ORT
/// contract this is the case for a **standard (non-precompiled) ONNX model**; only models
/// precompiled via `OrtCompileApi` embed an info string. A non-empty `Some(_)` is the
/// engine-allocated string, copied into an owned `String`.
pub fn compatibility_info_from_path(model_path: &str, ep_type: &str) -> Result<Option<String>> {
    let cpath =
        CString::new(model_path).map_err(|_| Error::new(-1, "model path contains a NUL byte"))?;
    let cep = CString::new(ep_type).map_err(|_| Error::new(-1, "ep type contains a NUL byte"))?;
    let alloc = Allocator::get_default()?;
    let mut info: *mut core::ffi::c_char = ptr::null_mut();
    check(unsafe {
        api().get_compatibility_info_from_model()(
            cpath.as_ptr(),
            cep.as_ptr(),
            alloc.alloc,
            &mut info,
        )
    })?;
    copy_and_free_cstr(info, &alloc)
}

/// Fetch the compatibility-info string from an in-memory model byte buffer
/// (`GetCompatibilityInfoFromModelBytes`). See [`compatibility_info_from_path`].
pub fn compatibility_info_from_bytes(model: &[u8], ep_type: &str) -> Result<Option<String>> {
    let cep = CString::new(ep_type).map_err(|_| Error::new(-1, "ep type contains a NUL byte"))?;
    let alloc = Allocator::get_default()?;
    let mut info: *mut core::ffi::c_char = ptr::null_mut();
    check(unsafe {
        api().get_compatibility_info_from_model_bytes()(
            model.as_ptr() as *const core::ffi::c_void,
            model.len(),
            cep.as_ptr(),
            alloc.alloc,
            &mut info,
        )
    })?;
    copy_and_free_cstr(info, &alloc)
}

/// Given a compatibility-info string (from [`compatibility_info_from_path`] /
/// [`compatibility_info_from_bytes`]) and one or more EP devices that **must belong to the same
/// execution provider**, return the group's compatibility verdict
/// (`GetModelCompatibilityForEpDevices`). Requires feature `ep`.
#[cfg(feature = "ep")]
pub fn compatibility_for_ep_devices(
    ep_devices: &[&crate::EpDevice], compatibility_info: &str,
) -> Result<CompiledModelCompatibility> {
    if ep_devices.is_empty() {
        return Err(Error::new(
            -1,
            "zrt: compatibility_for_ep_devices needs at least one EP device",
        ));
    }
    let first = ep_devices[0];
    let ep_name = first.ep_name()?;
    for device in ep_devices.iter().skip(1) {
        if !first.shares_device_environment(device) {
            return Err(Error::new(
                -1,
                "zrt: compatibility devices belong to different Environments",
            ));
        }
        if device.ep_name()? != ep_name {
            return Err(Error::new(
                -1,
                "zrt: compatibility devices belong to different execution providers",
            ));
        }
    }
    let cinfo = CString::new(compatibility_info)
        .map_err(|_| Error::new(-1, "compatibility info contains a NUL byte"))?;
    let ptrs: Vec<*const crate::sys::EpDeviceHandle> =
        ep_devices.iter().map(|d| d.as_ptr()).collect();
    let mut out: i32 = 0;
    check(unsafe {
        api().get_model_compatibility_for_ep_devices()(
            ptrs.as_ptr(),
            ptrs.len(),
            cinfo.as_ptr(),
            &mut out,
        )
    })?;
    CompiledModelCompatibility::from_i32(out)
}

/// Copy an engine-allocated C string into an owned `String`, then free it through the
/// allocator that produced it. A null pointer maps to `None` (no compat info) and is not freed.
fn copy_and_free_cstr(raw: *mut core::ffi::c_char, alloc: &Allocator) -> Result<Option<String>> {
    if raw.is_null() {
        return Ok(None);
    }
    let s = unsafe { crate::cstr_to_string(raw, "compatibility info") };
    // Best-effort free; a free error here is not actionable for the caller.
    let _ = unsafe { alloc.free(raw as *mut std::ffi::c_void) };
    s.map(Some)
}
