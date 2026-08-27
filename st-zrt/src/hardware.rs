//! Hardware device enumeration (ORT 1.27): discover the physical devices ORT knows about
//! and, for a given execution provider, why a specific hardware device is incompatible.
use crate::{Environment, Error, Result, api, check, sys};
use std::ffi::CString;
use std::ptr;
use std::sync::Arc;

/// The count of hardware devices known to `env` (`GetNumHardwareDevices`).
pub fn num_hardware_devices(env: &Environment) -> Result<usize> {
    let mut n: usize = 0;
    check(unsafe { api().get_num_hardware_devices()(env.as_ptr(), &mut n) })?;
    Ok(n)
}

/// Enumerate the hardware devices known to `env` (`GetHardwareDevices`). The returned
/// [`HardwareDevice`]s are borrowed from ORT — valid while `env` is alive; never released.
/// Empty on a host with no discoverable devices (e.g. a CPU-only build).
pub fn hardware_devices(env: &Environment) -> Result<Vec<HardwareDevice>> {
    let n = num_hardware_devices(env)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut ptrs: Vec<*const sys::HardwareDeviceHandle> = vec![ptr::null(); n];
    check(unsafe {
        api().get_hardware_devices()(
            env.as_ptr(),
            ptrs.as_mut_ptr() as *const *const sys::HardwareDeviceHandle,
            n,
        )
    })?;
    ptrs.into_iter()
        .map(|ptr| {
            let ptr =
                crate::ensure_non_null(ptr as *mut sys::HardwareDeviceHandle, "hardware device")?;
            Ok(HardwareDevice {
                ptr,
                env: env.share(),
            })
        })
        .collect()
}

/// An engine-owned `OrtHardwareDevice` discovered via [`hardware_devices`]. The native handle is
/// never released; this wrapper retains the environment that makes it valid.
#[derive(Clone)]
pub struct HardwareDevice {
    pub(crate) ptr: *const sys::HardwareDeviceHandle,
    env: Arc<crate::environment::EnvInner>,
}

impl HardwareDevice {
    /// Wrap an engine-owned (borrowed) hardware-device handle (e.g. from `EpDevice_Device`).
    ///
    /// # Safety
    /// `ptr` must remain valid for as long as this wrapper is used and must not be released.
    #[cfg(feature = "ep")]
    pub(crate) unsafe fn from_borrowed(
        ptr: *const sys::HardwareDeviceHandle, env: Arc<crate::environment::EnvInner>,
    ) -> Self {
        Self { ptr, env }
    }

    pub(crate) fn shares_environment(&self, env: &Environment) -> bool {
        Arc::ptr_eq(&self.env, &env.share())
    }

    /// The hardware-device class (`HardwareDevice_Type`, `OrtHardwareDeviceType`): CPU/GPU/FPGA/NPU
    /// etc. Raw `i32` — ORT does not publish the enum values as a named type.
    pub fn ty(&self) -> i32 {
        // SAFETY: the handle is engine-owned and valid for the borrow.
        unsafe { api().hardware_device__type()(self.ptr) }
    }

    /// The PCI/manufacturer vendor id (`HardwareDevice_VendorId`), e.g. `0x10de` for NVIDIA.
    pub fn vendor_id(&self) -> u32 {
        unsafe { api().hardware_device__vendor_id()(self.ptr) }
    }

    /// Human-readable vendor string (`HardwareDevice_Vendor`), e.g. `"NVIDIA"`. Empty if unset.
    pub fn vendor(&self) -> Result<String> {
        let p = unsafe { api().hardware_device__vendor()(self.ptr) };
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "hardware device vendor") }
        }
    }

    /// The device id within its vendor (`HardwareDevice_DeviceId`).
    pub fn device_id(&self) -> u32 {
        unsafe { api().hardware_device__device_id()(self.ptr) }
    }

    /// Vendor metadata key/value pairs (`HardwareDevice_Metadata`). Borrowed from the device —
    /// never released; returns an empty view when the device carries no metadata.
    pub fn metadata(&self) -> crate::allocator::KeyValuePairsView<'_> {
        let p = unsafe { api().hardware_device__metadata()(self.ptr) };
        // SAFETY: the handle is engine-owned and valid for the lifetime of this borrow.
        unsafe { crate::allocator::KeyValuePairsView::from_borrowed(p) }
    }
}

/// Per-EP incompatibility details for a hardware device (`OrtDeviceEpIncompatibilityDetails`),
/// returned by [`hardware_device_ep_incompatibility_details`] when an EP cannot use a device.
/// Released with `ReleaseDeviceEpIncompatibilityDetails` on drop.
pub struct DeviceEpIncompatibilityDetails {
    details: *mut sys::DeviceEpIncompatibilityDetailsHandle,
}

impl DeviceEpIncompatibilityDetails {
    /// Bitmask of the reasons the EP/device combination is incompatible
    /// (`DeviceEpIncompatibilityDetails_GetReasonsBitmask`).
    pub fn reasons_bitmask(&self) -> Result<u32> {
        let mut v: u32 = 0;
        check(unsafe {
            api().device_ep_incompatibility_details__get_reasons_bitmask()(
                self.details as *const sys::DeviceEpIncompatibilityDetailsHandle,
                &mut v,
            )
        })?;
        Ok(v)
    }

    /// The ORT error code associated with the incompatibility
    /// (`DeviceEpIncompatibilityDetails_GetErrorCode`).
    pub fn error_code(&self) -> Result<i32> {
        let mut v: i32 = 0;
        check(unsafe {
            api().device_ep_incompatibility_details__get_error_code()(
                self.details as *const sys::DeviceEpIncompatibilityDetailsHandle,
                &mut v,
            )
        })?;
        Ok(v)
    }

    /// Human-readable notes on the incompatibility (`DeviceEpIncompatibilityDetails_GetNotes`).
    pub fn notes(&self) -> Result<String> {
        let mut p: *const core::ffi::c_char = ptr::null();
        check(unsafe {
            api().device_ep_incompatibility_details__get_notes()(
                self.details as *const sys::DeviceEpIncompatibilityDetailsHandle,
                &mut p,
            )
        })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "incompatibility notes") }
        }
    }
}

impl Drop for DeviceEpIncompatibilityDetails {
    fn drop(&mut self) {
        if !self.details.is_null() {
            unsafe { api().release_device_ep_incompatibility_details()(self.details) }
        }
    }
}
unsafe impl Send for DeviceEpIncompatibilityDetails {}
unsafe impl Sync for DeviceEpIncompatibilityDetails {}

/// Probe why `hw` is incompatible with the execution provider named `ep_name`
/// (`GetHardwareDeviceEpIncompatibilityDetails`). Returns `Ok(None)` when there are no
/// incompatibility details (the device is compatible, or the EP does not apply to it).
pub fn hardware_device_ep_incompatibility_details(
    env: &Environment, ep_name: &str, hw: &HardwareDevice,
) -> Result<Option<DeviceEpIncompatibilityDetails>> {
    if !hw.shares_environment(env) {
        return Err(Error::new(
            -1,
            "hardware device belongs to a different Environment",
        ));
    }
    let cname = CString::new(ep_name).map_err(|_| Error::new(-1, "ep name contains a NUL byte"))?;
    let mut details: *mut sys::DeviceEpIncompatibilityDetailsHandle = ptr::null_mut();
    check(unsafe {
        api().get_hardware_device_ep_incompatibility_details()(
            env.as_ptr(),
            cname.as_ptr(),
            hw.ptr,
            &mut details,
        )
    })?;
    if details.is_null() {
        Ok(None)
    } else {
        Ok(Some(DeviceEpIncompatibilityDetails { details }))
    }
}
