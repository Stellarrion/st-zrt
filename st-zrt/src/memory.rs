//! `MemoryInfo` — describes where a tensor's backing memory lives.
use crate::{Result, api, check, sys};
use std::ffi::c_char;
use std::ptr;

pub struct MemoryInfo {
    pub(crate) info: *mut sys::MemoryInfoHandle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryInfoSnapshot {
    pub name: String,
    pub device_id: i32,
    pub alloc_type: sys::AllocatorType,
    pub mem_type: sys::MemType,
    pub device_type: i32,
    pub device_mem_type: i32,
    pub vendor_id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryDeviceSnapshot {
    pub device_type: i32,
    pub memory_type: i32,
    pub vendor_id: u32,
    pub device_id: u32,
}

impl MemoryInfoSnapshot {
    /// Whether a Rust slice may safely read/write this memory directly.
    #[inline]
    pub fn is_host_accessible(&self) -> bool {
        self.name == "Cpu" || self.name == "CudaPinned"
    }
}

impl MemoryInfo {
    /// CPU device memory (the configuration used by ORT's own zero-copy C samples).
    pub fn cpu() -> Result<Self> {
        let mut info: *mut sys::MemoryInfoHandle = ptr::null_mut();
        check(unsafe {
            api().create_cpu_memory_info()(
                sys::AllocatorType::Device,
                sys::MemType::Default,
                &mut info,
            )
        })?;
        let info = crate::ensure_non_null(info, "memory info")?;
        Ok(Self { info })
    }

    /// CUDA device memory (`CreateMemoryInfo("Cuda", Device, device_id, Default)`).
    ///
    /// Values allocated with this memory info live on the GPU. Do not expose them as Rust
    /// slices; use provider/device APIs to fill or read the raw device pointer, or bind them as
    /// device outputs and let ORT write into them.
    pub fn cuda(device_id: i32) -> Result<Self> {
        Self::new_named(
            "Cuda",
            sys::AllocatorType::Device,
            device_id,
            sys::MemType::Default,
        )
    }

    /// CUDA pinned host memory (`"CudaPinned"`). This is host-accessible memory associated with
    /// a CUDA device and can be used with Rust slices.
    pub fn cuda_pinned(device_id: i32) -> Result<Self> {
        Self::new_named(
            "CudaPinned",
            sys::AllocatorType::Device,
            device_id,
            sys::MemType::Default,
        )
    }

    /// General named constructor (`CreateMemoryInfo`, idx 68): a memory location identified by
    /// `name` (e.g. `"Cpu"`, `"CudaGPU"`) with an explicit allocator type, device id, and mem
    /// type. Use [`Self::cpu`] for the common CPU shortcut.
    pub fn new_named(
        name: &str, alloc_type: sys::AllocatorType, device_id: i32, mem_type: sys::MemType,
    ) -> Result<Self> {
        let cname = std::ffi::CString::new(name)
            .map_err(|_| crate::Error::new(-1, "memory name contains a NUL"))?;
        let mut info: *mut sys::MemoryInfoHandle = ptr::null_mut();
        check(unsafe {
            api().create_memory_info()(cname.as_ptr(), alloc_type, device_id, mem_type, &mut info)
        })?;
        let info = crate::ensure_non_null(info, "memory info")?;
        Ok(Self { info })
    }

    /// Provider name (e.g. `"Cpu"`). Borrowed from the engine; copied to an owned `String`.
    pub fn name(&self) -> Result<String> {
        let mut raw: *const c_char = ptr::null();
        check(unsafe {
            api().memory_info_get_name()(self.info as *const sys::MemoryInfoHandle, &mut raw)
        })?;
        if raw.is_null() {
            return Ok(String::new());
        }
        unsafe { crate::cstr_to_string(raw, "memory info name") }
    }

    /// Device id.
    pub fn device_id(&self) -> Result<i32> {
        let mut id: core::ffi::c_int = 0;
        check(unsafe {
            api().memory_info_get_id()(self.info as *const sys::MemoryInfoHandle, &mut id)
        })?;
        Ok(id)
    }

    /// Memory type (input/output/default).
    pub fn mem_type(&self) -> Result<sys::MemType> {
        let mut mt = sys::MemType::Default;
        check(unsafe {
            api().memory_info_get_mem_type()(self.info as *const sys::MemoryInfoHandle, &mut mt)
        })?;
        Ok(mt)
    }

    /// Allocator type (device/arena/…).
    pub fn alloc_type(&self) -> Result<sys::AllocatorType> {
        let mut at = sys::AllocatorType::Invalid;
        check(unsafe {
            api().memory_info_get_type()(self.info as *const sys::MemoryInfoHandle, &mut at)
        })?;
        Ok(at)
    }

    /// Copy the immutable ORT memory descriptor into Rust-owned data.
    pub fn snapshot(&self) -> Result<MemoryInfoSnapshot> {
        snapshot_from_ptr(self.info as *const sys::MemoryInfoHandle)
    }

    /// ORT 1.27 memory-device descriptor for this memory info.
    ///
    /// This is exposed through ORT's EP sub-API, so it is available when the `model-editor`
    /// feature is enabled in the current crate configuration.
    #[cfg(feature = "model-editor")]
    pub fn memory_device(&self) -> Result<MemoryDeviceSnapshot> {
        memory_device_from_memory_info(self.info as *const sys::MemoryInfoHandle)
    }

    /// Create a fresh ORT memory-info handle with the same descriptor fields.
    pub fn try_clone_descriptor(&self) -> Result<Self> {
        let snapshot = self.snapshot()?;
        if snapshot.name == "Cpu" {
            return Self::cpu();
        }
        Self::new_named(
            &snapshot.name,
            snapshot.alloc_type,
            snapshot.device_id,
            snapshot.mem_type,
        )
    }

    /// Whether a Rust slice may safely read/write this memory directly.
    pub fn is_host_accessible(&self) -> Result<bool> {
        Ok(self.snapshot()?.is_host_accessible())
    }
}

pub(crate) fn snapshot_from_ptr(info: *const sys::MemoryInfoHandle) -> Result<MemoryInfoSnapshot> {
    if info.is_null() {
        return Err(crate::Error::new(-1, "memory info pointer is null"));
    }

    let mut raw: *const c_char = ptr::null();
    check(unsafe { api().memory_info_get_name()(info, &mut raw) })?;
    let name = if raw.is_null() {
        String::new()
    } else {
        unsafe { crate::cstr_to_string(raw, "memory info name") }?
    };

    let mut device_id: core::ffi::c_int = 0;
    check(unsafe { api().memory_info_get_id()(info, &mut device_id) })?;

    let mut mem_type = sys::MemType::Default;
    check(unsafe { api().memory_info_get_mem_type()(info, &mut mem_type) })?;

    let mut alloc_type = sys::AllocatorType::Invalid;
    check(unsafe { api().memory_info_get_type()(info, &mut alloc_type) })?;

    let mut device_type = 0i32;
    unsafe { api().memory_info_get_device_type()(info, &mut device_type) };

    let device_mem_type = unsafe { api().memory_info_get_device_mem_type()(info) };
    let vendor_id = unsafe { api().memory_info_get_vendor_id()(info) };

    Ok(MemoryInfoSnapshot {
        name,
        device_id,
        alloc_type,
        mem_type,
        device_type,
        device_mem_type,
        vendor_id,
    })
}

#[cfg(feature = "model-editor")]
pub(crate) fn memory_device_snapshot_from_ptr(
    device: *const sys::MemoryDeviceHandle,
) -> Result<MemoryDeviceSnapshot> {
    if device.is_null() {
        return Err(crate::Error::new(-1, "memory device pointer is null"));
    }
    let ep =
        crate::model_editor::ep_api().ok_or_else(|| crate::Error::new(-1, "EpApi unavailable"))?;
    let device_type = unsafe {
        ep.MemoryDevice_GetDeviceType
            .ok_or_else(|| crate::Error::new(-1, "MemoryDevice_GetDeviceType unavailable"))?(
            device
        )
    };
    let memory_type = unsafe {
        ep.MemoryDevice_GetMemoryType
            .ok_or_else(|| crate::Error::new(-1, "MemoryDevice_GetMemoryType unavailable"))?(
            device
        )
    };
    let vendor_id = unsafe {
        ep.MemoryDevice_GetVendorId
            .ok_or_else(|| crate::Error::new(-1, "MemoryDevice_GetVendorId unavailable"))?(
            device
        )
    };
    let device_id = unsafe {
        ep.MemoryDevice_GetDeviceId
            .ok_or_else(|| crate::Error::new(-1, "MemoryDevice_GetDeviceId unavailable"))?(
            device
        )
    };
    Ok(MemoryDeviceSnapshot {
        device_type,
        memory_type,
        vendor_id,
        device_id,
    })
}

#[cfg(feature = "model-editor")]
pub(crate) fn memory_device_from_memory_info(
    info: *const sys::MemoryInfoHandle,
) -> Result<MemoryDeviceSnapshot> {
    if info.is_null() {
        return Err(crate::Error::new(-1, "memory info pointer is null"));
    }
    let ep =
        crate::model_editor::ep_api().ok_or_else(|| crate::Error::new(-1, "EpApi unavailable"))?;
    let device = unsafe {
        ep.MemoryInfo_GetMemoryDevice
            .ok_or_else(|| crate::Error::new(-1, "MemoryInfo_GetMemoryDevice unavailable"))?(
            info
        )
    };
    memory_device_snapshot_from_ptr(device)
}

impl Drop for MemoryInfo {
    fn drop(&mut self) {
        unsafe { api().release_memory_info()(self.info) }
    }
}
// OrtMemoryInfo is an immutable, thread-safe descriptor — safe to share.
unsafe impl Send for MemoryInfo {}
unsafe impl Sync for MemoryInfo {}
