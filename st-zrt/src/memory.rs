//! `MemoryInfo` — describes where a tensor's backing memory lives.
use crate::{Result, api, check, sys};
use std::ffi::{CStr, c_char};
use std::ptr;

#[cfg(test)]
thread_local! {
    static CLASS_FROM_PTR_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The device class a memory location belongs to (`OrtMemoryInfoDeviceType`, since v1.24).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryInfoDeviceType {
    Cpu = 0,
    Gpu = 1,
    Fpga = 2,
    Npu = 3,
}

/// Device-memory kind (`OrtDeviceMemoryType`, since v1.24): device-local vs host-accessible
/// (shared/pinned) memory for CPU↔device transfer.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryType {
    /// Device-local memory.
    Default = 0,
    /// Shared/pinned memory for transferring between CPU and the device.
    HostAccessible = 5,
}

/// Structural memory class, computed once when a [`MemoryInfo`] wrapper is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryClass {
    /// CPU memory, including structurally host-accessible memory from non-CUDA providers.
    Cpu,
    /// Device-local CUDA memory.
    CudaDevice,
    /// CUDA page-locked host memory.
    CudaPinned,
    /// Device-local memory owned by another execution provider.
    OtherDevice,
    /// A legacy descriptor with no recognizable provider name.
    Unknown,
}

impl MemoryClass {
    /// Whether Rust may safely expose the backing memory as a host slice.
    #[inline]
    pub const fn is_host_accessible(self) -> bool {
        matches!(self, Self::Cpu | Self::CudaPinned)
    }
}

pub struct MemoryInfo {
    pub(crate) info: *mut sys::MemoryInfoHandle,
    class: MemoryClass,
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
    pub class: MemoryClass,
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
    ///
    /// Prefers the v2 structural flag `device_mem_type == HostAccessible` (covers pinned/shared
    /// device memory and is robust to provider names we don't enumerate, e.g. a future `"RocmPinned"`),
    /// and keeps the name fallback for v1/legacy infos that carry no `device_mem_type`: CPU memory
    /// is inherently host-resident (ORT leaves its `device_mem_type` at `Default`), and `"CudaPinned"`
    /// for the pre-v2 pinned case.
    #[inline]
    pub const fn is_host_accessible(&self) -> bool {
        self.class.is_host_accessible()
    }
}

fn classify_memory(name: &[u8], device_type: i32, device_mem_type: i32) -> MemoryClass {
    if device_mem_type == DeviceMemoryType::HostAccessible as i32 {
        return if device_type == MemoryInfoDeviceType::Gpu as i32
            || matches!(name, b"Cuda" | b"CudaGPU" | b"CudaPinned")
        {
            MemoryClass::CudaPinned
        } else {
            // `MemoryClass` models host accessibility, so shared/pinned memory from a provider
            // other than CUDA uses the host-resident class as well.
            MemoryClass::Cpu
        };
    }
    // The name checks must precede the structural CPU fallback: legacy CreateMemoryInfo
    // descriptors may not carry a meaningful v2 device type.
    if name == b"Cpu" {
        return MemoryClass::Cpu;
    }
    if name == b"Cuda" || name == b"CudaGPU" {
        return MemoryClass::CudaDevice;
    }
    if name == b"CudaPinned" {
        return MemoryClass::CudaPinned;
    }
    if name.is_empty() || device_type == MemoryInfoDeviceType::Cpu as i32 {
        // Legacy `CreateMemoryInfo` descriptors may report the default CPU device type even for an
        // unrecognized provider name. Do not turn that ambiguous default into host access.
        MemoryClass::Unknown
    } else {
        MemoryClass::OtherDevice
    }
}

/// Classify an engine-owned memory-info pointer without allocating a provider-name string.
pub(crate) fn class_from_ptr(info: *const sys::MemoryInfoHandle) -> Result<MemoryClass> {
    if info.is_null() {
        return Err(crate::Error::new(-1, "memory info pointer is null"));
    }
    #[cfg(test)]
    CLASS_FROM_PTR_CALLS.with(|calls| calls.set(calls.get() + 1));
    let mut device_type = 0i32;
    unsafe { api().memory_info_get_device_type()(info, &mut device_type) };
    let device_mem_type = unsafe { api().memory_info_get_device_mem_type()(info) };
    let mut raw: *const c_char = ptr::null();
    check(unsafe { api().memory_info_get_name()(info, &mut raw) })?;
    let name = if raw.is_null() {
        &[][..]
    } else {
        unsafe { CStr::from_ptr(raw) }.to_bytes()
    };
    Ok(classify_memory(name, device_type, device_mem_type))
}

impl MemoryInfo {
    fn from_raw_owned(info: *mut sys::MemoryInfoHandle, what: &'static str) -> Result<Self> {
        let info = crate::ensure_non_null(info, what)?;
        match class_from_ptr(info as *const sys::MemoryInfoHandle) {
            Ok(class) => Ok(Self { info, class }),
            Err(error) => {
                unsafe { api().release_memory_info()(info) };
                Err(error)
            },
        }
    }

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
        Self::from_raw_owned(info, "memory info")
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
        Self::from_raw_owned(info, "memory info")
    }

    /// Richer v2 constructor (`CreateMemoryInfo_V2`, idx 320, since v1.24) — adds device class,
    /// vendor id, device-memory kind, and an explicit alignment over the legacy [`Self::new_named`].
    /// Prefer this for device/EP memory where the extra fields matter; [`Self::cpu`]/[`Self::cuda`]
    /// remain the common shortcuts.
    pub fn new_v2(
        name: &str, device_type: MemoryInfoDeviceType, vendor_id: u32, device_id: i32,
        mem_type: DeviceMemoryType, alignment: usize, allocator_type: sys::AllocatorType,
    ) -> Result<Self> {
        let cname = std::ffi::CString::new(name)
            .map_err(|_| crate::Error::new(-1, "memory name contains a NUL"))?;
        let mut info: *mut sys::MemoryInfoHandle = ptr::null_mut();
        check(unsafe {
            api().create_memory_info_v2()(
                cname.as_ptr(),
                device_type as core::ffi::c_int,
                vendor_id,
                device_id,
                mem_type as core::ffi::c_int,
                alignment,
                allocator_type,
                &mut info,
            )
        })?;
        Self::from_raw_owned(info, "memory info v2")
    }

    /// Whether this memory info describes the same location as `other` (`CompareMemoryInfo`).
    /// ORT writes `0` for equal, `-1` for not equal.
    pub fn equals(&self, other: &MemoryInfo) -> Result<bool> {
        let mut out: core::ffi::c_int = 0;
        check(unsafe {
            api().compare_memory_info()(
                self.info as *const sys::MemoryInfoHandle,
                other.info as *const sys::MemoryInfoHandle,
                &mut out,
            )
        })?;
        Ok(out == 0)
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

    /// Cached structural memory class (zero FFI calls).
    #[inline]
    pub const fn class(&self) -> MemoryClass {
        self.class
    }

    /// Whether a Rust slice may safely read/write this memory directly (zero FFI calls).
    #[inline]
    pub const fn is_host_accessible(&self) -> bool {
        self.class.is_host_accessible()
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

    let class = classify_memory(name.as_bytes(), device_type, device_mem_type);
    Ok(MemoryInfoSnapshot {
        name,
        device_id,
        alloc_type,
        mem_type,
        device_type,
        device_mem_type,
        vendor_id,
        class,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrecognized_legacy_names_remain_conservative() {
        assert_eq!(
            classify_memory(b"ExampleLegacy", MemoryInfoDeviceType::Cpu as i32, 0),
            MemoryClass::Unknown,
        );
        assert_eq!(
            classify_memory(b"ExampleNpu", MemoryInfoDeviceType::Npu as i32, 0),
            MemoryClass::OtherDevice,
        );
        assert_eq!(
            classify_memory(
                b"ExampleShared",
                MemoryInfoDeviceType::Cpu as i32,
                DeviceMemoryType::HostAccessible as i32,
            ),
            MemoryClass::Cpu,
        );
    }

    #[test]
    fn cached_class_and_host_access_issue_no_reclassification_ffi() {
        let before = CLASS_FROM_PTR_CALLS.with(std::cell::Cell::get);
        let info = MemoryInfo::cpu().expect("CPU memory info");
        let after_construction = CLASS_FROM_PTR_CALLS.with(std::cell::Cell::get);
        assert_eq!(after_construction, before + 1);

        for _ in 0..1_000 {
            assert_eq!(info.class(), MemoryClass::Cpu);
            assert!(info.is_host_accessible());
        }
        assert_eq!(
            CLASS_FROM_PTR_CALLS.with(std::cell::Cell::get),
            after_construction,
            "cached access unexpectedly re-entered the FFI classification path",
        );
    }
}
