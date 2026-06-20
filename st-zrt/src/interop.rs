//! External-resource interop (feature `model-editor`) — the `OrtInteropApi` zero-copy
//! GPU/graphics memory + semaphore import path (Vulkan / D3D12).
//!
//! These functions let an external graphics API share GPU buffers + timeline semaphores with
//! an ORT execution provider for zero-copy, synchronized inference. They are **GPU/EP-only**:
//! every entry point needs an `OrtEpDevice` handle (from `ep_api` device enumeration) and a
//! platform graphics handle (a Vulkan `VkDeviceMemory` / D3D12 `HANDLE` / fd), so the import /
//! wait / signal calls are type-checked + gateway-verified here but cannot run on a CPU-only
//! host. The CPU-runnable surface is the gateway check + the null-tolerant release functions.
//!
//! The descriptor structs + handle-type enums are hand-written `#[repr(C)]` here — the codegen
//! erased them to opaque handles. Layouts follow `onnxruntime_c_api.h` (field order + the C
//! ABI); `const` size-asserts pin the field count.
//!
//! # Codegen caveat
//! `CanImportMemory` / `CanImportSemaphore` pass `OrtExternalMemoryHandleType` /
//! `OrtExternalSemaphoreType` **by value** (a 4-byte `i32` enum), but the codegen emitted those
//! params as the 8-byte opaque `…Handle` structs. Calling through the generated signature would
//! be an ABI mismatch, so those two calls transmute the fn pointer to the true
//! `(importer, i32, &mut bool)` ABI (same C function underneath).
use crate::{check, sys, Result};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;

/// Borrow the live `InteropApi` table, or an error if the engine didn't populate it.
fn ia() -> Result<&'static sys::InteropApi> {
    crate::interop_api().ok_or_else(|| crate::Error::new(-1, "InteropApi unavailable"))
}

fn ia_fn<T: Copy>(f: Option<T>, function_name: &str) -> Result<T> {
    crate::model_editor::require_sub_api_fn(f, "InteropApi", function_name)
}

// ── handle-type / graphics-api enums (header: OrtExternalMemoryHandleType etc.) ──

/// External memory handle type (`OrtExternalMemoryHandleType`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalMemoryHandleType {
    /// D3D12 shared resource HANDLE.
    D3D12Resource = 0,
    /// D3D12 shared heap HANDLE.
    D3D12Heap = 1,
    /// Vulkan memory, Win32 HANDLE (`vkGetMemoryWin32HandleKHR`).
    VkMemoryWin32 = 2,
    /// Vulkan memory, opaque fd (`vkGetMemoryOpaqueFdKHR`).
    VkMemoryOpaqueFd = 3,
}

/// External semaphore type (`OrtExternalSemaphoreType`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalSemaphoreType {
    /// D3D12 fence HANDLE.
    D3D12Fence = 0,
    /// Vulkan timeline semaphore, Win32 HANDLE.
    VkTimelineSemaphoreWin32 = 1,
    /// Vulkan timeline semaphore, opaque fd.
    VkTimelineSemaphoreOpaqueFd = 2,
}

/// Graphics API for interop (`OrtGraphicsApi`, since v1.25).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphicsApi {
    /// No graphics interop (default).
    #[default]
    None = 0,
    /// Direct3D 12.
    D3D12 = 1,
    /// Vulkan.
    Vulkan = 2,
}

// ── descriptor structs (header: OrtExternalMemoryDescriptor etc.). `version` = API_VERSION. ──

/// Descriptor for importing external memory (`OrtExternalMemoryDescriptor`). `version` is
/// stamped to [`sys::API_VERSION`] (ORT requires it for forward compatibility).
#[repr(C)]
pub struct ExternalMemoryDescriptor {
    pub version: u32,
    pub handle_type: ExternalMemoryHandleType,
    pub native_handle: *mut c_void,
    pub size_bytes: usize,
    pub offset_bytes: usize,
}
const _: () = assert!(std::mem::size_of::<ExternalMemoryDescriptor>() == 32);

impl ExternalMemoryDescriptor {
    /// Build a descriptor for `native_handle` (a platform graphics handle) of `size_bytes`,
    /// imported as `handle_type`, at offset 0.
    pub fn new(
        handle_type: ExternalMemoryHandleType, native_handle: *mut c_void, size_bytes: usize,
    ) -> Self {
        Self {
            version: sys::API_VERSION,
            handle_type,
            native_handle,
            size_bytes,
            offset_bytes: 0,
        }
    }
}

/// Descriptor for importing an external semaphore (`OrtExternalSemaphoreDescriptor`).
#[repr(C)]
pub struct ExternalSemaphoreDescriptor {
    pub version: u32,
    pub semaphore_type: ExternalSemaphoreType,
    pub native_handle: *mut c_void,
}
const _: () = assert!(std::mem::size_of::<ExternalSemaphoreDescriptor>() == 16);

impl ExternalSemaphoreDescriptor {
    /// Build a descriptor for `native_handle` imported as `semaphore_type`.
    pub fn new(semaphore_type: ExternalSemaphoreType, native_handle: *mut c_void) -> Self {
        Self {
            version: sys::API_VERSION,
            semaphore_type,
            native_handle,
        }
    }
}

/// Descriptor for creating a tensor over imported memory (`OrtExternalTensorDescriptor`).
#[repr(C)]
pub struct ExternalTensorDescriptor<'a> {
    pub version: u32,
    pub element_type: sys::ElementType,
    pub shape: *const i64,
    pub rank: usize,
    pub offset_bytes: usize,
    _shape: PhantomData<&'a [i64]>,
}
const _: () = assert!(std::mem::size_of::<ExternalTensorDescriptor<'static>>() == 32);

impl<'a> ExternalTensorDescriptor<'a> {
    /// Build a descriptor for an `element_type` tensor of `shape`, at offset 0 within the
    /// imported memory. Borrows `shape` for the call that consumes the descriptor.
    pub fn new(element_type: sys::ElementType, shape: &'a [i64]) -> Self {
        Self {
            version: sys::API_VERSION,
            element_type,
            shape: shape.as_ptr(),
            rank: shape.len(),
            offset_bytes: 0,
            _shape: PhantomData,
        }
    }
}

/// Graphics-interop config (`OrtGraphicsInteropConfig`, since v1.25). `additional_options` is
/// an optional `OrtKeyValuePairs*` (pass null via [`Self::new`]); it is typed `*const c_void`
/// here as the codegen erased `OrtKeyValuePairs`.
#[repr(C)]
pub struct GraphicsInteropConfig {
    pub version: u32,
    pub graphics_api: GraphicsApi,
    pub command_queue: *mut c_void,
    pub additional_options: *const c_void,
}
const _: () = assert!(std::mem::size_of::<GraphicsInteropConfig>() == 24);

impl GraphicsInteropConfig {
    /// Build a config for `graphics_api` with an optional `command_queue` (D3D12
    /// `ID3D12CommandQueue*`; null for Vulkan) and no additional options.
    pub fn new(graphics_api: GraphicsApi, command_queue: *mut c_void) -> Self {
        Self {
            version: sys::API_VERSION,
            graphics_api,
            command_queue,
            additional_options: ptr::null(),
        }
    }
}

// ── owning handles ────────────────────────────────────────────────────────────

/// Owning `OrtExternalMemoryHandle` — released on drop (`ReleaseExternalMemoryHandle`).
/// Created by [`ExternalResourceImporter::import_memory`].
pub struct ExternalMemoryHandle {
    ptr: *mut sys::ExternalMemoryHandleHandle,
}
impl Drop for ExternalMemoryHandle {
    fn drop(&mut self) {
        if let Some(ia) = crate::interop_api() {
            // ReleaseExternalMemoryHandle tolerates null per the header.
            if let Some(release) = ia.ReleaseExternalMemoryHandle {
                unsafe { release(self.ptr) };
            }
        }
    }
}

/// Owning `OrtExternalSemaphoreHandle` — released on drop (`ReleaseExternalSemaphoreHandle`).
/// Created by [`ExternalResourceImporter::import_semaphore`].
pub struct ExternalSemaphoreHandle {
    ptr: *mut sys::ExternalSemaphoreHandleHandle,
}
impl Drop for ExternalSemaphoreHandle {
    fn drop(&mut self) {
        if let Some(ia) = crate::interop_api() {
            if let Some(release) = ia.ReleaseExternalSemaphoreHandle {
                unsafe { release(self.ptr) };
            }
        }
    }
}

/// Owning `OrtExternalResourceImporter` — the capability object for an EP device, created via
/// [`Self::for_device`]. Released on drop.
pub struct ExternalResourceImporter {
    ptr: *mut sys::ExternalResourceImporterHandle,
}

impl ExternalResourceImporter {
    /// Create an importer for `ep_device` (an `OrtEpDevice*` from device enumeration). Returns
    /// `Ok(None)` if the EP does not support external-resource import (the header sets
    /// `out_importer` null and returns success in that case).
    ///
    /// # Safety
    /// `ep_device` must be a valid `OrtEpDevice*` obtained from device enumeration.
    pub unsafe fn for_device(ep_device: *const sys::EpDeviceHandle) -> Result<Option<Self>> {
        let create = ia_fn(
            ia()?.CreateExternalResourceImporterForDevice,
            "CreateExternalResourceImporterForDevice",
        )?;
        let mut p: *mut sys::ExternalResourceImporterHandle = ptr::null_mut();
        check(unsafe { create(ep_device, &mut p) })?;
        Ok(if p.is_null() {
            None
        } else {
            Some(Self { ptr: p })
        })
    }

    /// Whether the importer can import `handle_type` memory.
    pub fn can_import_memory(&self, handle_type: ExternalMemoryHandleType) -> Result<bool> {
        let mut supported = false;
        // SAFETY: the codegen typed the by-value enum param as the 8-byte opaque
        // `ExternalMemoryHandleTypeHandle`; the true C ABI is a 4-byte i32 enum. Reinterpret
        // the fn pointer to the correct `(importer, i32, &mut bool)` signature — it is the
        // same C function underneath.
        let f: unsafe extern "C" fn(
            *const sys::ExternalResourceImporterHandle,
            i32,
            *mut bool,
        ) -> sys::StatusPtr =
            unsafe { std::mem::transmute(ia_fn(ia()?.CanImportMemory, "CanImportMemory")?) };
        check(unsafe { f(self.ptr, handle_type as i32, &mut supported) })?;
        Ok(supported)
    }

    /// Import external memory described by `desc` (BORROWED for the call). The returned handle
    /// owns the EP-side resource; release it (or let it drop) after the tensors built over it.
    pub fn import_memory(&self, desc: &ExternalMemoryDescriptor) -> Result<ExternalMemoryHandle> {
        let import = ia_fn(ia()?.ImportMemory, "ImportMemory")?;
        let mut p: *mut sys::ExternalMemoryHandleHandle = ptr::null_mut();
        check(unsafe {
            import(
                self.ptr,
                desc as *const ExternalMemoryDescriptor
                    as *const sys::ExternalMemoryDescriptorHandle,
                &mut p,
            )
        })?;
        let p = crate::ensure_non_null(p, "external memory handle")?;
        Ok(ExternalMemoryHandle { ptr: p })
    }

    /// Build a zero-copy tensor view over `mem_handle` per `tensor_desc` (both borrowed). The
    /// returned [`crate::OwnedValue`] is valid only while `mem_handle` stays alive.
    pub fn create_tensor_from_memory(
        &self, mem_handle: &ExternalMemoryHandle, tensor_desc: &ExternalTensorDescriptor<'_>,
    ) -> Result<crate::OwnedValue> {
        let create = ia_fn(ia()?.CreateTensorFromMemory, "CreateTensorFromMemory")?;
        let mut p: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            create(
                self.ptr,
                mem_handle.ptr,
                tensor_desc as *const ExternalTensorDescriptor
                    as *const sys::ExternalTensorDescriptorHandle,
                &mut p,
            )
        })?;
        let p = crate::ensure_non_null(p, "external tensor value")?;
        crate::OwnedValue::from_introspect(p)
    }

    /// Whether the importer can import `semaphore_type` semaphores.
    pub fn can_import_semaphore(&self, semaphore_type: ExternalSemaphoreType) -> Result<bool> {
        let mut supported = false;
        // SAFETY: same enum-by-value ABI caveat as `can_import_memory` above.
        let f: unsafe extern "C" fn(
            *const sys::ExternalResourceImporterHandle,
            i32,
            *mut bool,
        ) -> sys::StatusPtr =
            unsafe { std::mem::transmute(ia_fn(ia()?.CanImportSemaphore, "CanImportSemaphore")?) };
        check(unsafe { f(self.ptr, semaphore_type as i32, &mut supported) })?;
        Ok(supported)
    }

    /// Import an external semaphore described by `desc` (BORROWED for the call).
    pub fn import_semaphore(
        &self, desc: &ExternalSemaphoreDescriptor,
    ) -> Result<ExternalSemaphoreHandle> {
        let import = ia_fn(ia()?.ImportSemaphore, "ImportSemaphore")?;
        let mut p: *mut sys::ExternalSemaphoreHandleHandle = ptr::null_mut();
        check(unsafe {
            import(
                self.ptr,
                desc as *const ExternalSemaphoreDescriptor
                    as *const sys::ExternalSemaphoreDescriptorHandle,
                &mut p,
            )
        })?;
        let p = crate::ensure_non_null(p, "external semaphore handle")?;
        Ok(ExternalSemaphoreHandle { ptr: p })
    }

    /// Insert a wait on `semaphore` into the EP's `stream` until it reaches `value`
    /// (synchronize with external GPU work). `stream` is an `OrtSyncStream*` from
    /// `CreateSyncStreamForEpDevice`.
    ///
    /// # Safety
    /// `stream` must be a valid `OrtSyncStream*` for this EP, alive for the call.
    pub unsafe fn wait_semaphore(
        &self, semaphore: &ExternalSemaphoreHandle, stream: *mut sys::SyncStreamHandle, value: u64,
    ) -> Result<()> {
        let wait = ia_fn(ia()?.WaitSemaphore, "WaitSemaphore")?;
        check(unsafe { wait(self.ptr, semaphore.ptr, stream, value) })
    }

    /// Insert a signal of `value` on `semaphore` from the EP's `stream` (notify external GPU
    /// work that inference is complete).
    ///
    /// # Safety
    /// `stream` must be a valid `OrtSyncStream*` for this EP, alive for the call.
    pub unsafe fn signal_semaphore(
        &self, semaphore: &ExternalSemaphoreHandle, stream: *mut sys::SyncStreamHandle, value: u64,
    ) -> Result<()> {
        let signal = ia_fn(ia()?.SignalSemaphore, "SignalSemaphore")?;
        check(unsafe { signal(self.ptr, semaphore.ptr, stream, value) })
    }
}

impl Drop for ExternalResourceImporter {
    fn drop(&mut self) {
        if let Some(ia) = crate::interop_api() {
            if let Some(release) = ia.ReleaseExternalResourceImporter {
                unsafe { release(self.ptr) };
            }
        }
    }
}

// ── graphics interop (factory-level init/deinit) ──────────────────────────────

/// Initialize graphics interop for `ep_device` using `config` (`InitGraphicsInteropForEpDevice`,
/// since v1.25). Requests the EP factory set up D3D12/Vulkan interop. `ep_device` is an
/// `OrtEpDevice*`; `config` is borrowed.
///
/// # Safety
/// `ep_device` must be a valid `OrtEpDevice*` obtained from device enumeration.
pub unsafe fn init_graphics_interop_for_ep_device(
    ep_device: *const sys::EpDeviceHandle, config: &GraphicsInteropConfig,
) -> Result<()> {
    let init = ia_fn(
        ia()?.InitGraphicsInteropForEpDevice,
        "InitGraphicsInteropForEpDevice",
    )?;
    check(unsafe {
        init(
            ep_device,
            config as *const GraphicsInteropConfig as *const sys::GraphicsInteropConfigHandle,
        )
    })
}

/// Tear down graphics interop for `ep_device` (`DeinitGraphicsInteropForEpDevice`, since v1.25).
///
/// # Safety
/// `ep_device` must be a valid `OrtEpDevice*` obtained from device enumeration.
pub unsafe fn deinit_graphics_interop_for_ep_device(
    ep_device: *const sys::EpDeviceHandle,
) -> Result<()> {
    let deinit = ia_fn(
        ia()?.DeinitGraphicsInteropForEpDevice,
        "DeinitGraphicsInteropForEpDevice",
    )?;
    check(unsafe { deinit(ep_device) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU-runnable surface: the InteropApi gateway is populated, the release functions are
    /// reachable + null-tolerant, and the hand-written descriptors stamp `version` correctly.
    /// (The import / wait / signal calls need an EpDevice + a graphics handle — GPU-only.)
    #[test]
    fn interop_gateway_and_descriptors() {
        let ia = ia().expect("InteropApi populated");
        // Release fns are present and accept null (header: "May be nullptr").
        unsafe {
            ia_fn(
                ia.ReleaseExternalResourceImporter,
                "ReleaseExternalResourceImporter",
            )
            .expect("ReleaseExternalResourceImporter")(ptr::null_mut());
            ia_fn(
                ia.ReleaseExternalMemoryHandle,
                "ReleaseExternalMemoryHandle",
            )
            .expect("ReleaseExternalMemoryHandle")(ptr::null_mut());
            ia_fn(
                ia.ReleaseExternalSemaphoreHandle,
                "ReleaseExternalSemaphoreHandle",
            )
            .expect("ReleaseExternalSemaphoreHandle")(ptr::null_mut());
        }
        // Descriptors stamp version = API_VERSION and carry the given handle.
        let md = ExternalMemoryDescriptor::new(
            ExternalMemoryHandleType::VkMemoryOpaqueFd,
            0x1000 as *mut c_void,
            4096,
        );
        assert_eq!(md.version, sys::API_VERSION);
        assert_eq!(md.handle_type, ExternalMemoryHandleType::VkMemoryOpaqueFd);
        assert_eq!(md.size_bytes, 4096);

        let sd = ExternalSemaphoreDescriptor::new(
            ExternalSemaphoreType::D3D12Fence,
            0x2000 as *mut c_void,
        );
        assert_eq!(sd.version, sys::API_VERSION);

        let td = ExternalTensorDescriptor::new(sys::ElementType::Float, &[2, 3]);
        assert_eq!(td.version, sys::API_VERSION);
        assert_eq!(td.rank, 2);

        let gc = GraphicsInteropConfig::new(GraphicsApi::Vulkan, ptr::null_mut());
        assert_eq!(gc.version, sys::API_VERSION);
        assert_eq!(gc.graphics_api, GraphicsApi::Vulkan);
        eprintln!("InteropApi gateway populated; release fns null-tolerant; descriptors sound");
    }
}
