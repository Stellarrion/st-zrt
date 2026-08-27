//! LoRA adapter serving — load a base model once, then hot-swap low-rank adapter
//! weights per request without reloading the model.
//!
//! An [`LoraAdapter`] wraps an `OrtLoraAdapter` (loaded from a file or byte buffer with the
//! ORT default allocator, which is a process singleton). Attach one or more `Arc`-owned adapters to a
//! [`crate::RunOptions`] via [`crate::RunOptions::with_lora_adapter`] then [`crate::RunOptions::freeze`] it and pass the resulting
//! [`crate::MaterializedRunOptions`] to a run to activate the adapter for that run only.
use crate::{Allocator, Error, Result, api, check, ensure_non_null, sys};
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// An ORT LoRA adapter (`OrtLoraAdapter`): low-rank weights that modify a base model's
/// behaviour for the runs they are attached to. Cheap to load relative to a full model; a
/// typical serving setup loads the base model once and several adapters, then activates a
/// different adapter per request via [`crate::RunOptions::with_lora_adapter`].
///
/// The adapter is allocated through ORT's default allocator (a process singleton) and freed
/// with `ReleaseLoraAdapter` on drop. Thread-safe to share across runs via `Arc`.
pub struct LoraAdapter {
    adapter: *mut sys::LoraAdapterHandle,
}

impl LoraAdapter {
    /// Load a LoRA adapter from an ONNX adapter file on disk (`CreateLoraAdapter`).
    ///
    /// Uses ORT's default (CPU) allocator. ORT's lora path requires a non-CPU allocator with a
    /// data-transfer capability and rejects the CPU allocator, so this may error at load on a
    /// CPU-only setup — prefer [`Self::from_path_with_allocator`] with a device allocator from a
    /// session on the matching EP.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let cpath = CString::new(path.as_ref().to_string_lossy().into_owned())
            .map_err(|_| Error::new(-1, "lora adapter path contains a NUL byte"))?;
        Self::create(|alloc, out| unsafe {
            api().create_lora_adapter()(cpath.as_ptr(), alloc, out)
        })
    }

    /// Load a LoRA adapter from an in-memory byte buffer (`CreateLoraAdapterFromArray`).
    /// The buffer is copied by the engine; it does not need to outlive this call.
    ///
    /// Uses ORT's default (CPU) allocator — see [`Self::from_path`] for why that may be rejected;
    /// prefer [`Self::from_array_with_allocator`] on a device EP.
    pub fn from_array(bytes: &[u8]) -> Result<Self> {
        Self::create(|alloc, out| unsafe {
            api().create_lora_adapter_from_array()(
                bytes.as_ptr() as *const core::ffi::c_void,
                bytes.len(),
                alloc,
                out,
            )
        })
    }

    /// Load a LoRA adapter from disk against a specific `allocator` (`CreateLoraAdapter`). ORT's
    /// lora path needs a non-CPU allocator with a data-transfer capability (it rejects the CPU
    /// default), so pass a device allocator from a session on the matching EP — e.g.
    /// `Allocator::create(&cuda_session, &MemoryInfo::cuda(device_id))`.
    pub fn from_path_with_allocator<P: AsRef<Path>>(
        path: P, allocator: &Allocator,
    ) -> Result<Self> {
        let cpath = CString::new(path.as_ref().to_string_lossy().into_owned())
            .map_err(|_| Error::new(-1, "lora adapter path contains a NUL byte"))?;
        let mut adapter: *mut sys::LoraAdapterHandle = ptr::null_mut();
        check(unsafe {
            api().create_lora_adapter()(cpath.as_ptr(), allocator.alloc_handle(), &mut adapter)
        })?;
        let adapter = ensure_non_null(adapter, "lora adapter")?;
        Ok(Self { adapter })
    }

    /// Load a LoRA adapter from an in-memory byte buffer against a specific `allocator`
    /// (`CreateLoraAdapterFromArray`). See [`Self::from_path_with_allocator`] for why a device
    /// allocator is required.
    pub fn from_array_with_allocator(bytes: &[u8], allocator: &Allocator) -> Result<Self> {
        let mut adapter: *mut sys::LoraAdapterHandle = ptr::null_mut();
        check(unsafe {
            api().create_lora_adapter_from_array()(
                bytes.as_ptr() as *const core::ffi::c_void,
                bytes.len(),
                allocator.alloc_handle(),
                &mut adapter,
            )
        })?;
        let adapter = ensure_non_null(adapter, "lora adapter")?;
        Ok(Self { adapter })
    }

    /// Shared construction: borrow the default allocator for the call (it is a process
    /// singleton, so the local `Allocator` may drop immediately after), then null-check.
    fn create<F>(f: F) -> Result<Self>
    where
        F: FnOnce(*mut sys::AllocatorHandle, *mut *mut sys::LoraAdapterHandle) -> sys::StatusPtr,
    {
        let alloc = Allocator::get_default()?;
        let mut adapter: *mut sys::LoraAdapterHandle = ptr::null_mut();
        check(f(alloc.alloc, &mut adapter))?;
        let adapter = ensure_non_null(adapter, "lora adapter")?;
        Ok(Self { adapter })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::LoraAdapterHandle {
        self.adapter as *const sys::LoraAdapterHandle
    }
}

impl Drop for LoraAdapter {
    fn drop(&mut self) {
        if !self.adapter.is_null() {
            unsafe { api().release_lora_adapter()(self.adapter) }
        }
    }
}

// OrtLoraAdapter is an immutable, thread-safe handle once created.
unsafe impl Send for LoraAdapter {}
unsafe impl Sync for LoraAdapter {}
