//! `ModelMetadata` — the model's self-description (producer, graph name/description, domain,
//! version, custom metadata map). Obtained from a session via `SessionGetModelMetadata`
//! (idx 111, owning handle) and released on drop (`ReleaseModelMetadata`, idx 118).
use crate::allocator::Allocator;
use crate::{api, check, sys, Result};
use std::ffi::{c_char, c_void, CString};
use std::ptr;

pub struct ModelMetadata {
    meta: *mut sys::ModelMetadataHandle,
}

impl ModelMetadata {
    /// Wrap an owning handle from `SessionGetModelMetadata`. Owns it; releases on drop.
    pub(crate) unsafe fn from_owning(meta: *mut sys::ModelMetadataHandle) -> Self {
        Self { meta }
    }

    /// Engine-allocated string fields all share this signature: fetch, copy to an owned
    /// `String`, free the engine buffer. `None` when the field is absent (null).
    fn fetch_string(
        &self,
        f: unsafe extern "C" fn(
            *const sys::ModelMetadataHandle,
            *mut sys::AllocatorHandle,
            *mut *mut c_char,
        ) -> sys::StatusPtr,
    ) -> Result<Option<String>> {
        let alloc = Allocator::get_default()?;
        let mut raw: *mut c_char = ptr::null_mut();
        check(unsafe {
            f(
                self.meta as *const sys::ModelMetadataHandle,
                alloc.alloc,
                &mut raw,
            )
        })?;
        if raw.is_null() {
            return Ok(None);
        }
        let s = unsafe { crate::cstr_to_string(raw, "model metadata string") };
        let free = unsafe { alloc.free(raw as *mut c_void) };
        free?;
        Ok(Some(s?))
    }

    /// Producer name (e.g. `"pytorch"`, `"onnx"`) or `None`.
    pub fn producer_name(&self) -> Result<Option<String>> {
        self.fetch_string(unsafe { api().model_metadata_get_producer_name() })
    }
    /// Graph name or `None`.
    pub fn graph_name(&self) -> Result<Option<String>> {
        self.fetch_string(unsafe { api().model_metadata_get_graph_name() })
    }
    /// Model domain or `None`.
    pub fn domain(&self) -> Result<Option<String>> {
        self.fetch_string(unsafe { api().model_metadata_get_domain() })
    }
    /// Model description / doc string or `None` (`ModelMetadataGetDescription`, idx 115).
    pub fn description(&self) -> Result<Option<String>> {
        self.fetch_string(unsafe { api().model_metadata_get_description() })
    }
    /// Graph description or `None` (`ModelMetadataGetGraphDescription`, idx 158).
    pub fn graph_description(&self) -> Result<Option<String>> {
        self.fetch_string(unsafe { api().model_metadata_get_graph_description() })
    }

    /// Model version (`ModelMetadataGetVersion`, idx 117 — no allocator; returns int64).
    pub fn version(&self) -> Result<i64> {
        let mut v: i64 = 0;
        check(unsafe {
            api().model_metadata_get_version()(self.meta as *const sys::ModelMetadataHandle, &mut v)
        })?;
        Ok(v)
    }

    /// Look up a single key in the custom metadata map (`ModelMetadataLookupCustomMetadataMap`,
    /// idx 116). `None` if absent.
    pub fn lookup(&self, key: &str) -> Result<Option<String>> {
        let ckey =
            CString::new(key).map_err(|_| crate::Error::new(-1, "metadata key contains a NUL"))?;
        let alloc = Allocator::get_default()?;
        let mut raw: *mut c_char = ptr::null_mut();
        check(unsafe {
            api().model_metadata_lookup_custom_metadata_map()(
                self.meta as *const sys::ModelMetadataHandle,
                alloc.alloc,
                ckey.as_ptr(),
                &mut raw,
            )
        })?;
        if raw.is_null() {
            return Ok(None);
        }
        let s = unsafe { crate::cstr_to_string(raw, "custom metadata value") };
        let free = unsafe { alloc.free(raw as *mut c_void) };
        free?;
        Ok(Some(s?))
    }

    /// All keys in the custom metadata map (`ModelMetadataGetCustomMetadataMapKeys`, idx 123).
    pub fn custom_metadata_keys(&self) -> Result<Vec<String>> {
        let alloc = Allocator::get_default()?;
        let mut keys: *mut *mut c_char = ptr::null_mut();
        let mut num_keys: i64 = 0;
        check(unsafe {
            api().model_metadata_get_custom_metadata_map_keys()(
                self.meta as *const sys::ModelMetadataHandle,
                alloc.alloc,
                &mut keys,
                &mut num_keys,
            )
        })?;
        let mut out = Vec::with_capacity(num_keys.max(0) as usize);
        if num_keys > 0 && !keys.is_null() {
            let slice = unsafe { std::slice::from_raw_parts(keys, num_keys as usize) };
            let mut first_err = None;
            for &k in slice {
                if !k.is_null() {
                    let key = unsafe { crate::cstr_to_string(k, "custom metadata key") };
                    let free = unsafe { alloc.free(k as *mut c_void) };
                    match (key, free) {
                        (Ok(key), Ok(())) if first_err.is_none() => out.push(key),
                        (Err(err), _) if first_err.is_none() => first_err = Some(err),
                        (_, Err(err)) if first_err.is_none() => first_err = Some(err),
                        _ => {},
                    }
                }
            }
            // Free the array buffer (one allocation); individual strings freed above.
            let free_keys = unsafe { alloc.free(keys as *mut c_void) };
            if let Some(err) = first_err {
                free_keys?;
                return Err(err);
            }
            free_keys?;
        }
        Ok(out)
    }
}

impl Drop for ModelMetadata {
    fn drop(&mut self) {
        unsafe { api().release_model_metadata()(self.meta) }
    }
}
unsafe impl Send for ModelMetadata {}
unsafe impl Sync for ModelMetadata {}
