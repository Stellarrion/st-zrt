//! Prepacked-weight cache shared across ORT sessions.
use crate::{api, check, sys, Result};
use std::sync::Arc;

pub(crate) struct PrepackedWeightsInner {
    ptr: *mut sys::PrepackedWeightsContainerHandle,
}

/// ORT prepacked-weight cache.
///
/// Pass the same container to multiple compatible session creations to let ORT reuse
/// transformed/prepacked weights. Sessions built with this cache keep an internal `Arc`
/// reference to it, so the raw ORT container outlives every session relying on it.
#[derive(Clone)]
pub struct PrepackedWeightsContainer {
    pub(crate) inner: Arc<PrepackedWeightsInner>,
}

impl PrepackedWeightsContainer {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut sys::PrepackedWeightsContainerHandle = std::ptr::null_mut();
        check(unsafe { api().create_prepacked_weights_container()(&mut ptr) })?;
        let ptr = crate::ensure_non_null(ptr, "prepacked weights container")?;
        Ok(Self {
            inner: Arc::new(PrepackedWeightsInner { ptr }),
        })
    }

    #[inline]
    pub(crate) fn as_mut_ptr(&self) -> *mut sys::PrepackedWeightsContainerHandle {
        self.inner.ptr
    }

    #[inline]
    pub(crate) fn share(&self) -> Arc<PrepackedWeightsInner> {
        self.inner.clone()
    }
}

impl Drop for PrepackedWeightsInner {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                api().release_prepacked_weights_container()(self.ptr);
            }
        }
    }
}

unsafe impl Send for PrepackedWeightsInner {}
unsafe impl Sync for PrepackedWeightsInner {}
