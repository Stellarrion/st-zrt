//! Allocators. The default allocator is an engine singleton (not freed by us); a
//! [`Allocator::create`] allocator is session-scoped and owned (released on drop).
use crate::memory::MemoryInfo;
use crate::session::Session;
use crate::{Result, api, check, sys};
use std::ffi::c_void;
use std::ptr;

/// An ORT allocator. Either the process-wide default singleton (not owned — never released)
/// or a session-scoped allocator created via [`Allocator::create`] (owned — released on drop).
pub struct Allocator {
    pub(crate) alloc: *mut sys::AllocatorHandle,
    owned: bool,
}

/// A copied snapshot of ORT allocator stats.
///
/// The exact keys are allocator/provider-specific. CPU arena allocators commonly expose
/// current/peak byte counters; allocators that do not support stats return an ORT error.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocatorStats {
    entries: Vec<(String, String)>,
}

/// Numeric diff between two allocator stat snapshots.
///
/// ORT reports provider-specific stats as strings. This type includes keys that were present
/// in both snapshots, parsed as integers, and changed between snapshots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocatorStatsDelta {
    entries: Vec<(String, i128)>,
}

impl AllocatorStats {
    #[inline]
    pub fn entries(&self) -> &[(String, String)] {
        &self.entries
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v.as_str()))
    }

    /// Diff this snapshot against a later one, keeping changed integer counters.
    pub fn diff(&self, after: &AllocatorStats) -> AllocatorStatsDelta {
        let entries = self
            .entries
            .iter()
            .filter_map(|(key, before)| {
                let before = before.parse::<i128>().ok()?;
                let after = after.get(key)?.parse::<i128>().ok()?;
                let delta = after - before;
                (delta != 0).then(|| (key.clone(), delta))
            })
            .collect();
        AllocatorStatsDelta { entries }
    }
}

impl AllocatorStatsDelta {
    #[inline]
    pub fn entries(&self) -> &[(String, i128)] {
        &self.entries
    }

    #[inline]
    pub fn get(&self, key: &str) -> Option<i128> {
        self.entries
            .iter()
            .find_map(|(k, v)| (k == key).then_some(*v))
    }
}

impl Allocator {
    /// The ORT default allocator (a process singleton; releasing it is not our job).
    pub fn get_default() -> Result<Self> {
        let mut alloc: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe { api().get_allocator_with_default_options()(&mut alloc) })?;
        let alloc = crate::ensure_non_null(alloc, "default allocator")?;
        Ok(Self {
            alloc,
            owned: false,
        })
    }

    /// Create a session-scoped allocator for `mem` (`CreateAllocator`, idx 131; released via
    /// `ReleaseAllocator`, idx 132 on drop). Lets a caller allocate/free buffers through the
    /// same provider the session uses.
    pub fn create(session: &Session, mem: &MemoryInfo) -> Result<Self> {
        let mut alloc: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe {
            api().create_allocator()(
                session.as_ptr() as *const sys::SessionHandle,
                mem.info as *const sys::MemoryInfoHandle,
                &mut alloc,
            )
        })?;
        let alloc = crate::ensure_non_null(alloc, "session allocator")?;
        Ok(Self { alloc, owned: true })
    }

    /// Allocate `size` bytes (`AllocatorAlloc`, idx 75). The returned [`Allocation`] frees
    /// itself on drop (`AllocatorFree`, idx 76).
    pub fn allocate(&self, size: usize) -> Result<Allocation<'_>> {
        let mut p: *mut c_void = ptr::null_mut();
        check(unsafe { api().allocator_alloc()(self.alloc, size, &mut p) })?;
        Ok(Allocation {
            ptr: p,
            alloc: self,
        })
    }

    /// Snapshot allocator/provider stats via `AllocatorGetStats`.
    ///
    /// This is a diagnostic call and may allocate while copying ORT's returned key/value
    /// strings into Rust-owned memory. Do not place it inside the measured hot path.
    pub fn stats(&self) -> Result<AllocatorStats> {
        let mut kvps: *mut sys::KeyValuePairsHandle = ptr::null_mut();
        check(unsafe { api().allocator_get_stats()(self.alloc, &mut kvps) })?;
        if kvps.is_null() {
            return Ok(AllocatorStats::default());
        }

        let mut keys: *const *const core::ffi::c_char = ptr::null();
        let mut values: *const *const core::ffi::c_char = ptr::null();
        let mut len: usize = 0;
        unsafe { api().get_key_value_pairs()(kvps, &mut keys, &mut values, &mut len) };

        let mut entries = Vec::with_capacity(len);
        for i in 0..len {
            let key = unsafe { *keys.add(i) };
            let value = unsafe { *values.add(i) };
            let key = if key.is_null() {
                String::new()
            } else {
                unsafe { crate::cstr_to_string(key, "allocator stats key") }?
            };
            let value = if value.is_null() {
                String::new()
            } else {
                unsafe { crate::cstr_to_string(value, "allocator stats value") }?
            };
            entries.push((key, value));
        }
        unsafe { api().release_key_value_pairs()(kvps) };
        Ok(AllocatorStats { entries })
    }

    /// Free a buffer the engine allocated and handed back (e.g. an I/O name string).
    pub(crate) unsafe fn free(&self, p: *mut c_void) -> Result<()> {
        unsafe { check(api().allocator_free()(self.alloc, p)) }
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        if self.owned {
            unsafe { api().release_allocator()(self.alloc) };
        }
    }
}
unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

/// A byte buffer allocated by an [`Allocator`]; freed on drop.
pub struct Allocation<'a> {
    ptr: *mut c_void,
    alloc: &'a Allocator,
}

impl<'a> Allocation<'a> {
    /// Read-only pointer to the allocated bytes.
    #[inline]
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr
    }
    /// Mutable pointer to the allocated bytes.
    #[inline]
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        // Best-effort free; an error here is not actionable for the caller.
        let _ = unsafe { self.alloc.free(self.ptr) };
    }
}
unsafe impl Send for Allocation<'_> {}
unsafe impl Sync for Allocation<'_> {}
