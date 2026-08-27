//! Allocators. The default allocator is an engine singleton (not freed by us); a
//! [`Allocator::create`] allocator is session-scoped and owned (released on drop).
use crate::environment::EnvInner;
use crate::memory::MemoryInfo;
use crate::session::{Session, SessionInner};
use crate::{Error, Result, api, check, sys};
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

/// An ORT allocator. Either the process-wide default singleton (not owned — never released)
/// or a session-scoped allocator created via [`Allocator::create`] (owned — released on drop).
pub struct Allocator {
    pub(crate) alloc: *mut sys::AllocatorHandle,
    owned: bool,
    // Session-scoped allocators are provider-owned and may only be released/used while their
    // originating native session is alive. Default and environment-shared allocators use `None`.
    _session: Option<Arc<SessionInner>>,
    // Environment-shared allocator refs likewise must be released before their originating Env.
    _env: Option<Arc<EnvInner>>,
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
            _session: None,
            _env: None,
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
        Ok(Self {
            alloc,
            owned: true,
            _session: Some(session.share_inner()),
            _env: None,
        })
    }

    /// Adopt an owning allocator handle (e.g. one returned by `CreateSharedAllocator` /
    /// `GetSharedAllocator`). Released via `ReleaseAllocator` on drop.
    #[cfg(feature = "ep")]
    pub(crate) fn from_handle_owned(alloc: *mut sys::AllocatorHandle, env: Arc<EnvInner>) -> Self {
        Self {
            alloc,
            owned: true,
            _session: None,
            _env: Some(env),
        }
    }

    /// Transfer the originating-session guard into an owning value allocated by this allocator.
    /// The value must retain the guard through both `ReleaseValue` and `ReleaseAllocator`.
    #[inline]
    pub(crate) fn take_session_guard(&mut self) -> Option<Arc<SessionInner>> {
        self._session.take()
    }

    /// The raw allocator handle (crate-private; the env register/shared-allocator calls borrow it).
    #[inline]
    pub(crate) fn alloc_handle(&self) -> *mut sys::AllocatorHandle {
        self.alloc
    }

    /// Allocate `size` bytes (`AllocatorAlloc`, idx 75). The returned [`Allocation`] frees
    /// itself on drop (`AllocatorFree`, idx 76).
    pub fn allocate(&self, size: usize) -> Result<Allocation<'_>> {
        let mut p: *mut c_void = ptr::null_mut();
        check(unsafe { api().allocator_alloc()(self.alloc, size, &mut p) })?;
        Ok(Allocation {
            ptr: p,
            len: size,
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

    /// The memory-info describing this allocator's allocations (`AllocatorGetInfo`). Borrowed from
    /// the engine — captured into an owned [`crate::MemoryInfoSnapshot`] (no handle to release).
    pub fn memory_info(&self) -> Result<crate::MemoryInfoSnapshot> {
        let mut info: *const sys::MemoryInfoHandle = ptr::null();
        check(unsafe { api().allocator_get_info()(self.alloc, &mut info) })?;
        if info.is_null() {
            return Err(crate::Error::new(
                -1,
                "zrt: allocator returned null memory info",
            ));
        }
        crate::memory::snapshot_from_ptr(info)
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        if self.owned {
            unsafe { api().release_allocator()(self.alloc) };
        }
    }
}
// SAFETY: ORT allocators (default + BFCArena + CUDA) are designed for concurrent alloc/free across
// threads, so a shared `&Allocator` is safe to use from multiple threads. This assumes the
// underlying ORT allocator is not device-thread-affine (true for ORT's built-in allocators).
unsafe impl Send for Allocator {}
unsafe impl Sync for Allocator {}

/// A byte buffer allocated by an [`Allocator`]; freed on drop.
pub struct Allocation<'a> {
    ptr: *mut c_void,
    len: usize,
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
    /// Allocated byte length (the `size` passed to [`Allocator::allocate`]).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether this allocation is zero bytes (from `Allocator::allocate(0)`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// The allocator handle that owns this buffer (the one that will free it).
    #[inline]
    pub(crate) fn allocator_handle(&self) -> *mut sys::AllocatorHandle {
        self.alloc.alloc
    }
    /// Detach the buffer from this `Allocation`, suppressing its `Drop` (so the bytes are no
    /// longer Rust-freed). Returns `(ptr, byte_len)`. The caller becomes responsible for freeing
    /// `ptr` through [`allocator_handle`] (e.g. by handing it to an ORT value with the allocator
    /// as its deleter).
    #[inline]
    pub(crate) fn into_raw_parts(self) -> (*mut c_void, usize) {
        let p = self.ptr;
        let n = self.len;
        std::mem::forget(self);
        (p, n)
    }
}

impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        // Best-effort free; an error here is not actionable for the caller.
        let _ = unsafe { self.alloc.free(self.ptr) };
    }
}
// SAFETY: an `Allocation` is a unique borrow of one allocator-owned buffer; it is `Send`/`Sync`
// under the same assumption as `Allocator` (ORT allocators are concurrency-safe). The caller must
// still avoid aliasing a single allocation's buffer across threads for mutation.
unsafe impl Send for Allocation<'_> {}
unsafe impl Sync for Allocation<'_> {}

/// An owning `OrtKeyValuePairs` container: a string→string map used by allocator stats and
/// session-config introspection. Built with [`KeyValuePairs::new`] + [`KeyValuePairs::add`];
/// released with `ReleaseKeyValuePairs` on drop.
pub struct KeyValuePairs {
    kvps: *mut sys::KeyValuePairsHandle,
}

impl KeyValuePairs {
    /// Create an empty key/value-pairs container (`CreateKeyValuePairs`).
    pub fn new() -> Result<Self> {
        let mut kvps: *mut sys::KeyValuePairsHandle = ptr::null_mut();
        // CreateKeyValuePairs returns void; a null result means allocation failed.
        unsafe { api().create_key_value_pairs()(&mut kvps) };
        let kvps = crate::ensure_non_null(kvps, "key/value pairs")?;
        Ok(Self { kvps })
    }

    /// Adopt an owning `OrtKeyValuePairs` handle returned by ORT (e.g.
    /// `GetSessionOptionsConfigEntries`). The wrapper releases it on drop.
    ///
    /// # Safety
    /// `kvps` must be a freshly-allocated owning handle that nothing else will release.
    pub(crate) unsafe fn from_handle(kvps: *mut sys::KeyValuePairsHandle) -> Self {
        Self { kvps }
    }

    /// Add or replace `key` → `value` (`AddKeyValuePair`).
    pub fn add(&mut self, key: &str, value: &str) -> Result<()> {
        let ck = CString::new(key).map_err(|_| Error::new(-1, "key/value key contains a NUL"))?;
        let cv =
            CString::new(value).map_err(|_| Error::new(-1, "key/value value contains a NUL"))?;
        unsafe { api().add_key_value_pair()(self.kvps, ck.as_ptr(), cv.as_ptr()) };
        Ok(())
    }

    /// Look up the value for `key` (`GetKeyValue`). Returns `None` if absent.
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let ck = CString::new(key).map_err(|_| Error::new(-1, "key/value key contains a NUL"))?;
        let p = unsafe { api().get_key_value()(self.kvps, ck.as_ptr()) };
        if p.is_null() {
            Ok(None)
        } else {
            unsafe { crate::cstr_to_string(p, "key/value pair value") }.map(Some)
        }
    }

    /// Remove `key` if present (`RemoveKeyValuePair`).
    pub fn remove(&mut self, key: &str) -> Result<()> {
        let ck = CString::new(key).map_err(|_| Error::new(-1, "key/value key contains a NUL"))?;
        unsafe { api().remove_key_value_pair()(self.kvps, ck.as_ptr()) };
        Ok(())
    }

    /// The raw `OrtKeyValuePairs*` (crate-private; ORT copies it at the one call site that reads it).
    pub(crate) fn raw_ptr(&self) -> *const sys::KeyValuePairsHandle {
        self.kvps as *const sys::KeyValuePairsHandle
    }
}

impl Drop for KeyValuePairs {
    fn drop(&mut self) {
        if !self.kvps.is_null() {
            unsafe { api().release_key_value_pairs()(self.kvps) }
        }
    }
}
unsafe impl Send for KeyValuePairs {}
unsafe impl Sync for KeyValuePairs {}

/// Borrowed view over an engine-owned `OrtKeyValuePairs` — e.g. an `EpDevice`'s metadata
/// or a [`crate::HardwareDevice`]'s metadata (returned by `EpDevice_EpMetadata` /
/// `HardwareDevice_Metadata`). **Borrows** the handle for the lifetime of its parent; never
/// releases it (the parent owns it). For an owning container you build and release yourself, use
/// [`KeyValuePairs`].
pub struct KeyValuePairsView<'a> {
    kvps: *const sys::KeyValuePairsHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> KeyValuePairsView<'a> {
    /// Wrap an engine-owned (borrowed) key/value-pairs handle. A null handle is tolerated —
    /// [`Self::get`] then returns `None`.
    ///
    /// # Safety
    /// `kvps` must remain valid for the lifetime `'a` and must not be released by the caller.
    pub(crate) unsafe fn from_borrowed(kvps: *const sys::KeyValuePairsHandle) -> Self {
        Self {
            kvps,
            _life: PhantomData,
        }
    }

    /// Look up the value for `key` (`GetKeyValue`). Returns `None` if absent, or if this view
    /// wraps a null handle (no metadata present).
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        if self.kvps.is_null() {
            return Ok(None);
        }
        let ck = CString::new(key).map_err(|_| Error::new(-1, "key/value key contains a NUL"))?;
        let p = unsafe { api().get_key_value()(self.kvps, ck.as_ptr()) };
        if p.is_null() {
            Ok(None)
        } else {
            unsafe { crate::cstr_to_string(p, "key/value pair value") }.map(Some)
        }
    }
}
