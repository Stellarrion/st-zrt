//! `ArenaCfg` — arena allocator tuning (the E1 knobs: max memory, extend strategy, chunk
//! sizes). Built via `CreateArenaCfg` (idx 156) or the newer key/value `CreateArenaCfgV2`
//! (idx 164); released on drop (`ReleaseArenaCfg`, idx 157). Consumed by
//! [`crate::Environment::register_allocator`].
use crate::{Error, Result, api, check, sys};
use std::ffi::CString;
use std::ptr;

/// How the arena grows a chunk when more memory is needed (ORT convention).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ArenaExtendStrategy {
    /// Round up to the next power of two (default; amortizes growth).
    NextPowerOfTwo = 0,
    /// Allocate exactly what is requested (lower fragmentation, more calls).
    SameAsRequested = 1,
}

pub struct ArenaCfg {
    pub(crate) cfg: *mut sys::ArenaCfgHandle,
}

impl ArenaCfg {
    /// Build an arena config (`CreateArenaCfg`, idx 156):
    /// - `max_mem` — arena size ceiling in bytes (use `usize::MAX` for unlimited).
    /// - `strategy` — chunk growth strategy.
    /// - `initial_chunk_size_bytes` — size of the first chunk.
    /// - `max_dead_bytes_per_chunk` — dead bytes allowed within a chunk before a new one is
    ///   allocated.
    pub fn new(
        max_mem: usize, strategy: ArenaExtendStrategy, initial_chunk_size_bytes: i32,
        max_dead_bytes_per_chunk: i32,
    ) -> Result<Self> {
        let mut cfg: *mut sys::ArenaCfgHandle = ptr::null_mut();
        check(unsafe {
            api().create_arena_cfg()(
                max_mem,
                strategy as core::ffi::c_int,
                initial_chunk_size_bytes,
                max_dead_bytes_per_chunk,
                &mut cfg,
            )
        })?;
        let cfg = crate::ensure_non_null(cfg, "arena config")?;
        Ok(Self { cfg })
    }

    /// Build an arena config from arbitrary key/value entries (`CreateArenaCfgV2`, idx 164) —
    /// the escape hatch for EP-specific arena keys (see ORT's `arena_extend_strategy`,
    /// `initial_gpu_chunk_size_in_bytes`, …).
    pub fn with_entries(entries: &[(&str, usize)]) -> Result<Self> {
        let keys: Vec<CString> = entries
            .iter()
            .map(|(k, _)| CString::new(*k).map_err(|_| Error::new(-1, "arena key contains a NUL")))
            .collect::<Result<_>>()?;
        let values: Vec<usize> = entries.iter().map(|(_, v)| *v).collect();
        let key_ptrs: Vec<*const core::ffi::c_char> = keys.iter().map(|c| c.as_ptr()).collect();
        let mut cfg: *mut sys::ArenaCfgHandle = ptr::null_mut();
        check(unsafe {
            api().create_arena_cfg_v2()(key_ptrs.as_ptr(), values.as_ptr(), entries.len(), &mut cfg)
        })?;
        let cfg = crate::ensure_non_null(cfg, "arena config")?;
        Ok(Self { cfg })
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::ArenaCfgHandle {
        self.cfg as *const sys::ArenaCfgHandle
    }
}

impl Drop for ArenaCfg {
    fn drop(&mut self) {
        unsafe { api().release_arena_cfg()(self.cfg) }
    }
}
unsafe impl Send for ArenaCfg {}
unsafe impl Sync for ArenaCfg {}
