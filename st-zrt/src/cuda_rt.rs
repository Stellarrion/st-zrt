//! Minimal CUDA-runtime binding (feature `cuda`) for the cuda-graph device-input refresh path.
//!
//! ORT 1.27's built-in CUDA EP is a legacy provider — it surfaces no `OrtEpDevice` and registers no
//! env-level `IDataTransfer`, so explicit host↔CUDA `CopyTensors` finds no transfer route. To refresh
//! a device-resident lane input before each cuda-graph replay we bypass ORT's copy and use the CUDA
//! runtime directly: an `Arc<CudaStream>` is retained by the CUDA configuration and handed to ORT via
//! [`crate::CudaConfig::with_user_stream`] (ORT then replays the captured graph on that
//! stream via `cudaGraphLaunch`), and the host→device refresh is enqueued on the **same** stream with
//! `cudaMemcpyAsync` — same-stream ordering makes it race-free with the replay.
//!
//! `libcudart` is resolved/rpath'd by `st-zrt-sys` for the `cuda` build and its link-search path
//! propagates here, so `#[link(name = "cudart")]` resolves these symbols.
#![cfg(feature = "cuda")]

use crate::{Error, Result};
use std::ffi::c_void;
use std::os::raw::c_char;

const CUDA_SUCCESS: i32 = 0;
/// `cudaMemcpyKind::cudaMemcpyHostToDevice`.
const MEMCPY_HOST_TO_DEVICE: u32 = 1;
/// `cudaMemcpyKind::cudaMemcpyDeviceToDevice`.
const MEMCPY_DEVICE_TO_DEVICE: u32 = 3;
/// `cudaStreamCreateWithFlags` flag: non-blocking stream.
const STREAM_NON_BLOCKING: u32 = 1;

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void, src: *const c_void, count: usize, kind: u32, stream: *mut c_void,
    ) -> i32;
    fn cudaGetErrorString(err: i32) -> *const c_char;
}

fn check(err: i32, what: &str) -> Result<()> {
    if err == CUDA_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let p = cudaGetErrorString(err);
            if p.is_null() {
                format!("zrt: CUDA {what} failed (error {err})")
            } else {
                let s = std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
                format!("zrt: CUDA {what} failed (error {err}): {s}")
            }
        };
        Err(Error::new(-1, msg))
    }
}

/// An owning `cudaStream_t` created on a specific device. Pass an `Arc<Self>` to
/// [`crate::CudaConfig::with_user_stream`] (so ORT replays the captured graph on this
/// stream) AND to the device-input lane's per-run refresh — the same stream guarantees the
/// host→device copy is ordered before the replay.
///
/// **Device binding:** a CUDA stream is bound to the device current at creation ([`Self::new`] calls
/// `cudaSetDevice`). Any *manual* enqueue on this stream (e.g. a raw `cudaMemcpyAsync`) from a thread
/// whose current device is not [`Self::device_id`] requires a `cudaSetDevice(device_id)` first, or the
/// call targets the wrong device. Safe ZRT paths activate and validate the stream's device before
/// direct CUDA work and retain the stream through native session teardown.
#[derive(Debug, PartialEq, Eq)]
pub struct CudaStream {
    raw: *mut c_void,
    device: i32,
}

impl CudaStream {
    /// Create a non-blocking CUDA stream on `device_id`.
    pub fn new(device_id: i32) -> Result<Self> {
        ensure_device_active(device_id)?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cudaStreamCreateWithFlags(&mut raw, STREAM_NON_BLOCKING) },
            "cudaStreamCreateWithFlags",
        )?;
        Ok(Self {
            raw,
            device: device_id,
        })
    }

    /// The device this stream was created on (the `device_id` passed to [`Self::new`]). A manual
    /// enqueue requires `cudaSetDevice(device_id)` first — see the type docs.
    pub fn device_id(&self) -> i32 {
        self.device
    }

    /// The raw `cudaStream_t` handle (for `user_compute_stream` + the device-input refresh).
    pub fn as_ptr(&self) -> *mut c_void {
        self.raw
    }

    /// Block the host until all work enqueued on this stream is done (`cudaStreamSynchronize`).
    /// Use during capture/warmup to make graph capture deterministic; keep it off the replay hot path.
    pub fn synchronize(&self) -> Result<()> {
        ensure_device_active(self.device)?;
        check(
            unsafe { cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // A stream is device-bound. Restore its device before best-effort sync + destroy;
            // errors cannot propagate from Drop.
            unsafe { cudaSetDevice(self.device) };
            unsafe { cudaStreamSynchronize(self.raw) };
            unsafe { cudaStreamDestroy(self.raw) };
        }
    }
}
// CUDA streams are safe to share across threads — the runtime serializes work submitted to a stream.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// Number of CUDA devices visible to the process.
pub fn device_count() -> Result<i32> {
    let mut count = 0;
    check(
        unsafe { cudaGetDeviceCount(&mut count) },
        "cudaGetDeviceCount",
    )?;
    Ok(count)
}

/// Block the host until all device work is done (`cudaDeviceSynchronize`). Use during capture/warmup
/// to flush device-wide cuda-graph capture state between lanes (capture is device-wide serial); keep
/// it off the replay hot path.
pub fn device_synchronize() -> Result<()> {
    check(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
}

/// Enqueue a host→device copy on `stream` (`cudaMemcpyAsync`, `cudaMemcpyHostToDevice`). When
/// submitted on the same stream as a captured-graph replay it is ordered before that replay. When
/// copying a whole pinned buffer, pass `PinnedBuffer::as_ptr()` with `len() * size_of::<T>()` bytes.
///
/// # Safety
///
/// - `src` points to at least `count` valid, initialized bytes that remain valid and unmutated until
///   `stream` completes the copy.
/// - `dst` is a valid device pointer with at least `count` bytes allocated, and is not freed before
///   the copy completes.
/// - `stream` is a valid `cudaStream_t`.
/// - Neither `src` nor `dst` is freed before `stream` completes the copy.
pub unsafe fn memcpy_async_h2d(
    dst: *mut c_void, src: *const c_void, count: usize, stream: *mut c_void,
) -> Result<()> {
    check(
        unsafe { cudaMemcpyAsync(dst, src, count, MEMCPY_HOST_TO_DEVICE, stream) },
        "cudaMemcpyAsync(host→device)",
    )
}

/// Enqueue a device→device copy on an owned stream.
///
/// # Safety
///
/// `src` and `dst` must be valid same-device CUDA allocations of at least `count` bytes, must not
/// overlap, and must remain alive until `stream` completes.
pub unsafe fn memcpy_async_d2d(
    dst: *mut c_void, src: *const c_void, count: usize, stream: &CudaStream,
) -> Result<()> {
    ensure_device_active(stream.device_id())?;
    check(
        unsafe { cudaMemcpyAsync(dst, src, count, MEMCPY_DEVICE_TO_DEVICE, stream.as_ptr()) },
        "cudaMemcpyAsync(device→device)",
    )
}

/// Page-locked ("pinned") host memory from `cudaMallocHost`. Pinned memory enables truly-async DMA
/// host→device copies — a pageable source can force the CUDA runtime to stage the copy through
/// pinned memory and synchronize, quietly ruining stream overlap. This is the right staging buffer
/// for a per-run `memcpy_async_h2d` refresh on a retained CUDA stream. Freed with
/// `cudaFreeHost` on drop.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> PinnedBuffer<T> {
    /// Allocate `len` zero-initialized elements of pinned host memory.
    pub fn zeros(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::new(-1, "zrt: pinned buffer byte length overflows usize"))?;
        let mut base: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cudaMallocHost(&mut base, bytes) },
            "cudaMallocHost",
        )?;
        if bytes > 0 {
            // Zero as raw bytes (host-accessible memory).
            unsafe { std::ptr::write_bytes(base as *mut u8, 0, bytes) };
        }
        Ok(Self {
            ptr: base as *mut T,
            len,
        })
    }

    /// Element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the pinned region as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Borrow the pinned region as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Raw host pointer — the pinned source for `memcpy_async_h2d`.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Best-effort free; errors cannot propagate from Drop.
            unsafe { cudaFreeHost(self.ptr as *mut c_void) };
        }
    }
}

// Pinned host memory is safe to share across threads.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

// ─── CUDA events ─────────────────────────────────────────────────────────────

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventQuery(event: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaStreamWaitEvent(stream: *mut c_void, event: *mut c_void, flags: u32) -> i32;
}

/// `cudaEventCreateWithFlags` flag: disable timing (lower overhead).
const EVENT_DISABLE_TIMING: u32 = 2;
/// CUDA's stable `cudaErrorNotReady` value.
const ERROR_NOT_READY: i32 = 600;

/// An owning, device-bound CUDA event for dependency-driven synchronization.
///
/// Events are created with timing disabled. This primitive does not by itself prove that an ORT
/// execution provider used a particular stream; callers must only use it with an explicitly
/// configured `user_compute_stream` whose ordering has been validated on the target provider.
pub struct CudaEvent {
    raw: *mut c_void,
    device: i32,
}

impl CudaEvent {
    /// Create a synchronization-only CUDA event on `device_id`.
    pub fn new(device_id: i32) -> Result<Self> {
        ensure_device_active(device_id)?;
        let mut raw: *mut c_void = std::ptr::null_mut();
        check(
            unsafe { cudaEventCreateWithFlags(&mut raw, EVENT_DISABLE_TIMING) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Self {
            raw,
            device: device_id,
        })
    }

    /// Record this event after all work currently queued on `stream`.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        if self.device != stream.device_id() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: CUDA event device {} does not match stream device {}",
                    self.device,
                    stream.device_id()
                ),
            ));
        }
        ensure_device_active(self.device)?;
        check(
            unsafe { cudaEventRecord(self.raw, stream.as_ptr()) },
            "cudaEventRecord",
        )
    }

    /// Record after work on a raw stream whose ownership and device identity were validated by a
    /// higher-level ZRT lane.
    pub(crate) fn record_raw_stream(&self, stream: *mut c_void) -> Result<()> {
        ensure_device_active(self.device)?;
        check(
            unsafe { cudaEventRecord(self.raw, stream) },
            "cudaEventRecord",
        )
    }

    /// Return `true` once all work preceding the event has completed.
    pub fn is_complete(&self) -> Result<bool> {
        ensure_device_active(self.device)?;
        self.query_raw()
    }

    fn query_raw(&self) -> Result<bool> {
        let rc = unsafe { cudaEventQuery(self.raw) };
        match rc {
            CUDA_SUCCESS => Ok(true),
            ERROR_NOT_READY => Ok(false),
            _ => {
                check(rc, "cudaEventQuery")?;
                unreachable!("CUDA success and not-ready handled above")
            },
        }
    }

    /// Block the host until all work preceding the event completes.
    pub fn synchronize(&self) -> Result<()> {
        ensure_device_active(self.device)?;
        check(
            unsafe { cudaEventSynchronize(self.raw) },
            "cudaEventSynchronize",
        )
    }

    /// Device on which this event was created.
    #[inline]
    pub fn device_id(&self) -> i32 {
        self.device
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Best effort: an event is device-bound, so restore its device before destruction.
            unsafe { cudaSetDevice(self.device) };
            unsafe { cudaEventDestroy(self.raw) };
        }
    }
}

// CUDA runtime event operations are thread-safe; device selection is verified on each operation.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

/// Read-only borrowed view of an in-flight CUDA completion event.
#[derive(Clone, Copy)]
pub struct CompletionEventRef<'a> {
    event: &'a CudaEvent,
}

impl<'a> From<&'a CudaEvent> for CompletionEventRef<'a> {
    fn from(event: &'a CudaEvent) -> Self {
        Self { event }
    }
}

impl<'a> CompletionEventRef<'a> {
    pub(crate) fn new(event: &'a CudaEvent) -> Self {
        Self { event }
    }

    /// Device to which the event belongs.
    pub fn device_id(self) -> i32 {
        self.event.device_id()
    }

    /// Query this event with ordinary per-event device validation.
    pub fn is_complete(self) -> Result<bool> {
        self.event.is_complete()
    }

    /// Block until this event completes.
    pub fn synchronize(self) -> Result<()> {
        self.event.synchronize()
    }
}

/// Device-bound nonblocking batch CUDA completion poller.
///
/// Every query validates the calling thread's CUDA device once, then queries all supplied events
/// without repeated `cudaGetDevice` calls. This primitive intentionally implements no spin, sleep,
/// deadline, or backoff policy.
#[derive(Debug, Clone, Copy)]
pub struct CudaCompletionPoller {
    device: i32,
}

impl CudaCompletionPoller {
    /// Construct a poller and validate `device_id` on the current thread.
    pub fn new(device_id: i32) -> Result<Self> {
        ensure_device_active(device_id)?;
        Ok(Self { device: device_id })
    }

    /// Bound CUDA device.
    pub fn device_id(self) -> i32 {
        self.device
    }

    pub(crate) fn validate_current_device(self) -> Result<()> {
        ensure_device_active(self.device)
    }

    pub(crate) fn query_validated(self, event: CompletionEventRef<'_>) -> Result<bool> {
        if event.device_id() != self.device {
            return Err(Error::new(
                -1,
                "zrt: CUDA completion batch contains an event from another device",
            ));
        }
        event.event.query_raw()
    }

    /// Perform one strictly nonblocking batch query into caller-owned storage.
    ///
    /// Device and length validation happens before any result is written. If an individual CUDA
    /// event query itself fails, earlier entries may already have been updated; callers must treat
    /// the result buffer as unspecified when this method returns `Err`.
    pub fn query(&self, events: &[CompletionEventRef<'_>], ready: &mut [bool]) -> Result<()> {
        if ready.len() < events.len() {
            return Err(Error::new(
                -1,
                "zrt: CUDA completion result buffer is smaller than the event batch",
            ));
        }
        if events.iter().any(|event| event.device_id() != self.device) {
            return Err(Error::new(
                -1,
                "zrt: CUDA completion batch contains an event from another device",
            ));
        }
        if events.is_empty() {
            return Ok(());
        }
        self.validate_current_device()?;
        for (event, status) in events.iter().copied().zip(ready) {
            *status = self.query_validated(event)?;
        }
        Ok(())
    }
}

/// Make `stream` wait for `event`. Work submitted to `stream` after this call cannot begin until
/// the event's preceding work completes. CUDA permits cross-device event waits; the current device
/// must still match the destination stream and is set here.
pub fn stream_wait_event(stream: &CudaStream, event: &CudaEvent) -> Result<()> {
    ensure_device_active(stream.device_id())?;
    check(
        unsafe { cudaStreamWaitEvent(stream.as_ptr(), event.raw, 0) },
        "cudaStreamWaitEvent",
    )
}

// ─── CUDA worker device selection ─────────────────────────────────────────────

/// Verify that `device_id` is current on this thread and select it when necessary.
///
/// CUDA device selection is thread-local and can be changed by ORT or by other CUDA users in the
/// process. Therefore this function calls `cudaGetDevice` rather than trusting a wrapper-local
/// cache. It avoids redundant `cudaSetDevice` calls while remaining correct after foreign changes.
pub(crate) fn ensure_device_active(device_id: i32) -> Result<()> {
    let mut current = -1;
    check(unsafe { cudaGetDevice(&mut current) }, "cudaGetDevice")?;
    if current != device_id {
        check(unsafe { cudaSetDevice(device_id) }, "cudaSetDevice")?;
    }
    Ok(())
}
