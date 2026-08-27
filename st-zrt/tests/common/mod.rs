//! Shared helpers for the cuda-graph test binaries (`cuda_ep` and future CUDA suites).
//!
//! CUDA-graph **capture** (`cudaStreamBeginCapture` / `cudaStreamEndCapture` /
//! `cudaGraphInstantiate`) is a device-wide serial operation: it fails with CUDA errors 900/901
//! ("operation not permitted when stream is capturing") when another stream is concurrently
//! capturing. Concurrent **replay** (`cudaGraphLaunch`) is safe in production. Tests serialize their complete CUDA
//! lifetimes because ordinary setup/teardown work on another test thread must not overlap capture.
//!
//! Two scopes must be covered:
//!
//! - **Within one test binary** — cargo runs a binary's tests on a thread pool, so two capturing
//!   tests in the *same* binary can race. A process-local `Mutex` serializes them.
//! - **Across test binaries** — cargo may run multiple CUDA suites
//!   concurrently (or a CI matrix may invoke them as parallel jobs). Each binary links this module
//!   independently, so the in-process `Mutex` does not reach across processes. An exclusive
//!   `flock(2)` on a fixed temp file serializes capture process-wide.
//!
//! `flock` is keyed per *open file description*: two `open()` calls in the same process get
//!   independent locks, so the in-process `Mutex` is still required (it is the fast path; the
//!   `flock` only ever contends across processes).

use std::fs::{File, OpenOptions};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

unsafe extern "C" {
    fn flock(fd: RawFd, operation: i32) -> i32;
}

/// `LOCK_EX` (exclusive) for `flock(2)` — `<sys/file.h>` on Linux/BSD.
const LOCK_EX: i32 = 2;

static CAPTURE_MUTEX: Mutex<()> = Mutex::new(());

/// Guard returned by [`cuda_graph_capture_lock`]; holds the in-process mutex and the cross-process
/// `flock` until dropped.
pub struct CudaGraphCaptureGuard {
    _in_process: MutexGuard<'static, ()>,
    _flock: File,
}

fn lock_path() -> PathBuf {
    std::env::temp_dir().join("st-zrt-cuda-graph-capture.lock")
}

/// Take the device-wide CUDA test lock for the full duration of a CUDA test (in-process mutex +
/// cross-process `flock`). All tests in a binary that contains graph-capture coverage acquire it:
/// ordinary CUDA work must not overlap another thread's capture. Production replay is not serialized.
pub fn cuda_graph_capture_lock() -> CudaGraphCaptureGuard {
    let in_process = CAPTURE_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = lock_path();
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("zrt: open cuda-graph capture lock {path:?}: {e}"));
    // Blocking exclusive flock — serializes capture across processes (replay stays concurrent).
    let rc = unsafe { flock(file.as_raw_fd(), LOCK_EX) };
    assert_eq!(rc, 0, "zrt: flock(LOCK_EX) on {path:?} failed");
    CudaGraphCaptureGuard {
        _in_process: in_process,
        _flock: file,
    }
}
