//! ORT global thread-pool configuration.

use crate::{api, check, sys, Error, Result};
use std::any::Any;
use std::ffi::{c_void, CString};
use std::ptr;

/// User-supplied thread manager for ORT global thread pools.
///
/// ORT calls [`Self::create`] when constructing pool workers and later calls [`Self::join`] while
/// destroying the pool. ZRT stores the manager inside [`crate::Environment`], so it stays alive
/// until after `ReleaseEnv`.
pub trait ThreadManager: Send + Sync + 'static {
    /// A handle returned by [`Self::create`] and later passed to [`Self::join`].
    type Thread: Send + 'static;

    /// Spawn one ORT worker thread. The new thread must call `work`.
    fn create(&self, work: impl FnOnce() + Send + 'static) -> Result<Self::Thread>;

    /// Join a worker previously returned by [`Self::create`].
    fn join(thread: Self::Thread) -> Result<()>;
}

/// Global ORT thread-pool options (`OrtThreadingOptions`).
///
/// Pass this to [`crate::Environment::new_with_global_thread_pools`]. Sessions created under that
/// environment use the global pool by default; call
/// [`crate::SessionOptions::use_per_session_threads`] to opt out for a specific session.
pub struct ThreadingOptions {
    handle: *mut sys::ThreadingOptionsHandle,
    manager: Option<Box<dyn Any + Send + Sync>>,
}

impl ThreadingOptions {
    /// Create empty ORT threading options.
    pub fn new() -> Result<Self> {
        let mut handle: *mut sys::ThreadingOptionsHandle = ptr::null_mut();
        check(unsafe { api().create_threading_options()(&mut handle) })?;
        let handle = crate::ensure_non_null(handle, "threading options")?;
        Ok(Self {
            handle,
            manager: None,
        })
    }

    /// Set the global intra-op thread count.
    pub fn with_intra_threads(self, n: i32) -> Result<Self> {
        check(unsafe { api().set_global_intra_op_num_threads()(self.handle, n) })?;
        Ok(self)
    }

    /// Set the global inter-op thread count.
    pub fn with_inter_threads(self, n: i32) -> Result<Self> {
        check(unsafe { api().set_global_inter_op_num_threads()(self.handle, n) })?;
        Ok(self)
    }

    /// Configure whether ORT worker threads may spin while waiting for work.
    pub fn with_spin_control(self, allow_spinning: bool) -> Result<Self> {
        check(unsafe { api().set_global_spin_control()(self.handle, i32::from(allow_spinning)) })?;
        Ok(self)
    }

    /// Disable worker-thread spinning.
    pub fn disable_spinning(self) -> Result<Self> {
        self.with_spin_control(false)
    }

    /// Pin global intra-op worker threads with ORT's affinity string format.
    ///
    /// ORT applies this only to the global intra-op thread pool. Pass a semicolon-separated
    /// CPU-list string such as `"1;2;3"` or `"1,2;3,4"` depending on the desired worker/core
    /// mapping.
    pub fn with_intra_thread_affinity(self, affinity: &str) -> Result<Self> {
        let affinity = CString::new(affinity)
            .map_err(|_| Error::local("thread affinity contains a NUL byte"))?;
        check(unsafe {
            api().set_global_intra_op_thread_affinity()(self.handle, affinity.as_ptr())
        })?;
        Ok(self)
    }

    /// Enable denormal-as-zero behavior for global worker threads.
    pub fn with_denormal_as_zero(self) -> Result<Self> {
        check(unsafe { api().set_global_denormal_as_zero()(self.handle) })?;
        Ok(self)
    }

    /// Install a custom thread manager for the global ORT thread pool.
    pub fn with_thread_manager<T>(mut self, manager: T) -> Result<Self>
    where
        T: ThreadManager,
    {
        if self.manager.is_some() {
            return Err(Error::new(
                -1,
                "zrt: global thread manager already configured",
            ));
        }

        let mut manager = Box::new(manager);
        let manager_ptr = (&mut *manager) as *mut T as *mut c_void;
        check(unsafe {
            api().set_global_custom_thread_creation_options()(self.handle, manager_ptr)
        })?;
        check(unsafe {
            api().set_global_custom_create_thread_fn()(self.handle, Some(thread_create::<T>))
        })?;
        check(unsafe {
            api().set_global_custom_join_thread_fn()(self.handle, Some(thread_join::<T>))
        })?;
        self.manager = Some(manager);
        Ok(self)
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::ThreadingOptionsHandle {
        self.handle as *const sys::ThreadingOptionsHandle
    }
}

impl Drop for ThreadingOptions {
    fn drop(&mut self) {
        unsafe { api().release_threading_options()(self.handle) }
    }
}

unsafe impl Send for ThreadingOptions {}
unsafe impl Sync for ThreadingOptions {}

struct SendablePtr(*mut c_void);
unsafe impl Send for SendablePtr {}

unsafe extern "C" fn thread_create<T>(
    options: *mut c_void, worker: sys::ThreadWorkerFn, worker_param: *mut c_void,
) -> sys::CustomThreadHandle
where
    T: ThreadManager,
{
    let worker_param = SendablePtr(worker_param);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let manager = unsafe { &*(options as *const T) };
        manager.create(move || {
            let worker_param = worker_param;
            unsafe { worker(worker_param.0) };
        })
    }));

    match result {
        Ok(Ok(thread)) => Box::into_raw(Box::new(thread)) as sys::CustomThreadHandle,
        Ok(Err(_)) | Err(_) => ptr::null(),
    }
}

unsafe extern "C" fn thread_join<T>(handle: sys::CustomThreadHandle)
where
    T: ThreadManager,
{
    if handle.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let thread = unsafe { Box::from_raw(handle as *mut T::Thread) };
        T::join(*thread)
    }));
    let _ = result;
}
