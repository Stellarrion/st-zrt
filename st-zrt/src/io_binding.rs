//! IoBinding — the zero-copy *output* path. Bind outputs to caller-preallocated buffers
//! so ORT writes results directly into them (`BindOutput(name, value)`), eliminating the
//! per-run output allocation (the E2 anti-pattern fix). This is the output analog of
//! [`crate::TensorView`]: inputs are zero-copy views, and with IoBinding so are outputs.
use crate::allocator::Allocator;
use crate::element::TensorElement;
use crate::memory::{MemoryClass, MemoryInfo};
use crate::session::{Session, SessionInner};
use crate::tensor::{AllocatedTensor, OwnedValue, RunInput, TensorBuffer};
use crate::type_info::checked_element_count;
use crate::{Error, Result, api, check, sys};
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

/// A caller-owned mutable buffer wrapped as an ORT value, for binding as a zero-copy
/// output via [`IoBinding`]. Constructed with `CreateTensorWithDataAsOrtValue`; the engine
/// holds the buffer pointer and, when this value is bound as an output, writes the computed
/// result directly into it. The buffer is NOT freed by the engine (`buf` remains the
/// caller's); only the `OrtValue` handle is released on drop.
pub struct OutputValue<'a> {
    value: *mut sys::ValueHandle,
    elem_type: sys::ElementType,
    count: usize,
    data: *mut c_void,
    memory_class: MemoryClass,
    _life: PhantomData<&'a mut [u8]>,
}

impl<'a> OutputValue<'a> {
    /// Wrap `buf` as a tensor value of `shape`. The engine will write the bound output's
    /// result into `buf` in place. `buf.len()` must equal the product of `shape`; the shape
    /// must match the model's actual output shape.
    pub fn from_buffer<T: TensorElement>(
        buf: &'a mut [T], shape: &[i64], mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_shape_len(shape, buf.len())?;
        if !mem.is_host_accessible() {
            let info = mem.snapshot()?;
            return Err(Error::new(
                -1,
                format!(
                    "OutputValue wraps a Rust slice and requires host-accessible memory, got {} device {} ({:?}/{:?})",
                    info.name, info.device_id, info.alloc_type, info.mem_type
                ),
            ));
        }
        let bytes = std::mem::size_of_val(buf);
        let data = buf.as_mut_ptr() as *mut c_void;
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_with_data_as_ort_value()(
                mem.info as *const sys::MemoryInfoHandle,
                buf.as_mut_ptr() as *mut c_void,
                bytes,
                shape.as_ptr(),
                shape.len(),
                T::ELEM,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "output value")?;
        Ok(Self {
            value,
            elem_type: T::ELEM,
            count: buf.len(),
            data,
            memory_class: mem.class(),
            _life: PhantomData,
        })
    }

    #[inline]
    pub(crate) fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }

    /// Zero-copy read of the result buffer as a typed slice (`GetTensorMutableData`).
    /// After [`Session::run_binding`], holds the computed output. Tied to `&self` so the
    /// borrow is released with the `OutputValue`.
    pub fn as_slice<T: TensorElement>(&self) -> Result<&[T]> {
        if self.elem_type as i32 != T::ELEM as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: OutputValue::as_slice<{}> on a {:?} buffer",
                    std::any::type_name::<T>(),
                    self.elem_type
                ),
            ));
        }
        if !self.memory_class.is_host_accessible() {
            return Err(Error::new(
                -1,
                format!(
                    "output tensor memory class is not host-accessible: {:?}",
                    self.memory_class
                ),
            ));
        }
        let data = crate::slice_data_ptr(self.data as *mut T, self.count, "output tensor data")?;
        // SAFETY: `data` is the caller buffer pointer cached at construction. `OutputValue`'s
        // exclusive lifetime marker keeps that buffer stable and borrowed for the value's life.
        Ok(unsafe { std::slice::from_raw_parts(data as *const T, self.count) })
    }

    /// Zero-copy mutable access to the cached caller-owned output buffer.
    ///
    /// Call this only when no run using the value is in flight. The exclusive `&mut self` prevents
    /// simultaneous safe reads or writes through this wrapper.
    pub fn as_mut_slice<T: TensorElement>(&mut self) -> Result<&mut [T]> {
        if self.elem_type as i32 != T::ELEM as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: OutputValue::as_mut_slice<{}> on a {:?} buffer",
                    std::any::type_name::<T>(),
                    self.elem_type
                ),
            ));
        }
        if !self.memory_class.is_host_accessible() {
            return Err(Error::new(
                -1,
                format!(
                    "output tensor memory class is not host-accessible: {:?}",
                    self.memory_class
                ),
            ));
        }
        let data = crate::slice_data_ptr(self.data as *mut T, self.count, "output tensor data")?;
        Ok(unsafe { std::slice::from_raw_parts_mut(data, self.count) })
    }
}

impl std::fmt::Debug for OutputValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputValue")
            .field("value", &self.value)
            .field("elem_type", &self.elem_type)
            .field("count", &self.count)
            .field("data", &self.data)
            .field("memory_class", &self.memory_class)
            .finish()
    }
}

impl Drop for OutputValue<'_> {
    fn drop(&mut self) {
        // Releases the OrtValue handle only; the backing buffer is the caller's.
        unsafe { api().release_value()(self.value) }
    }
}
unsafe impl Send for OutputValue<'_> {}

fn validate_shape_len(shape: &[i64], len: usize) -> Result<()> {
    let expected = checked_element_count(shape)?;
    if expected != len {
        return Err(Error::new(
            -1,
            format!("output tensor shape expects {expected} elements, got {len}"),
        ));
    }
    Ok(())
}

/// An IoBinding: a name→value map for inputs and outputs, bound once and reused across
/// [`Session::run_binding`] calls. Bind-once-mutate-in-place is the intended pattern: build
/// the binding once, mutate the input/output buffers between runs, and never rebind.
pub struct IoBinding {
    binding: *mut sys::IoBindingHandle,
    // ORT creates a binding for one specific session. Keep that session alive through
    // `ReleaseIoBinding` and reject attempts to run the binding through another session.
    session: Arc<SessionInner>,
}

impl std::fmt::Debug for IoBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IoBinding")
            .field("binding", &self.binding)
            .finish()
    }
}

impl IoBinding {
    /// Create a binding owned by `sess`. Released on drop (`ReleaseIoBinding`, idx 135).
    pub fn new(sess: &Session) -> Result<Self> {
        let mut binding: *mut sys::IoBindingHandle = ptr::null_mut();
        check(unsafe { api().create_io_binding()(sess.as_ptr(), &mut binding) })?;
        let binding = crate::ensure_non_null(binding, "I/O binding")?;
        Ok(Self {
            binding,
            session: sess.share_inner(),
        })
    }

    #[inline]
    pub(crate) fn belongs_to(&self, session: &Session) -> bool {
        session.shares_inner(&self.session)
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const sys::IoBindingHandle {
        self.binding as *const sys::IoBindingHandle
    }

    /// Bind the input `name` to `input` (`BindInput`, idx 136).
    pub fn bind_input(&mut self, name: &str, input: &dyn RunInput) -> Result<()> {
        let cname = CString::new(name).map_err(|_| Error::new(-1, "input name contains a NUL"))?;
        self.bind_input_cstr(&cname, input)
    }

    pub(crate) fn bind_input_cstr(&mut self, name: &CStr, input: &dyn RunInput) -> Result<()> {
        check(unsafe { api().bind_input()(self.binding, name.as_ptr(), input.as_value_ptr()) })
    }

    /// Bind the output `name` to a caller-owned buffer (zero-copy: ORT writes the result
    /// directly into the buffer). `value` must be a tensor of the model's output type/shape.
    /// (`BindOutput`, idx 137.)
    pub fn bind_output(&mut self, name: &str, value: &OutputValue<'_>) -> Result<()> {
        let cname = CString::new(name).map_err(|_| Error::new(-1, "output name contains a NUL"))?;
        check(unsafe { api().bind_output()(self.binding, cname.as_ptr(), value.as_value_ptr()) })
    }

    /// Bind the output `name` to a reusable owned tensor buffer. This is the lane-local
    /// variant of [`Self::bind_output`]: the buffer owns its backing `Vec<T>` and can be
    /// mutated/read between runs without rebuilding the binding.
    pub fn bind_output_buffer<T: TensorElement>(
        &mut self, name: &str, value: &TensorBuffer<T>,
    ) -> Result<()> {
        let cname = CString::new(name).map_err(|_| Error::new(-1, "output name contains a NUL"))?;
        check(unsafe { api().bind_output()(self.binding, cname.as_ptr(), value.as_value_ptr()) })
    }

    /// Bind the output `name` to an ORT allocator-owned tensor. This supports both CPU and
    /// provider/device allocations such as CUDA.
    pub fn bind_output_allocated<T: TensorElement>(
        &mut self, name: &str, value: &AllocatedTensor<T>,
    ) -> Result<()> {
        let cname = CString::new(name).map_err(|_| Error::new(-1, "output name contains a NUL"))?;
        check(unsafe { api().bind_output()(self.binding, cname.as_ptr(), value.as_value_ptr()) })
    }

    /// Bind the output `name` to a memory location, letting ORT allocate the result tensor.
    /// Use this for dynamic-shape outputs; retrieve the values after the run with
    /// [`Self::output_values`]. (`BindOutputToDevice`, idx 138.)
    pub fn bind_output_device(&mut self, name: &str, mem: &MemoryInfo) -> Result<()> {
        let cname = CString::new(name).map_err(|_| Error::new(-1, "output name contains a NUL"))?;
        check(unsafe {
            api().bind_output_to_device()(
                self.binding,
                cname.as_ptr(),
                mem.info as *const sys::MemoryInfoHandle,
            )
        })
    }

    /// Synchronize bound outputs (`SynchronizeBoundOutputs`) — a no-op on the CPU EP, needed
    /// for async/device EPs so the result is visible before reading the buffers.
    pub fn synchronize_outputs(&self) -> Result<()> {
        check(unsafe { api().synchronize_bound_outputs()(self.binding) })
    }

    /// Synchronize bound inputs (`SynchronizeBoundInputs`).
    pub fn synchronize_inputs(&self) -> Result<()> {
        check(unsafe { api().synchronize_bound_inputs()(self.binding) })
    }

    /// Drop all input bindings.
    pub fn clear_inputs(&mut self) {
        unsafe { api().clear_bound_inputs()(self.binding) }
    }
    /// Drop all output bindings.
    pub fn clear_outputs(&mut self) {
        unsafe { api().clear_bound_outputs()(self.binding) }
    }

    /// Retrieve the output values from a device-bound run (`GetBoundOutputValues`, idx 140).
    /// The values are engine-allocated and owned by the caller (released on drop). The array
    /// holding the handles is freed; the individual values are not.
    pub fn output_values(&self) -> Result<Vec<OwnedValue>> {
        let alloc = Allocator::get_default()?;
        let mut out: *mut *mut sys::ValueHandle = ptr::null_mut();
        let mut count: usize = 0;
        check(unsafe {
            api().get_bound_output_values()(
                self.binding as *const sys::IoBindingHandle,
                alloc.alloc,
                &mut out,
                &mut count,
            )
        })?;
        // `out` is an engine-allocated array of `count` owning value handles.
        let handles: &[*mut sys::ValueHandle] = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(out, count) }
        };
        let values = OwnedValue::collect_from_raw(handles);
        // Free the array buffer (one allocation); the values keep their own handles.
        let free = if out.is_null() {
            Ok(())
        } else {
            unsafe { alloc.free(out as *mut c_void) }
        };
        match (values, free) {
            (Ok(values), Ok(())) => Ok(values),
            (Err(err), _) => Err(err),
            (Ok(_), Err(err)) => Err(err),
        }
    }

    /// The names of the outputs currently bound to this binding (`GetBoundOutputNames`, idx 139).
    /// Returns the names in bind order. The engine allocates the string buffer and a parallel
    /// lengths array via the default allocator; both are freed here (the names are copied into
    /// Rust-owned `String`s).
    pub fn output_names(&self) -> Result<Vec<String>> {
        let alloc = Allocator::get_default()?;
        let mut buffer: *mut core::ffi::c_char = ptr::null_mut();
        let mut lengths: *mut usize = ptr::null_mut();
        let mut count: usize = 0;
        let status = check(unsafe {
            api().get_bound_output_names()(
                self.binding as *const sys::IoBindingHandle,
                alloc.alloc,
                &mut buffer,
                &mut lengths,
                &mut count,
            )
        });
        let result = collect_bound_names(status, buffer, lengths, count);
        // Free both engine-allocated arrays (best-effort); errors here are not actionable.
        if !buffer.is_null() {
            let _ = unsafe { alloc.free(buffer as *mut c_void) };
        }
        if !lengths.is_null() {
            let _ = unsafe { alloc.free(lengths as *mut c_void) };
        }
        result
    }
}

/// Copy the engine-allocated bound-output names into owned `String`s. The engine returns a single
/// contiguous, **non-NUL-terminated** UTF-8 buffer plus a parallel `lengths` array (one length per
/// name); name `i` spans `lengths[i]` bytes starting at the running offset. `status` is consumed
/// so `?` short-circuits on an ORT error; the caller frees `buffer`/`lengths` regardless.
fn collect_bound_names(
    status: Result<()>, buffer: *mut core::ffi::c_char, lengths: *mut usize, count: usize,
) -> Result<Vec<String>> {
    status?;
    let mut names = Vec::with_capacity(count);
    if count != 0 && !buffer.is_null() && !lengths.is_null() {
        let lens = unsafe { std::slice::from_raw_parts(lengths, count) };
        let mut offset = 0usize;
        for &n in lens {
            let name = if n == 0 {
                String::new()
            } else {
                let bytes =
                    unsafe { std::slice::from_raw_parts(buffer.add(offset) as *const u8, n) };
                std::str::from_utf8(bytes)
                    .map(str::to_owned)
                    .map_err(|_| Error::new(-1, "zrt: bound output name is not valid UTF-8"))?
            };
            names.push(name);
            offset += n;
        }
    }
    Ok(names)
}

impl Drop for IoBinding {
    fn drop(&mut self) {
        unsafe { api().release_io_binding()(self.binding) }
    }
}
unsafe impl Send for IoBinding {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_value_rejects_dynamic_and_mismatched_shapes() {
        let mem = MemoryInfo::cpu().unwrap();
        let mut buf = [0.0f32; 4];
        assert!(OutputValue::from_buffer(&mut buf, &[-1, 4], &mem).is_err());
        assert!(OutputValue::from_buffer(&mut buf, &[5], &mem).is_err());
        let mut value = OutputValue::from_buffer(&mut buf, &[2, 2], &mem).unwrap();
        value
            .as_mut_slice::<f32>()
            .unwrap()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(value.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn output_value_rejects_cuda_device_memory() {
        let mem = MemoryInfo::cuda(0).unwrap();
        let mut buf = [0.0f32; 4];
        assert!(OutputValue::from_buffer(&mut buf, &[2, 2], &mem).is_err());
    }
}
