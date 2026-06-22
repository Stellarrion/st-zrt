//! IoBinding — the zero-copy *output* path. Bind outputs to caller-preallocated buffers
//! so ORT writes results directly into them (`BindOutput(name, value)`), eliminating the
//! per-run output allocation (the E2 anti-pattern fix). This is the output analog of
//! [`crate::TensorView`]: inputs are zero-copy views, and with IoBinding so are outputs.
use crate::allocator::Allocator;
use crate::element::TensorElement;
use crate::memory::MemoryInfo;
use crate::session::Session;
use crate::tensor::{AllocatedTensor, OwnedValue, RunInput, TensorBuffer, tensor_memory_info};
use crate::type_info::checked_element_count;
use crate::{Error, Result, api, check, sys};
use std::ffi::{CStr, CString, c_void};
use std::marker::PhantomData;
use std::ptr;

/// A caller-owned mutable buffer wrapped as an ORT value, for binding as a zero-copy
/// output via [`IoBinding`]. Constructed with `CreateTensorWithDataAsOrtValue`; the engine
/// holds the buffer pointer and, when this value is bound as an output, writes the computed
/// result directly into it. The buffer is NOT freed by the engine (`buf` remains the
/// caller's); only the `OrtValue` handle is released on drop.
pub struct OutputValue<'a> {
    value: *mut sys::ValueHandle,
    elem_type: sys::ElementType,
    count: usize,
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
        if !mem.is_host_accessible()? {
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
        let info = tensor_memory_info(self.value as *const sys::ValueHandle)?;
        if !info.is_host_accessible() {
            return Err(Error::new(
                -1,
                format!(
                    "output tensor memory is not host-accessible: {} device {} ({:?}/{:?})",
                    info.name, info.device_id, info.alloc_type, info.mem_type
                ),
            ));
        }
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut T, self.count, "output tensor data")?;
        // SAFETY: data points into the caller-owned buffer (ours), contiguous and aligned,
        // holding `self.count` elements of T. Lives for at least 'a (the buffer's lifetime).
        Ok(unsafe { std::slice::from_raw_parts(data as *const T, self.count) })
    }
}

impl Drop for OutputValue<'_> {
    fn drop(&mut self) {
        // Releases the OrtValue handle only; the backing buffer is the caller's.
        unsafe { api().release_value()(self.value) }
    }
}
unsafe impl Send for OutputValue<'_> {}
unsafe impl Sync for OutputValue<'_> {}

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
}

impl IoBinding {
    /// Create a binding owned by `sess`. Released on drop (`ReleaseIoBinding`, idx 135).
    pub fn new(sess: &Session) -> Result<Self> {
        let mut binding: *mut sys::IoBindingHandle = ptr::null_mut();
        check(unsafe { api().create_io_binding()(sess.as_ptr(), &mut binding) })?;
        let binding = crate::ensure_non_null(binding, "I/O binding")?;
        Ok(Self { binding })
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
        assert!(OutputValue::from_buffer(&mut buf, &[2, 2], &mem).is_ok());
    }

    #[test]
    fn output_value_rejects_cuda_device_memory() {
        let mem = MemoryInfo::cuda(0).unwrap();
        let mut buf = [0.0f32; 4];
        assert!(OutputValue::from_buffer(&mut buf, &[2, 2], &mem).is_err());
    }
}
