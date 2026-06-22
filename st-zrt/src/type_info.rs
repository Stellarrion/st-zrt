//! Shape/type introspection: `TensorTypeAndShapeInfo` over the owning
//! `GetTensorTypeAndShape` path (idx 65 — returns an OWNING handle, one release).
use crate::{Error, Result, api, check, sys};
use std::ffi::{CStr, c_char};
use std::ptr;

/// Owning wrapper over `OrtTensorTypeAndShapeInfo` obtained from a value via
/// `GetTensorTypeAndShape`. Released on drop (`ReleaseTensorTypeAndShapeInfo`, idx 99).
pub struct TensorTypeAndShapeInfo {
    info: *mut sys::TensorTypeAndShapeInfoHandle,
}

impl TensorTypeAndShapeInfo {
    /// Wrap an owning handle returned by `GetTensorTypeAndShape` (idx 65). The wrapper
    /// assumes ownership and will release it on drop.
    ///
    /// # Safety
    /// `info` must be a freshly-allocated owning handle from `GetTensorTypeAndShape`.
    pub(crate) unsafe fn from_owning(info: *mut sys::TensorTypeAndShapeInfoHandle) -> Self {
        Self { info }
    }

    /// Build a fresh, empty type+shape info (`CreateTensorTypeAndShapeInfo`). Fill it with
    /// [`Self::set_element_type`] + [`Self::set_dimensions`], then hand it to a shape-inference
    /// context (`ShapeInferContext::set_output_type_shape` with the `custom-ops` feature) or inspect it. Owning —
    /// released on drop.
    pub fn new() -> Result<Self> {
        let mut info: *mut sys::TensorTypeAndShapeInfoHandle = ptr::null_mut();
        check(unsafe { api().create_tensor_type_and_shape_info()(&mut info) })?;
        let info = crate::ensure_non_null(info, "tensor type and shape info")?;
        // SAFETY: CreateTensorTypeAndShapeInfo allocates an owning handle.
        Ok(unsafe { Self::from_owning(info) })
    }

    /// Set the element type (`SetTensorElementType`).
    pub fn set_element_type(&mut self, ty: sys::ElementType) -> Result<()> {
        check(unsafe { api().set_tensor_element_type()(self.info, ty) })
    }

    /// Set the concrete dimensions (`SetDimensions`).
    pub fn set_dimensions(&mut self, dims: &[i64]) -> Result<()> {
        check(unsafe { api().set_dimensions()(self.info, dims.as_ptr(), dims.len()) })
    }

    /// The raw owning handle (`pub(crate)` — shape-inference and model-editor wrappers pass it
    /// to ORT APIs that borrow it).
    #[cfg(any(feature = "custom-ops", feature = "model-editor"))]
    pub(crate) fn as_ptr(&self) -> *const sys::TensorTypeAndShapeInfoHandle {
        self.info as *const sys::TensorTypeAndShapeInfoHandle
    }

    /// Element type of the tensor.
    pub fn element_type(&self) -> Result<sys::ElementType> {
        let mut et = sys::ElementType::Undefined;
        check(unsafe {
            api().get_tensor_element_type()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
                &mut et,
            )
        })?;
        Ok(et)
    }

    /// Number of dimensions (rank).
    pub fn rank(&self) -> Result<usize> {
        let mut n: usize = 0;
        check(unsafe {
            api().get_dimensions_count()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
                &mut n,
            )
        })?;
        Ok(n)
    }

    /// Total element count (product of dimensions).
    ///
    /// This computes from dimensions in Rust instead of calling ORT
    /// `GetTensorShapeElementCount`, because ORT may report a SafeInt overflow for static
    /// symbolic shapes such as `[-1, 1000]`. If any dimension is dynamic/unknown, this returns
    /// a controlled ZRT error.
    pub fn element_count(&self) -> Result<usize> {
        checked_element_count(&self.dims()?)
    }

    /// Concrete dimensions, e.g. `[1, 1, 28, 28]`.
    pub fn dims(&self) -> Result<Vec<i64>> {
        let n = self.rank()?;
        let mut out = vec![0i64; n];
        check(unsafe {
            api().get_dimensions()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
                out.as_mut_ptr(),
                n,
            )
        })?;
        Ok(out)
    }

    /// Symbolic (named) dimensions: `Some("batch")` where the model declared a symbolic
    /// dim, `None` where it is concrete. Length equals `rank()`. The strings are borrowed
    /// from the engine-owned handle for the lifetime of `self`.
    pub fn symbolic_dims(&self) -> Result<Vec<Option<&str>>> {
        let n = self.rank()?;
        let mut ptrs: Vec<*const c_char> = vec![ptr::null(); n];
        check(unsafe {
            api().get_symbolic_dimensions()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
                ptrs.as_mut_ptr(),
                n,
            )
        })?;
        ptrs.iter()
            .map(|&p| {
                if p.is_null() {
                    Ok(None)
                } else {
                    // SAFETY: the engine guarantees a NUL-terminated UTF-8-ish C string for the
                    // lifetime of the handle. We only borrow it; we do not free it.
                    unsafe { CStr::from_ptr(p) }
                        .to_str()
                        .map(Some)
                        .map_err(|_| {
                            Error::new(-1, "zrt: symbolic dimension name is not valid UTF-8")
                        })
                }
            })
            .collect()
    }
}

pub(crate) fn checked_element_count(dims: &[i64]) -> Result<usize> {
    let mut count = 1usize;
    for &dim in dims {
        if dim < 0 {
            return Err(Error::new(
                -1,
                format!("tensor shape contains a dynamic/unknown dimension ({dim})"),
            ));
        }
        let dim = usize::try_from(dim)
            .map_err(|_| Error::new(-1, "tensor dimension does not fit usize"))?;
        count = count
            .checked_mul(dim)
            .ok_or_else(|| Error::new(-1, "tensor shape element count overflows usize"))?;
    }
    Ok(count)
}

impl Drop for TensorTypeAndShapeInfo {
    fn drop(&mut self) {
        unsafe { api().release_tensor_type_and_shape_info()(self.info) }
    }
}

/// Introspect a tensor value's full type+shape (owning path). The value MUST be a tensor;
/// for map/sequence values use `OwnedValue::value_type` instead.
pub(crate) fn tensor_type_and_shape(
    value: *const sys::ValueHandle,
) -> Result<TensorTypeAndShapeInfo> {
    let mut info: *mut sys::TensorTypeAndShapeInfoHandle = ptr::null_mut();
    check(unsafe { api().get_tensor_type_and_shape()(value, &mut info) })?;
    let info = crate::ensure_non_null(info, "tensor type and shape info")?;
    Ok(unsafe { TensorTypeAndShapeInfo::from_owning(info) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Engine-backed round trip of the builder: create → set element type + dims → read back.
    /// No model needed; exercises CreateTensorTypeAndShapeInfo + SetTensorElementType +
    /// SetDimensions (+ the read accessors), all released on drop.
    #[test]
    fn type_and_shape_info_builder_round_trip() {
        let mut info = TensorTypeAndShapeInfo::new().expect("new");
        info.set_element_type(sys::ElementType::Float)
            .expect("set elem type");
        info.set_dimensions(&[2, 3]).expect("set dims");
        assert_eq!(info.element_type().unwrap(), sys::ElementType::Float);
        assert_eq!(info.dims().unwrap(), vec![2, 3]);
        assert_eq!(info.rank().unwrap(), 2);
        assert_eq!(info.element_count().unwrap(), 6);
        eprintln!("type_and_shape_info_builder_round_trip: create + set + read OK");
    }

    #[test]
    fn type_info_accepts_newer_quantized_metadata_element_types() {
        for ty in [
            sys::ElementType::Float8E4M3FN,
            sys::ElementType::Float8E5M2,
            sys::ElementType::Uint4,
            sys::ElementType::Int4,
            sys::ElementType::Float4E2M1,
        ] {
            let mut info = TensorTypeAndShapeInfo::new().expect("new");
            info.set_element_type(ty).expect("set elem type");
            assert_eq!(info.element_type().unwrap(), ty);
        }
    }

    #[test]
    fn checked_element_count_rejects_dynamic_and_overflow() {
        assert_eq!(checked_element_count(&[1, 1000]).unwrap(), 1000);
        assert!(checked_element_count(&[-1, 1000]).is_err());
        assert!(checked_element_count(&[i64::MAX, 3]).is_err());
    }
}
