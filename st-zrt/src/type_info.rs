//! Shape/type introspection: `TensorTypeAndShapeInfo` over the owning
//! `GetTensorTypeAndShape` path (idx 65 — returns an OWNING handle, one release).
use crate::{Error, Result, api, check, sys};
use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
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

    /// Attach symbolic dimension names to the dims (`SetSymbolicDimensions`, idx 271). `dim_params`
    /// is one name per dimension (e.g. `["batch", "sequence"]`); the count must match the rank set
    /// via [`Self::set_dimensions`]. Meaningful on an owning TSI built with [`Self::new`].
    pub fn set_symbolic_dimensions(&mut self, dim_params: &[&str]) -> Result<()> {
        let cnames: Vec<std::ffi::CString> = dim_params
            .iter()
            .map(|s| {
                std::ffi::CString::new(*s)
                    .map_err(|_| Error::new(-1, "symbolic dimension name contains a NUL"))
            })
            .collect::<Result<_>>()?;
        let ptrs: Vec<*const c_char> = cnames.iter().map(|c| c.as_ptr()).collect();
        check(unsafe { api().set_symbolic_dimensions()(self.info, ptrs.as_ptr(), ptrs.len()) })
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

    /// Whether the shape is known (`TensorTypeAndShape_HasShape`).
    pub fn has_shape(&self) -> bool {
        unsafe {
            api().tensor_type_and_shape__has_shape()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
            )
        }
    }

    /// The element count as computed by the engine (`GetTensorShapeElementCount`). Distinct from
    /// [`Self::element_count`], which multiplies dims in Rust to avoid ORT's SafeInt overflow for
    /// symbolic shapes. Prefer [`Self::element_count`] unless you specifically want the engine's value.
    pub fn shape_element_count(&self) -> Result<usize> {
        let mut n: usize = 0;
        check(unsafe {
            api().get_tensor_shape_element_count()(
                self.info as *const sys::TensorTypeAndShapeInfoHandle,
                &mut n,
            )
        })?;
        Ok(n)
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

/// Owning wrapper over a generic `OrtTypeInfo`. Obtained from session type introspection (e.g.
/// overridable initializers); released with `ReleaseTypeInfo` on drop. Named `RuntimeTypeInfo` to
/// avoid clashing with the model-editor construction type `model_editor::TypeInfo`.
pub struct RuntimeTypeInfo {
    info: *mut sys::TypeInfoHandle,
}

impl RuntimeTypeInfo {
    /// Wrap a freshly-allocated owning `OrtTypeInfo` handle.
    ///
    /// # Safety
    /// `info` must be an owning handle the caller transferred (e.g. from
    /// `SessionGetOverridableInitializerTypeInfo`).
    pub(crate) unsafe fn from_owning(info: *mut sys::TypeInfoHandle) -> Self {
        Self { info }
    }

    /// The ONNX value kind (`GetOnnxTypeFromTypeInfo`) — Tensor, Sequence, Map, or Optional.
    pub fn onnx_type(&self) -> Result<sys::OnnxType> {
        let mut ty = sys::OnnxType::Unknown;
        check(unsafe {
            api().get_onnx_type_from_type_info()(self.info as *const sys::TypeInfoHandle, &mut ty)
        })?;
        Ok(ty)
    }

    /// The type denotation string (`GetDenotationFromTypeInfo`). Empty if none.
    pub fn denotation(&self) -> Result<String> {
        let mut p: *const c_char = ptr::null();
        let mut len: usize = 0;
        check(unsafe {
            api().get_denotation_from_type_info()(
                self.info as *const sys::TypeInfoHandle,
                &mut p as *mut _ as *const *const c_char,
                &mut len,
            )
        })?;
        if p.is_null() || len == 0 {
            return Ok(String::new());
        }
        let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, len) };
        std::str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| Error::new(-1, "zrt: type-info denotation is not valid UTF-8"))
    }

    /// Cast to a borrowed tensor type+shape view (`CastTypeInfoToTensorInfo`). `None` if this type
    /// is not a tensor. The view borrows this [`RuntimeTypeInfo`] — do not release it.
    pub fn cast_to_tensor(&self) -> Result<Option<TensorTypeAndShapeInfoView<'_>>> {
        let mut out: *const sys::TensorTypeAndShapeInfoHandle = ptr::null();
        check(unsafe {
            api().cast_type_info_to_tensor_info()(
                self.info as *const sys::TypeInfoHandle,
                &mut out as *mut _ as *const *const sys::TensorTypeAndShapeInfoHandle,
            )
        })?;
        if out.is_null() {
            Ok(None)
        } else {
            Ok(Some(TensorTypeAndShapeInfoView {
                info: out,
                _life: PhantomData,
            }))
        }
    }

    /// Cast to a borrowed map type-info view (`CastTypeInfoToMapTypeInfo`). `None` if not a map.
    pub fn cast_to_map(&self) -> Result<Option<MapTypeInfo<'_>>> {
        let mut out: *const sys::MapTypeInfoHandle = ptr::null();
        check(unsafe {
            api().cast_type_info_to_map_type_info()(
                self.info as *const sys::TypeInfoHandle,
                &mut out as *mut _ as *const *const sys::MapTypeInfoHandle,
            )
        })?;
        if out.is_null() {
            Ok(None)
        } else {
            Ok(Some(MapTypeInfo {
                info: out,
                _life: PhantomData,
            }))
        }
    }

    /// Cast to a borrowed sequence type-info view (`CastTypeInfoToSequenceTypeInfo`). `None` if not
    /// a sequence.
    pub fn cast_to_sequence(&self) -> Result<Option<SequenceTypeInfo<'_>>> {
        let mut out: *const sys::SequenceTypeInfoHandle = ptr::null();
        check(unsafe {
            api().cast_type_info_to_sequence_type_info()(
                self.info as *const sys::TypeInfoHandle,
                &mut out as *mut _ as *const *const sys::SequenceTypeInfoHandle,
            )
        })?;
        if out.is_null() {
            Ok(None)
        } else {
            Ok(Some(SequenceTypeInfo {
                info: out,
                _life: PhantomData,
            }))
        }
    }

    /// Cast to a borrowed optional type-info view (`CastTypeInfoToOptionalTypeInfo`). `None` if not
    /// an optional.
    pub fn cast_to_optional(&self) -> Result<Option<OptionalTypeInfo<'_>>> {
        let mut out: *const sys::OptionalTypeInfoHandle = ptr::null();
        check(unsafe {
            api().cast_type_info_to_optional_type_info()(
                self.info as *const sys::TypeInfoHandle,
                &mut out as *mut _ as *const *const sys::OptionalTypeInfoHandle,
            )
        })?;
        if out.is_null() {
            Ok(None)
        } else {
            Ok(Some(OptionalTypeInfo {
                info: out,
                _life: PhantomData,
            }))
        }
    }
}

/// Borrowed tensor type+shape view from a TypeInfo cast. Not owning — borrowed from the parent
/// [`RuntimeTypeInfo`]; never released.
pub struct TensorTypeAndShapeInfoView<'a> {
    info: *const sys::TensorTypeAndShapeInfoHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> TensorTypeAndShapeInfoView<'a> {
    /// Element type (`GetTensorElementType`).
    pub fn element_type(&self) -> Result<sys::ElementType> {
        let mut et = sys::ElementType::Undefined;
        check(unsafe { api().get_tensor_element_type()(self.info, &mut et) })?;
        Ok(et)
    }
    /// Whether the shape is known (`TensorTypeAndShape_HasShape`).
    pub fn has_shape(&self) -> bool {
        unsafe { api().tensor_type_and_shape__has_shape()(self.info) }
    }
    /// Engine-computed element count (`GetTensorShapeElementCount`).
    pub fn shape_element_count(&self) -> Result<usize> {
        let mut n: usize = 0;
        check(unsafe { api().get_tensor_shape_element_count()(self.info, &mut n) })?;
        Ok(n)
    }
}

/// Borrowed map type-info view (`OrtMapTypeInfo`). Borrowed from the parent [`RuntimeTypeInfo`].
pub struct MapTypeInfo<'a> {
    info: *const sys::MapTypeInfoHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> MapTypeInfo<'a> {
    /// The map's key element type (`GetMapKeyType`). Keys are restricted to scalar types.
    pub fn key_type(&self) -> Result<sys::ElementType> {
        let mut et = sys::ElementType::Undefined;
        check(unsafe { api().get_map_key_type()(self.info, &mut et) })?;
        Ok(et)
    }
    /// Owning type-info for the map's value type (`GetMapValueType`). Released on drop.
    pub fn value_type(&self) -> Result<RuntimeTypeInfo> {
        let mut ti: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { api().get_map_value_type()(self.info, &mut ti) })?;
        let ti = crate::ensure_non_null(ti, "map value type info")?;
        // SAFETY: `ti` is a freshly-allocated owning handle from ORT.
        Ok(unsafe { RuntimeTypeInfo::from_owning(ti) })
    }
}

/// Borrowed sequence type-info view (`OrtSequenceTypeInfo`). Borrowed from the parent
/// [`RuntimeTypeInfo`].
pub struct SequenceTypeInfo<'a> {
    info: *const sys::SequenceTypeInfoHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> SequenceTypeInfo<'a> {
    /// Owning type-info for the sequence's element type (`GetSequenceElementType`). Released on drop.
    pub fn element_type(&self) -> Result<RuntimeTypeInfo> {
        let mut ti: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { api().get_sequence_element_type()(self.info, &mut ti) })?;
        let ti = crate::ensure_non_null(ti, "sequence element type info")?;
        // SAFETY: `ti` is a freshly-allocated owning handle from ORT.
        Ok(unsafe { RuntimeTypeInfo::from_owning(ti) })
    }
}

/// Borrowed optional type-info view (`OrtOptionalTypeInfo`). Borrowed from the parent
/// [`RuntimeTypeInfo`].
pub struct OptionalTypeInfo<'a> {
    info: *const sys::OptionalTypeInfoHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> OptionalTypeInfo<'a> {
    /// Owning type-info for the optional's contained type (`GetOptionalContainedTypeInfo`).
    /// Released on drop.
    pub fn contained(&self) -> Result<RuntimeTypeInfo> {
        let mut ti: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { api().get_optional_contained_type_info()(self.info, &mut ti) })?;
        let ti = crate::ensure_non_null(ti, "optional contained type info")?;
        // SAFETY: `ti` is a freshly-allocated owning handle from ORT.
        Ok(unsafe { RuntimeTypeInfo::from_owning(ti) })
    }
}

impl Drop for RuntimeTypeInfo {
    fn drop(&mut self) {
        if !self.info.is_null() {
            unsafe { api().release_type_info()(self.info) }
        }
    }
}
unsafe impl Send for RuntimeTypeInfo {}
unsafe impl Sync for RuntimeTypeInfo {}

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
        // Symbolic dim names (SetSymbolicDimensions) — one per rank.
        info.set_symbolic_dimensions(&["batch", "feature"])
            .expect("set symbolic dimensions");
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
