//! Session initializers backed by caller-owned tensors.
use crate::element::TensorElement;
use crate::tensor::{RunInput, TensorBuffer};
use crate::{sys, Error, Result};
use std::ffi::{c_char, CString};

trait InitializerValue: Send + Sync {
    fn as_value_ptr(&self) -> *const sys::ValueHandle;
}

impl<T> InitializerValue for TensorBuffer<T>
where
    T: TensorElement + Send + Sync,
{
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        RunInput::as_value_ptr(self)
    }
}

/// An initializer tensor owned by ZRT and kept alive by the session using it.
///
/// Use this when a model initializer/weight should come from external caller memory
/// instead of being copied from the model file. Construction is zero-copy with respect to
/// the tensor backing buffer: ORT sees the `TensorBuffer`'s storage directly.
pub struct OwnedInitializer {
    name: CString,
    value: Box<dyn InitializerValue>,
}

impl OwnedInitializer {
    pub fn tensor<T>(name: &str, value: TensorBuffer<T>) -> Result<Self>
    where
        T: TensorElement + Send + Sync + 'static,
    {
        Ok(Self {
            name: CString::new(name)
                .map_err(|_| Error::new(-1, "initializer name contains a NUL"))?,
            value: Box::new(value),
        })
    }

    #[inline]
    pub fn name(&self) -> &str {
        self.name
            .to_str()
            .expect("initializer names are constructed from Rust UTF-8")
    }

    #[inline]
    pub(crate) fn name_ptr(&self) -> *const c_char {
        self.name.as_ptr()
    }

    #[inline]
    pub(crate) fn value_ptr(&self) -> *const sys::ValueHandle {
        self.value.as_value_ptr()
    }
}

unsafe impl Send for OwnedInitializer {}
unsafe impl Sync for OwnedInitializer {}
