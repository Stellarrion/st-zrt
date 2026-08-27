//! Session initializers backed by caller-owned tensors.
use crate::element::TensorElement;
use crate::tensor::{RunInput, TensorBuffer};
use crate::{Error, Result, api, check, ensure_non_null, sys};
use std::ffi::{CString, c_char};
use std::ptr;

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

/// A description of a single initializer stored in an **external file** (`OrtExternalInitializerInfo`,
/// ORT 1.23+): a file path, a byte offset within it, and a byte length. ORT memory-maps the
/// described region instead of copying the weights into the model — important for multi-GB
/// models whose weights live on disk.
///
/// Built with [`ExternalInitializerInfo::new`]; the three getters round-trip the construction
/// values. Released with `ReleaseExternalInitializerInfo` on drop.
pub struct ExternalInitializerInfo {
    info: *mut sys::ExternalInitializerInfoHandle,
}

impl ExternalInitializerInfo {
    /// Describe an external initializer at `file_offset` bytes into `file_path`, spanning
    /// `byte_size` bytes (`CreateExternalInitializerInfo`). ORT copies the path string.
    pub fn new(file_path: &str, file_offset: i64, byte_size: usize) -> Result<Self> {
        let cpath = CString::new(file_path)
            .map_err(|_| Error::new(-1, "external initializer file path contains a NUL byte"))?;
        let mut info: *mut sys::ExternalInitializerInfoHandle = ptr::null_mut();
        check(unsafe {
            api().create_external_initializer_info()(
                cpath.as_ptr(),
                file_offset,
                byte_size,
                &mut info,
            )
        })?;
        let info = ensure_non_null(info, "external initializer info")?;
        Ok(Self { info })
    }

    /// The file path this info was built with (`ExternalInitializerInfo_GetFilePath`).
    pub fn file_path(&self) -> Result<String> {
        let p = unsafe {
            api().external_initializer_info__get_file_path()(
                self.info as *const sys::ExternalInitializerInfoHandle,
            )
        };
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "external initializer file path") }
        }
    }

    /// The byte offset within the file (`ExternalInitializerInfo_GetFileOffset`).
    #[inline]
    pub fn file_offset(&self) -> i64 {
        unsafe {
            api().external_initializer_info__get_file_offset()(
                self.info as *const sys::ExternalInitializerInfoHandle,
            )
        }
    }

    /// The byte length of the initializer data (`ExternalInitializerInfo_GetByteSize`).
    #[inline]
    pub fn byte_size(&self) -> usize {
        unsafe {
            api().external_initializer_info__get_byte_size()(
                self.info as *const sys::ExternalInitializerInfoHandle,
            )
        }
    }
}

impl Drop for ExternalInitializerInfo {
    fn drop(&mut self) {
        if !self.info.is_null() {
            unsafe { api().release_external_initializer_info()(self.info) }
        }
    }
}
unsafe impl Send for ExternalInitializerInfo {}
unsafe impl Sync for ExternalInitializerInfo {}
