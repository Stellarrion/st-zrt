//! Engine error type and `Result` alias.
use std::ffi::CString;

/// A failure reported by the engine, with its ORT error code and message.
#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub message: String,
}

impl Error {
    /// Construct a st-zrt-local error (not from an ORT status).
    pub(crate) fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Construct a user/local error that did not originate from an ORT status.
    pub fn local(message: impl Into<String>) -> Self {
        Self::new(-1, message)
    }

    /// The ORT error code, if this error came from an ORT status (`code` in the known
    /// range). `None` for st-zrt-local errors (negative codes) or unrecognized codes.
    pub fn ort_code(&self) -> Option<crate::sys::OrtErrorCode> {
        crate::sys::OrtErrorCode::from_c_int(self.code)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match crate::sys::OrtErrorCode::from_c_int(self.code) {
            Some(c) => write!(f, "zrt error [{:?} ({})]: {}", c, self.code, self.message),
            None => write!(f, "zrt error [{}]: {}", self.code, self.message),
        }
    }
}
impl std::error::Error for Error {}

impl From<(i32, CString)> for Error {
    fn from((code, msg): (i32, CString)) -> Self {
        Self {
            code,
            message: msg.to_string_lossy().into_owned(),
        }
    }
}

/// A NUL byte in a string being marshaled to a C string.
impl From<std::ffi::NulError> for Error {
    fn from(_: std::ffi::NulError) -> Self {
        Self::new(-1, "string contains a NUL byte")
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_code_maps_known_and_local() {
        assert_eq!(
            Error::new(2, "bad arg").ort_code(),
            Some(crate::sys::OrtErrorCode::InvalidArgument)
        );
        assert_eq!(
            Error::new(14, "missing").ort_code(),
            Some(crate::sys::OrtErrorCode::NotFound)
        );
        // A st-zrt-local error (negative code) is not an ORT code.
        assert_eq!(Error::new(-1, "local").ort_code(), None);
        // An unknown (future-ORT) code does not map.
        assert_eq!(Error::new(999, "future").ort_code(), None);
        // Display names a known ORT code.
        assert!(
            Error::new(2, "bad arg")
                .to_string()
                .contains("InvalidArgument")
        );
    }
}
