//! serde helpers for the config types (feature `serde`).
//!
//! Several config fields store [`CString`](std::ffi::CString) — session/EP config-entry
//! keys+values, EP-option device/cache paths. serde has no built-in `CString` support (a
//! `CString` may hold non-UTF-8 bytes, which `String` cannot represent), so these helpers
//! (de)serialize them as `String`, erroring at deserialize time if a value carries a NUL
//! byte — mirroring the builder methods, which take `&str` and reject NULs.
//!
//! [`sys::GraphOptimizationLevel`] lives in `st-zrt-sys` (no serde dep), so it is
//! (de)serialized as its `#[repr(i32)]` discriminant.
//!
//! The macro-generated EP types (`CudaOptions`, …) hold live runtime handles, not config —
//! they are intentionally **not** serializable. The serializable EP config is
//! [`crate::ep::EpConfig`] (the queued key/value path) plus the flat-struct EP options
//! (`MigraphxOptions`, `OpenvinoOptions`).
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::ffi::CString;

/// (De)serialize `Option<CString>` as `Option<String>`.
#[allow(dead_code)]
pub mod opt_cstr {
    use super::*;
    pub fn serialize<S: Serializer>(c: &Option<CString>, s: S) -> Result<S::Ok, S::Error> {
        match c {
            Some(v) => s.serialize_some(v.to_str().map_err(serde::ser::Error::custom)?),
            None => s.serialize_none(),
        }
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<CString>, D::Error> {
        match Option::<String>::deserialize(d)? {
            Some(v) => CString::new(v).map(Some).map_err(de::Error::custom),
            None => Ok(None),
        }
    }
}

/// (De)serialize `Vec<(CString, CString)>` as `Vec<(String, String)>` — the shape of the
/// session/EP config-entry queues.
pub mod kv_pairs {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[(CString, CString)], s: S) -> Result<S::Ok, S::Error> {
        let mapped: Vec<(&str, &str)> = v
            .iter()
            .map(|(k, val)| {
                Ok((
                    k.to_str().map_err(serde::ser::Error::custom)?,
                    val.to_str().map_err(serde::ser::Error::custom)?,
                ))
            })
            .collect::<Result<_, S::Error>>()?;
        mapped.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<(CString, CString)>, D::Error> {
        <Vec<(String, String)> as Deserialize>::deserialize(d)?
            .into_iter()
            .map(|(k, v)| CString::new(k).and_then(|k| CString::new(v).map(|v| (k, v))))
            .collect::<Result<_, _>>()
            .map_err(de::Error::custom)
    }
}

/// (De)serialize `Vec<(CString, i64)>` as `Vec<(String, i64)>` for free-dimension
/// override queues.
pub mod kv_i64_pairs {
    use super::*;
    pub fn serialize<S: Serializer>(v: &[(CString, i64)], s: S) -> Result<S::Ok, S::Error> {
        let mapped: Vec<(&str, i64)> = v
            .iter()
            .map(|(k, val)| Ok((k.to_str().map_err(serde::ser::Error::custom)?, *val)))
            .collect::<Result<_, S::Error>>()?;
        mapped.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<(CString, i64)>, D::Error> {
        <Vec<(String, i64)> as Deserialize>::deserialize(d)?
            .into_iter()
            .map(|(k, v)| CString::new(k).map(|k| (k, v)))
            .collect::<Result<_, _>>()
            .map_err(de::Error::custom)
    }
}

/// (De)serialize [`sys::GraphOptimizationLevel`] as its `i32` discriminant (the ORT levels:
/// `DisableAll=0`, `Basic=1`, `Extended=2`, `Layout=3`, `All=99`). Unknown values error.
pub mod graph_opt {
    use super::*;
    use crate::sys::GraphOptimizationLevel;
    pub fn serialize<S: Serializer>(v: &GraphOptimizationLevel, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_i32(*v as i32)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<GraphOptimizationLevel, D::Error> {
        Ok(match i32::deserialize(d)? {
            0 => GraphOptimizationLevel::DisableAll,
            1 => GraphOptimizationLevel::Basic,
            2 => GraphOptimizationLevel::Extended,
            3 => GraphOptimizationLevel::Layout,
            99 => GraphOptimizationLevel::All,
            i => {
                return Err(de::Error::custom(format!(
                    "unknown GraphOptimizationLevel discriminant: {i}"
                )));
            },
        })
    }
}
