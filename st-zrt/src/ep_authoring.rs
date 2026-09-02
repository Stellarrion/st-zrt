//! EP-authoring helpers (feature `model-editor`) — safe wrappers over the `OrtEpApi` helper table.
//!
//! These are the building blocks for implementing a custom execution provider in Rust: declare
//! which operators your EP implements ([`KernelDefBuilder`]→[`KernelDef`]) and register them
//! ([`KernelRegistry`]). This module wraps the **independently-creatable** helpers — ones whose
//! handles do not require an EP-factory callback to obtain.
//!
//! The factory-tied surface (`CreateEpDevice`, `EpGraphSupportInfo` partitioning,
//! `ProfilingEventsContainer`) lives in the EP-factory callback flow: its handles only exist
//! inside a hand-written `CreateEpFactories` (the `cdylib` entry ORT dlopens), so it is reached
//! via the [`crate::ep_api`] gateway there. [`EpAuthor`] provides the high-level instance contract,
//! and [`custom_ep!`](crate::custom_ep) emits the `cdylib` entry points for an EP factory.
//!
//! Reached via [`crate::model_editor::ep_api`] (`GetEpApi`) + [`crate::model_editor::require_sub_api_fn`].
use crate::model_editor::{ep_api, require_sub_api_fn};
use crate::{Error, Result, check, ensure_non_null, sys};
use std::borrow::Cow;
use std::ffi::{CString, c_char, c_int};
use std::marker::PhantomData;
use std::ptr;

#[cfg(test)]
use std::ffi::CStr;

/// Borrow the live `OrtEpApi` helper table, or an error if the engine did not populate it.
fn ep() -> Result<&'static sys::EpApi> {
    ep_api().ok_or_else(|| Error::new(-1, "OrtEpApi unavailable"))
}

/// Build a `CString`, rejecting interior NUL (which the C side would silently truncate).
fn cstr(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::new(-1, format!("zrt: {what} contains a NUL byte")))
}

/// Resolve an [`sys::ElementType`] to its static `DataTypeHandle` (`GetTensorDataType`). The handle
/// is engine-owned and not released.
fn tensor_data_type(elem: sys::ElementType) -> Result<*const sys::DataTypeHandle> {
    let f = require_sub_api_fn(ep()?.GetTensorDataType, "EpApi", "GetTensorDataType")?;
    let mut p: *const sys::DataTypeHandle = ptr::null();
    check(unsafe { f(elem, &mut p as *mut _ as *const *const sys::DataTypeHandle) })?;
    if p.is_null() {
        Err(Error::new(-1, "zrt: data type pointer is null"))
    } else {
        Ok(p)
    }
}

/// Owning `OrtKernelDefBuilder` — incrementally describes one operator implementation your EP
/// provides, finalized into a [`KernelDef`] via [`Self::build`].
///
/// Builder methods return `&mut Self` for chaining. The builder handle is released on drop
/// (`ReleaseKernelDefBuilder`); [`Self::build`] does not consume it (ORT's `Build` leaves the
/// builder releasable), so a builder may be dropped at any scope end after building.
pub struct KernelDefBuilder {
    raw: *mut sys::KernelDefBuilderHandle,
}

impl KernelDefBuilder {
    /// Create a fresh kernel-def builder (`CreateKernelDefBuilder`).
    pub fn new() -> Result<Self> {
        let create = require_sub_api_fn(
            ep()?.CreateKernelDefBuilder,
            "EpApi",
            "CreateKernelDefBuilder",
        )?;
        let mut raw: *mut sys::KernelDefBuilderHandle = ptr::null_mut();
        check(unsafe { create(&mut raw) })?;
        Ok(Self {
            raw: ensure_non_null(raw, "kernel def builder")?,
        })
    }

    /// Set the operator type this kernel implements, e.g. `"Conv"` (`KernelDefBuilder_SetOperatorType`).
    pub fn set_operator_type(&mut self, op_type: &str) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetOperatorType,
            "EpApi",
            "KernelDefBuilder_SetOperatorType",
        )?;
        let s = cstr(op_type, "operator type")?;
        check(unsafe { f(self.raw, s.as_ptr()) })?;
        Ok(self)
    }

    /// Set the operator domain, e.g. `""` (ai.onnx) or `"com.acme"` (`KernelDefBuilder_SetDomain`).
    pub fn set_domain(&mut self, domain: &str) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetDomain,
            "EpApi",
            "KernelDefBuilder_SetDomain",
        )?;
        let s = cstr(domain, "domain")?;
        check(unsafe { f(self.raw, s.as_ptr()) })?;
        Ok(self)
    }

    /// Set the opset version range this kernel supports (`KernelDefBuilder_SetSinceVersion`).
    pub fn set_since_version(&mut self, start: i32, end: i32) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetSinceVersion,
            "EpApi",
            "KernelDefBuilder_SetSinceVersion",
        )?;
        check(unsafe { f(self.raw, start as c_int, end as c_int) })?;
        Ok(self)
    }

    /// Set the execution-provider name this kernel belongs to
    /// (`KernelDefBuilder_SetExecutionProvider`).
    pub fn set_execution_provider(&mut self, ep_name: &str) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetExecutionProvider,
            "EpApi",
            "KernelDefBuilder_SetExecutionProvider",
        )?;
        let s = cstr(ep_name, "execution provider name")?;
        check(unsafe { f(self.raw, s.as_ptr()) })?;
        Ok(self)
    }

    /// Set input `input_index`'s memory type (`KernelDefBuilder_SetInputMemType`).
    pub fn set_input_mem_type(
        &mut self, input_index: usize, mem_type: sys::MemType,
    ) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetInputMemType,
            "EpApi",
            "KernelDefBuilder_SetInputMemType",
        )?;
        check(unsafe { f(self.raw, input_index, mem_type) })?;
        Ok(self)
    }

    /// Set output `output_index`'s memory type (`KernelDefBuilder_SetOutputMemType`).
    pub fn set_output_mem_type(
        &mut self, output_index: usize, mem_type: sys::MemType,
    ) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_SetOutputMemType,
            "EpApi",
            "KernelDefBuilder_SetOutputMemType",
        )?;
        check(unsafe { f(self.raw, output_index, mem_type) })?;
        Ok(self)
    }

    /// Add a type constraint: input/output argument `arg_name` may bind any of `elem_types`
    /// (`KernelDefBuilder_AddTypeConstraint`).
    pub fn add_type_constraint(
        &mut self, arg_name: &str, elem_types: &[sys::ElementType],
    ) -> Result<&mut Self> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_AddTypeConstraint,
            "EpApi",
            "KernelDefBuilder_AddTypeConstraint",
        )?;
        let name = cstr(arg_name, "type constraint arg name")?;
        let handles: Vec<*const sys::DataTypeHandle> = elem_types
            .iter()
            .map(|&e| tensor_data_type(e))
            .collect::<Result<_>>()?;
        check(unsafe { f(self.raw, name.as_ptr(), handles.as_ptr(), handles.len()) })?;
        Ok(self)
    }

    /// Finalize the builder into an owning [`KernelDef`] (`KernelDefBuilder_Build`).
    pub fn build(&mut self) -> Result<KernelDef> {
        let f = require_sub_api_fn(
            ep()?.KernelDefBuilder_Build,
            "EpApi",
            "KernelDefBuilder_Build",
        )?;
        let mut raw: *mut sys::KernelDefHandle = ptr::null_mut();
        check(unsafe { f(self.raw, &mut raw) })?;
        Ok(KernelDef {
            raw: ensure_non_null(raw, "kernel def")?,
        })
    }
}

impl Drop for KernelDefBuilder {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseKernelDefBuilder) {
            unsafe { release(self.raw) }
        }
    }
}
// The builder handle is touched by ORT during EP setup; the wrapper only releases it on drop.
unsafe impl Send for KernelDefBuilder {}

/// Owning `OrtKernelDef` — the finalized description of one EP kernel, produced by
/// [`KernelDefBuilder::build`]. Released on drop (`ReleaseKernelDef`).
pub struct KernelDef {
    raw: *mut sys::KernelDefHandle,
}

impl KernelDef {
    /// The operator type (`KernelDef_GetOperatorType`). Empty if the engine stores none.
    pub fn operator_type(&self) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.KernelDef_GetOperatorType,
            "EpApi",
            "KernelDef_GetOperatorType",
        )?;
        let p: *const c_char = unsafe { f(self.raw) };
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "kernel def operator type") }
        }
    }

    /// The operator domain (`KernelDef_GetDomain`).
    pub fn domain(&self) -> Result<String> {
        let f = require_sub_api_fn(ep()?.KernelDef_GetDomain, "EpApi", "KernelDef_GetDomain")?;
        let p: *const c_char = unsafe { f(self.raw) };
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "kernel def domain") }
        }
    }

    /// The execution-provider name (`KernelDef_GetExecutionProvider`).
    pub fn execution_provider(&self) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.KernelDef_GetExecutionProvider,
            "EpApi",
            "KernelDef_GetExecutionProvider",
        )?;
        let p: *const c_char = unsafe { f(self.raw) };
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "kernel def execution provider") }
        }
    }

    /// The supported opset version range `(start, end)` (`KernelDef_GetSinceVersion`).
    pub fn since_version(&self) -> Result<(i32, i32)> {
        let f = require_sub_api_fn(
            ep()?.KernelDef_GetSinceVersion,
            "EpApi",
            "KernelDef_GetSinceVersion",
        )?;
        let mut start: c_int = 0;
        let mut end: c_int = 0;
        check(unsafe { f(self.raw, &mut start, &mut end) })?;
        Ok((start as i32, end as i32))
    }

    /// Input `input_index`'s memory type (`KernelDef_GetInputMemType`).
    pub fn input_mem_type(&self, input_index: usize) -> Result<sys::MemType> {
        let f = require_sub_api_fn(
            ep()?.KernelDef_GetInputMemType,
            "EpApi",
            "KernelDef_GetInputMemType",
        )?;
        let mut mt: sys::MemType = sys::MemType::Default;
        check(unsafe { f(self.raw, input_index, &mut mt) })?;
        Ok(mt)
    }

    /// Output `output_index`'s memory type (`KernelDef_GetOutputMemType`).
    pub fn output_mem_type(&self, output_index: usize) -> Result<sys::MemType> {
        let f = require_sub_api_fn(
            ep()?.KernelDef_GetOutputMemType,
            "EpApi",
            "KernelDef_GetOutputMemType",
        )?;
        let mut mt: sys::MemType = sys::MemType::Default;
        check(unsafe { f(self.raw, output_index, &mut mt) })?;
        Ok(mt)
    }
}

impl Drop for KernelDef {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseKernelDef) {
            unsafe { release(self.raw) }
        }
    }
}
unsafe impl Send for KernelDef {}

/// Owning `OrtOpSchema` — the contract for one operator, fetched via [`OpSchema::get`]. Released
/// on drop (`ReleaseOpSchema`).
pub struct OpSchema {
    raw: *mut sys::OpSchemaHandle,
}

impl OpSchema {
    /// Fetch a built-in or registered op schema (`GetOpSchema`). `domain` `""` is `ai.onnx`;
    /// `max_inclusive_version` selects the highest schema at or below that opset (pass a high value
    /// such as `21` for the latest registered schema).
    pub fn get(name: &str, max_inclusive_version: i32, domain: &str) -> Result<Self> {
        let f = require_sub_api_fn(ep()?.GetOpSchema, "EpApi", "GetOpSchema")?;
        let n = cstr(name, "op name")?;
        let d = cstr(domain, "domain")?;
        let mut raw: *mut sys::OpSchemaHandle = ptr::null_mut();
        check(unsafe {
            f(
                n.as_ptr(),
                max_inclusive_version as c_int,
                d.as_ptr(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: ensure_non_null(raw, "op schema")?,
        })
    }

    /// The opset version this schema was introduced at (`OpSchema_GetSinceVersion`).
    pub fn since_version(&self) -> Result<i32> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetSinceVersion,
            "EpApi",
            "OpSchema_GetSinceVersion",
        )?;
        let mut out: c_int = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out as i32)
    }

    /// Number of declared inputs (`OpSchema_GetNumInputs`).
    pub fn num_inputs(&self) -> Result<usize> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetNumInputs,
            "EpApi",
            "OpSchema_GetNumInputs",
        )?;
        let mut out: usize = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// Input `index`'s name (`OpSchema_GetInputName`). Empty if the schema stores none.
    pub fn input_name(&self, index: usize) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetInputName,
            "EpApi",
            "OpSchema_GetInputName",
        )?;
        let mut p: *const c_char = ptr::null();
        check(unsafe { f(self.raw, index, &mut p as *mut _ as *const *const c_char) })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "op schema input name") }
        }
    }

    /// Number of declared outputs (`OpSchema_GetNumOutputs`).
    pub fn num_outputs(&self) -> Result<usize> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetNumOutputs,
            "EpApi",
            "OpSchema_GetNumOutputs",
        )?;
        let mut out: usize = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// Output `index`'s name (`OpSchema_GetOutputName`).
    pub fn output_name(&self, index: usize) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetOutputName,
            "EpApi",
            "OpSchema_GetOutputName",
        )?;
        let mut p: *const c_char = ptr::null();
        check(unsafe { f(self.raw, index, &mut p as *mut _ as *const *const c_char) })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "op schema output name") }
        }
    }

    /// Number of type constraints (`OpSchema_GetTypeConstraintCount`).
    pub fn type_constraint_count(&self) -> Result<usize> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetTypeConstraintCount,
            "EpApi",
            "OpSchema_GetTypeConstraintCount",
        )?;
        let mut out: usize = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// Type constraint `index` (`OpSchema_GetTypeConstraint`), borrowed from this schema.
    pub fn type_constraint(&self, index: usize) -> Result<OpSchemaTypeConstraint<'_>> {
        let f = require_sub_api_fn(
            ep()?.OpSchema_GetTypeConstraint,
            "EpApi",
            "OpSchema_GetTypeConstraint",
        )?;
        let mut p: *const sys::OpSchemaTypeConstraintHandle = ptr::null();
        check(unsafe {
            f(
                self.raw,
                index,
                &mut p as *mut _ as *const *const sys::OpSchemaTypeConstraintHandle,
            )
        })?;
        if p.is_null() {
            Err(Error::new(-1, "zrt: op schema type constraint is null"))
        } else {
            // SAFETY: the constraint handle is borrowed from this schema for `&self`.
            Ok(unsafe { OpSchemaTypeConstraint::from_borrowed(p) })
        }
    }
}

impl Drop for OpSchema {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseOpSchema) {
            unsafe { release(self.raw) }
        }
    }
}
unsafe impl Send for OpSchema {}

/// Borrowed `OrtOpSchemaTypeConstraint` — one named type constraint of an [`OpSchema`] (its allowed
/// element types and the input/output indices it binds). Borrowed from the parent schema; never
/// released.
pub struct OpSchemaTypeConstraint<'a> {
    raw: *const sys::OpSchemaTypeConstraintHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> OpSchemaTypeConstraint<'a> {
    /// # Safety
    /// `raw` must remain valid for `'a` and must not be released by the caller.
    pub(crate) unsafe fn from_borrowed(raw: *const sys::OpSchemaTypeConstraintHandle) -> Self {
        Self {
            raw,
            _life: PhantomData,
        }
    }

    /// The constraint's type-parameter name, e.g. `"T"` (`OpSchemaTypeConstraint_GetTypeParamName`).
    pub fn type_param_name(&self) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.OpSchemaTypeConstraint_GetTypeParamName,
            "EpApi",
            "OpSchemaTypeConstraint_GetTypeParamName",
        )?;
        let mut p: *const c_char = ptr::null();
        check(unsafe { f(self.raw, &mut p as *mut _ as *const *const c_char) })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "type param name") }
        }
    }

    /// The allowed element-type strings, e.g. `["tensor(float)"]`
    /// (`OpSchemaTypeConstraint_GetAllowedTypes`).
    pub fn allowed_types(&self) -> Result<Vec<String>> {
        let f = require_sub_api_fn(
            ep()?.OpSchemaTypeConstraint_GetAllowedTypes,
            "EpApi",
            "OpSchemaTypeConstraint_GetAllowedTypes",
        )?;
        let mut types_ptr: *const *const c_char = ptr::null();
        let mut count: usize = 0;
        check(unsafe {
            f(
                self.raw,
                &mut types_ptr as *mut _ as *const *const *const c_char,
                &mut count,
            )
        })?;
        if types_ptr.is_null() || count == 0 {
            return Ok(Vec::new());
        }
        (0..count)
            .map(|i| {
                let p = unsafe { *types_ptr.add(i) };
                if p.is_null() {
                    Ok(String::new())
                } else {
                    unsafe { crate::cstr_to_string(p, "allowed type") }
                }
            })
            .collect()
    }

    /// The input indices this constraint binds (`OpSchemaTypeConstraint_GetInputIndices`).
    pub fn input_indices(&self) -> Result<Vec<usize>> {
        let f = require_sub_api_fn(
            ep()?.OpSchemaTypeConstraint_GetInputIndices,
            "EpApi",
            "OpSchemaTypeConstraint_GetInputIndices",
        )?;
        read_index_array(f, self.raw)
    }

    /// The output indices this constraint binds (`OpSchemaTypeConstraint_GetOutputIndices`).
    pub fn output_indices(&self) -> Result<Vec<usize>> {
        let f = require_sub_api_fn(
            ep()?.OpSchemaTypeConstraint_GetOutputIndices,
            "EpApi",
            "OpSchemaTypeConstraint_GetOutputIndices",
        )?;
        read_index_array(f, self.raw)
    }
}

/// Read a `(out: *const *const usize, count: *mut usize)` index-array pair — the shape of both
/// `OpSchemaTypeConstraint_GetInputIndices` and `_GetOutputIndices`.
fn read_index_array(
    f: unsafe extern "C" fn(
        *const sys::OpSchemaTypeConstraintHandle,
        *const *const usize,
        *mut usize,
    ) -> sys::StatusPtr,
    raw: *const sys::OpSchemaTypeConstraintHandle,
) -> Result<Vec<usize>> {
    let mut ptr: *const usize = ptr::null();
    let mut count: usize = 0;
    check(unsafe { f(raw, &mut ptr as *mut _ as *const *const usize, &mut count) })?;
    if ptr.is_null() || count == 0 {
        return Ok(Vec::new());
    }
    Ok((0..count).map(|i| unsafe { *ptr.add(i) }).collect())
}

/// Owning `OrtProfilingEvent` — one profiling event an EP profiler emits, created via
/// [`ProfilingEvent::new`] and read back field-by-field. Released on drop
/// (`ReleaseProfilingEvent`).
///
/// This wraps the **independently-creatable** profiling surface. An EP author builds events with
/// [`Self::new`] and normally hands them to `ProfilingEventsContainer_AddEvents` inside the EP's
/// `EndProfiling` callback; that container is factory-tied (it is handed to the EP inside the
/// profile callback, not constructible here) and lives in the `EpAuthor` ergonomics layer. Only the
/// create/read primitives an EP author needs to build and inspect an event are wrapped here.
pub struct ProfilingEvent {
    raw: *mut sys::ProfilingEventHandle,
}

impl ProfilingEvent {
    /// Create a profiling event (`CreateProfilingEvent`). `process_id`/`thread_id` may be `-1` when
    /// not applicable. `args` are key/value string pairs ORT copies (the caller's strings may drop
    /// once this returns). `timestamp_us`/`duration_us` are microseconds relative to the profiling
    /// start time.
    pub fn new(
        category: sys::ProfilingEventCategory, process_id: i32, thread_id: i32, event_name: &str,
        timestamp_us: i64, duration_us: i64, args: &[(&str, &str)],
    ) -> Result<Self> {
        let f = require_sub_api_fn(ep()?.CreateProfilingEvent, "EpApi", "CreateProfilingEvent")?;
        let name = cstr(event_name, "event name")?;
        // ORT copies the arg strings, so the owned CStrings may drop after the call.
        let keys: Vec<CString> = args
            .iter()
            .map(|(k, _)| cstr(k, "event arg key"))
            .collect::<Result<_>>()?;
        let vals: Vec<CString> = args
            .iter()
            .map(|(_, v)| cstr(v, "event arg value"))
            .collect::<Result<_>>()?;
        let key_ptrs: Vec<*const c_char> = keys.iter().map(|s| s.as_ptr()).collect();
        let val_ptrs: Vec<*const c_char> = vals.iter().map(|s| s.as_ptr()).collect();
        let mut raw: *mut sys::ProfilingEventHandle = ptr::null_mut();
        check(unsafe {
            f(
                category,
                process_id,
                thread_id,
                name.as_ptr(),
                timestamp_us,
                duration_us,
                key_ptrs.as_ptr(),
                val_ptrs.as_ptr(),
                key_ptrs.len(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: ensure_non_null(raw, "profiling event")?,
        })
    }

    /// The event category (`ProfilingEvent_GetCategory`).
    pub fn category(&self) -> Result<sys::ProfilingEventCategory> {
        let f = require_sub_api_fn(
            ep()?.ProfilingEvent_GetCategory,
            "EpApi",
            "ProfilingEvent_GetCategory",
        )?;
        let mut out: sys::ProfilingEventCategory = sys::ProfilingEventCategory::Session;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// The event name (`ProfilingEvent_GetName`). Empty if the engine stores none.
    pub fn name(&self) -> Result<String> {
        let f = require_sub_api_fn(
            ep()?.ProfilingEvent_GetName,
            "EpApi",
            "ProfilingEvent_GetName",
        )?;
        let mut p: *const c_char = ptr::null();
        check(unsafe { f(self.raw, &mut p as *mut _ as *const *const c_char) })?;
        if p.is_null() {
            Ok(String::new())
        } else {
            unsafe { crate::cstr_to_string(p, "profiling event name") }
        }
    }

    /// The start timestamp in microseconds (`ProfilingEvent_GetTimestampUs`).
    pub fn timestamp_us(&self) -> Result<i64> {
        let f = require_sub_api_fn(
            ep()?.ProfilingEvent_GetTimestampUs,
            "EpApi",
            "ProfilingEvent_GetTimestampUs",
        )?;
        let mut out: i64 = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// The duration in microseconds (`ProfilingEvent_GetDurationUs`).
    pub fn duration_us(&self) -> Result<i64> {
        let f = require_sub_api_fn(
            ep()?.ProfilingEvent_GetDurationUs,
            "EpApi",
            "ProfilingEvent_GetDurationUs",
        )?;
        let mut out: i64 = 0;
        check(unsafe { f(self.raw, &mut out) })?;
        Ok(out)
    }

    /// Look up an event argument's value by key (`ProfilingEvent_GetArgValue`); `None` if absent.
    pub fn arg_value(&self, key: &str) -> Result<Option<String>> {
        let f = require_sub_api_fn(
            ep()?.ProfilingEvent_GetArgValue,
            "EpApi",
            "ProfilingEvent_GetArgValue",
        )?;
        let k = cstr(key, "event arg key")?;
        let mut p: *const c_char = ptr::null();
        check(unsafe {
            f(
                self.raw,
                k.as_ptr(),
                &mut p as *mut _ as *const *const c_char,
            )
        })?;
        if p.is_null() {
            Ok(None)
        } else {
            unsafe { crate::cstr_to_string(p, "profiling event arg value") }.map(Some)
        }
    }
}

impl Drop for ProfilingEvent {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseProfilingEvent) {
            unsafe { release(self.raw) }
        }
    }
}
// The event handle is only released on drop; it is not touched concurrently from the Rust side.
unsafe impl Send for ProfilingEvent {}

// ── EP instance authoring: the `EpAuthor` trait + vtable glue ────────────────
//
// An `OrtEp` (see `sys::ep_vtables::EpVTable`) is a `repr(C)` table of function pointers ORT
// indexes by slot. ORT hands the EP author the ONLY handle to an instance — the `this_ptr`, which
// points at the vtable — so the EP's Rust state must be recoverable from that pointer. We use the
// standard C "vtable as first field" idiom: [`EpInstance<T>`] is `repr(C)` with the vtable first,
// so `vtable_ptr == instance_ptr` and a trampoline casts `this as *const EpInstance<T>` to reach
// `state`. The factory's `CreateEp` boxes such an instance, leaks it (`Box::into_raw`), and hands
// ORT the vtable pointer; ORT returns it as `this_ptr` to every callback, and `ReleaseEp` drops it
// (`Box::from_raw`). Factory glue, the `cdylib` export, and in-process validation are provided below.

/// Borrowed view of an `OrtGraph` passed to [`EpAuthor::get_capability`] (read-only — the EP
/// inspects it to decide which nodes to claim). Methods are added as EP authors need them; a stub
/// EP that claims no nodes never dereferences it.
pub struct EpGraphRef<'a> {
    #[allow(dead_code)] // retained for future graph-inspection API extensions
    raw: *const sys::GraphHandle,
    _life: PhantomData<&'a ()>,
}
impl<'a> EpGraphRef<'a> {
    /// # Safety
    /// `raw` must be a valid `OrtGraph*` for `'a` (or null if the EP will not read it).
    pub(crate) unsafe fn from_borrowed(raw: *const sys::GraphHandle) -> Self {
        Self {
            raw,
            _life: PhantomData,
        }
    }
}

/// Borrowed view of an `OrtEpGraphSupportInfo` the EP fills during [`EpAuthor::get_capability`] to
/// declare claimed nodes. A stub EP that claims nothing leaves it untouched.
pub struct EpGraphSupportInfoRef<'a> {
    #[allow(dead_code)] // retained for future node-claim API extensions
    raw: *mut sys::EpGraphSupportInfoHandle,
    _life: PhantomData<&'a ()>,
}
impl<'a> EpGraphSupportInfoRef<'a> {
    /// # Safety
    /// `raw` must be a valid `OrtEpGraphSupportInfo*` for `'a` (or null if unused).
    pub(crate) unsafe fn from_borrowed(raw: *mut sys::EpGraphSupportInfoHandle) -> Self {
        Self {
            raw,
            _life: PhantomData,
        }
    }
}

/// One execution-provider instance. The default implementations make a *stub* EP that claims no
/// graph nodes (so every node falls back to the CPU EP) — override [`Self::name`] and, if your EP
/// takes nodes, [`Self::get_capability`].
///
/// The instance is `Send` (ORT may call its callbacks from worker threads); an EP that needs shared
/// mutable state across concurrent runs must guard it itself.
pub trait EpAuthor: Sized + Send + 'static {
    /// The EP's name, returned to ORT (which copies it). Should be stable and non-empty.
    fn name(&self) -> Cow<'_, str>;

    /// Declare which graph nodes this EP claims (`OrtEp::GetCapability`). Default: claim none.
    ///
    /// `graph` is the (possibly nested) model graph; `support` is the builder ORT reads back. A
    /// real EP will use future `support` helpers to claim nodes or fuse subgraphs;
    /// the stub default does nothing, leaving every node on the CPU EP.
    fn get_capability(
        &mut self, _graph: &EpGraphRef<'_>, _support: &mut EpGraphSupportInfoRef<'_>,
    ) -> Result<()> {
        Ok(())
    }
}

/// `repr(C)` EP instance: the [`sys::ep_vtables::EpVTable`] (first field, so its address IS the
/// instance address) followed by the author's state. Built by [`ep_instance`] / the factory's
/// `CreateEp`; recovered by every trampoline via `this as *const EpInstance<T>`.
#[repr(C)]
pub struct EpInstance<T: EpAuthor> {
    /// Must be the first field — its pointer is the `OrtEp*` ORT stores and passes back.
    pub vtable: sys::ep_vtables::EpVTable,
    pub state: T,
}

/// Build an [`EpInstance`] for `state`, wiring the vtable to `state`'s [`EpAuthor`] impl. The
/// instance is returned boxed and leaked (`Box::into_raw`-ready) so it can be handed to ORT; the
/// matching `ReleaseEp` reclaims it. In-process unit tests may also use the returned
/// `Box` directly and let it drop normally.
pub fn ep_instance<T: EpAuthor>(state: T) -> Box<EpInstance<T>> {
    Box::new(EpInstance {
        vtable: ep_vtable::<T>(),
        state,
    })
}

/// Build a fully-populated [`sys::ep_vtables::EpVTable`] dispatching to `T`'s [`EpAuthor`] impl.
/// `GetName` + `GetCapability` are wired (the stub surface); every other slot is `None` (ORT applies
/// its documented per-slot default — e.g. NULL `GetPreferredDataLayout` ⇒ NCHW). Optional callbacks
/// that are not exposed by the current authoring API remain unwired.
pub fn ep_vtable<T: EpAuthor>() -> sys::ep_vtables::EpVTable {
    sys::ep_vtables::EpVTable {
        ort_version_supported: sys::API_VERSION,
        GetName: Some(__priv::get_name::<T>),
        GetCapability: Some(__priv::get_capability::<T>),
        Compile: None,
        ReleaseNodeComputeInfos: None,
        GetPreferredDataLayout: None,
        ShouldConvertDataLayoutForOp: None,
        SetDynamicOptions: None,
        OnRunStart: None,
        OnRunEnd: None,
        CreateAllocator: None,
        CreateSyncStreamForDevice: None,
        GetCompiledModelCompatibilityInfo: None,
        GetKernelRegistry: None,
        IsConcurrentRunSupported: None,
        Sync: None,
        CreateProfiler: None,
        IsGraphCaptureEnabled: None,
        IsGraphCaptured: None,
        ReplayGraph: None,
        GetGraphCaptureNodeAssignmentPolicy: None,
        GetAvailableResource: None,
        OnSessionInitializationEnd: None,
        GetDefaultMemoryDevice: None,
        ReleaseCapturedGraph: None,
    }
}

// ── EP factory authoring: the `EpFactoryAuthor` trait + vtable glue ───────────
//
// The factory (`OrtEpFactory`) mints [`EpAuthor`] instances: ORT obtains it (in-process, or via the
// dlopened `CreateEpFactories` symbol emitted by [`custom_ep!`](crate::custom_ep)), calls `CreateEp` per session to get
// an `OrtEp`, and `ReleaseEp` when done. Like [`EpInstance`], the factory state is recovered from
// the vtable pointer via the first-field idiom (`EpFactoryInstance<F>` is `repr(C)`, vtable first).

/// An execution-provider factory. `Send + Sync` because ORT may call factory methods from multiple
/// threads. Implement [`Self::name`] and [`Self::create_ep`]; `vendor` defaults to empty.
pub trait EpFactoryAuthor: Sized + Send + Sync + 'static {
    /// The EP type this factory produces.
    type Ep: EpAuthor;

    /// The factory (and EP) name, returned to ORT (which copies it). Should be stable and non-empty.
    fn name(&self) -> Cow<'_, str>;

    /// The vendor name. Default: empty.
    fn vendor(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }

    /// Build one EP instance for a session (`OrtEpFactory::CreateEp`). The C `devices`/
    /// `session_options`/`logger` args are not yet exposed — a stub EP that claims no nodes ignores
    /// them; an EP that needs device/session context reads it from the graph at `get_capability`
    /// time instead.
    fn create_ep(&self) -> Result<Self::Ep>;
}

/// `repr(C)` factory instance: the [`sys::ep_vtables::EpFactoryVTable`] (first field) followed by
/// the author's state. Built by [`ep_factory_instance`]; recovered by every trampoline via
/// `this as *const EpFactoryInstance<F>`.
#[repr(C)]
pub struct EpFactoryInstance<F: EpFactoryAuthor> {
    /// Must be the first field — its pointer is the `OrtEpFactory*` ORT stores and passes back.
    pub vtable: sys::ep_vtables::EpFactoryVTable,
    pub state: F,
}

/// Build an [`EpFactoryInstance`] for `state`, wiring the vtable to `state`'s [`EpFactoryAuthor`]
/// impl. Boxed and leaked-ready to hand to ORT (in-process or via `CreateEpFactories`); the matching
/// `ReleaseEpFactory` emitted by [`custom_ep!`](crate::custom_ep) reclaims it.
pub fn ep_factory_instance<F: EpFactoryAuthor>(state: F) -> Box<EpFactoryInstance<F>> {
    Box::new(EpFactoryInstance {
        vtable: ep_factory_vtable::<F>(),
        state,
    })
}

/// Build a fully-populated [`sys::ep_vtables::EpFactoryVTable`] dispatching to `F`'s
/// [`EpFactoryAuthor`] impl. `GetName`/`GetVendor`/`GetVendorId`/`GetVersion`/`CreateEp`/
/// `ReleaseEp` are wired; every other slot is `None`, including hardware-device discovery through
/// `GetSupportedDevices`.
pub fn ep_factory_vtable<F: EpFactoryAuthor>() -> sys::ep_vtables::EpFactoryVTable {
    sys::ep_vtables::EpFactoryVTable {
        ort_version_supported: sys::API_VERSION,
        GetName: Some(__priv::get_factory_name::<F>),
        GetVendor: Some(__priv::get_vendor::<F>),
        GetSupportedDevices: None,
        CreateEp: Some(__priv::create_ep::<F>),
        ReleaseEp: Some(__priv::release_ep::<F>),
        GetVendorId: Some(__priv::get_vendor_id),
        GetVersion: Some(__priv::get_version),
        ValidateCompiledModelCompatibilityInfo: None,
        CreateAllocator: None,
        ReleaseAllocator: None,
        CreateDataTransfer: None,
        IsStreamAware: None,
        CreateSyncStreamForDevice: None,
        GetHardwareDeviceIncompatibilityDetails: None,
        CreateExternalResourceImporterForDevice: None,
        GetNumCustomOpDomains: None,
        GetCustomOpDomains: None,
        InitGraphicsInterop: None,
        DeinitGraphicsInterop: None,
    }
}

// ── In-process EP registration primitives (EpApi-bound) ──────────────────────
//
// For in-process EP registration (no `cdylib`/`dlopen`), the host constructs a hardware device +
// binds the factory to it via `CreateEpDevice`, then appends the resulting `EpDevice` to session
// options. These owning wrappers integrate with [`crate::SessionOptions::append_ep_device`] for the
// final `SessionOptionsAppendExecutionProvider_V2` step.

/// Owning `OrtHardwareDevice` created via the EpApi (`CreateHardwareDevice`). Released on drop
/// (`ReleaseHardwareDevice`). `device_type` is `OrtHardwareDeviceType`: `0`=CPU, `1`=GPU, `2`=NPU.
pub struct OwnedHardwareDevice {
    raw: *mut sys::HardwareDeviceHandle,
}
impl OwnedHardwareDevice {
    /// Create a hardware device (`CreateHardwareDevice`); no metadata.
    pub fn new(
        device_type: i32, vendor_id: u32, device_id: u32, vendor_name: &str,
    ) -> Result<Self> {
        let f = require_sub_api_fn(ep()?.CreateHardwareDevice, "EpApi", "CreateHardwareDevice")?;
        let name = cstr(vendor_name, "vendor name")?;
        let mut raw: *mut sys::HardwareDeviceHandle = ptr::null_mut();
        check(unsafe {
            f(
                device_type,
                vendor_id,
                device_id,
                name.as_ptr(),
                ptr::null(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: ensure_non_null(raw, "hardware device")?,
        })
    }

    /// Borrow the raw `OrtHardwareDevice*` (e.g. for [`OwnedEpDevice::new`]).
    pub(crate) fn as_ptr(&self) -> *const sys::HardwareDeviceHandle {
        self.raw
    }
}
impl Drop for OwnedHardwareDevice {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseHardwareDevice) {
            unsafe { release(self.raw) }
        }
    }
}
unsafe impl Send for OwnedHardwareDevice {}

/// Owning `OrtEpDevice` created via the EpApi (`CreateEpDevice`) — binds a factory to a hardware
/// device for in-process registration. Released on drop (`ReleaseEpDevice`).
pub struct OwnedEpDevice {
    raw: *mut sys::EpDeviceHandle,
}
impl OwnedEpDevice {
    /// Create an EpDevice binding `factory` (a leaked [`EpFactoryInstance`] pointer — the factory
    /// must outlive this device) to `hardware_device`; no EP metadata/options.
    ///
    /// # Safety
    /// `factory` must be a valid `OrtEpFactory*` (a leaked [`EpFactoryInstance`] vtable pointer)
    /// that outlives the returned `OwnedEpDevice`.
    pub unsafe fn new(
        factory: *mut sys::EpFactoryHandle, hardware_device: &OwnedHardwareDevice,
    ) -> Result<Self> {
        let f = require_sub_api_fn(ep()?.CreateEpDevice, "EpApi", "CreateEpDevice")?;
        let mut raw: *mut sys::EpDeviceHandle = ptr::null_mut();
        check(unsafe {
            f(
                factory,
                hardware_device.as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut raw,
            )
        })?;
        Ok(Self {
            raw: ensure_non_null(raw, "ep device")?,
        })
    }

    /// Borrow the raw `OrtEpDevice*` for `SessionOptions::append_ep_device` when the `ep` feature
    /// is enabled.
    #[cfg_attr(not(feature = "ep"), allow(dead_code))]
    pub(crate) fn as_ptr(&self) -> *const sys::EpDeviceHandle {
        self.raw
    }
}
impl Drop for OwnedEpDevice {
    fn drop(&mut self) {
        if let Some(release) = ep_api().and_then(|t| t.ReleaseEpDevice) {
            unsafe { release(self.raw) }
        }
    }
}
unsafe impl Send for OwnedEpDevice {}

// Trampolines the vtable names. Generic over `T: EpAuthor` / `F: EpFactoryAuthor`; recover the
// instance from the vtable pointer. Panics never unwind across the FFI boundary (caught → a null
// status / fallback name; release paths abort since they have no status return).
pub mod __priv {
    use super::{
        Cow, EpAuthor, EpFactoryAuthor, EpFactoryInstance, EpGraphRef, EpGraphSupportInfoRef,
        EpInstance, ep_instance,
    };
    use crate::{Error, api, sys};
    use std::ffi::{CString, c_char};
    use std::os::raw::c_int;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr;

    // `thread_local` scratch for the `*const c_char` a status-free getter must return. ORT copies
    // the string synchronously within the call that produced it, so overwriting it on the next call
    // is sound.
    thread_local! {
        static LAST_NAME: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
    }

    /// Build an `OrtStatus` (`ORT_FAIL`) from a st-zrt error.
    pub(crate) fn error_to_status(e: Error) -> sys::StatusPtr {
        match CString::new(e.to_string()) {
            Ok(msg) => unsafe {
                api().create_status()(
                    sys::OrtErrorCode::Fail as c_int,
                    msg.as_ptr() as *const c_char,
                )
            },
            Err(_) => {
                static FALLBACK: &[u8] = b"st-zrt ep error (NUL in message)\0";
                unsafe {
                    api().create_status()(
                        sys::OrtErrorCode::Fail as c_int,
                        FALLBACK.as_ptr() as *const c_char,
                    )
                }
            },
        }
    }

    /// `OrtEp::GetName` — return the EP name as a stable C string (ORT copies it).
    pub unsafe extern "C" fn get_name<T: EpAuthor>(this: *const sys::EpHandle) -> *const c_char {
        // SAFETY: `this` points at the vtable, which is the first field of `EpInstance<T>`.
        let res = catch_unwind(AssertUnwindSafe(|| {
            let inst = unsafe { &*(this as *const EpInstance<T>) };
            inst.state.name()
        }));
        match res {
            Ok(name) => store_name(name),
            Err(_) => {
                static PANIC: &[u8] = b"st-zrt ep name panicked\0";
                PANIC.as_ptr() as *const c_char
            },
        }
    }

    /// `OrtEp::GetCapability` — let the EP claim nodes. Default claims none (Ok).
    pub unsafe extern "C" fn get_capability<T: EpAuthor>(
        ep: *mut sys::EpHandle, graph: *const sys::GraphHandle,
        support: *mut sys::EpGraphSupportInfoHandle,
    ) -> sys::StatusPtr {
        // SAFETY: `ep` points at the vtable (first field of `EpInstance<T>`).
        let res = catch_unwind(AssertUnwindSafe(|| {
            let inst = unsafe { &mut *(ep as *mut EpInstance<T>) };
            let g = unsafe { EpGraphRef::from_borrowed(graph) };
            let mut s = unsafe { EpGraphSupportInfoRef::from_borrowed(support) };
            inst.state.get_capability(&g, &mut s)
        }));
        match res {
            Ok(Ok(())) => ptr::null_mut(),
            Ok(Err(e)) => error_to_status(e),
            Err(_) => error_to_status(Error::new(
                sys::OrtErrorCode::Fail as i32,
                "st-zrt ep get_capability panicked",
            )),
        }
    }

    fn store_name(name: Cow<'_, str>) -> *const c_char {
        let c = CString::new(name.as_ref()).unwrap_or_else(|_| CString::new("").unwrap());
        let p = c.as_ptr();
        LAST_NAME.with(|cell| *cell.borrow_mut() = Some(c));
        p
    }

    /// `OrtEpFactory::GetName` — return the factory name as a stable C string (ORT copies it).
    pub unsafe extern "C" fn get_factory_name<F: EpFactoryAuthor>(
        this: *const sys::EpFactoryHandle,
    ) -> *const c_char {
        let res = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: `this` points at the vtable, the first field of `EpFactoryInstance<F>`.
            let f = unsafe { &*(this as *const EpFactoryInstance<F>) };
            f.state.name()
        }));
        match res {
            Ok(name) => store_name(name),
            Err(_) => {
                static PANIC: &[u8] = b"st-zrt factory name panicked\0";
                PANIC.as_ptr() as *const c_char
            },
        }
    }

    /// `OrtEpFactory::GetVendor`.
    pub unsafe extern "C" fn get_vendor<F: EpFactoryAuthor>(
        this: *const sys::EpFactoryHandle,
    ) -> *const c_char {
        let res = catch_unwind(AssertUnwindSafe(|| {
            let f = unsafe { &*(this as *const EpFactoryInstance<F>) };
            f.state.vendor()
        }));
        match res {
            Ok(name) => store_name(name),
            Err(_) => {
                static PANIC: &[u8] = b"st-zrt factory vendor panicked\0";
                PANIC.as_ptr() as *const c_char
            },
        }
    }

    /// `OrtEpFactory::GetVendorId` — default 0 (unknown). ORT reads this during `CreateEpDevice`.
    /// Non-generic: the default is independent of the factory type (a trait method can override later).
    pub unsafe extern "C" fn get_vendor_id(_this: *const sys::EpFactoryHandle) -> u32 {
        0
    }

    /// `OrtEpFactory::GetVersion` — default `"0.0.0"`. ORT reads this during `CreateEpDevice`.
    pub unsafe extern "C" fn get_version(_this: *const sys::EpFactoryHandle) -> *const c_char {
        store_name(Cow::Borrowed("0.0.0"))
    }

    /// `OrtEpFactory::CreateEp` — mint an `OrtEp` (`EpInstance<F::Ep>`) for one session, leak it
    /// (ORT owns it until `ReleaseEp`), and write its vtable pointer to `*ep_out`.
    pub unsafe extern "C" fn create_ep<F: EpFactoryAuthor>(
        factory: *mut sys::EpFactoryHandle, _devices: *const *const sys::HardwareDeviceHandle,
        _ep_metadata_pairs: *const *const sys::KeyValuePairsHandle, _num_devices: usize,
        _session_options: *const sys::SessionOptionsHandle, _logger: *const sys::LoggerHandle,
        ep_out: *mut *mut sys::EpHandle,
    ) -> sys::StatusPtr {
        let res = catch_unwind(AssertUnwindSafe(|| {
            let f = unsafe { &*(factory as *const EpFactoryInstance<F>) };
            let ep = f.state.create_ep()?;
            // `ep_instance` boxes an `EpInstance<F::Ep>`; `Box::into_raw` yields the vtable pointer
            // (vtable is the first field), which is the `OrtEp*` we hand ORT.
            let raw = Box::into_raw(ep_instance(ep));
            unsafe { *ep_out = raw as *mut sys::EpHandle };
            Ok(())
        }));
        match res {
            Ok(Ok(())) => ptr::null_mut(),
            Ok(Err(e)) => error_to_status(e),
            Err(_) => error_to_status(Error::new(
                sys::OrtErrorCode::Fail as i32,
                "st-zrt create_ep panicked",
            )),
        }
    }

    /// `OrtEpFactory::ReleaseEp` — reclaim the `EpInstance<F::Ep>` `create_ep` leaked. No status
    /// return exists, so a panic aborts (cannot unwind across the FFI boundary).
    pub unsafe extern "C" fn release_ep<F: EpFactoryAuthor>(
        _factory: *mut sys::EpFactoryHandle, ep: *mut sys::ep_vtables::EpVTable,
    ) {
        let res = catch_unwind(AssertUnwindSafe(|| {
            if !ep.is_null() {
                // SAFETY: `ep` was produced by `create_ep` (an `EpInstance<F::Ep>` boxed + leaked);
                // its vtable pointer IS the instance pointer.
                let _ = unsafe { Box::from_raw(ep as *mut EpInstance<F::Ep>) };
            }
        }));
        if res.is_err() {
            std::process::abort();
        }
    }
}

/// Declare an execution provider as a loadable shared library. Invoke once per `cdylib` crate with
/// a factory type that is [`Default`] + [`EpFactoryAuthor`]; expands to the two `#[no_mangle]` C
/// symbols ORT `dlopen`s — `CreateEpFactories` (mints one [`EpFactoryInstance`] via
/// `F::default()` + [`ep_factory_instance`], leaks it, writes `factories[0]`, `*num = 1`) and
/// `ReleaseEpFactory` (reclaims it). Mirrors the `custom_op!` macro available with the `custom-ops` feature.
///
/// The `ort_api_base`/`default_logger` params are not yet forwarded (a real EP that needs the ORT
/// Api accesses it through `ort_api_base->GetApi`); construction must not panic. For in-process use
/// (no `cdylib`), enable the `ep` feature, build an [`OwnedEpDevice`], and pass it to
/// `SessionOptions::append_ep_device`.
#[macro_export]
macro_rules! custom_ep {
    ($F:ty) => {
        /// `CreateEpFactories` — ORT `dlopen`s this symbol to obtain the EP's factory instance.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn CreateEpFactories(
            _registered_name: *const ::std::ffi::c_char, _ort_api_base: *const ::std::ffi::c_void,
            _default_logger: *const ::std::ffi::c_void,
            factories: *mut *mut $crate::sys::EpFactoryHandle, max_factories: usize,
            num_factories: *mut usize,
        ) -> $crate::sys::StatusPtr {
            let n = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                let inst = $crate::ep_factory_instance(<$F>::default());
                let raw = ::std::boxed::Box::into_raw(inst);
                if max_factories == 0 {
                    // No room in the caller's array — reclaim and report zero factories.
                    let _ = unsafe { ::std::boxed::Box::from_raw(raw) };
                    0
                } else {
                    unsafe { *factories.add(0) = raw as *mut $crate::sys::EpFactoryHandle };
                    1
                }
            }))
            .unwrap_or(0);
            unsafe { *num_factories = n };
            ::core::ptr::null_mut()
        }

        /// `ReleaseEpFactory` — ORT `dlopen`s this symbol to release a factory instance.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn ReleaseEpFactory(
            factory: *mut $crate::sys::EpFactoryHandle,
        ) -> $crate::sys::StatusPtr {
            if !factory.is_null() {
                let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                    // SAFETY: `factory` was produced by `CreateEpFactories` (an `EpFactoryInstance<$F>`
                    // boxed + leaked); its vtable pointer IS the instance pointer.
                    let _ = unsafe {
                        ::std::boxed::Box::from_raw(factory as *mut $crate::EpFactoryInstance<$F>)
                    };
                }));
                if res.is_err() {
                    ::std::process::abort();
                }
            }
            ::core::ptr::null_mut()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `KernelDef` end-to-end and read every field back: the builder setters round-trip
    /// through the `KernelDef` getters on a CPU host (no EP device required).
    #[test]
    fn kernel_def_builder_roundtrip() {
        let mut b = KernelDefBuilder::new().expect("builder");
        b.set_operator_type("Conv").expect("op type");
        b.set_domain("com.acme").expect("domain");
        b.set_since_version(1, 3).expect("since version");
        b.set_execution_provider("AcmeExecutionProvider")
            .expect("ep name");
        let def = b.build().expect("build");

        assert_eq!(def.operator_type().unwrap(), "Conv");
        assert_eq!(def.domain().unwrap(), "com.acme");
        assert_eq!(def.execution_provider().unwrap(), "AcmeExecutionProvider");
        assert_eq!(def.since_version().unwrap(), (1, 3));
    }

    /// Fetch a built-in op schema (`GetOpSchema("Add", ...)`) and read its contract: since-version,
    /// input/output counts + names, and type constraints. Proves the whole `OpSchema` read path on a
    /// CPU host. `Add` has 2 inputs / 1 output across every opset, so those counts are stable asserts.
    #[test]
    fn op_schema_reads_built_in_op_contract() {
        // The schema registry is populated when an `Environment` is created; look up `Add` (a core
        // op with 2 inputs / 1 output across every opset) in the default domain `""`.
        let _envs = crate::lock_default_env_creation();
        let _env = crate::Environment::new().expect("env");
        let schema = OpSchema::get("Add", 21, "").expect("fetch Add schema");

        let since = schema.since_version().expect("since version");
        assert!(since >= 1, "Add since_version should be >= 1, got {since}");

        let n_in = schema.num_inputs().expect("num inputs");
        assert_eq!(n_in, 2, "Add has 2 inputs");
        let n_out = schema.num_outputs().expect("num outputs");
        assert_eq!(n_out, 1, "Add has 1 output");

        for i in 0..n_in {
            let name = schema.input_name(i).expect("input name");
            eprintln!("Add input[{i}] name = {name:?}");
            assert!(!name.is_empty(), "Add input name must be non-empty");
        }
        for i in 0..n_out {
            let name = schema.output_name(i).expect("output name");
            eprintln!("Add output[{i}] name = {name:?}");
            assert!(!name.is_empty(), "Add output name must be non-empty");
        }

        let tcc = schema
            .type_constraint_count()
            .expect("type constraint count");
        eprintln!("Add has {tcc} type constraint(s)");
        let mut saw_allowed_types = false;
        for i in 0..tcc {
            let tc = schema.type_constraint(i).expect("type constraint");
            let param = tc.type_param_name().expect("type param name");
            let allowed = tc.allowed_types().expect("allowed types");
            let in_idx = tc.input_indices().expect("input indices");
            eprintln!("  constraint[{i}] {param:?} -> {allowed:?} (inputs {in_idx:?})");
            if !allowed.is_empty() {
                saw_allowed_types = true;
            }
        }
        assert!(
            saw_allowed_types,
            "Add must advertise at least one allowed type"
        );
    }

    /// Set input/output memory types and a type constraint on the builder, then read the memory
    /// types back from the built `KernelDef` (the EpApi has no type-constraint getter, so the
    /// constraint is asserted only to succeed). `GetTensorDataType` is exercised via the constraint.
    #[test]
    fn kernel_def_mem_type_and_type_constraint_roundtrip() {
        let mut b = KernelDefBuilder::new().expect("builder");
        b.set_operator_type("Gemm").expect("op type");
        b.set_input_mem_type(0, sys::MemType::CpuInput)
            .expect("set input mem type");
        b.set_output_mem_type(0, sys::MemType::CpuOutput)
            .expect("set output mem type");
        b.add_type_constraint("T", &[sys::ElementType::Float, sys::ElementType::Double])
            .expect("add type constraint");
        let def = b.build().expect("build");

        assert_eq!(def.input_mem_type(0).unwrap(), sys::MemType::CpuInput);
        assert_eq!(def.output_mem_type(0).unwrap(), sys::MemType::CpuOutput);
    }

    /// Create a profiling event with a category, timing, and args, then read every field back on a
    /// CPU host (no EP device or profiling session required — `CreateProfilingEvent` is a pure
    /// constructor that just stores the fields). Exercises the category enum, the timestamp/duration
    /// ints, the name, and the arg lookup (hit + miss).
    #[test]
    fn profiling_event_create_and_read_back() {
        let ev = ProfilingEvent::new(
            sys::ProfilingEventCategory::Kernel,
            1234,
            5678,
            "MyKernel",
            1_000,
            42,
            &[("op", "Conv"), ("shape", "1x2x3")],
        )
        .expect("create profiling event");

        assert_eq!(ev.category().unwrap(), sys::ProfilingEventCategory::Kernel);
        assert_eq!(ev.name().unwrap(), "MyKernel");
        assert_eq!(ev.timestamp_us().unwrap(), 1_000);
        assert_eq!(ev.duration_us().unwrap(), 42);
        assert_eq!(ev.arg_value("op").unwrap(), Some("Conv".to_string()));
        assert_eq!(ev.arg_value("shape").unwrap(), Some("1x2x3".to_string()));
        assert_eq!(ev.arg_value("missing").unwrap(), None);
    }

    /// Build a stub `EpAuthor`, wrap it in an `EpInstance`, and invoke the `GetName`/`GetCapability`
    /// vtable slots directly — proving the trampolines recover the instance state from the vtable
    /// pointer (the `vtable == instance` `repr(C)` idiom) without involving ORT. The default
    /// `get_capability` claims nothing and returns success (null status). This is the mechanism the
    /// factory glue and the in-process load test use.
    #[test]
    fn ep_instance_dispatches_via_vtable() {
        struct StubEp;
        impl EpAuthor for StubEp {
            fn name(&self) -> Cow<'_, str> {
                "stub-ep".into()
            }
        }
        let mut inst = ep_instance(StubEp);
        let get_name = inst.vtable.GetName.expect("GetName wired");
        let get_capability = inst.vtable.GetCapability.expect("GetCapability wired");
        // The vtable is the first field, so its address IS the `OrtEp*` ORT would store.
        let ep_ptr: *mut sys::EpHandle =
            &mut *inst as *mut EpInstance<StubEp> as *mut sys::EpHandle;

        let name_ptr = unsafe { get_name(ep_ptr as *const sys::EpHandle) };
        let name = unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap();
        assert_eq!(name, "stub-ep");

        // Default capability returns null status (success) — the stub claims no nodes.
        let st = unsafe { get_capability(ep_ptr, ptr::null(), ptr::null_mut()) };
        assert!(st.is_null(), "default get_capability must succeed");
    }

    /// Build a stub `EpFactoryAuthor`, then drive the factory vtable directly (no ORT): `GetName`
    /// returns the factory name; `CreateEp` mints an `EpInstance` whose `GetName` returns the EP
    /// name; `ReleaseEp` reclaims it. Proves the factory→EP lifecycle wiring and the leak/reclaim
    /// (`Box::into_raw` / `Box::from_raw`) exercised by the in-process load test.
    #[test]
    fn ep_factory_dispatches_create_and_release() {
        struct StubEp;
        impl EpAuthor for StubEp {
            fn name(&self) -> Cow<'_, str> {
                "stub-ep".into()
            }
        }
        struct StubFactory;
        impl EpFactoryAuthor for StubFactory {
            type Ep = StubEp;
            fn name(&self) -> Cow<'_, str> {
                "stub-factory".into()
            }
            fn create_ep(&self) -> Result<StubEp> {
                Ok(StubEp)
            }
        }

        let factory = ep_factory_instance(StubFactory);
        let get_name = factory.vtable.GetName.expect("factory GetName");
        let create_ep_fn = factory.vtable.CreateEp.expect("factory CreateEp");
        let release_ep_fn = factory.vtable.ReleaseEp.expect("factory ReleaseEp");
        let f_ptr: *mut sys::EpFactoryHandle =
            &*factory as *const EpFactoryInstance<StubFactory> as *mut sys::EpFactoryHandle;

        let name_ptr = unsafe { get_name(f_ptr as *const sys::EpFactoryHandle) };
        assert_eq!(
            unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap(),
            "stub-factory"
        );

        // CreateEp mints an EpInstance; its vtable pointer is the OrtEp*.
        let mut ep_out: *mut sys::EpHandle = ptr::null_mut();
        let st = unsafe {
            create_ep_fn(
                f_ptr,
                ptr::null(),
                ptr::null(),
                0,
                ptr::null(),
                ptr::null(),
                &mut ep_out,
            )
        };
        assert!(st.is_null(), "CreateEp must succeed");
        assert!(!ep_out.is_null(), "CreateEp must set ep_out");
        let ep_vt = ep_out as *const sys::ep_vtables::EpVTable;
        let ep_name_ptr = unsafe { ((*ep_vt).GetName.unwrap())(ep_out as *const sys::EpHandle) };
        assert_eq!(
            unsafe { CStr::from_ptr(ep_name_ptr) }.to_str().unwrap(),
            "stub-ep"
        );

        // ReleaseEp reclaims the leaked EpInstance (ep_out is dangling afterwards).
        unsafe { release_ep_fn(f_ptr, ep_out as *mut sys::ep_vtables::EpVTable) };
    }

    // Stub factory for the `custom_ep!` macro test below.
    struct MacroEp;
    impl EpAuthor for MacroEp {
        fn name(&self) -> Cow<'_, str> {
            "macro-ep".into()
        }
    }
    struct MacroFactory;
    impl Default for MacroFactory {
        fn default() -> Self {
            Self
        }
    }
    impl EpFactoryAuthor for MacroFactory {
        type Ep = MacroEp;
        fn name(&self) -> Cow<'_, str> {
            "macro-factory".into()
        }
        fn create_ep(&self) -> Result<MacroEp> {
            Ok(MacroEp)
        }
    }
    custom_ep!(MacroFactory);

    /// Invoke `custom_ep!`'s `#[no_mangle] CreateEpFactories`/`ReleaseEpFactory` directly (the same
    /// symbols ORT would `dlopen`): it mints one factory whose `GetName` returns the factory name,
    /// then reclaims it. Proves the cdylib-export macro end-to-end without a loader.
    #[test]
    fn custom_ep_macro_produces_factory() {
        let mut factories: [*mut sys::EpFactoryHandle; 4] = [ptr::null_mut(); 4];
        let mut num = 0usize;
        let st = unsafe {
            CreateEpFactories(
                c"name".as_ptr(),
                ptr::null(),
                ptr::null(),
                factories.as_mut_ptr(),
                4,
                &mut num,
            )
        };
        assert!(
            st.is_null(),
            "CreateEpFactories returns null status on success"
        );
        assert_eq!(num, 1, "one factory produced");
        let f = factories[0];
        assert!(!f.is_null());
        // The factory pointer IS the vtable pointer; invoke GetName directly.
        let vt = f as *const sys::ep_vtables::EpFactoryVTable;
        let name_ptr = unsafe { ((*vt).GetName.unwrap())(f as *const sys::EpFactoryHandle) };
        assert_eq!(
            unsafe { CStr::from_ptr(name_ptr) }.to_str().unwrap(),
            "macro-factory"
        );
        let st = unsafe { ReleaseEpFactory(f) };
        assert!(st.is_null(), "ReleaseEpFactory returns null status");
    }

    /// In-process EP registration primitives work standalone on a CPU host (no session): create a
    /// hardware device, bind a factory to it via `CreateEpDevice`, drop both. (`new` returns `Ok`
    /// only when the engine handed back a non-null handle, so construction success is the assert.)
    #[test]
    fn owned_hardware_and_ep_device_create_on_cpu() {
        struct NoopEp;
        impl EpAuthor for NoopEp {
            fn name(&self) -> Cow<'_, str> {
                "noop-ep".into()
            }
        }
        struct NoopFactory;
        impl EpFactoryAuthor for NoopFactory {
            type Ep = NoopEp;
            fn name(&self) -> Cow<'_, str> {
                "noop-factory".into()
            }
            fn create_ep(&self) -> Result<NoopEp> {
                Ok(NoopEp)
            }
        }
        let _envs = crate::lock_default_env_creation();
        let _env = crate::Environment::new().expect("env");
        let factory = ep_factory_instance(NoopFactory);
        let factory_ptr =
            &*factory as *const EpFactoryInstance<NoopFactory> as *mut sys::EpFactoryHandle;
        let hw = OwnedHardwareDevice::new(0 /*CPU*/, 0, 0, "st-zrt-test").expect("hardware device");
        let _ep_device = unsafe { OwnedEpDevice::new(factory_ptr, &hw) }.expect("ep device");
        // Both drop cleanly (ReleaseEpDevice / ReleaseHardwareDevice) before the factory Box.
    }

    /// The end-to-end in-process EP validation bar: author a stub EP (claims no nodes ⇒ every node
    /// falls back to the CPU EP), attach it to a real session built over a tiny `Y = Add(X1, X2)`
    /// model, run it, and confirm the full factory lifecycle fires through real ORT — `CreateEp` +
    /// `GetCapability` during session init, `ReleaseEp` (observed via the EP's `Drop`) at session
    /// teardown — and that the run produces the correct CPU result. Needs `ep` (the V2 attach
    /// path) on top of this module's `model-editor`.
    #[cfg(feature = "ep")]
    #[test]
    fn stub_ep_session_runs_with_cpu_fallback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CREATE_EP: AtomicUsize = AtomicUsize::new(0);
        static GET_CAPABILITY: AtomicUsize = AtomicUsize::new(0);
        static DROP_EP: AtomicUsize = AtomicUsize::new(0);

        struct StubEp;
        impl EpAuthor for StubEp {
            fn name(&self) -> Cow<'_, str> {
                "st-zrt-stub".into()
            }
            fn get_capability(
                &mut self, _g: &EpGraphRef<'_>, _s: &mut EpGraphSupportInfoRef<'_>,
            ) -> Result<()> {
                // Claim nothing ⇒ ORT keeps every node on the CPU EP (full fallback).
                GET_CAPABILITY.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }
        impl Drop for StubEp {
            fn drop(&mut self) {
                // Fires when `ReleaseEp` reclaims the leaked `EpInstance` (session teardown).
                DROP_EP.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct StubFactory;
        impl EpFactoryAuthor for StubFactory {
            type Ep = StubEp;
            fn name(&self) -> Cow<'_, str> {
                "st-zrt-stub-factory".into()
            }
            fn create_ep(&self) -> Result<StubEp> {
                CREATE_EP.fetch_add(1, Ordering::SeqCst);
                Ok(StubEp)
            }
        }

        let _envs = crate::lock_default_env_creation();
        let env = crate::Environment::new().expect("env");

        // Build `Y = Add(X1, X2)` (float[1]) in memory.
        let mem = crate::MemoryInfo::cpu().expect("cpu mem");
        let mut tsi = crate::TensorTypeAndShapeInfo::new().expect("tsi");
        tsi.set_element_type(crate::ElementType::Float)
            .expect("elem");
        tsi.set_dimensions(&[1]).expect("dims");
        let ty = crate::TypeInfo::tensor(&tsi).expect("type");
        let g = crate::Graph::new().expect("graph");
        g.set_inputs(vec![
            crate::ValueInfo::new("X1", &ty).expect("X1"),
            crate::ValueInfo::new("X2", &ty).expect("X2"),
        ])
        .expect("inputs");
        g.set_outputs(vec![crate::ValueInfo::new("Y", &ty).expect("Y")])
            .expect("outputs");
        g.add_node(crate::Node::new("Add", "", "add", &["X1", "X2"], &["Y"]).expect("node"))
            .expect("add node");
        let model = crate::Model::new(&[("", 21)]).expect("model");
        model.add_graph(g).expect("add graph");
        let bytes = model
            .to_bytes(&env, &crate::SessionOptions::new())
            .expect("to_bytes");

        // Author the EP: leak the factory (ORT holds it; `ReleaseEp` needs it), bind it to a CPU
        // hardware device. The factory + device + ep-device outlive the session below (declared
        // first, dropped last).
        let factory = ep_factory_instance(StubFactory);
        let factory_ptr = Box::into_raw(factory) as *mut sys::EpFactoryHandle;
        let hw = OwnedHardwareDevice::new(0 /*CPU*/, 0, 0, "st-zrt-test").expect("hardware device");
        let ep_device = unsafe { OwnedEpDevice::new(factory_ptr, &hw) }.expect("ep device");

        // Attach the authored EP device, then build the session.
        let opts = unsafe {
            crate::SessionOptions::new()
                .append_ep_device(&ep_device, &[])
                .expect("append ep device")
        };
        let sess = crate::Session::from_bytes(&env, &bytes, opts).expect("session");

        // Session init drove `CreateEp` + `GetCapability` through the factory/EP vtables.
        assert!(
            CREATE_EP.load(Ordering::SeqCst) >= 1,
            "factory CreateEp must fire during session init"
        );
        assert!(
            GET_CAPABILITY.load(Ordering::SeqCst) >= 1,
            "EP GetCapability must fire during session init"
        );

        // Run ⇒ CPU fallback computes Y = X1 + X2 = 2 + 3 = 5.
        let v1 = crate::Tensor::from_buffer(&[2.0_f32], &[1], &mem).expect("X1");
        let v2 = crate::Tensor::from_buffer(&[3.0_f32], &[1], &mem).expect("X2");
        let inputs: [&dyn crate::RunInput; 2] = [&v1, &v2];
        let mut out: Vec<Option<crate::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&inputs, &mut out).expect("run");
        let y: &[f32] = out[0].as_ref().expect("out").as_slice().expect("read");
        assert_eq!(y[0], 5.0, "CPU fallback must compute Add correctly");

        // Drop the session ⇒ ORT calls the factory's `ReleaseEp` ⇒ reclaims the `EpInstance` ⇒ the
        // `StubEp` drops. Only then is it safe to release the author-owned factory/device.
        drop(sess);
        assert!(
            DROP_EP.load(Ordering::SeqCst) >= 1,
            "ReleaseEp must fire on session drop"
        );

        drop(ep_device);
        drop(hw);
        // SAFETY: the session is dropped (no ORT references remain); the leaked factory is ours.
        unsafe {
            let _ = Box::from_raw(factory_ptr as *mut EpFactoryInstance<StubFactory>);
        }
    }
}
