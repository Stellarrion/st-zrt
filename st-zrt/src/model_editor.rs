//! Safe accessors for the ORT sub-API function tables (feature `model-editor`).
//!
//! `libonnxruntime` exposes four deref-style sub-APIs beyond the main `OrtApi`, each
//! obtained via a gateway getter:
//! - [`model_editor_api`] — graph/model editing (`OrtModelEditorApi`)
//! - [`compile_api`] — AOT model compilation (`OrtCompileApi`)
//! - [`ep_api`] — the execution-provider registry + EP authoring (`OrtEpApi`); this is
//!   the modern surface where DirectML / QNN / CoreML attach
//! - [`interop_api`] — external-resource interop (`OrtInteropApi`)
//!
//! This module wraps the **graph/model-surgery** part of `OrtModelEditorApi` — the
//! build-a-model-from-scratch → run path ([`Model`], [`Graph`], [`Node`], [`ValueInfo`],
//! [`TypeInfo`]) — plus [`crate::Session::from_model`] / [`crate::Session::opset_for_domain`].
//! The other sub-APIs (AOT compile, EP authoring, GPU/Vulkan interop) and the remaining
//! `ModelEditorApi` fns are reached via the gateway accessors below, added on demand.
//!
//! # Ownership (move semantics)
//! `Create*` are owning (`_Outptr_` → `Release*` on drop). The graph TAKES OWNERSHIP of the
//! nodes / value-infos / initializers added to it, and the model takes ownership of the graph
//! (per the ORT header — "do NOT call Release*"). So the consuming methods
//! ([`Graph::add_node`], [`Graph::set_inputs`], [`Graph::set_outputs`], [`Model::add_graph`])
//! `mem::forget` the moved wrapper on success (the graph/model now owns the handle); on error
//! the wrapper drops normally and releases. [`crate::Session::from_model`] BORROWS the model
//! (released when the [`Model`] drops).
use crate::{api, check, sys, Result, RunInput};
use std::ffi::{c_char, c_void, CString};
use std::ptr;

/// The graph/model-editing sub-API (`GetModelEditorApi`). Process-static; `None` only if
/// the engine didn't populate it (version skew).
pub fn model_editor_api() -> Option<&'static sys::ModelEditorApi> {
    let p = unsafe { api().get_model_editor_api()() };
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The AOT model-compilation sub-API (`GetCompileApi`).
pub fn compile_api() -> Option<&'static sys::CompileApi> {
    let p = unsafe { api().get_compile_api()() };
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The execution-provider registry / EP-authoring sub-API (`GetEpApi`) — the modern
/// surface for registering EPs and enumerating devices (where DirectML / QNN / CoreML
/// attach).
pub fn ep_api() -> Option<&'static sys::EpApi> {
    let p = unsafe { api().get_ep_api()() };
    (!p.is_null()).then(|| unsafe { &*p })
}

/// The external-resource interop sub-API (`GetInteropApi`).
pub fn interop_api() -> Option<&'static sys::InteropApi> {
    let p = unsafe { api().get_interop_api()() };
    (!p.is_null()).then(|| unsafe { &*p })
}

/// Borrow the live `ModelEditorApi` table, or an error if the engine didn't populate it.
fn me() -> Result<&'static sys::ModelEditorApi> {
    model_editor_api().ok_or_else(|| crate::Error::new(-1, "ModelEditorApi unavailable"))
}

/// Borrow the live `CompileApi` table, or an error if the engine didn't populate it.
fn ca() -> Result<&'static sys::CompileApi> {
    compile_api().ok_or_else(|| crate::Error::new(-1, "CompileApi unavailable"))
}

pub(crate) fn require_sub_api_fn<T: Copy>(
    f: Option<T>, api_name: &str, function_name: &str,
) -> Result<T> {
    f.ok_or_else(|| crate::Error::new(-1, format!("{api_name}.{function_name} unavailable")))
}

// ── owning sub-handles (graph/model surgery) ─────────────────────────────────

/// `OrtTypeInfo` — owning (`ReleaseTypeInfo`). Borrowed by [`ValueInfo::new`].
pub struct TypeInfo {
    ptr: *mut sys::TypeInfoHandle,
}
impl TypeInfo {
    /// A tensor type from a (built) [`crate::TensorTypeAndShapeInfo`].
    pub fn tensor(info: &crate::TensorTypeAndShapeInfo) -> Result<Self> {
        let create = require_sub_api_fn(
            me()?.CreateTensorTypeInfo,
            "ModelEditorApi",
            "CreateTensorTypeInfo",
        )?;
        let mut p: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { create(info.as_ptr(), &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor type info")?;
        Ok(Self { ptr: p })
    }
    /// A sparse-tensor type from a (built) [`crate::TensorTypeAndShapeInfo`].
    pub fn sparse_tensor(info: &crate::TensorTypeAndShapeInfo) -> Result<Self> {
        let create = require_sub_api_fn(
            me()?.CreateSparseTensorTypeInfo,
            "ModelEditorApi",
            "CreateSparseTensorTypeInfo",
        )?;
        let mut p: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { create(info.as_ptr(), &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor type info")?;
        Ok(Self { ptr: p })
    }
    /// A map type (`{key_type: value_type}`); `value_type` is borrowed.
    pub fn map(key_type: sys::ElementType, value_type: &TypeInfo) -> Result<Self> {
        let create = require_sub_api_fn(
            me()?.CreateMapTypeInfo,
            "ModelEditorApi",
            "CreateMapTypeInfo",
        )?;
        let mut p: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { create(key_type, value_type.ptr, &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor type info")?;
        Ok(Self { ptr: p })
    }
    /// A sequence type (`[element]`); `element` is borrowed.
    pub fn sequence(element: &TypeInfo) -> Result<Self> {
        let create = require_sub_api_fn(
            me()?.CreateSequenceTypeInfo,
            "ModelEditorApi",
            "CreateSequenceTypeInfo",
        )?;
        let mut p: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { create(element.ptr, &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor type info")?;
        Ok(Self { ptr: p })
    }
    /// An optional type (`Option<contained>`); `contained` is borrowed.
    pub fn optional(contained: &TypeInfo) -> Result<Self> {
        let create = require_sub_api_fn(
            me()?.CreateOptionalTypeInfo,
            "ModelEditorApi",
            "CreateOptionalTypeInfo",
        )?;
        let mut p: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { create(contained.ptr, &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor type info")?;
        Ok(Self { ptr: p })
    }
    pub(crate) fn as_ptr(&self) -> *const sys::TypeInfoHandle {
        self.ptr
    }
}
impl Drop for TypeInfo {
    fn drop(&mut self) {
        unsafe { api().release_type_info()(self.ptr) }
    }
}

/// `OrtValueInfo` — owning (`ReleaseValueInfo`). MOVED into a graph by
/// [`Graph::set_inputs`] / [`Graph::set_outputs`] (the graph takes ownership).
pub struct ValueInfo {
    ptr: *mut sys::ValueInfoHandle,
}
impl ValueInfo {
    /// `name` + a borrowed [`TypeInfo`]. The type is copied into the value-info.
    pub fn new(name: &str, ty: &TypeInfo) -> Result<Self> {
        let create =
            require_sub_api_fn(me()?.CreateValueInfo, "ModelEditorApi", "CreateValueInfo")?;
        let cname = CString::new(name)?;
        let mut p: *mut sys::ValueInfoHandle = ptr::null_mut();
        check(unsafe { create(cname.as_ptr(), ty.as_ptr(), &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor value info")?;
        Ok(Self { ptr: p })
    }
}
impl Drop for ValueInfo {
    fn drop(&mut self) {
        unsafe { api().release_value_info()(self.ptr) }
    }
}

/// `OrtNode` — owning (`ReleaseNode`). MOVED into a graph by [`Graph::add_node`].
pub struct Node {
    ptr: *mut sys::NodeHandle,
}

/// `OrtOpAttr` — owning (`ReleaseOpAttr`). MOVED into [`Node::with_attributes`].
///
/// This is intentionally scoped to graph construction. It covers the scalar and array
/// attributes needed by normal ONNX ops without enabling the custom-op feature.
pub struct NodeAttr {
    ptr: *mut sys::OpAttrHandle,
}

impl NodeAttr {
    fn new(name: &str, data: &[u8], len: usize, ty: sys::OpAttrType) -> Result<Self> {
        let name = CString::new(name)?;
        let len = i32::try_from(len)
            .map_err(|_| crate::Error::new(-1, "model-editor attribute length overflows i32"))?;
        let mut out: *mut sys::OpAttrHandle = ptr::null_mut();
        check(unsafe {
            api().create_op_attr()(
                name.as_ptr(),
                data.as_ptr() as *const c_void,
                len,
                ty,
                &mut out,
            )
        })?;
        let out = crate::ensure_non_null(out, "model-editor op attribute")?;
        Ok(Self { ptr: out })
    }

    /// Scalar int64 attribute.
    pub fn int(name: &str, value: i64) -> Result<Self> {
        Self::new(
            name,
            value.to_ne_bytes().as_slice(),
            1,
            sys::OpAttrType::Int,
        )
    }

    /// Scalar float attribute.
    pub fn float(name: &str, value: f32) -> Result<Self> {
        Self::new(
            name,
            value.to_ne_bytes().as_slice(),
            1,
            sys::OpAttrType::Float,
        )
    }

    /// int64 array attribute.
    pub fn ints(name: &str, values: &[i64]) -> Result<Self> {
        Self::new(
            name,
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            },
            values.len(),
            sys::OpAttrType::Ints,
        )
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::OpAttrHandle {
        self.ptr
    }
}

impl Drop for NodeAttr {
    fn drop(&mut self) {
        unsafe { api().release_op_attr()(self.ptr) }
    }
}

impl Node {
    /// An op node (no attributes). `inputs` / `outputs` are the names of the graph values
    /// the node reads / writes (graph inputs, initializer names, or other nodes' outputs) —
    /// not [`ValueInfo`]s.
    pub fn new(
        op: &str, domain: &str, name: &str, inputs: &[&str], outputs: &[&str],
    ) -> Result<Self> {
        Self::with_attributes(op, domain, name, inputs, outputs, Vec::new())
    }

    /// An op node with ONNX attributes. CONSUMES the attributes because ORT takes ownership
    /// when node creation succeeds.
    pub fn with_attributes(
        op: &str, domain: &str, name: &str, inputs: &[&str], outputs: &[&str],
        attributes: Vec<NodeAttr>,
    ) -> Result<Self> {
        let cop = CString::new(op)?;
        let cdom = CString::new(domain)?;
        let cname = CString::new(name)?;
        let in_c: Vec<CString> = inputs
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<_, _>>()?;
        let out_c: Vec<CString> = outputs
            .iter()
            .map(|s| CString::new(*s))
            .collect::<std::result::Result<_, _>>()?;
        let in_p: Vec<*const c_char> = in_c.iter().map(|c| c.as_ptr()).collect();
        let out_p: Vec<*const c_char> = out_c.iter().map(|c| c.as_ptr()).collect();
        let mut attr_p: Vec<*mut sys::OpAttrHandle> =
            attributes.iter().map(|attr| attr.as_ptr()).collect();
        let attr_ptr = if attr_p.is_empty() {
            ptr::null_mut()
        } else {
            attr_p.as_mut_ptr()
        };
        let create = require_sub_api_fn(me()?.CreateNode, "ModelEditorApi", "CreateNode")?;
        let mut p: *mut sys::NodeHandle = ptr::null_mut();
        check(unsafe {
            create(
                cop.as_ptr(),
                cdom.as_ptr(),
                cname.as_ptr(),
                in_p.as_ptr(),
                in_p.len(),
                out_p.as_ptr(),
                out_p.len(),
                attr_ptr,
                attr_p.len(),
                &mut p,
            )
        })?;
        let p = crate::ensure_non_null(p, "model-editor node")?;
        for attr in attributes {
            std::mem::forget(attr); // node took ownership — do not ReleaseOpAttr
        }
        Ok(Self { ptr: p })
    }
}
impl Drop for Node {
    fn drop(&mut self) {
        unsafe { api().release_node()(self.ptr) }
    }
}

/// `OrtGraph` — owning (`ReleaseGraph`). MOVED into a model by [`Model::add_graph`].
pub struct Graph {
    ptr: *mut sys::GraphHandle,
}
impl Graph {
    pub fn new() -> Result<Self> {
        let create = require_sub_api_fn(me()?.CreateGraph, "ModelEditorApi", "CreateGraph")?;
        let mut p: *mut sys::GraphHandle = ptr::null_mut();
        check(unsafe { create(&mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor graph")?;
        Ok(Self { ptr: p })
    }
    /// Set the graph's inputs. CONSUMES the [`ValueInfo`]s (the graph takes ownership).
    pub fn set_inputs(&self, inputs: Vec<ValueInfo>) -> Result<()> {
        let set_inputs =
            require_sub_api_fn(me()?.SetGraphInputs, "ModelEditorApi", "SetGraphInputs")?;
        let mut ptrs: Vec<*mut sys::ValueInfoHandle> = inputs.iter().map(|v| v.ptr).collect();
        check(unsafe { set_inputs(self.ptr, ptrs.as_mut_ptr(), ptrs.len()) })?;
        for v in inputs {
            std::mem::forget(v); // graph took ownership — do not ReleaseValueInfo
        }
        Ok(())
    }
    /// Set the graph's outputs. CONSUMES the [`ValueInfo`]s.
    pub fn set_outputs(&self, outputs: Vec<ValueInfo>) -> Result<()> {
        let set_outputs =
            require_sub_api_fn(me()?.SetGraphOutputs, "ModelEditorApi", "SetGraphOutputs")?;
        let mut ptrs: Vec<*mut sys::ValueInfoHandle> = outputs.iter().map(|v| v.ptr).collect();
        check(unsafe { set_outputs(self.ptr, ptrs.as_mut_ptr(), ptrs.len()) })?;
        for v in outputs {
            std::mem::forget(v);
        }
        Ok(())
    }
    /// Add a node. CONSUMES the [`Node`] (the graph takes ownership).
    pub fn add_node(&self, node: Node) -> Result<()> {
        let add = require_sub_api_fn(me()?.AddNodeToGraph, "ModelEditorApi", "AddNodeToGraph")?;
        check(unsafe { add(self.ptr, node.ptr) })?;
        std::mem::forget(node);
        Ok(())
    }

    /// Add an initializer — a named constant tensor (a weight) — to the graph. CONSUMES the
    /// tensor's `OrtValue`: ORT takes ownership of it ("do NOT call ReleaseOrtValue"), exactly
    /// like [`Graph::add_node`]. The constant is registered under `name`; reference `name` from
    /// a node's inputs to consume it. An initializer is **not** a graph input (do not list it in
    /// [`Graph::set_inputs`]).
    ///
    /// `data_is_external` must match how the tensor was built:
    /// - `false` → engine-allocated ([`crate::Tensor::copy_from_slice`]); works for **any** size.
    /// - `true` → caller-managed memory ([`crate::Tensor::from_buffer`]); per the ORT header the
    ///   data **must be ≥128 bytes** — small external tensors are unsupported, so use
    ///   `copy_from_slice` for them.
    pub fn add_initializer(
        &self, name: &str, tensor: crate::Tensor<'_>, data_is_external: bool,
    ) -> Result<()> {
        let add = require_sub_api_fn(
            me()?.AddInitializerToGraph,
            "ModelEditorApi",
            "AddInitializerToGraph",
        )?;
        let cname = CString::new(name)?;
        let value = tensor.as_value_ptr() as *mut sys::ValueHandle;
        check(unsafe { add(self.ptr, cname.as_ptr(), value, data_is_external) })?;
        std::mem::forget(tensor); // ORT took ownership of the OrtValue — do not ReleaseValue
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // Only releases if NOT moved into a model (`Model::add_graph` forgets on success).
        unsafe { api().release_graph()(self.ptr) }
    }
}

/// `OrtModel` — owning (`ReleaseModel`). BORROWED by [`crate::Session::from_model`]
/// (released when this drops, after the session is created).
pub struct Model {
    ptr: *mut sys::ModelHandle,
}
impl Model {
    /// `opsets` = `[(domain, since_version)]` (e.g. `[("", 21)]` for the ONNX default domain).
    pub fn new(opsets: &[(&str, i32)]) -> Result<Self> {
        let doms: Vec<CString> = opsets
            .iter()
            .map(|(d, _)| CString::new(*d))
            .collect::<std::result::Result<_, _>>()?;
        let vers: Vec<i32> = opsets.iter().map(|(_, v)| *v).collect();
        let dom_p: Vec<*const c_char> = doms.iter().map(|c| c.as_ptr()).collect();
        let create = require_sub_api_fn(me()?.CreateModel, "ModelEditorApi", "CreateModel")?;
        let mut p: *mut sys::ModelHandle = ptr::null_mut();
        check(unsafe { create(dom_p.as_ptr(), vers.as_ptr(), opsets.len(), &mut p) })?;
        let p = crate::ensure_non_null(p, "model-editor model")?;
        Ok(Self { ptr: p })
    }
    /// Add the (single) main graph. CONSUMES the [`Graph`] (the model takes ownership).
    pub fn add_graph(&self, graph: Graph) -> Result<()> {
        let add = require_sub_api_fn(me()?.AddGraphToModel, "ModelEditorApi", "AddGraphToModel")?;
        check(unsafe { add(self.ptr, graph.ptr) })?;
        std::mem::forget(graph);
        Ok(())
    }

    /// Serialize this model to ONNX bytes via the AOT-compile path (`CompileApi`): build
    /// compile options from `opts`, set this model as the input + a buffer as the output,
    /// compile. The result is a self-contained ONNX blob — reload it with
    /// [`crate::Session::from_bytes`]. `opts` supplies the graph-optimization level.
    pub fn to_bytes(
        &self, env: &crate::Environment, opts: &crate::SessionOptions,
    ) -> Result<Vec<u8>> {
        let ca = compile_api().ok_or_else(|| crate::Error::new(-1, "CompileApi unavailable"))?;
        let create_options = require_sub_api_fn(
            ca.CreateModelCompilationOptionsFromSessionOptions,
            "CompileApi",
            "CreateModelCompilationOptionsFromSessionOptions",
        )?;
        let opts_handle = opts.build_handle()?;
        let mut copts: *mut sys::ModelCompilationOptionsHandle = ptr::null_mut();
        let build = check(unsafe {
            create_options(
                env.as_ptr(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut copts,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        build?;
        let copts = crate::ensure_non_null(copts, "model compilation options")?;

        // Input = this model (borrowed); output = a buffer allocated by the default allocator.
        let outcome: Result<Vec<u8>> = (|| {
            let set_input = require_sub_api_fn(
                ca.ModelCompilationOptions_SetInputModel,
                "CompileApi",
                "ModelCompilationOptions_SetInputModel",
            )?;
            check(unsafe { set_input(copts, self.as_ptr()) })?;
            let alloc = crate::allocator::Allocator::get_default()?;
            let mut buf_ptr: *mut c_void = ptr::null_mut();
            let mut buf_size: usize = 0;
            let set_output_buffer = require_sub_api_fn(
                ca.ModelCompilationOptions_SetOutputModelBuffer,
                "CompileApi",
                "ModelCompilationOptions_SetOutputModelBuffer",
            )?;
            check(unsafe { set_output_buffer(copts, alloc.alloc, &mut buf_ptr, &mut buf_size) })?;
            let compile = require_sub_api_fn(ca.CompileModel, "CompileApi", "CompileModel")?;
            let compile = check(unsafe { compile(env.as_ptr(), copts) });
            let bytes = if buf_ptr.is_null() || buf_size == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(buf_ptr as *const u8, buf_size).to_vec() }
            };
            let free = if buf_ptr.is_null() {
                Ok(())
            } else {
                unsafe { alloc.free(buf_ptr) }
            };
            match (compile, free) {
                (Ok(()), Ok(())) => Ok(bytes),
                (Err(err), _) => Err(err),
                (Ok(()), Err(err)) => Err(err),
            }
        })();

        if let Some(release) = ca.ReleaseModelCompilationOptions {
            unsafe { release(copts) };
        }
        outcome
    }

    /// Compile this model and write it to an ONNX file at `path` (the AOT-compile
    /// `SetOutputModelPath` path) — the file-based counterpart of [`Model::to_bytes`]. `opts`
    /// supplies the graph-optimization level. For EP-context / flags / external-initializers
    /// control, build a [`ModelCompilationOptions`] directly.
    pub fn to_file(
        &self, env: &crate::Environment, opts: &crate::SessionOptions, path: &str,
    ) -> Result<()> {
        let copts = ModelCompilationOptions::new(env, opts)?;
        copts.set_input_model(self)?;
        copts.set_output_model_path(path)?;
        // `copts` drops (releases the handle) after compile returns.
        copts.compile(env)
    }

    pub(crate) fn as_ptr(&self) -> *const sys::ModelHandle {
        self.ptr
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe { api().release_model()(self.ptr) }
    }
}

// ── AOT model compilation (CompileApi) ───────────────────────────────────────
//
// `CompileApi` (`GetCompileApi`) ahead-of-time compiles a model: serialize it ([`Model::to_bytes`]
// / [`Model::to_file`]) OR produce an EP-specific compiled artifact (EPContext binary, external
// initializers). `ModelCompilationOptions` is the config handle; pick an input source + output
// destination + EP-context/flags, then [`ModelCompilationOptions::compile`].

/// Owning wrapper for `OrtModelCompilationOptions` (feature `model-editor`) — the AOT
/// model-compilation handle. Built from a [`crate::SessionOptions`] (which seeds the
/// graph-optimization level + EP config); set an input source, an output destination, and any
/// EP-context/flags, then [`Self::compile`].
///
/// Output-to-buffer is served by [`Model::to_bytes`] (it manages the engine-allocated buffer
/// inline); this builder covers output-to-file ([`Self::set_output_model_path`]). The two
/// output-callback setters (`SetOutputModelWriteFunc`, `SetOutputModelGetInitializerLocationFunc`)
/// are **not wrapped**: the codegen erased their `OrtWriteBufferFunc` /
/// `OrtGetInitializerLocationFunc` fn-pointer types into opaque handles, so they can't be
/// constructed from a Rust fn — reach them via [`compile_api`] if needed.
pub struct ModelCompilationOptions {
    ptr: *mut sys::ModelCompilationOptionsHandle,
}

impl ModelCompilationOptions {
    /// Create compile options from a [`crate::SessionOptions`]
    /// (`CreateModelCompilationOptionsFromSessionOptions`). The session options' optimization
    /// level + EP config seed the compile.
    pub fn new(env: &crate::Environment, opts: &crate::SessionOptions) -> Result<Self> {
        let ca = ca()?;
        let create_options = require_sub_api_fn(
            ca.CreateModelCompilationOptionsFromSessionOptions,
            "CompileApi",
            "CreateModelCompilationOptionsFromSessionOptions",
        )?;
        let opts_handle = opts.build_handle()?;
        let mut p: *mut sys::ModelCompilationOptionsHandle = ptr::null_mut();
        let r = check(unsafe {
            create_options(
                env.as_ptr(),
                opts_handle as *const sys::SessionOptionsHandle,
                &mut p,
            )
        });
        unsafe { api().release_session_options()(opts_handle) };
        r?;
        let p = crate::ensure_non_null(p, "model compilation options")?;
        Ok(Self { ptr: p })
    }

    /// Input: an in-memory [`Model`] built via the model-editor API (BORROWED — the model must
    /// outlive [`Self::compile`]).
    pub fn set_input_model(&self, model: &Model) -> Result<()> {
        let set_input = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetInputModel,
            "CompileApi",
            "ModelCompilationOptions_SetInputModel",
        )?;
        check(unsafe { set_input(self.ptr, model.as_ptr()) })
    }

    /// Input: the bytes of a serialized ONNX model (BORROWED for the duration of compile).
    pub fn set_input_model_from_buffer(&self, bytes: &[u8]) -> Result<()> {
        let set_input = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetInputModelFromBuffer,
            "CompileApi",
            "ModelCompilationOptions_SetInputModelFromBuffer",
        )?;
        check(unsafe { set_input(self.ptr, bytes.as_ptr() as *const c_void, bytes.len()) })
    }

    /// Input: a path to an ONNX model file (`path` copied; must be NUL-free).
    pub fn set_input_model_path(&self, path: &str) -> Result<()> {
        let set_input = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetInputModelPath,
            "CompileApi",
            "ModelCompilationOptions_SetInputModelPath",
        )?;
        let c = CString::new(path)?;
        check(unsafe { set_input(self.ptr, c.as_ptr()) })
    }

    /// Output: write the compiled model to `path` (copied; NUL-free) — the file counterpart of
    /// [`Model::to_bytes`].
    pub fn set_output_model_path(&self, path: &str) -> Result<()> {
        let set_output = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetOutputModelPath,
            "CompileApi",
            "ModelCompilationOptions_SetOutputModelPath",
        )?;
        let c = CString::new(path)?;
        check(unsafe { set_output(self.ptr, c.as_ptr()) })
    }

    /// Store initializers larger than `threshold` bytes in an external file at `path` rather
    /// than in the model (only initializers for nodes NOT compiled go here — compiled nodes
    /// embed their data in EPContext nodes). `path` copied; NUL-free.
    pub fn set_output_model_external_initializers_file(
        &self, path: &str, threshold: usize,
    ) -> Result<()> {
        let set_output = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetOutputModelExternalInitializersFile,
            "CompileApi",
            "ModelCompilationOptions_SetOutputModelExternalInitializersFile",
        )?;
        let c = CString::new(path)?;
        check(unsafe { set_output(self.ptr, c.as_ptr(), threshold) })
    }

    /// Embed EPContext binary data into the model's EPContext nodes' `ep_cache_context`
    /// attribute (`true`), or store just a path to an external context file (`false`, the
    /// default). See the ORT EPContext design doc.
    pub fn set_ep_context_embed_mode(&self, embed: bool) -> Result<()> {
        let set_mode = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetEpContextEmbedMode,
            "CompileApi",
            "ModelCompilationOptions_SetEpContextEmbedMode",
        )?;
        check(unsafe { set_mode(self.ptr, embed) })
    }

    /// Set compile flags — a `u32` bitmask OR of `OrtCompileApiFlags` (e.g. the quantization
    /// pre-pass flag; since v1.23).
    pub fn set_flags(&self, flags: u32) -> Result<()> {
        let set_flags = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetFlags,
            "CompileApi",
            "ModelCompilationOptions_SetFlags",
        )?;
        check(unsafe { set_flags(self.ptr, flags) })
    }

    /// EP context binary location: the output `directory` + `model_name` the EP uses to name its
    /// context binary file (used when compiling with in-memory I/O + non-embedded EP context).
    /// Both copied; NUL-free.
    pub fn set_ep_context_binary_information(
        &self, directory: &str, model_name: &str,
    ) -> Result<()> {
        let set_info = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetEpContextBinaryInformation,
            "CompileApi",
            "ModelCompilationOptions_SetEpContextBinaryInformation",
        )?;
        let d = CString::new(directory)?;
        let m = CString::new(model_name)?;
        check(unsafe { set_info(self.ptr, d.as_ptr(), m.as_ptr()) })
    }

    /// Override the graph-optimization level (otherwise inherited from the seed
    /// [`crate::SessionOptions`]).
    pub fn set_graph_optimization_level(&self, level: sys::GraphOptimizationLevel) -> Result<()> {
        let set_level = require_sub_api_fn(
            ca()?.ModelCompilationOptions_SetGraphOptimizationLevel,
            "CompileApi",
            "ModelCompilationOptions_SetGraphOptimizationLevel",
        )?;
        check(unsafe { set_level(self.ptr, level) })
    }

    /// Run the compilation (`CompileModel`), writing to the configured output (path, or buffer
    /// via [`Model::to_bytes`]).
    pub fn compile(&self, env: &crate::Environment) -> Result<()> {
        let compile = require_sub_api_fn(ca()?.CompileModel, "CompileApi", "CompileModel")?;
        check(unsafe { compile(env.as_ptr(), self.ptr) })
    }
}

impl Drop for ModelCompilationOptions {
    fn drop(&mut self) {
        if let Some(ca) = compile_api() {
            if let Some(release) = ca.ReleaseModelCompilationOptions {
                unsafe { release(self.ptr) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All four sub-APIs are populated (non-null) at ORT API 26 on this host.
    #[test]
    fn sub_apis_available() {
        assert!(model_editor_api().is_some(), "ModelEditorApi missing");
        assert!(compile_api().is_some(), "CompileApi missing");
        assert!(ep_api().is_some(), "EpApi missing");
        assert!(interop_api().is_some(), "InteropApi missing");
        eprintln!("all four sub-APIs available via the safe accessors");
    }
}
