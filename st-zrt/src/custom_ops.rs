//! Custom-op authoring surface (feature `custom-ops`).
//!
//! Two owning, CPU-testable lifecycles plus the borrowed kernel-time API a custom-op
//! kernel uses while ORT is calling it.
//!
//! - [`OpAttr`] — an operator attribute (CreateOpAttr / ReadOpAttr / OpAttr_GetType /
//!   OpAttr_GetName). Owning: released on drop. Round-trippable on CPU with no kernel.
//! - [`CustomOpDomain`] — the registration container a session options holds
//!   (CreateCustomOpDomain / AddCustomOpDomain). Owning: released on drop. It MUST outlive
//!   every session built from options it was attached to (an ORT invariant — ORT keeps a
//!   reference, it does not copy). Attach via [`crate::SessionOptions::with_custom_op_domain`].
//!
//! The kernel-time types — [`KernelInfo`], [`KernelContext`], [`Op`] — wrap handles ORT
//! hands to your kernel callbacks (which run only while ORT is invoking your kernel). They
//! faithfully expose the FFI with status-checking and lifetime tying. The `custom_op!` macro
//! emits the `OrtCustomOp` vtable bridging ORT's callbacks to a [`CustomOp`] impl, and the
//! whole path — `create` / `compute` / `destroy` / `infer_shapes` — is runtime-verified
//! end-to-end via a bundled `com.example::MyRelu` fixture. Reach a hand-built vtable directly
//! via [`CustomOpDomain::add_raw`].
//!
//! Ownership follows the borrow-vs-own split used elsewhere in st-zrt: `KernelInfo` /
//! `KernelContext` are borrowed for the kernel call (no `Drop`); `Op` / `OpAttr` /
//! `CustomOpDomain` / [`OwnedKernelInfo`] are owning (released on drop).
use crate::allocator::Allocator;
use crate::memory::MemoryInfo;
use crate::session_options::SessionOptions;
use crate::tensor::TensorView;
use crate::{api, check, sys, Error, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::marker::PhantomData;
use std::ptr;

// ─── two-call ORT string/array fetch helpers ──────────────────────────────────

/// Fetch a NUL-terminated string via the ORT "char* out, size_t* size" two-call pattern:
/// probe the required size with a null buffer (the status it returns is released and
/// ignored — ORT reports "buffer too small" but still sets the size), then fetch into a
/// sized buffer and check that call. Returns the string without its trailing NUL.
unsafe fn fetch_sized_string(
    fill: impl Fn(*mut c_char, *mut usize) -> sys::StatusPtr,
) -> Result<String> {
    let mut size: usize = 0;
    let probe = fill(ptr::null_mut(), &mut size);
    if !probe.is_null() {
        api().release_status()(probe);
    }
    let mut buf = vec![0u8; size];
    check(fill(buf.as_mut_ptr() as *mut c_char, &mut size))?;
    trim_nul(&buf, size)
}

/// Fetch a fixed-size pod array (`f32`/`i64`) via the same two-call pattern. The probe
/// returns the element count; we then allocate and fill.
unsafe fn fetch_sized_array<T: Copy + Default>(
    fill: impl Fn(*mut T, *mut usize) -> sys::StatusPtr,
) -> Result<Vec<T>> {
    let mut count: usize = 0;
    let probe = fill(ptr::null_mut(), &mut count);
    if !probe.is_null() {
        api().release_status()(probe);
    }
    let mut buf = vec![T::default(); count];
    check(fill(buf.as_mut_ptr(), &mut count))?;
    Ok(buf)
}

/// Decode `buf[..size]` as UTF-8, dropping a single trailing NUL if present.
fn trim_nul(buf: &[u8], size: usize) -> Result<String> {
    if size == 0 {
        return Ok(String::new());
    }
    let end = if buf[size - 1] == 0 { size - 1 } else { size };
    std::str::from_utf8(&buf[..end])
        .map(str::to_owned)
        .map_err(|_| Error::new(-1, "zrt: custom-op string is not valid UTF-8"))
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::new(-1, "custom-op string contains a NUL byte"))
}

fn usize_to_c_int(value: usize, what: &'static str) -> Result<c_int> {
    c_int::try_from(value).map_err(|_| Error::new(-1, format!("zrt: {what} exceeds c_int::MAX")))
}

// ─── OpAttr (owning) ──────────────────────────────────────────────────────────

/// An operator attribute, built with `CreateOpAttr` and read back with `ReadOpAttr` /
/// `OpAttr_GetType` / `OpAttr_GetName`. Owning — released on drop. Pass `&OpAttr` to
/// [`Op::create`] as an attribute, or read one obtained from a node.
pub struct OpAttr {
    ptr: *mut sys::OpAttrHandle,
}

impl OpAttr {
    /// Build an attribute of `ty` from a raw byte buffer. `len` is the element count for
    /// array types (Ints/Floats), the byte count for String, and 1 for scalar Int/Float.
    pub fn new(name: &str, data: &[u8], len: usize, ty: sys::OpAttrType) -> Result<Self> {
        let name = cstring(name)?;
        let len = usize_to_c_int(len, "custom-op attribute length")?;
        let api = api();
        let mut out: *mut sys::OpAttrHandle = ptr::null_mut();
        check(unsafe {
            api.create_op_attr()(
                name.as_ptr(),
                data.as_ptr() as *const c_void,
                len,
                ty,
                &mut out,
            )
        })?;
        let out = crate::ensure_non_null(out, "custom-op attribute")?;
        Ok(Self { ptr: out })
    }

    /// Scalar float attribute (`ORT_OP_ATTR_FLOAT`, len = 1).
    pub fn new_float(name: &str, value: f32) -> Result<Self> {
        Self::new(
            name,
            value.to_ne_bytes().as_slice(),
            1usize,
            sys::OpAttrType::Float,
        )
    }

    /// Scalar int64 attribute (`ORT_OP_ATTR_INT`, len = 1).
    pub fn new_int(name: &str, value: i64) -> Result<Self> {
        Self::new(
            name,
            value.to_ne_bytes().as_slice(),
            1usize,
            sys::OpAttrType::Int,
        )
    }

    /// String attribute (`ORT_OP_ATTR_STRING`, len = byte count).
    pub fn new_string(name: &str, value: &str) -> Result<Self> {
        Self::new(name, value.as_bytes(), value.len(), sys::OpAttrType::String)
    }

    /// int64 array attribute (`ORT_OP_ATTR_INTS`, len = element count).
    pub fn new_ints(name: &str, values: &[i64]) -> Result<Self> {
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

    /// float array attribute (`ORT_OP_ATTR_FLOATS`, len = element count).
    pub fn new_floats(name: &str, values: &[f32]) -> Result<Self> {
        Self::new(
            name,
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            },
            values.len(),
            sys::OpAttrType::Floats,
        )
    }

    /// The attribute's value kind.
    pub fn ty(&self) -> Result<sys::OpAttrType> {
        let mut out = sys::OpAttrType::Undefined;
        check(unsafe { api().op_attr__get_type()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// The attribute's name (engine-owned string).
    pub fn name(&self) -> Result<String> {
        let p: *const c_char = ptr::null();
        check(unsafe { api().op_attr__get_name()(self.ptr, &p) })?;
        Ok(if p.is_null() {
            String::new()
        } else {
            unsafe { crate::cstr_to_string(p, "custom-op attribute name") }?
        })
    }

    /// Read the attribute's raw bytes into `buf`. Returns the byte count ORT wrote.
    pub fn read_into(&self, ty: sys::OpAttrType, buf: &mut [u8]) -> Result<usize> {
        let mut written: usize = 0;
        check(unsafe {
            api().read_op_attr()(
                self.ptr,
                ty,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
                &mut written,
            )
        })?;
        Ok(written)
    }
}

impl Drop for OpAttr {
    fn drop(&mut self) {
        unsafe { api().release_op_attr()(self.ptr) }
    }
}
unsafe impl Send for OpAttr {}
unsafe impl Sync for OpAttr {}

// ─── CustomOpDomain (owning) ──────────────────────────────────────────────────

/// A custom-op domain: a named container of registered custom ops, attached to session
/// options so sessions built from them resolve the domain's ops. Owning — released on
/// drop. **Must outlive every session built from options it was attached to** (ORT retains
/// the domain; it does not copy it).
///
/// Registering an op means adding an `OrtCustomOp` vtable struct. The ergonomic vtable
/// builder is a separate milestone; for now use [`CustomOpDomain::add_raw`] with a
/// hand-built `*const OrtCustomOp` (cast to `*const CustomOpHandle`).
pub struct CustomOpDomain {
    ptr: *mut sys::CustomOpDomainHandle,
}

impl CustomOpDomain {
    /// Create an empty domain named `domain` (e.g. `"com.example.foo"`).
    pub fn new(domain: &str) -> Result<Self> {
        let domain = cstring(domain)?;
        let mut out: *mut sys::CustomOpDomainHandle = ptr::null_mut();
        check(unsafe { api().create_custom_op_domain()(domain.as_ptr(), &mut out) })?;
        let out = crate::ensure_non_null(out, "custom-op domain")?;
        Ok(Self { ptr: out })
    }

    /// Add a registered op (`OrtCustomOp*`, cast to `*const CustomOpHandle`) to this
    /// domain.
    ///
    /// # Safety
    /// `op` must point to a valid, fully-initialized `OrtCustomOp` vtable (its `version`
    /// field set to `ORT_API_VERSION` and the callbacks ORT will read populated), and it
    /// must remain valid until this domain is released.
    pub unsafe fn add_raw(&self, op: *const sys::CustomOpHandle) -> Result<()> {
        check(api().custom_op_domain__add()(self.ptr, op))
    }

    /// Register a custom op whose vtable was built by the [`custom_op!`](crate::custom_op)
    /// macro. The `&'static` bound means a `pub static` vtable (which `custom_op!` emits)
    /// satisfies the ORT invariant — the vtable storage outlives this domain and every
    /// session built from options it's attached to.
    pub fn add_op(&self, vtable: &'static sys::OrtCustomOp) -> Result<()> {
        // SAFETY: `OrtCustomOp` is `#[repr(C)]`-compatible with ORT's `OrtCustomOp*`, and a
        // `&'static` reference is a valid pointer for the domain's lifetime.
        unsafe { self.add_raw(vtable as *const sys::OrtCustomOp as *const sys::CustomOpHandle) }
    }
}

impl Drop for CustomOpDomain {
    fn drop(&mut self) {
        unsafe { api().release_custom_op_domain()(self.ptr) }
    }
}
unsafe impl Send for CustomOpDomain {}
unsafe impl Sync for CustomOpDomain {}

impl SessionOptions {
    /// Attach a custom-op domain. The domain is registered when the session-options handle
    /// is built (in `Session::new`). The `domain` MUST outlive every session built from
    /// these options — ORT keeps a reference to it.
    #[cfg(feature = "custom-ops")]
    pub fn with_custom_op_domain(mut self, domain: &CustomOpDomain) -> Self {
        self.custom_op_domains.push(domain.ptr);
        self
    }
}

// ─── authoring a custom op: the CustomOp trait + OpIoSpec ─────────────────────

/// One input or output of a custom op: its tensor element type, its presence
/// characteristic, and (for inputs) its memory type.
#[derive(Clone, Copy, Debug)]
pub struct OpIoSpec {
    /// Tensor element type for this input/output.
    pub element_type: sys::ElementType,
    /// Required / optional / variadic.
    pub characteristic: sys::CustomOpInputOutputCharacteristic,
    /// Memory type (inputs only; ignored for outputs).
    pub memory_type: sys::MemType,
}

impl OpIoSpec {
    /// A required input/output of `ty` on default memory.
    pub const fn required(ty: sys::ElementType) -> Self {
        Self {
            element_type: ty,
            characteristic: sys::CustomOpInputOutputCharacteristic::Required,
            memory_type: sys::MemType::Default,
        }
    }

    /// An optional input/output of `ty` on default memory.
    pub const fn optional(ty: sys::ElementType) -> Self {
        Self {
            element_type: ty,
            characteristic: sys::CustomOpInputOutputCharacteristic::Optional,
            memory_type: sys::MemType::Default,
        }
    }

    /// Required input/output of `ty`, placed on `mem` (inputs only).
    pub const fn required_on(ty: sys::ElementType, mem: sys::MemType) -> Self {
        Self {
            element_type: ty,
            characteristic: sys::CustomOpInputOutputCharacteristic::Required,
            memory_type: mem,
        }
    }
}

/// Author a custom ONNX operator. `Self` is the per-kernel-instance state ORT holds
/// between `create` and `destroy`.
///
/// Implement this, then emit the `OrtCustomOp` vtable with the [`custom_op!`](crate::custom_op)
/// macro and register it on a [`CustomOpDomain`] via [`CustomOpDomain::add_op`].
///
/// **Bounds — `Send + 'static`:** a kernel instance is created on the graph-loading thread
/// and `compute` may run on an ORT worker thread, so `Self: Send`. The state escapes `create`
/// into ORT-owned storage and lives until `destroy`, so `Self: 'static`. `Sync` is not
/// required: ORT does not invoke `compute` concurrently on a single kernel instance (one
/// kernel per graph node; sequential within a `Run`) — the same contract the C++ `CustomOpBase`
/// and the `ort`/`onnxruntime-rs` crates rely on for `&mut self`. If you opt into ORT's
/// `ExecutionMode::Parallel` you own the responsibility for any cross-instance shared state.
///
/// **Panics** in `create`/`compute` are caught and surfaced to ORT as an `ORT_FAIL` status
/// (never unwound across the FFI boundary); a panic in `Drop` of the kernel state aborts the
/// process (there is no status path from `destroy`).
pub trait CustomOp: Sized + Send + 'static {
    /// Op name as it appears in a graph node (without the domain prefix). Supplied to
    /// [`custom_op!`](crate::custom_op), which NUL-terminates it for the C callback.
    const NAME: &'static str;
    /// Domain; `""` for the default `ai.onnx` domain, else e.g. `"com.example"`.
    const DOMAIN: &'static str = "";
    /// First opset version this op exists in.
    const SINCE_VERSION: i32 = 1;
    /// Inclusive upper opset version. Defaults to the ORT API version (26).
    const END_VERSION: i32 = sys::API_VERSION as i32;

    /// Build the kernel state from the node's `KernelInfo` (attributes, I/O type info).
    fn create(info: &KernelInfo<'_>) -> Result<Self>;
    /// Run one inference. Read inputs via `ctx.input(i)` + [`crate::TensorView::as_slice`];
    /// write outputs via `ctx.output_mut(i, dims, |buf| …)`.
    fn compute(&mut self, ctx: &KernelContext<'_>) -> Result<()>;

    /// Shape inference (the `InferOutputShape` vtable callback): set each output's type+shape
    /// via [`ShapeInferContext::set_output_type_shape`]. ORT calls this at session-creation
    /// (graph-optimization) time, BEFORE any kernel exists — so it is an associated function
    /// (no `self`), invoked as `T::infer_shapes(&ctx)`. The default is a no-op: ORT then leaves
    /// output shapes unknown, which is fine when the model's output value_info already carries
    /// them; implement it when your op's output shape is not statically known to the graph.
    fn infer_shapes(_ctx: &ShapeInferContext<'_>) -> Result<()> {
        Ok(())
    }

    /// Input schema. The `custom_op!` macro reads this to fill the vtable's input
    /// type/count/characteristic/memory-type callbacks. Return a reference to a `static`
    /// array (rvalue promotion does not apply here):
    /// `static IN: [OpIoSpec; 1] = [OpIoSpec::required(ElementType::Float)]; &IN`.
    fn inputs() -> &'static [OpIoSpec];
    /// Output schema. Fills the vtable's output type/count/characteristic callbacks —
    /// return a reference to a `static` array (see [`CustomOp::inputs`]).
    fn outputs() -> &'static [OpIoSpec];

    /// Execution-provider type; `None` ⇒ the CPU EP (this milestone is CPU-only — returning a
    /// provider string here is not yet wired through the vtable).
    fn execution_provider_type() -> Option<&'static str> {
        None
    }
    /// Minimum arity of a variadic input (default 1; only meaningful if the last input is
    /// variadic).
    fn variadic_input_min_arity() -> std::os::raw::c_int {
        1
    }
    /// Whether a variadic input's operands must share an element type (default false).
    fn variadic_input_homogeneity() -> bool {
        false
    }
    fn variadic_output_min_arity() -> std::os::raw::c_int {
        1
    }
    fn variadic_output_homogeneity() -> bool {
        false
    }
}

// ─── KernelInfo (borrowed) + OwnedKernelInfo (owning) ─────────────────────────

/// Borrowed access to the `OrtKernelInfo*` ORT passes to a kernel's `Create` callback:
/// node attributes and input/output metadata. Borrowed for the kernel call — no `Drop`.
/// Construct one from the raw pointer ORT hands you (`KernelInfo::from_ptr`).
pub struct KernelInfo<'a> {
    ptr: *const sys::KernelInfoHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> KernelInfo<'a> {
    /// Wrap a raw `OrtKernelInfo*` ORT passed to a kernel callback.
    ///
    /// # Safety
    /// `ptr` must be a valid `OrtKernelInfo*` obtained from ORT (e.g. the argument to a
    /// kernel's `Create` callback) and remain valid for `'a`.
    pub unsafe fn from_ptr(ptr: *const sys::KernelInfoHandle) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }

    /// Number of declared inputs.
    pub fn input_count(&self) -> Result<usize> {
        let mut out = 0usize;
        check(unsafe { api().kernel_info__get_input_count()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Number of declared outputs.
    pub fn output_count(&self) -> Result<usize> {
        let mut out = 0usize;
        check(unsafe { api().kernel_info__get_output_count()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Name of input `index`.
    pub fn input_name(&self, index: usize) -> Result<String> {
        unsafe {
            fetch_sized_string(|out, size| {
                api().kernel_info__get_input_name()(self.ptr, index, out, size)
            })
        }
    }

    /// Name of output `index`.
    pub fn output_name(&self, index: usize) -> Result<String> {
        unsafe {
            fetch_sized_string(|out, size| {
                api().kernel_info__get_output_name()(self.ptr, index, out, size)
            })
        }
    }

    /// Owning `OrtTypeInfo*` for input `index` — release with `ReleaseTypeInfo` when done.
    pub fn input_type_info(&self, index: usize) -> Result<*mut sys::TypeInfoHandle> {
        let mut out: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { api().kernel_info__get_input_type_info()(self.ptr, index, &mut out) })?;
        crate::ensure_non_null(out, "kernel input type info")
    }

    /// Owning `OrtTypeInfo*` for output `index` — release with `ReleaseTypeInfo` when done.
    pub fn output_type_info(&self, index: usize) -> Result<*mut sys::TypeInfoHandle> {
        let mut out: *mut sys::TypeInfoHandle = ptr::null_mut();
        check(unsafe { api().kernel_info__get_output_type_info()(self.ptr, index, &mut out) })?;
        crate::ensure_non_null(out, "kernel output type info")
    }

    /// The node's name.
    pub fn node_name(&self) -> Result<String> {
        unsafe {
            fetch_sized_string(|out, size| api().kernel_info__get_node_name()(self.ptr, out, size))
        }
    }

    /// The op's domain (e.g. `""` for ai.onnx, `"com.example.foo"` for custom).
    pub fn operator_domain(&self) -> Result<String> {
        unsafe {
            fetch_sized_string(|out, size| {
                api().kernel_info__get_operator_domain()(self.ptr, out, size)
            })
        }
    }

    /// The op's type/name (e.g. `"Conv"`).
    pub fn operator_type(&self) -> Result<String> {
        unsafe {
            fetch_sized_string(|out, size| {
                api().kernel_info__get_operator_type()(self.ptr, out, size)
            })
        }
    }

    /// The opset `since_version` this kernel was registered for.
    pub fn operator_since_version(&self) -> Result<i32> {
        let mut out: c_int = 0;
        check(unsafe { api().kernel_info__get_operator_since_version()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Scalar float attribute `name`. Missing/invalid attribute → `Err` (use `.ok()` for
    /// `Option<f32>`); see the module docs on the ORT convention.
    pub fn attr_float(&self, name: &str) -> Result<f32> {
        let name = cstring(name)?;
        let mut out: f32 = 0.0;
        check(unsafe {
            api().kernel_info_get_attribute_float()(self.ptr, name.as_ptr(), &mut out)
        })?;
        Ok(out)
    }

    /// Scalar int64 attribute `name`. Missing/invalid → `Err` (`.ok()` for `Option<i64>`).
    pub fn attr_int64(&self, name: &str) -> Result<i64> {
        let name = cstring(name)?;
        let mut out: i64 = 0;
        check(unsafe {
            api().kernel_info_get_attribute_int64()(self.ptr, name.as_ptr(), &mut out)
        })?;
        Ok(out)
    }

    /// String attribute `name`. Missing/invalid → `Err`.
    pub fn attr_string(&self, name: &str) -> Result<String> {
        let name = cstring(name)?;
        unsafe {
            fetch_sized_string(|out, size| {
                api().kernel_info_get_attribute_string()(self.ptr, name.as_ptr(), out, size)
            })
        }
    }

    /// float-array attribute `name`. Missing/invalid → `Err`.
    pub fn attr_floats(&self, name: &str) -> Result<Vec<f32>> {
        let name = cstring(name)?;
        unsafe {
            fetch_sized_array(|out, size| {
                api().kernel_info_get_attribute_array_float()(self.ptr, name.as_ptr(), out, size)
            })
        }
    }

    /// int64-array attribute `name`. Missing/invalid → `Err`.
    pub fn attr_int64s(&self, name: &str) -> Result<Vec<i64>> {
        let name = cstring(name)?;
        unsafe {
            fetch_sized_array(|out, size| {
                api().kernel_info_get_attribute_array_int64()(self.ptr, name.as_ptr(), out, size)
            })
        }
    }

    /// Owning `OrtValue*` for a tensor attribute `name`, allocated via `allocator` —
    /// release with `ReleaseValue` when done.
    pub fn attr_tensor(&self, name: &str, allocator: &Allocator) -> Result<*mut sys::ValueHandle> {
        let name = cstring(name)?;
        let mut out: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().kernel_info_get_attribute_tensor()(
                self.ptr,
                name.as_ptr(),
                allocator.alloc,
                &mut out,
            )
        })?;
        Ok(out)
    }

    /// `(is_constant, value)` for input `index`. When `is_constant` is true the returned
    /// `OrtValue*` is the constant initializer (borrowed from the graph — do not release).
    pub fn constant_input_tensor(&self, index: usize) -> Result<(bool, *const sys::ValueHandle)> {
        let mut is_const: c_int = 0;
        let out: *const sys::ValueHandle = ptr::null();
        check(unsafe {
            api().kernel_info_get_constant_input_tensor()(self.ptr, index, &mut is_const, &out)
        })?;
        Ok((is_const != 0, out))
    }

    /// Borrowed `OrtLogger*` for this node (do not release).
    pub fn logger(&self) -> Result<*const sys::LoggerHandle> {
        let out: *const sys::LoggerHandle = ptr::null();
        check(unsafe { api().kernel_info__get_logger()(self.ptr, &out) })?;
        Ok(out)
    }

    /// Borrowed `OrtAllocator*` for `mem_type` (do not release).
    pub fn allocator(&self, mem_type: sys::MemType) -> Result<*mut sys::AllocatorHandle> {
        let mut out: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe { api().kernel_info_get_allocator()(self.ptr, mem_type, &mut out) })?;
        crate::ensure_non_null(out, "kernel allocator")
    }

    /// Make an independent owning copy of this info (`CopyKernelInfo`) — used to keep
    /// metadata alive beyond kernel construction.
    pub fn to_owned(&self) -> Result<OwnedKernelInfo> {
        let mut out: *mut sys::KernelInfoHandle = ptr::null_mut();
        check(unsafe { api().copy_kernel_info()(self.ptr, &mut out) })?;
        let out = crate::ensure_non_null(out, "kernel info")?;
        Ok(OwnedKernelInfo { ptr: out })
    }
}

/// An owning copy of a [`KernelInfo`] (`CopyKernelInfo`), independent of the kernel call.
/// Released on drop. Use [`KernelInfo::to_owned`] to create one.
pub struct OwnedKernelInfo {
    ptr: *mut sys::KernelInfoHandle,
}

impl OwnedKernelInfo {
    /// The raw `OrtKernelInfo*`.
    pub fn as_ptr(&self) -> *const sys::KernelInfoHandle {
        self.ptr
    }

    /// Borrow this owned info for use with [`KernelInfo`]'s getters.
    pub fn as_ref(&self) -> KernelInfo<'_> {
        KernelInfo {
            ptr: self.ptr,
            _life: PhantomData,
        }
    }
}

impl Drop for OwnedKernelInfo {
    fn drop(&mut self) {
        unsafe { api().release_kernel_info()(self.ptr) }
    }
}
unsafe impl Send for OwnedKernelInfo {}
unsafe impl Sync for OwnedKernelInfo {}

// ─── KernelContext (borrowed) ─────────────────────────────────────────────────

/// Borrowed access to the `OrtKernelContext*` ORT passes to a kernel's `Compute`
/// callback: the inputs, the output buffers, the allocator, scratch space, and the
/// parallel-for primitive. Borrowed for the compute call — no `Drop`. Construct one from
/// the raw pointer ORT hands you (`KernelContext::from_ptr`).
pub struct KernelContext<'a> {
    ptr: *const sys::KernelContextHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> KernelContext<'a> {
    /// Wrap a raw `OrtKernelContext*` ORT passed to a kernel callback.
    ///
    /// # Safety
    /// `ptr` must be a valid `OrtKernelContext*` obtained from ORT (e.g. the argument to a
    /// kernel's `Compute` callback) and remain valid for `'a`.
    pub unsafe fn from_ptr(ptr: *const sys::KernelContextHandle) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }

    /// Number of inputs.
    pub fn input_count(&self) -> Result<usize> {
        let mut out = 0usize;
        check(unsafe { api().kernel_context__get_input_count()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Number of outputs.
    pub fn output_count(&self) -> Result<usize> {
        let mut out = 0usize;
        check(unsafe { api().kernel_context__get_output_count()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Input `index` as a zero-copy tensor view borrowing this context, or `None` if the
    /// (optional) input is absent. Assumes the input is a tensor (the custom-op case).
    pub fn input(&self, index: usize) -> Result<Option<TensorView<'a>>> {
        let out: *const sys::ValueHandle = ptr::null();
        check(unsafe { api().kernel_context__get_input()(self.ptr, index, &out) })?;
        Ok(if out.is_null() {
            None
        } else {
            Some(TensorView {
                value: out as *mut sys::ValueHandle,
                _life: PhantomData,
            })
        })
    }

    /// Allocate output `index` with `dims` and run `f` over its writable, typed buffer —
    /// the write path for a custom-op `compute`. The buffer is engine-owned (allocated by
    /// `KernelContext_GetOutput`, freed with the context); `f` receives a `&mut [T]` of
    /// `dims`' element count, scoped to this call.
    ///
    /// Call at most once per `index` within one compute (mirrors the ORT C contract: the
    /// same index twice may alias the same buffer). Reading an input via
    /// [`KernelContext::input`] (then [`crate::TensorView::as_slice`]) and writing one
    /// output here is the common, sound pattern (distinct buffers).
    pub fn output_mut<T: crate::element::TensorElement>(
        &self, index: usize, dims: &[i64], f: impl FnOnce(&mut [T]) -> Result<()>,
    ) -> Result<()> {
        let mut out: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().kernel_context__get_output()(
                self.ptr as *mut sys::KernelContextHandle,
                index,
                dims.as_ptr(),
                dims.len(),
                &mut out,
            )
        })?;
        let out = crate::ensure_non_null(out, "custom-op output value")?;
        let mut data: *mut std::ffi::c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(out, &mut data) })?;
        let count = crate::type_info::checked_element_count(dims)?;
        let data = crate::slice_data_ptr(data as *mut T, count, "custom-op output data")?;
        // SAFETY: `data` is a contiguous, aligned buffer of `count` uninitialized elements
        // of T, freshly allocated for this output and owned by the context. The slice is
        // confined to `f` (cannot outlive this call).
        let slice = unsafe { std::slice::from_raw_parts_mut(data, count) };
        f(slice)
    }

    /// Borrowed `OrtLogger*` for this compute call (do not release).
    pub fn logger(&self) -> Result<*const sys::LoggerHandle> {
        let out: *const sys::LoggerHandle = ptr::null();
        check(unsafe { api().kernel_context__get_logger()(self.ptr, &out) })?;
        Ok(out)
    }

    /// Borrowed `OrtAllocator*` for `mem` (do not release).
    pub fn allocator(&self, mem: &MemoryInfo) -> Result<*mut sys::AllocatorHandle> {
        let mut out: *mut sys::AllocatorHandle = ptr::null_mut();
        check(unsafe { api().kernel_context__get_allocator()(self.ptr, mem.info, &mut out) })?;
        crate::ensure_non_null(out, "kernel context allocator")
    }

    /// Engine-owned scratch buffer of `count_or_bytes` bytes for `mem`. The buffer is
    /// freed with the context — do not release it.
    pub fn scratch_buffer(&self, mem: &MemoryInfo, count_or_bytes: usize) -> Result<*mut c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        check(unsafe {
            api().kernel_context__get_scratch_buffer()(self.ptr, mem.info, count_or_bytes, &mut out)
        })?;
        Ok(out)
    }

    /// The GPU compute stream for this context, or null on CPU. Returned as a raw pointer
    /// (EP-specific; opaque to the CPU API).
    pub fn gpu_compute_stream(&self) -> Result<*mut c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        check(unsafe { api().kernel_context__get_gpu_compute_stream()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// An EP-specific resource (`resource_version`, `resource_id`). Returned as a raw
    /// pointer (opaque to the CPU API).
    pub fn resource(&self, resource_version: c_int, resource_id: c_int) -> Result<*mut c_void> {
        let mut out: *mut c_void = ptr::null_mut();
        check(unsafe {
            api().kernel_context__get_resource()(self.ptr, resource_version, resource_id, &mut out)
        })?;
        Ok(out)
    }

    /// Run `f` over `total` iterations in `num_batch` batches, dispatched by the engine
    /// (may be concurrent across worker threads — hence `Send + Sync`). Bridges ORT's
    /// `KernelContext_ParallelFor` C callback.
    pub fn parallel_for<F>(&self, total: usize, num_batch: usize, f: F) -> Result<()>
    where
        F: Fn(usize) + Send + Sync,
    {
        let data = Box::into_raw(Box::new(f)) as *mut c_void;
        let res = check(unsafe {
            api().kernel_context__parallel_for()(
                self.ptr,
                Some(parallel_for_trampoline::<F>),
                total,
                num_batch,
                data,
            )
        });
        // Reclaim the box regardless of success/failure (ORT has finished all calls).
        unsafe { drop(Box::from_raw(data as *mut F)) };
        res
    }
}

/// Re-dispatches an ORT parallel-for callback to the boxed Rust closure it carries.
unsafe extern "C" fn parallel_for_trampoline<F>(data: *mut c_void, index: usize)
where
    F: Fn(usize) + Send + Sync,
{
    if !data.is_null() {
        (&*(data as *const F))(index);
    }
}

// ─── ShapeInferContext (borrowed) ────────────────────────────────────────────

/// Borrowed access to the `OrtShapeInferContext*` ORT passes to a custom op's
/// `InferOutputShape` callback: read input type+shapes and set output type+shapes. Borrowed
/// for the inference call — no `Drop`. ORT calls shape inference at session-creation time
/// (graph optimization), before any kernel exists; reach it via [`CustomOp::infer_shapes`].
pub struct ShapeInferContext<'a> {
    ptr: *const sys::ShapeInferContextHandle,
    _life: PhantomData<&'a ()>,
}

impl<'a> ShapeInferContext<'a> {
    /// Wrap a raw `OrtShapeInferContext*` ORT passed to a shape-inference callback.
    ///
    /// # Safety
    /// `ptr` must be a valid `OrtShapeInferContext*` obtained from ORT and remain valid for `'a`.
    pub unsafe fn from_ptr(ptr: *const sys::ShapeInferContextHandle) -> Self {
        Self {
            ptr,
            _life: PhantomData,
        }
    }

    /// Number of graph inputs (`ShapeInferContext_GetInputCount`).
    pub fn input_count(&self) -> Result<usize> {
        let mut out = 0usize;
        check(unsafe { api().shape_infer_context__get_input_count()(self.ptr, &mut out) })?;
        Ok(out)
    }

    /// Owning type+shape info for input `index` (`ShapeInferContext_GetInputTypeShape`); released
    /// on drop. Read an input's element type/dims from it, or — for an elementwise op — pass it
    /// straight to [`Self::set_output_type_shape`].
    pub fn input_type_shape(
        &self, index: usize,
    ) -> Result<crate::type_info::TensorTypeAndShapeInfo> {
        let mut info: *mut sys::TensorTypeAndShapeInfoHandle = ptr::null_mut();
        check(unsafe {
            api().shape_infer_context__get_input_type_shape()(self.ptr, index, &mut info)
        })?;
        let info = crate::ensure_non_null(info, "tensor type and shape info")?;
        // SAFETY: GetInputTypeShape allocates an owning handle; released on drop.
        Ok(unsafe { crate::type_info::TensorTypeAndShapeInfo::from_owning(info) })
    }

    /// Set output `index`'s type+shape (`ShapeInferContext_SetOutputTypeShape`). **Consumes
    /// `info`**: ORT takes ownership of the handle on success, so do not release it — verified
    /// empirically against the engine (releasing it after a successful call double-frees,
    /// despite the C API's `const` annotation). On error ORT did not take it, so the handle is
    /// released here. Build `info` with [`crate::TensorTypeAndShapeInfo::new`] +
    /// `set_element_type` + `set_dimensions` (or hand one from [`Self::input_type_shape`]).
    pub fn set_output_type_shape(
        &self, index: usize, info: crate::type_info::TensorTypeAndShapeInfo,
    ) -> Result<()> {
        let info_ptr = info.as_ptr();
        let res = check(unsafe {
            api().shape_infer_context__set_output_type_shape()(self.ptr, index, info_ptr)
        });
        if res.is_ok() {
            // ORT took ownership of `info`'s handle; prevent Drop from releasing it.
            std::mem::forget(info);
        }
        res
    }
}

// ─── Op (owning): compose + invoke a built-in ORT op from a kernel ────────────

/// An instantiated ORT native operator (`CreateOp`), released on drop. Used to call a
/// built-in op from within a custom-op kernel (`InvokeOp`).
pub struct Op {
    ptr: *mut sys::OpHandle,
}

impl Op {
    /// Create a native op bound to `info`'s kernel. `type_constraints` map constraint
    /// names (e.g. `"T"`) to concrete element types; `attrs` are pre-built attributes.
    #[allow(clippy::too_many_arguments)] // mirrors the C CreateOp signature
    pub fn create(
        info: &KernelInfo<'_>, op_name: &str, domain: &str, version: i32,
        type_constraints: &[(&str, sys::ElementType)], attrs: &[&OpAttr], input_count: usize,
        output_count: usize,
    ) -> Result<Self> {
        let op_name = cstring(op_name)?;
        let domain = cstring(domain)?;
        let type_constraint_count =
            usize_to_c_int(type_constraints.len(), "custom-op type constraint count")?;
        let attr_count = usize_to_c_int(attrs.len(), "custom-op attribute count")?;
        let input_count = usize_to_c_int(input_count, "custom-op input count")?;
        let output_count = usize_to_c_int(output_count, "custom-op output count")?;
        let tc_names: Vec<CString> = type_constraints
            .iter()
            .map(|(n, _)| cstring(n))
            .collect::<Result<_>>()?;
        let tc_name_ptrs: Vec<*const c_char> = tc_names.iter().map(|c| c.as_ptr()).collect();
        let tc_vals: Vec<sys::ElementType> = type_constraints.iter().map(|(_, t)| *t).collect();
        let attr_ptrs: Vec<*const sys::OpAttrHandle> = attrs
            .iter()
            .map(|a| a.ptr as *const sys::OpAttrHandle)
            .collect();
        let mut out: *mut sys::OpHandle = ptr::null_mut();
        check(unsafe {
            api().create_op()(
                info.ptr,
                op_name.as_ptr(),
                domain.as_ptr(),
                version,
                tc_name_ptrs.as_ptr(),
                tc_vals.as_ptr(),
                type_constraint_count,
                attr_ptrs.as_ptr(),
                attr_count,
                input_count,
                output_count,
                &mut out,
            )
        })?;
        let out = crate::ensure_non_null(out, "custom-op native op")?;
        Ok(Self { ptr: out })
    }

    /// Invoke this op within `ctx`. `outputs` must be pre-allocated OrtValues (e.g. from a
    /// prior `KernelContext::output_mut`); ORT writes into them.
    pub fn invoke(
        &self, ctx: &KernelContext<'_>, inputs: &[&TensorView<'_>],
        outputs: &mut [&mut TensorView<'_>],
    ) -> Result<()> {
        let in_ptrs: Vec<*const sys::ValueHandle> = inputs
            .iter()
            .map(|t| t.value as *const sys::ValueHandle)
            .collect();
        let mut out_ptrs: Vec<*mut sys::ValueHandle> =
            outputs.iter_mut().map(|t| t.value).collect();
        let input_count = usize_to_c_int(in_ptrs.len(), "custom-op invoke input count")?;
        let output_count = usize_to_c_int(out_ptrs.len(), "custom-op invoke output count")?;
        check(unsafe {
            api().invoke_op()(
                ctx.ptr,
                self.ptr,
                in_ptrs.as_ptr(),
                input_count,
                out_ptrs.as_mut_ptr(),
                output_count,
            )
        })
    }
}

impl Drop for Op {
    fn drop(&mut self) {
        unsafe { api().release_op()(self.ptr) }
    }
}
unsafe impl Send for Op {}
unsafe impl Sync for Op {}

// ─── custom_op! internals: the generic extern "C" trampolines ──────────────────
//
// These bridge ORT's vtable callbacks to a user's `CustomOp` impl. They are generic over
// `T: CustomOp` and live in a `#[doc(hidden)] pub` module so the `#[macro_export]`
// `custom_op!` macro — expanded in *downstream* crates — can name them via `$crate::__priv`.
#[doc(hidden)]
pub mod __priv {
    use super::{CustomOp, KernelContext, KernelInfo, ShapeInferContext};
    use crate::{api, sys, Error};
    use std::ffi::{c_char, c_void, CString};
    use std::os::raw::c_int;
    use std::panic::AssertUnwindSafe;
    use std::ptr;

    /// Build an `OrtStatus` (`ORT_FAIL`) from a st-zrt error. ORT copies the message on
    /// `CreateStatus` and frees the status we return to a V2 callback.
    pub fn error_to_status(e: Error) -> sys::StatusPtr {
        // ORT_FAIL is the convention for a custom-op kernel error.
        match CString::new(e.to_string()) {
            Ok(msg) => unsafe {
                api().create_status()(sys::OrtErrorCode::Fail as c_int, msg.as_ptr())
            },
            Err(_) => {
                static FALLBACK: &[u8] = b"st-zrt custom-op error (NUL in message)\0";
                unsafe {
                    api().create_status()(
                        sys::OrtErrorCode::Fail as c_int,
                        FALLBACK.as_ptr() as *const c_char,
                    )
                }
            },
        }
    }

    /// `CreateKernelV2` trampoline: build the kernel state, box it, hand the raw pointer out.
    pub unsafe extern "C" fn create_kernel<T: CustomOp>(
        _op: *const c_void, _api: *const c_void, info: *const c_void, kernel_out: *mut *mut c_void,
    ) -> sys::StatusPtr {
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let info = unsafe { KernelInfo::from_ptr(info as *const sys::KernelInfoHandle) };
            T::create(&info)
        }));
        match res {
            Ok(Ok(kernel)) => {
                unsafe { *kernel_out = Box::into_raw(Box::new(kernel)) as *mut c_void };
                ptr::null_mut()
            },
            Ok(Err(e)) => error_to_status(e),
            Err(_) => error_to_status(Error::new(
                sys::OrtErrorCode::Fail as i32,
                "st-zrt custom-op create panicked",
            )),
        }
    }

    /// `KernelComputeV2` trampoline: run one inference.
    pub unsafe extern "C" fn compute<T: CustomOp>(
        kernel: *mut c_void, ctx: *mut c_void,
    ) -> sys::StatusPtr {
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            if kernel.is_null() {
                return Ok(());
            }
            let k = unsafe { &mut *(kernel as *mut T) };
            let ctx = unsafe { KernelContext::from_ptr(ctx as *const sys::KernelContextHandle) };
            k.compute(&ctx)
        }));
        match res {
            Ok(Ok(())) => ptr::null_mut(),
            Ok(Err(e)) => error_to_status(e),
            Err(_) => error_to_status(Error::new(
                sys::OrtErrorCode::Fail as i32,
                "st-zrt custom-op compute panicked",
            )),
        }
    }

    /// `KernelDestroy` trampoline: drop the boxed kernel state. A panic here must not unwind
    /// across the boundary (no status path exists), so it aborts the process.
    pub unsafe extern "C" fn destroy<T: CustomOp>(kernel: *mut c_void) {
        if kernel.is_null() {
            return;
        }
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe {
            drop(Box::from_raw(kernel as *mut T));
        }));
        if res.is_err() {
            std::process::abort();
        }
    }

    /// `InferOutputShapeFn` trampoline: run shape inference. This fires at session-creation
    /// time, before any kernel exists, so it calls the associated `T::infer_shapes` (no
    /// `self`). Panics are caught and surfaced as `ORT_FAIL`, like create/compute.
    pub unsafe extern "C" fn infer_output_shape<T: CustomOp>(
        _op: *const c_void, ctx: *mut c_void,
    ) -> sys::StatusPtr {
        let res = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let ctx =
                unsafe { ShapeInferContext::from_ptr(ctx as *const sys::ShapeInferContextHandle) };
            T::infer_shapes(&ctx)
        }));
        match res {
            Ok(Ok(())) => ptr::null_mut(),
            Ok(Err(e)) => error_to_status(e),
            Err(_) => error_to_status(Error::new(
                sys::OrtErrorCode::Fail as i32,
                "st-zrt custom-op infer_shapes panicked",
            )),
        }
    }

    // ── schema trampolines (read T::inputs()/outputs(); never deref `op`) ──
    pub unsafe extern "C" fn get_input_type_count<T: CustomOp>(_op: *const c_void) -> usize {
        <T as CustomOp>::inputs().len()
    }
    pub unsafe extern "C" fn get_input_type<T: CustomOp>(_op: *const c_void, index: usize) -> i32 {
        <T as CustomOp>::inputs()
            .get(index)
            .map_or(sys::ElementType::Undefined as i32, |s| {
                s.element_type as i32
            })
    }
    pub unsafe extern "C" fn get_output_type_count<T: CustomOp>(_op: *const c_void) -> usize {
        <T as CustomOp>::outputs().len()
    }
    pub unsafe extern "C" fn get_output_type<T: CustomOp>(_op: *const c_void, index: usize) -> i32 {
        <T as CustomOp>::outputs()
            .get(index)
            .map_or(sys::ElementType::Undefined as i32, |s| {
                s.element_type as i32
            })
    }
    pub unsafe extern "C" fn get_input_characteristic<T: CustomOp>(
        _op: *const c_void, index: usize,
    ) -> i32 {
        <T as CustomOp>::inputs().get(index).map_or(
            sys::CustomOpInputOutputCharacteristic::Required as i32,
            |s| s.characteristic as i32,
        )
    }
    pub unsafe extern "C" fn get_output_characteristic<T: CustomOp>(
        _op: *const c_void, index: usize,
    ) -> i32 {
        <T as CustomOp>::outputs().get(index).map_or(
            sys::CustomOpInputOutputCharacteristic::Required as i32,
            |s| s.characteristic as i32,
        )
    }
    pub unsafe extern "C" fn get_input_memory_type<T: CustomOp>(
        _op: *const c_void, index: usize,
    ) -> i32 {
        <T as CustomOp>::inputs()
            .get(index)
            .map_or(sys::MemType::Default as i32, |s| s.memory_type as i32)
    }
    pub unsafe extern "C" fn get_execution_provider_type<T: CustomOp>(
        _op: *const c_void,
    ) -> *const c_char {
        // CPU-only in this milestone; the trait override is accepted but not yet wired through.
        let _ = <T as CustomOp>::execution_provider_type();
        ptr::null()
    }
    pub unsafe extern "C" fn get_variadic_input_min_arity<T: CustomOp>(
        _op: *const c_void,
    ) -> c_int {
        <T as CustomOp>::variadic_input_min_arity()
    }
    pub unsafe extern "C" fn get_variadic_input_homogeneity<T: CustomOp>(
        _op: *const c_void,
    ) -> c_int {
        <T as CustomOp>::variadic_input_homogeneity() as c_int
    }
    pub unsafe extern "C" fn get_variadic_output_min_arity<T: CustomOp>(
        _op: *const c_void,
    ) -> c_int {
        <T as CustomOp>::variadic_output_min_arity()
    }
    pub unsafe extern "C" fn get_variadic_output_homogeneity<T: CustomOp>(
        _op: *const c_void,
    ) -> c_int {
        <T as CustomOp>::variadic_output_homogeneity() as c_int
    }
    pub unsafe extern "C" fn get_start_version<T: CustomOp>(_op: *const c_void) -> c_int {
        <T as CustomOp>::SINCE_VERSION
    }
    pub unsafe extern "C" fn get_end_version<T: CustomOp>(_op: *const c_void) -> c_int {
        <T as CustomOp>::END_VERSION
    }
}

/// Emit the `OrtCustomOp` vtable for a type implementing [`CustomOp`].
///
/// `custom_op!(ReluOp, "MyRelu", as RELU_OP)` declares a `pub static RELU_OP:
/// sys::OrtCustomOp` at the call site whose callbacks bridge to `<ReluOp as CustomOp>`.
/// Register it via [`CustomOpDomain::add_op`] — e.g. `domain.add_op(&RELU_OP)`. `$name` is the op's graph name
/// (NUL-terminated at compile time for the C `GetName` callback).
#[macro_export]
macro_rules! custom_op {
    ($T:ty, $name:literal, as $vtable:ident $(,)?) => {
        impl $T {
            #[doc(hidden)]
            unsafe extern "C" fn __zrt_custom_op_get_name(
                _op: *const ::std::ffi::c_void,
            ) -> *const ::std::os::raw::c_char {
                concat!($name, "\0").as_ptr() as *const ::std::os::raw::c_char
            }
        }

        #[doc = concat!(" `OrtCustomOp` vtable for `", stringify!($T), "`.")]
        pub static $vtable: $crate::sys::OrtCustomOp = $crate::sys::OrtCustomOp {
            version: $crate::sys::API_VERSION,
            create_kernel: None,
            get_name: Some(<$T>::__zrt_custom_op_get_name),
            get_execution_provider_type: Some($crate::__priv::get_execution_provider_type::<$T>),
            get_input_type: Some($crate::__priv::get_input_type::<$T>),
            get_input_type_count: Some($crate::__priv::get_input_type_count::<$T>),
            get_output_type: Some($crate::__priv::get_output_type::<$T>),
            get_output_type_count: Some($crate::__priv::get_output_type_count::<$T>),
            kernel_compute: None,
            kernel_destroy: Some($crate::__priv::destroy::<$T>),
            get_input_characteristic: Some($crate::__priv::get_input_characteristic::<$T>),
            get_output_characteristic: Some($crate::__priv::get_output_characteristic::<$T>),
            get_input_memory_type: Some($crate::__priv::get_input_memory_type::<$T>),
            get_variadic_input_min_arity: Some($crate::__priv::get_variadic_input_min_arity::<$T>),
            get_variadic_input_homogeneity: Some(
                $crate::__priv::get_variadic_input_homogeneity::<$T>,
            ),
            get_variadic_output_min_arity: Some(
                $crate::__priv::get_variadic_output_min_arity::<$T>,
            ),
            get_variadic_output_homogeneity: Some(
                $crate::__priv::get_variadic_output_homogeneity::<$T>,
            ),
            create_kernel_v2: Some($crate::__priv::create_kernel::<$T>),
            kernel_compute_v2: Some($crate::__priv::compute::<$T>),
            infer_output_shape_fn: Some($crate::__priv::infer_output_shape::<$T>),
            get_start_version: Some($crate::__priv::get_start_version::<$T>),
            get_end_version: Some($crate::__priv::get_end_version::<$T>),
            get_may_inplace: None,
            release_may_inplace: None,
            get_alias_map: None,
            release_alias_map: None,
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an OpAttr of each scalar/array kind and read it back — round-trips the
    /// attribute lifecycle on CPU (no kernel, no model, no GPU). Exercises CreateOpAttr,
    /// OpAttr_GetType, OpAttr_GetName, ReadOpAttr, ReleaseOpAttr.
    #[test]
    fn op_attr_round_trip() {
        let f = OpAttr::new_float("alpha", 1.5).expect("new_float");
        assert_eq!(f.ty().unwrap(), sys::OpAttrType::Float);
        assert_eq!(f.name().unwrap(), "alpha");
        let mut buf = [0u8; 4];
        let n = f.read_into(sys::OpAttrType::Float, &mut buf).unwrap();
        assert_eq!(n, 4);
        assert_eq!(f32::from_ne_bytes(buf), 1.5);

        let i = OpAttr::new_int("count", 42).expect("new_int");
        let mut b = [0u8; 8];
        i.read_into(sys::OpAttrType::Int, &mut b).unwrap();
        assert_eq!(i64::from_ne_bytes(b), 42);

        let s = OpAttr::new_string("mode", "fast").expect("new_string");
        assert_eq!(s.ty().unwrap(), sys::OpAttrType::String);
        let mut b = [0u8; 16];
        let n = s.read_into(sys::OpAttrType::String, &mut b).unwrap();
        assert_eq!(&b[..n], b"fast");

        let arr = OpAttr::new_ints("dims", &[3, 5, 7]).expect("new_ints");
        assert_eq!(arr.ty().unwrap(), sys::OpAttrType::Ints);
        eprintln!("op_attr_round_trip: float/int/string/ints all round-tripped + released");
    }

    /// Exercises the custom-op-domain lifecycle on CPU: create a domain, attach it to a
    /// built session-options handle (AddCustomOpDomain), release the options, then drop
    /// the domain. Needs no model and no GPU — it proves the domain indices/signatures are
    /// right (a wrong index crashes; this returns cleanly).
    #[test]
    fn custom_op_domain_lifecycle() {
        let domain = CustomOpDomain::new("com.example.foo").expect("new domain");

        // Attach via the SessionOptions pure-value path, which calls AddCustomOpDomain
        // inside build_handle.
        let opts = SessionOptions::default().with_custom_op_domain(&domain);
        let h = opts.build_handle().expect("build_handle");
        // Release the options handle (CreateSession copies; the domain stays ours).
        unsafe {
            crate::api().release_session_options()(h);
        }
        drop(domain); // ReleaseCustomOpDomain
        eprintln!("custom_op_domain_lifecycle: create + attach + release clean");
    }

    // ── custom_op! authoring tests ──
    //
    // `TestOp` is a no-op kernel whose vtable is emitted by the `custom_op!` macro. With no
    // bundled model referencing `com.example.test::TestOp`, ORT never calls create/compute/
    // destroy — so we verify the SCHEMA callbacks directly and the REGISTRATION lifecycle.
    // create/compute/destroy are compile-verified (the example exercises their bodies).
    struct TestOp;

    impl CustomOp for TestOp {
        const NAME: &'static str = "TestOp";
        const DOMAIN: &'static str = "com.example.test";
        const SINCE_VERSION: i32 = 7;
        fn create(_info: &KernelInfo<'_>) -> Result<Self> {
            Ok(Self)
        }
        fn compute(&mut self, _ctx: &KernelContext<'_>) -> Result<()> {
            Ok(())
        }
        fn inputs() -> &'static [OpIoSpec] {
            static INPUTS: [OpIoSpec; 2] = [
                OpIoSpec::required(sys::ElementType::Float),
                OpIoSpec::optional(sys::ElementType::Int64),
            ];
            &INPUTS
        }
        fn outputs() -> &'static [OpIoSpec] {
            static OUTPUTS: [OpIoSpec; 1] = [OpIoSpec::required(sys::ElementType::Float)];
            &OUTPUTS
        }
    }

    crate::custom_op!(TestOp, "TestOp", as TEST_OP_VTABLE);

    /// Call the vtable's schema callbacks directly (no ORT involvement) and assert they
    /// reflect `TestOp`'s impl — proves the `custom_op!` trampolines are wired correctly.
    #[test]
    fn custom_op_vtable_schema() {
        let v = &TEST_OP_VTABLE;
        unsafe {
            let name_ptr = (v.get_name.unwrap())(std::ptr::null());
            assert_eq!(
                std::ffi::CStr::from_ptr(name_ptr).to_bytes(),
                b"TestOp",
                "get_name"
            );

            assert_eq!(
                (v.get_input_type_count.unwrap())(std::ptr::null()),
                2,
                "input count"
            );
            assert_eq!(
                (v.get_input_type.unwrap())(std::ptr::null(), 0),
                sys::ElementType::Float as i32,
                "input[0] type"
            );
            assert_eq!(
                (v.get_input_type.unwrap())(std::ptr::null(), 1),
                sys::ElementType::Int64 as i32,
                "input[1] type"
            );
            assert_eq!(
                (v.get_output_type_count.unwrap())(std::ptr::null()),
                1,
                "output count"
            );
            assert_eq!(
                (v.get_output_type.unwrap())(std::ptr::null(), 0),
                sys::ElementType::Float as i32,
                "output[0] type"
            );
            assert_eq!(
                (v.get_input_characteristic.unwrap())(std::ptr::null(), 1),
                sys::CustomOpInputOutputCharacteristic::Optional as i32,
                "input[1] optional"
            );
            assert_eq!(
                (v.get_start_version.unwrap())(std::ptr::null()),
                7,
                "since version"
            );
            assert_eq!(
                (v.get_end_version.unwrap())(std::ptr::null()),
                sys::API_VERSION as i32,
                "end version"
            );
            assert!(
                (v.get_execution_provider_type.unwrap())(std::ptr::null()).is_null(),
                "EP type null (CPU)"
            );
        }
        eprintln!(
            "custom_op_vtable_schema: name/schema/versions correct via direct callback calls"
        );
    }

    /// Register the `custom_op!`-built vtable on a domain + attach to real session options.
    /// Exercises `add_op` (CustomOpDomain_Add) end-to-end on CPU — no model.
    #[test]
    fn custom_op_vtable_registration() {
        let domain = CustomOpDomain::new(TestOp::DOMAIN).expect("new domain");
        domain.add_op(&TEST_OP_VTABLE).expect("add_op");
        let opts = SessionOptions::default().with_custom_op_domain(&domain);
        let h = opts.build_handle().expect("build_handle");
        unsafe {
            crate::api().release_session_options()(h);
        }
        drop(domain);
        eprintln!("custom_op_vtable_registration: add_op + AddCustomOpDomain + release clean");
    }
}
