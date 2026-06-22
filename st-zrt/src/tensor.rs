//! Tensor value types: numeric session inputs, owning string inputs, reusable zero-copy buffers,
//! allocator-owned tensors, and engine-owned output values.
use crate::allocator::Allocator;
use crate::element::TensorElement;
use crate::error::Error;
use crate::memory::MemoryInfo;
use crate::type_info::{TensorTypeAndShapeInfo, checked_element_count, tensor_type_and_shape};
use crate::{Result, api, check, packed_element_bits, sys, tensor_byte_len};
use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::File;
use std::marker::PhantomData;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::Path;
use std::ptr;
use std::ptr::NonNull;

// ─── inputs: anything passable to Session::run ──────────────────────────────

mod input_sealed {
    pub trait Sealed {}
    impl Sealed for super::Tensor<'_> {}
    impl<T: super::TensorElement> Sealed for super::AllocatedTensor<T> {}
    impl<T: super::TensorElement> Sealed for super::SparseTensor<'_, T> {}
    impl<T: super::TensorElement> Sealed for super::TensorBuffer<T> {}
    impl Sealed for super::StringTensor {}
}

/// A value passable to `Session::run` as an input. Sealed — the only implementors are
/// [`Tensor`], [`TensorBuffer`], [`AllocatedTensor`], [`SparseTensor`], and [`StringTensor`].
/// Downstream code receives it as `&dyn RunInput`.
pub trait RunInput: input_sealed::Sealed {
    #[doc(hidden)]
    fn as_value_ptr(&self) -> *const sys::ValueHandle;
}

// ─── borrowed tensor view (read-only; never releases its handle) ─────────────

/// A borrowed, read-only view over an ORT tensor value handle. **Never releases the
/// handle** — it is owned elsewhere: an owning [`Tensor`], an [`AllocatedTensor`], or ORT's
/// kernel context (a custom-op input via `KernelContext::input` when the `custom-ops` feature is
/// enabled). All tensor read accessors live here.
///
/// For a session input you own, build a [`Tensor`] and read through the `Deref`-forwarded
/// methods here (or [`Tensor::as_view`]); a custom-op kernel receives one of these directly
/// from `KernelContext::input` and must not release it.
pub struct TensorView<'a> {
    pub(crate) value: *mut sys::ValueHandle,
    pub(crate) _life: PhantomData<&'a [u8]>,
}

impl<'a> TensorView<'a> {
    /// Full type+shape introspection (engine-owned handle; released when dropped).
    pub fn tensor_type_and_shape(&self) -> Result<TensorTypeAndShapeInfo> {
        tensor_type_and_shape(self.value as *const sys::ValueHandle)
    }

    /// Element type of the tensor.
    pub fn element_type(&self) -> Result<sys::ElementType> {
        self.tensor_type_and_shape()?.element_type()
    }

    /// Element count (product of the dimensions).
    pub fn element_count(&self) -> Result<usize> {
        self.tensor_type_and_shape()?.element_count()
    }

    /// Total numeric tensor backing-buffer size in bytes.
    ///
    /// Uses ORT's `GetTensorSizeInBytes`, so packed sub-byte tensor storage is measured by the
    /// engine. ORT returns an error for string tensors and non-tensor values.
    pub fn byte_len(&self) -> Result<usize> {
        tensor_value_byte_len(self.value as *const sys::ValueHandle)
    }

    /// Dimensions of the tensor.
    pub fn dims(&self) -> Result<Vec<i64>> {
        self.tensor_type_and_shape()?.dims()
    }

    /// Zero-copy typed read of the backing buffer via `GetTensorMutableData`. Returns `Err`
    /// when the tensor's element type does not match `T` or the value is backed by
    /// device-only memory. Works for any
    /// host-accessible `ValueHandle` the view wraps — a kernel input, an engine-owned CPU tensor
    /// (`copy_from_slice`), or ORT's zero-copy wrapper of a `from_buffer` buffer.
    pub fn as_slice<T: TensorElement>(&self) -> Result<&[T]> {
        let tsi = self.tensor_type_and_shape()?;
        let elem = tsi.element_type()?;
        if elem as i32 != T::ELEM as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: as_slice<{}> on a {:?} tensor",
                    std::any::type_name::<T>(),
                    elem
                ),
            ));
        }
        let count = tsi.element_count()?;
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut T, count, "tensor data")?;
        // SAFETY: `data` is a contiguous, aligned buffer of `count` elements of T, owned by
        // the value and live for at least the lifetime of this borrow.
        Ok(unsafe { std::slice::from_raw_parts(data as *const T, count) })
    }

    /// Zero-copy read as raw bytes. For packed 2-bit/4-bit tensors this returns the packed
    /// backing storage; use [`Self::element_type`] to interpret the bit layout.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        let n = self.byte_len()?;
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut u8, n, "tensor data")?;
        Ok(unsafe { std::slice::from_raw_parts(data as *const u8, n) })
    }
}

unsafe impl Send for TensorView<'_> {}
unsafe impl Sync for TensorView<'_> {}

// ─── owning tensor: a session input (releases its handle on drop) ────────────

/// An owning tensor value for session inputs — owns its `OrtValue` handle (released on
/// drop). Construct zero-copy from a caller buffer ([`Tensor::from_buffer`]) or as a copied,
/// ORT-default-allocator tensor ([`Tensor::copy_from_slice`]); pass it to
/// [`crate::Session::run`]. Read it through the `Deref`-forwarded [`TensorView`] accessors or
/// [`Tensor::as_view`].
pub struct Tensor<'a> {
    view: TensorView<'a>,
}

impl<'a> Tensor<'a> {
    /// Wrap `buf` as a zero-copy tensor of the given `shape`. No allocation, no copy.
    /// The engine reads directly from `buf`; it does NOT copy it and does NOT free it.
    /// `buf` must outlive every use of this tensor.
    pub fn from_buffer<T: TensorElement>(
        buf: &'a [T], shape: &[i64], mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_shape_len(shape, buf.len())?;
        ensure_memory_host_accessible(mem)?;
        let bytes = std::mem::size_of_val(buf);
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_with_data_as_ort_value()(
                mem.info as *const sys::MemoryInfoHandle,
                buf.as_ptr() as *mut c_void,
                bytes,
                shape.as_ptr(),
                shape.len(),
                T::ELEM,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "tensor value")?;
        Ok(Self {
            view: TensorView {
                value,
                _life: PhantomData,
            },
        })
    }

    /// Wrap an already-packed sub-byte tensor as a zero-copy ORT value.
    ///
    /// `elem_type` must be one of `Uint4`, `Int4`, `Float4E2M1`, `Uint2`, or `Int2`.
    /// `buf` length is validated against `ceil(product(shape) * bits_per_element / 8)`.
    /// The bit/nibble value layout is ORT/ONNX packed storage; ZRT intentionally exposes it as
    /// bytes instead of pretending each logical element is a Rust scalar.
    /// ORT may still reject creation for packed metadata types it can report but not wrap.
    pub fn from_packed_bytes(
        buf: &'a [u8], shape: &[i64], elem_type: sys::ElementType, mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_packed_bytes_len(shape, elem_type, buf.len())?;
        ensure_memory_host_accessible(mem)?;
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_with_data_as_ort_value()(
                mem.info as *const sys::MemoryInfoHandle,
                buf.as_ptr() as *mut c_void,
                buf.len(),
                shape.as_ptr(),
                shape.len(),
                elem_type,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "packed tensor value")?;
        Ok(Self {
            view: TensorView {
                value,
                _life: PhantomData,
            },
        })
    }

    /// Create an engine-allocated tensor and copy `buf` into it (`CreateTensorAsOrtValue` +
    /// fill via `GetTensorMutableData`). NOT zero-copy — the engine owns the buffer — so it
    /// sidesteps the alignment/arena caveats of external buffers (and is what the `ort` crate
    /// does by default). The returned tensor borrows nothing.
    pub fn copy_from_slice<T: TensorElement>(buf: &[T], shape: &[i64]) -> Result<Self> {
        validate_shape_len(shape, buf.len())?;
        let alloc = Allocator::get_default()?;
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_as_ort_value()(
                alloc.alloc,
                shape.as_ptr(),
                shape.len(),
                T::ELEM,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "tensor value")?;
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut T, buf.len(), "tensor data")?;
        // The fresh tensor's buffer is uninitialized; copy buf in (engine-aligned).
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), data, buf.len()) };
        Ok(Self {
            view: TensorView {
                value,
                _life: PhantomData,
            },
        })
    }

    /// Borrow this owning tensor as a read-only [`TensorView`].
    pub fn as_view(&self) -> &TensorView<'a> {
        &self.view
    }
}

impl<'a> std::ops::Deref for Tensor<'a> {
    type Target = TensorView<'a>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl RunInput for Tensor<'_> {
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.view.value as *const sys::ValueHandle
    }
}

impl Drop for Tensor<'_> {
    fn drop(&mut self) {
        // For `from_buffer`, this releases only the wrapper handle. For `copy_from_slice`,
        // ORT also releases the allocator-owned backing storage.
        unsafe { api().release_value()(self.view.value) }
    }
}
unsafe impl Send for Tensor<'_> {}
unsafe impl Sync for Tensor<'_> {}

// ─── allocator-owned tensor: CPU or device memory owned by ORT ──────────────

/// An allocator-owned tensor value.
///
/// Unlike [`TensorBuffer`], this does not use Rust `Vec` storage. ORT allocates the tensor backing
/// memory through an [`Allocator`], so the memory can live on CPU or on an execution-provider
/// device such as CUDA. Host-accessible tensors can be read through [`Self::as_slice`]; device
/// tensors expose only raw pointers and metadata.
pub struct AllocatedTensor<T: TensorElement> {
    value: *mut sys::ValueHandle,
    allocator: Allocator,
    shape: Vec<i64>,
    count: usize,
    elem_type: sys::ElementType,
    _ty: PhantomData<T>,
}

impl<T: TensorElement> AllocatedTensor<T> {
    /// Allocate a tensor with `allocator` and `shape` via ORT `CreateTensorAsOrtValue`.
    pub fn new(allocator: Allocator, shape: &[i64]) -> Result<Self> {
        let count = shape_element_count(shape)?;
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_as_ort_value()(
                allocator.alloc,
                shape.as_ptr(),
                shape.len(),
                T::ELEM,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "allocated tensor value")?;
        Ok(Self {
            value,
            allocator,
            shape: shape.to_vec(),
            count,
            elem_type: T::ELEM,
            _ty: PhantomData,
        })
    }

    /// Allocate a tensor from a session-scoped allocator for `mem`.
    pub fn for_session(session: &crate::Session, mem: &MemoryInfo, shape: &[i64]) -> Result<Self> {
        Self::new(Allocator::create(session, mem)?, shape)
    }

    /// Allocate a CUDA device tensor from a session-scoped CUDA allocator.
    #[cfg(feature = "cuda")]
    pub fn cuda(session: &crate::Session, device_id: i32, shape: &[i64]) -> Result<Self> {
        let mem = MemoryInfo::cuda(device_id)?;
        Self::for_session(session, &mem, shape)
    }

    /// Copy host data into an allocator-owned host-accessible tensor.
    ///
    /// This intentionally returns an error for device-only allocations; use [`Self::raw_mut_ptr`]
    /// with an EP/runtime-specific copy operation for CUDA device tensors.
    pub fn copy_from_slice(allocator: Allocator, shape: &[i64], data: &[T]) -> Result<Self> {
        validate_shape_len(shape, data.len())?;
        let mut tensor = Self::new(allocator, shape)?;
        let dst = tensor.as_mut_slice()?;
        dst.copy_from_slice(data);
        Ok(tensor)
    }

    #[inline]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn element_type(&self) -> sys::ElementType {
        self.elem_type
    }

    pub fn byte_len(&self) -> Result<usize> {
        self.count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::new(-1, "allocated tensor byte length overflows usize"))
    }

    /// Memory descriptor for the tensor backing allocation.
    pub fn memory_info(&self) -> Result<crate::memory::MemoryInfoSnapshot> {
        tensor_memory_info(self.value as *const sys::ValueHandle)
    }

    /// Raw backing pointer returned by ORT. For CUDA tensors this is a device pointer.
    pub fn raw_mut_ptr(&self) -> Result<*mut c_void> {
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        Ok(crate::slice_data_ptr(data as *mut u8, self.byte_len()?, "tensor data")? as *mut c_void)
    }

    /// Raw typed backing pointer. For CUDA tensors this is a device pointer.
    pub fn raw_typed_ptr(&self) -> Result<*mut T> {
        Ok(self.raw_mut_ptr()? as *mut T)
    }

    /// Host-accessible read.
    pub fn as_slice(&self) -> Result<&[T]> {
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let data = self.raw_typed_ptr()?;
        Ok(unsafe { std::slice::from_raw_parts(data as *const T, self.count) })
    }

    /// Host-accessible mutable read/write.
    pub fn as_mut_slice(&mut self) -> Result<&mut [T]> {
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let data = self.raw_typed_ptr()?;
        Ok(unsafe { std::slice::from_raw_parts_mut(data, self.count) })
    }

    /// Borrow this value as a tensor view for type/shape introspection.
    pub fn as_view(&self) -> TensorView<'_> {
        TensorView {
            value: self.value,
            _life: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }

    #[inline]
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
}

impl<T: TensorElement> RunInput for AllocatedTensor<T> {
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }
}

impl<T: TensorElement> Drop for AllocatedTensor<T> {
    fn drop(&mut self) {
        unsafe { api().release_value()(self.value) }
    }
}

unsafe impl<T: TensorElement + Send> Send for AllocatedTensor<T> {}
unsafe impl<T: TensorElement + Sync> Sync for AllocatedTensor<T> {}

// ─── reusable owned zero-copy tensor buffer ─────────────────────────────────

/// An owned, reusable tensor buffer backed by caller memory.
///
/// `TensorBuffer` owns stable backing storage (`Vec`, aligned allocation, or dense mmap) and wraps
/// it in a stable ORT value via `CreateTensorWithDataAsOrtValue`. Mutate the slice between runs and
/// reuse the same value handle in a prepared binding. This is the building block for lane-local
/// input/output buffers and external dense initializers: no per-request allocation, no copy, no
/// rebind.
pub struct TensorBuffer<T: TensorElement> {
    value: *mut sys::ValueHandle,
    data: TensorStorage<T>,
    shape: Vec<i64>,
    elem_type: sys::ElementType,
}

/// Options for mapping dense tensor bytes from a file.
///
/// This is for already-dense typed tensor data. ORT will dereference the mapped memory as `T`
/// directly; compressed formats cannot be decoded lazily through this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmapTensorOptions {
    /// Byte offset of the dense tensor payload inside the file.
    pub byte_offset: u64,
    /// Apply `MADV_SEQUENTIAL` on Linux.
    pub sequential: bool,
    /// Apply `MADV_HUGEPAGE` on Linux. This is a transparent-hugepage hint, not `MAP_HUGETLB`.
    pub hugepage: bool,
    /// Lock the mapped pages in RAM. On Linux this calls `mlock` and returns an error if the OS
    /// refuses it, commonly due to `RLIMIT_MEMLOCK`.
    pub locked: bool,
}

impl Default for MmapTensorOptions {
    fn default() -> Self {
        Self {
            byte_offset: 0,
            sequential: true,
            hugepage: false,
            locked: false,
        }
    }
}

enum TensorStorage<T: TensorElement> {
    Vec(Vec<T>),
    Aligned(AlignedBuffer<T>),
    Mmap(MmapBuffer<T>),
}

impl<T: TensorElement> TensorStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        match self {
            Self::Vec(v) => v.as_slice(),
            Self::Aligned(v) => v.as_slice(),
            Self::Mmap(v) => v.as_slice(),
        }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Vec(v) => v.as_mut_slice(),
            Self::Aligned(v) => v.as_mut_slice(),
            Self::Mmap(v) => v.as_mut_slice(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_slice().len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct AlignedBuffer<T: TensorElement> {
    ptr: NonNull<T>,
    len: usize,
    layout: Option<Layout>,
    locked_bytes: usize,
}

impl<T: TensorElement> AlignedBuffer<T> {
    fn zeroed(len: usize, alignment: usize) -> Result<Self> {
        let min_align = std::mem::align_of::<T>();
        let alignment = alignment.max(min_align);
        if !alignment.is_power_of_two() {
            return Err(Error::new(
                -1,
                format!("alignment must be a power of two, got {alignment}"),
            ));
        }
        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                layout: None,
                locked_bytes: 0,
            });
        }
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::new(-1, "aligned tensor byte size overflows usize"))?;
        let layout = Layout::from_size_align(bytes, alignment)
            .map_err(|_| Error::new(-1, "invalid aligned tensor layout"))?;
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).unwrap_or_else(|| handle_alloc_error(layout));
        Ok(Self {
            ptr,
            len,
            layout: Some(layout),
            locked_bytes: 0,
        })
    }

    fn lock_pages(&mut self) -> Result<()> {
        let Some(layout) = self.layout else {
            return Ok(());
        };
        lock_pages_raw(self.ptr.as_ptr().cast(), layout.size())?;
        self.locked_bytes = layout.size();
        Ok(())
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: TensorElement> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            if self.locked_bytes != 0 {
                unlock_pages_raw(self.ptr.as_ptr().cast(), self.locked_bytes);
            }
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout) }
        }
    }
}

struct MmapBuffer<T: TensorElement> {
    mapping_ptr: NonNull<u8>,
    mapping_len: usize,
    data_ptr: NonNull<T>,
    len: usize,
    locked: bool,
}

impl<T: TensorElement> MmapBuffer<T> {
    fn map_file(path: &Path, len: usize, options: MmapTensorOptions) -> Result<Self> {
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::new(-1, "mmap tensor byte size overflows usize"))?;
        let file = File::open(path).map_err(|err| {
            Error::new(
                -1,
                format!("failed to open mmap tensor file {}: {err}", path.display()),
            )
        })?;
        let file_len = file
            .metadata()
            .map_err(|err| {
                Error::new(
                    -1,
                    format!("failed to stat mmap tensor file {}: {err}", path.display()),
                )
            })?
            .len();
        let end = options
            .byte_offset
            .checked_add(bytes as u64)
            .ok_or_else(|| Error::new(-1, "mmap tensor file range overflows u64"))?;
        if end > file_len {
            return Err(Error::new(
                -1,
                format!(
                    "mmap tensor range [{}..{}) exceeds file size {} for {}",
                    options.byte_offset,
                    end,
                    file_len,
                    path.display()
                ),
            ));
        }
        if bytes == 0 {
            return Ok(Self {
                mapping_ptr: NonNull::dangling(),
                mapping_len: 0,
                data_ptr: NonNull::dangling(),
                len,
                locked: false,
            });
        }

        let mapping = map_file_range(&file, options.byte_offset, bytes)?;
        let data_addr = mapping.ptr.as_ptr().wrapping_add(mapping.data_offset) as usize;
        let align = std::mem::align_of::<T>();
        if data_addr % align != 0 {
            unsafe { unmap_raw(mapping.ptr.as_ptr(), mapping.len) };
            return Err(Error::new(
                -1,
                format!(
                    "mmap tensor data pointer is not aligned for {}",
                    std::any::type_name::<T>()
                ),
            ));
        }

        let mut out = Self {
            mapping_ptr: mapping.ptr,
            mapping_len: mapping.len,
            data_ptr: NonNull::new(data_addr as *mut T)
                .ok_or_else(|| Error::new(-1, "mmap tensor data pointer is null"))?,
            len,
            locked: false,
        };
        if options.sequential {
            advise_sequential_raw(out.mapping_ptr.as_ptr(), out.mapping_len);
        }
        if options.hugepage {
            advise_hugepage_raw(out.mapping_ptr.as_ptr(), out.mapping_len);
        }
        if options.locked {
            lock_pages_raw(out.mapping_ptr.as_ptr(), out.mapping_len)?;
            out.locked = true;
        }
        Ok(out)
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr.as_ptr(), self.len) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr.as_ptr(), self.len) }
    }
}

impl<T: TensorElement> Drop for MmapBuffer<T> {
    fn drop(&mut self) {
        if self.mapping_len == 0 {
            return;
        }
        if self.locked {
            unlock_pages_raw(self.mapping_ptr.as_ptr(), self.mapping_len);
        }
        unsafe { unmap_raw(self.mapping_ptr.as_ptr(), self.mapping_len) };
    }
}

impl<T: TensorElement> TensorBuffer<T> {
    /// Create from an existing vector. `data.len()` must equal the product of `shape`.
    pub fn from_vec(data: Vec<T>, shape: &[i64], mem: &MemoryInfo) -> Result<Self> {
        Self::from_storage(TensorStorage::Vec(data), shape, mem)
    }

    /// Create a tensor buffer backed by dense typed bytes from a memory-mapped file.
    ///
    /// The file data must already be laid out as contiguous native-endian `T` values matching
    /// `shape`. ORT sees the mapped bytes directly through `CreateTensorWithDataAsOrtValue`; no
    /// read/copy into a `Vec` is performed. This cannot be used for compressed weights because ORT
    /// has no C API hook to decode data on element access.
    pub fn from_mmap_file<P: AsRef<Path>>(
        path: P, shape: &[i64], mem: &MemoryInfo,
    ) -> Result<Self> {
        Self::from_mmap_file_with_options(path, shape, mem, MmapTensorOptions::default())
    }

    /// Create a memory-mapped tensor buffer with explicit mmap/advice options.
    pub fn from_mmap_file_with_options<P: AsRef<Path>>(
        path: P, shape: &[i64], mem: &MemoryInfo, options: MmapTensorOptions,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let data = TensorStorage::Mmap(MmapBuffer::map_file(path.as_ref(), len, options)?);
        Self::from_storage(data, shape, mem)
    }

    fn from_storage(data: TensorStorage<T>, shape: &[i64], mem: &MemoryInfo) -> Result<Self> {
        validate_shape_len(shape, data.len())?;
        ensure_memory_host_accessible(mem)?;
        let bytes = std::mem::size_of_val(data.as_slice());
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_with_data_as_ort_value()(
                mem.info as *const sys::MemoryInfoHandle,
                data.as_ptr() as *mut c_void,
                bytes,
                shape.as_ptr(),
                shape.len(),
                T::ELEM,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "tensor buffer value")?;
        Ok(Self {
            value,
            data,
            shape: shape.to_vec(),
            elem_type: T::ELEM,
        })
    }

    /// Create a zero-initialized reusable tensor buffer.
    pub fn zeros(shape: &[i64], mem: &MemoryInfo) -> Result<Self>
    where
        T: Clone + Default,
    {
        let len = shape_element_count(shape)?;
        Self::from_vec(vec![T::default(); len], shape, mem)
    }

    /// Create a zero-initialized reusable tensor buffer and touch one element per page
    /// before binding it to ORT. This moves first-touch page faults out of the request path.
    pub fn zeros_prefaulted(shape: &[i64], mem: &MemoryInfo) -> Result<Self>
    where
        T: Clone + Default,
    {
        let len = shape_element_count(shape)?;
        let mut data = vec![T::default(); len];
        prefault_slice(&mut data);
        Self::from_vec(data, shape, mem)
    }

    /// Create a zero-initialized reusable tensor buffer with an explicit byte alignment.
    /// `alignment` must be a power of two and at least `align_of::<T>()`.
    pub fn zeros_aligned(shape: &[i64], alignment: usize, mem: &MemoryInfo) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let data = TensorStorage::Aligned(AlignedBuffer::zeroed(len, alignment)?);
        Self::from_storage(data, shape, mem)
    }

    /// Create an aligned reusable tensor buffer and prefault it before binding to ORT.
    pub fn zeros_aligned_prefaulted(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = TensorStorage::Aligned(AlignedBuffer::zeroed(len, alignment)?);
        prefault_slice(data.as_mut_slice());
        Self::from_storage(data, shape, mem)
    }

    /// Create an aligned reusable tensor buffer and lock its pages in RAM where supported.
    ///
    /// On Linux this calls `mlock` and returns an error if the kernel rejects the request
    /// (commonly due to `RLIMIT_MEMLOCK`). On other platforms this is currently a no-op.
    pub fn zeros_aligned_mlocked(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = AlignedBuffer::zeroed(len, alignment)?;
        data.lock_pages()?;
        Self::from_storage(TensorStorage::Aligned(data), shape, mem)
    }

    /// Create an aligned reusable tensor buffer, prefault it, and lock its pages in RAM.
    pub fn zeros_aligned_mlocked_prefaulted(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = AlignedBuffer::zeroed(len, alignment)?;
        prefault_slice(data.as_mut_slice());
        data.lock_pages()?;
        Self::from_storage(TensorStorage::Aligned(data), shape, mem)
    }

    /// Create an aligned reusable tensor buffer and apply a best-effort hugepage hint before
    /// binding it to ORT.
    pub fn zeros_aligned_hugepage(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let data = TensorStorage::Aligned(AlignedBuffer::zeroed(len, alignment)?);
        advise_hugepage(data.as_slice());
        Self::from_storage(data, shape, mem)
    }

    /// Create an aligned reusable tensor buffer, apply a best-effort hugepage hint, and prefault
    /// it before binding to ORT.
    pub fn zeros_aligned_hugepage_prefaulted(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = TensorStorage::Aligned(AlignedBuffer::zeroed(len, alignment)?);
        advise_hugepage(data.as_slice());
        prefault_slice(data.as_mut_slice());
        Self::from_storage(data, shape, mem)
    }

    /// Create an aligned reusable tensor buffer, apply a hugepage hint, and lock its pages in RAM.
    pub fn zeros_aligned_hugepage_mlocked(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = AlignedBuffer::zeroed(len, alignment)?;
        advise_hugepage(data.as_slice());
        data.lock_pages()?;
        Self::from_storage(TensorStorage::Aligned(data), shape, mem)
    }

    /// Create an aligned reusable tensor buffer, apply a hugepage hint, prefault it, and lock
    /// its pages in RAM.
    pub fn zeros_aligned_hugepage_mlocked_prefaulted(
        shape: &[i64], alignment: usize, mem: &MemoryInfo,
    ) -> Result<Self> {
        let len = shape_element_count(shape)?;
        let mut data = AlignedBuffer::zeroed(len, alignment)?;
        advise_hugepage(data.as_slice());
        prefault_slice(data.as_mut_slice());
        data.lock_pages()?;
        Self::from_storage(TensorStorage::Aligned(data), shape, mem)
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    #[inline]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn byte_len(&self) -> Result<usize> {
        self.data
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::new(-1, "tensor buffer byte length overflows usize"))
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn element_type(&self) -> sys::ElementType {
        self.elem_type
    }

    /// Pointer ORT sees for this tensor's backing storage (`GetTensorMutableData`).
    /// For a correctly wrapped zero-copy buffer this is identical to `as_slice().as_ptr()`.
    pub fn engine_data_ptr(&self) -> Result<*const T> {
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut T, self.len(), "tensor buffer data")?;
        Ok(data as *const T)
    }

    /// Memory descriptor for this wrapped tensor buffer.
    pub fn memory_info(&self) -> Result<crate::memory::MemoryInfoSnapshot> {
        tensor_memory_info(self.value as *const sys::ValueHandle)
    }

    #[inline]
    pub(crate) fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }
}

fn prefault_slice<T: TensorElement>(slice: &mut [T]) {
    if slice.is_empty() {
        return;
    }
    let elem_size = std::mem::size_of::<T>().max(1);
    let stride = (4096 / elem_size).max(1);
    let ptr = slice.as_mut_ptr();
    for i in (0..slice.len()).step_by(stride) {
        unsafe {
            let p = ptr.add(i);
            let value = ptr::read_volatile(p);
            ptr::write_volatile(p, value);
        }
    }
    let last = slice.len() - 1;
    unsafe {
        let p = ptr.add(last);
        let value = ptr::read_volatile(p);
        ptr::write_volatile(p, value);
    }
}

fn advise_hugepage<T: TensorElement>(slice: &[T]) {
    if slice.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(slice);
    if bytes == 0 {
        return;
    }
    advise_hugepage_raw(slice.as_ptr().cast(), bytes);
}

#[cfg(target_os = "linux")]
fn advise_hugepage_raw(ptr: *const u8, bytes: usize) {
    const MADV_HUGEPAGE: i32 = 14;
    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }
    // Best-effort hint: kernels without THP, disabled THP, or unsupported mappings may reject it.
    let _ = unsafe { madvise(ptr as *mut c_void, bytes, MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
fn advise_hugepage_raw(_ptr: *const u8, _bytes: usize) {}

#[cfg(target_os = "linux")]
struct MappedRange {
    ptr: NonNull<u8>,
    len: usize,
    data_offset: usize,
}

#[cfg(target_os = "linux")]
fn map_file_range(file: &File, byte_offset: u64, bytes: usize) -> Result<MappedRange> {
    const PROT_READ: c_int = 0x1;
    const PROT_WRITE: c_int = 0x2;
    const MAP_PRIVATE: c_int = 0x02;
    const MAP_FAILED: *mut c_void = !0usize as *mut c_void;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64,
        ) -> *mut c_void;
    }

    let page_size = page_size()?;
    let page_mask = (page_size as u64) - 1;
    let map_offset = byte_offset & !page_mask;
    let data_offset = (byte_offset - map_offset) as usize;
    let map_len = data_offset
        .checked_add(bytes)
        .ok_or_else(|| Error::new(-1, "mmap length overflows usize"))?;
    let ptr = unsafe {
        mmap(
            ptr::null_mut(),
            map_len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE,
            file.as_raw_fd(),
            map_offset as i64,
        )
    };
    if ptr == MAP_FAILED {
        return Err(Error::new(
            -1,
            format!(
                "mmap failed for {map_len} bytes at offset {map_offset}: {}",
                std::io::Error::last_os_error()
            ),
        ));
    }
    let ptr = NonNull::new(ptr as *mut u8)
        .ok_or_else(|| Error::new(-1, "mmap returned a null pointer"))?;
    Ok(MappedRange {
        ptr,
        len: map_len,
        data_offset,
    })
}

#[cfg(not(target_os = "linux"))]
struct MappedRange {
    ptr: NonNull<u8>,
    len: usize,
    data_offset: usize,
}

#[cfg(not(target_os = "linux"))]
fn map_file_range(_file: &File, _byte_offset: u64, _bytes: usize) -> Result<MappedRange> {
    Err(Error::new(
        -1,
        "mmap-backed tensor buffers are currently implemented on Linux only",
    ))
}

#[cfg(target_os = "linux")]
fn page_size() -> Result<usize> {
    const _SC_PAGESIZE: c_int = 30;
    unsafe extern "C" {
        fn sysconf(name: c_int) -> isize;
    }
    let value = unsafe { sysconf(_SC_PAGESIZE) };
    if value <= 0 {
        Err(Error::new(
            -1,
            format!(
                "sysconf(_SC_PAGESIZE) failed: {}",
                std::io::Error::last_os_error()
            ),
        ))
    } else {
        Ok(value as usize)
    }
}

#[cfg(target_os = "linux")]
unsafe fn unmap_raw(ptr: *mut u8, bytes: usize) {
    unsafe extern "C" {
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
    let _ = unsafe { munmap(ptr.cast(), bytes) };
}

#[cfg(not(target_os = "linux"))]
unsafe fn unmap_raw(_ptr: *mut u8, _bytes: usize) {}

#[cfg(target_os = "linux")]
fn advise_sequential_raw(ptr: *const u8, bytes: usize) {
    const MADV_SEQUENTIAL: i32 = 2;
    unsafe extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: i32) -> i32;
    }
    let _ = unsafe { madvise(ptr as *mut c_void, bytes, MADV_SEQUENTIAL) };
}

#[cfg(not(target_os = "linux"))]
fn advise_sequential_raw(_ptr: *const u8, _bytes: usize) {}

#[cfg(target_os = "linux")]
fn lock_pages_raw(ptr: *const u8, bytes: usize) -> Result<()> {
    unsafe extern "C" {
        fn mlock(addr: *const c_void, len: usize) -> c_int;
    }
    if unsafe { mlock(ptr.cast(), bytes) } == 0 {
        Ok(())
    } else {
        Err(Error::new(
            -1,
            format!(
                "mlock failed for {bytes} bytes: {}",
                std::io::Error::last_os_error()
            ),
        ))
    }
}

#[cfg(not(target_os = "linux"))]
fn lock_pages_raw(_ptr: *const u8, _bytes: usize) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn unlock_pages_raw(ptr: *const u8, bytes: usize) {
    unsafe extern "C" {
        fn munlock(addr: *const c_void, len: usize) -> c_int;
    }
    let _ = unsafe { munlock(ptr.cast(), bytes) };
}

#[cfg(not(target_os = "linux"))]
fn unlock_pages_raw(_ptr: *const u8, _bytes: usize) {}

impl<T: TensorElement> RunInput for TensorBuffer<T> {
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }
}

impl<T: TensorElement> Drop for TensorBuffer<T> {
    fn drop(&mut self) {
        unsafe { api().release_value()(self.value) }
    }
}

unsafe impl<T: TensorElement + Send> Send for TensorBuffer<T> {}
unsafe impl<T: TensorElement + Sync> Sync for TensorBuffer<T> {}

fn shape_element_count(shape: &[i64]) -> Result<usize> {
    checked_element_count(shape)
}

fn validate_shape_len(shape: &[i64], len: usize) -> Result<()> {
    let expected = shape_element_count(shape)?;
    if expected != len {
        return Err(Error::new(
            -1,
            format!("tensor shape expects {expected} elements, got {len}"),
        ));
    }
    Ok(())
}

fn validate_packed_bytes_len(shape: &[i64], elem_type: sys::ElementType, len: usize) -> Result<()> {
    if packed_element_bits(elem_type).is_none() {
        return Err(Error::new(
            -1,
            format!(
                "zrt: {:?} is not a packed sub-byte tensor element type",
                elem_type
            ),
        ));
    }
    let count = shape_element_count(shape)?;
    let expected = tensor_byte_len(elem_type, count)?;
    if expected != len {
        return Err(Error::new(
            -1,
            format!(
                "packed {:?} tensor shape expects {expected} bytes for {count} elements, got {len}",
                elem_type
            ),
        ));
    }
    Ok(())
}

pub(crate) fn tensor_memory_info(
    value: *const sys::ValueHandle,
) -> Result<crate::memory::MemoryInfoSnapshot> {
    let mut info: *const sys::MemoryInfoHandle = ptr::null();
    check(unsafe { api().get_tensor_memory_info()(value, &mut info) })?;
    crate::memory::snapshot_from_ptr(info)
}

fn tensor_value_byte_len(value: *const sys::ValueHandle) -> Result<usize> {
    let mut bytes = 0usize;
    check(unsafe { api().get_tensor_size_in_bytes()(value, &mut bytes) })?;
    Ok(bytes)
}

fn ensure_value_host_accessible(value: *const sys::ValueHandle) -> Result<()> {
    let info = tensor_memory_info(value)?;
    if !info.is_host_accessible() {
        return Err(Error::new(
            -1,
            format!(
                "tensor memory is not host-accessible: {} device {} ({:?}/{:?})",
                info.name, info.device_id, info.alloc_type, info.mem_type
            ),
        ));
    }
    Ok(())
}

fn ensure_memory_host_accessible(mem: &MemoryInfo) -> Result<()> {
    let info = mem.snapshot()?;
    if !info.is_host_accessible() {
        return Err(Error::new(
            -1,
            format!(
                "Rust slice-backed tensors require host-accessible memory, got {} device {} ({:?}/{:?})",
                info.name, info.device_id, info.alloc_type, info.mem_type
            ),
        ));
    }
    Ok(())
}

// ─── sparse tensor inputs/readback ──────────────────────────────────────────

/// An owning sparse tensor value for session inputs.
///
/// `copy_*` constructors allocate sparse storage through ORT and copy values/indices once.
/// `from_*_buffer` constructors are zero-copy over caller-owned values and indices; those
/// buffers must outlive every use of the sparse tensor.
pub struct SparseTensor<'a, T: TensorElement> {
    value: *mut sys::ValueHandle,
    values_host_accessible: bool,
    _life: PhantomData<&'a mut [T]>,
}

impl<T: TensorElement> SparseTensor<'static, T> {
    /// Create a COO sparse tensor by copying values and indices into ORT-owned storage.
    pub fn copy_coo(
        values: &[T], dense_shape: &[i64], values_shape: &[i64], indices: &[i64],
        data_mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_coo_indices(values.len(), dense_shape.len(), indices.len())?;
        let alloc = Allocator::get_default()?;
        let value = create_empty_sparse::<T>(&alloc, dense_shape)?;
        let result = check(unsafe {
            api().fill_sparse_tensor_coo()(
                value,
                data_mem.info as *const sys::MemoryInfoHandle,
                values_shape.as_ptr(),
                values_shape.len(),
                sparse_data_ptr(values),
                indices.as_ptr(),
                indices.len(),
            )
        });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: true,
            _life: PhantomData,
        })
    }

    /// Create a CSR sparse tensor by copying values and indices into ORT-owned storage.
    pub fn copy_csr(
        values: &[T], dense_shape: &[i64], values_shape: &[i64], inner_indices: &[i64],
        outer_indices: &[i64], data_mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_csr_indices(
            values.len(),
            dense_shape,
            inner_indices.len(),
            outer_indices.len(),
        )?;
        let alloc = Allocator::get_default()?;
        let value = create_empty_sparse::<T>(&alloc, dense_shape)?;
        let result = check(unsafe {
            api().fill_sparse_tensor_csr()(
                value,
                data_mem.info as *const sys::MemoryInfoHandle,
                values_shape.as_ptr(),
                values_shape.len(),
                sparse_data_ptr(values),
                inner_indices.as_ptr(),
                inner_indices.len(),
                outer_indices.as_ptr(),
                outer_indices.len(),
            )
        });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: true,
            _life: PhantomData,
        })
    }

    /// Create a block-sparse tensor by copying values and indices into ORT-owned storage.
    pub fn copy_block_sparse(
        values: &[T], dense_shape: &[i64], values_shape: &[i64], indices_shape: &[i64],
        indices: &[i32], data_mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_shape_len(indices_shape, indices.len())?;
        let alloc = Allocator::get_default()?;
        let value = create_empty_sparse::<T>(&alloc, dense_shape)?;
        let result = check(unsafe {
            api().fill_sparse_tensor_block_sparse()(
                value,
                data_mem.info as *const sys::MemoryInfoHandle,
                values_shape.as_ptr(),
                values_shape.len(),
                sparse_data_ptr(values),
                indices_shape.as_ptr(),
                indices_shape.len(),
                indices.as_ptr(),
            )
        });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: true,
            _life: PhantomData,
        })
    }
}

impl<'a, T: TensorElement> SparseTensor<'a, T> {
    /// Create a zero-copy COO sparse tensor over caller-owned values and indices.
    pub fn from_coo_buffer(
        values: &'a mut [T], dense_shape: &[i64], values_shape: &[i64], indices: &'a mut [i64],
        mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_coo_indices(values.len(), dense_shape.len(), indices.len())?;
        let host = mem.snapshot()?.is_host_accessible();
        let value = create_sparse_with_values::<T>(values, dense_shape, values_shape, mem)?;
        let result =
            check(unsafe { api().use_coo_indices()(value, indices.as_mut_ptr(), indices.len()) });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: host,
            _life: PhantomData,
        })
    }

    /// Create a zero-copy CSR sparse tensor over caller-owned values and indices.
    pub fn from_csr_buffer(
        values: &'a mut [T], dense_shape: &[i64], values_shape: &[i64],
        inner_indices: &'a mut [i64], outer_indices: &'a mut [i64], mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_csr_indices(
            values.len(),
            dense_shape,
            inner_indices.len(),
            outer_indices.len(),
        )?;
        let host = mem.snapshot()?.is_host_accessible();
        let value = create_sparse_with_values::<T>(values, dense_shape, values_shape, mem)?;
        let result = check(unsafe {
            api().use_csr_indices()(
                value,
                inner_indices.as_mut_ptr(),
                inner_indices.len(),
                outer_indices.as_mut_ptr(),
                outer_indices.len(),
            )
        });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: host,
            _life: PhantomData,
        })
    }

    /// Create a zero-copy block-sparse tensor over caller-owned values and indices.
    pub fn from_block_sparse_buffer(
        values: &'a mut [T], dense_shape: &[i64], values_shape: &[i64], indices_shape: &[i64],
        indices: &'a mut [i32], mem: &MemoryInfo,
    ) -> Result<Self> {
        validate_sparse_shapes::<T>(values.len(), dense_shape, values_shape)?;
        validate_shape_len(indices_shape, indices.len())?;
        let host = mem.snapshot()?.is_host_accessible();
        let value = create_sparse_with_values::<T>(values, dense_shape, values_shape, mem)?;
        let result = check(unsafe {
            api().use_block_sparse_indices()(
                value,
                indices_shape.as_ptr(),
                indices_shape.len(),
                indices.as_mut_ptr(),
            )
        });
        if let Err(err) = result {
            unsafe { api().release_value()(value) };
            return Err(err);
        }
        Ok(Self {
            value,
            values_host_accessible: host,
            _life: PhantomData,
        })
    }

    /// Whether ORT reports this value as sparse.
    pub fn is_sparse(&self) -> Result<bool> {
        let mut out = 0;
        check(unsafe {
            api().is_sparse_tensor()(self.value as *const sys::ValueHandle, &mut out)
        })?;
        Ok(out != 0)
    }

    /// Sparse storage format (COO, CSR/CSRC, or block-sparse).
    pub fn format(&self) -> Result<sys::SparseFormat> {
        let mut format = sys::SparseFormat::Undefined;
        check(unsafe {
            api().get_sparse_tensor_format()(self.value as *const sys::ValueHandle, &mut format)
        })?;
        Ok(format)
    }

    /// Type and shape of the non-zero values buffer.
    pub fn values_type_and_shape(&self) -> Result<TensorTypeAndShapeInfo> {
        let mut info: *mut sys::TensorTypeAndShapeInfoHandle = ptr::null_mut();
        check(unsafe {
            api().get_sparse_tensor_values_type_and_shape()(
                self.value as *const sys::ValueHandle,
                &mut info,
            )
        })?;
        let info = crate::ensure_non_null(info, "sparse tensor values type and shape info")?;
        Ok(unsafe { TensorTypeAndShapeInfo::from_owning(info) })
    }

    /// Type and shape for one sparse-index buffer.
    pub fn indices_type_and_shape(
        &self, format: sys::SparseIndicesFormat,
    ) -> Result<TensorTypeAndShapeInfo> {
        let mut info: *mut sys::TensorTypeAndShapeInfoHandle = ptr::null_mut();
        check(unsafe {
            api().get_sparse_tensor_indices_type_shape()(
                self.value as *const sys::ValueHandle,
                format,
                &mut info,
            )
        })?;
        let info = crate::ensure_non_null(info, "sparse tensor indices type and shape info")?;
        Ok(unsafe { TensorTypeAndShapeInfo::from_owning(info) })
    }

    /// Raw pointer to the sparse values buffer. For CUDA/device sparse values this may be a
    /// provider pointer.
    pub fn values_data_ptr(&self) -> Result<*const T> {
        let mut data: *const c_void = ptr::null();
        check(unsafe {
            api().get_sparse_tensor_values()(
                self.value as *const sys::ValueHandle,
                &mut data as *mut *const c_void as *const *const c_void,
            )
        })?;
        let count = self.values_type_and_shape()?.element_count()?;
        Ok(crate::slice_data_ptr(data as *mut T, count, "sparse tensor values")? as *const T)
    }

    /// Host-accessible zero-copy read of sparse values.
    pub fn values_as_slice(&self) -> Result<&[T]> {
        if !self.values_host_accessible {
            return Err(Error::new(
                -1,
                "sparse tensor values are not host-accessible",
            ));
        }
        let info = self.values_type_and_shape()?;
        let elem = info.element_type()?;
        if elem as i32 != T::ELEM as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: sparse values as_slice<{}> on a {:?} tensor",
                    std::any::type_name::<T>(),
                    elem
                ),
            ));
        }
        let count = info.element_count()?;
        let ptr = self.values_data_ptr()?;
        Ok(unsafe { std::slice::from_raw_parts(ptr, count) })
    }

    /// Raw pointer and element count for a sparse-index buffer.
    pub fn indices_data_ptr(
        &self, format: sys::SparseIndicesFormat,
    ) -> Result<(*const c_void, usize)> {
        let mut count = 0usize;
        let mut data: *const c_void = ptr::null();
        check(unsafe {
            api().get_sparse_tensor_indices()(
                self.value as *const sys::ValueHandle,
                format,
                &mut count,
                &mut data as *mut *const c_void as *const *const c_void,
            )
        })?;
        let data = crate::slice_data_ptr(data as *mut u8, count, "sparse tensor indices")?;
        Ok((data as *const c_void, count))
    }

    /// Host-accessible zero-copy read of COO or CSR index buffers (`i64`).
    pub fn indices_i64(&self, format: sys::SparseIndicesFormat) -> Result<&[i64]> {
        if format == sys::SparseIndicesFormat::BlockSparse {
            return Err(Error::new(
                -1,
                "block-sparse indices are i32; use block_sparse_indices",
            ));
        }
        let (ptr, count) = self.indices_data_ptr(format)?;
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const i64, count) })
    }

    /// Host-accessible zero-copy read of block-sparse index buffers (`i32`).
    pub fn block_sparse_indices(&self) -> Result<&[i32]> {
        let (ptr, count) = self.indices_data_ptr(sys::SparseIndicesFormat::BlockSparse)?;
        Ok(unsafe { std::slice::from_raw_parts(ptr as *const i32, count) })
    }
}

impl<T: TensorElement> RunInput for SparseTensor<'_, T> {
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }
}

impl<T: TensorElement> Drop for SparseTensor<'_, T> {
    fn drop(&mut self) {
        unsafe { api().release_value()(self.value) }
    }
}

unsafe impl<T: TensorElement + Send> Send for SparseTensor<'_, T> {}
unsafe impl<T: TensorElement + Sync> Sync for SparseTensor<'_, T> {}

fn create_empty_sparse<T: TensorElement>(
    alloc: &Allocator, dense_shape: &[i64],
) -> Result<*mut sys::ValueHandle> {
    shape_element_count(dense_shape)?;
    let mut value: *mut sys::ValueHandle = ptr::null_mut();
    check(unsafe {
        api().create_sparse_tensor_as_ort_value()(
            alloc.alloc,
            dense_shape.as_ptr(),
            dense_shape.len(),
            T::ELEM,
            &mut value,
        )
    })?;
    crate::ensure_non_null(value, "sparse tensor value")
}

fn create_sparse_with_values<T: TensorElement>(
    values: &mut [T], dense_shape: &[i64], values_shape: &[i64], mem: &MemoryInfo,
) -> Result<*mut sys::ValueHandle> {
    let mut value: *mut sys::ValueHandle = ptr::null_mut();
    check(unsafe {
        api().create_sparse_tensor_with_values_as_ort_value()(
            mem.info as *const sys::MemoryInfoHandle,
            values.as_mut_ptr() as *mut c_void,
            dense_shape.as_ptr(),
            dense_shape.len(),
            values_shape.as_ptr(),
            values_shape.len(),
            T::ELEM,
            &mut value,
        )
    })?;
    crate::ensure_non_null(value, "sparse tensor value")
}

fn validate_sparse_shapes<T: TensorElement>(
    values_len: usize, dense_shape: &[i64], values_shape: &[i64],
) -> Result<()> {
    shape_element_count(dense_shape)?;
    validate_shape_len(values_shape, values_len)?;
    if T::ELEM == sys::ElementType::String {
        return Err(Error::new(
            -1,
            "sparse string tensors must use ORT copied string APIs, which ZRT does not expose yet",
        ));
    }
    Ok(())
}

fn validate_coo_indices(values_len: usize, dense_rank: usize, indices_len: usize) -> Result<()> {
    let ok = if values_len == 0 {
        indices_len == 0
    } else {
        indices_len == values_len
            || indices_len
                == values_len
                    .checked_mul(dense_rank)
                    .ok_or_else(|| Error::new(-1, "COO index length overflows usize"))?
    };
    if !ok {
        return Err(Error::new(
            -1,
            format!("COO indices must have 0, nnz, or nnz*dense_rank entries; got {indices_len}"),
        ));
    }
    Ok(())
}

fn validate_csr_indices(
    values_len: usize, dense_shape: &[i64], inner_len: usize, outer_len: usize,
) -> Result<()> {
    if inner_len != values_len {
        return Err(Error::new(
            -1,
            format!("CSR inner indices must match nnz ({values_len}), got {inner_len}"),
        ));
    }
    let rows = dense_shape
        .first()
        .copied()
        .ok_or_else(|| Error::new(-1, "CSR sparse tensors require rank >= 1"))?;
    if rows < 0 {
        return Err(Error::new(
            -1,
            "CSR dense shape cannot contain dynamic rows",
        ));
    }
    let expected_outer = usize::try_from(rows)
        .map_err(|_| Error::new(-1, "CSR row count does not fit usize"))?
        .checked_add(1)
        .ok_or_else(|| Error::new(-1, "CSR outer-index length overflows usize"))?;
    if outer_len != expected_outer {
        return Err(Error::new(
            -1,
            format!(
                "CSR outer indices must have rows + 1 entries ({expected_outer}), got {outer_len}"
            ),
        ));
    }
    Ok(())
}

fn sparse_data_ptr<T: TensorElement>(values: &[T]) -> *const c_void {
    values.as_ptr() as *const c_void
}

// ─── owning string input ────────────────────────────────────────────────────

/// An owning string-tensor input. ONNX string tensors cannot be zero-copy over caller
/// memory: the engine materializes them via `CreateTensorAsOrtValue` (STRING) +
/// `FillStringTensor`. The `&str` slices are copied into the engine allocator once at
/// construction; the hot path then reuses the stable value handle.
pub struct StringTensor {
    value: *mut sys::ValueHandle,
}

impl StringTensor {
    /// Build a rank-N string tensor of `shape` from `strings`. `strings.len()` must equal
    /// the product of `shape` (the element count); the engine validates this.
    pub fn new(strings: &[&str], shape: &[i64]) -> Result<Self> {
        validate_shape_len(shape, strings.len())?;
        let alloc = Allocator::get_default()?;
        let mut value: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().create_tensor_as_ort_value()(
                alloc.alloc,
                shape.as_ptr(),
                shape.len(),
                sys::ElementType::String,
                &mut value,
            )
        })?;
        let value = crate::ensure_non_null(value, "string tensor value")?;
        // FillStringTensor requires an array of NUL-terminated UTF-8 strings.
        let cstrings: Vec<CString> = strings
            .iter()
            .map(|s| {
                CString::new(*s).map_err(|_| Error::new(-1, "string tensor element contains a NUL"))
            })
            .collect::<Result<_>>()?;
        let ptrs: Vec<*const c_char> = cstrings.iter().map(|c| c.as_ptr()).collect();
        check(unsafe { api().fill_string_tensor()(value, ptrs.as_ptr(), ptrs.len()) })?;
        Ok(Self { value })
    }
}

impl RunInput for StringTensor {
    #[inline]
    fn as_value_ptr(&self) -> *const sys::ValueHandle {
        self.value as *const sys::ValueHandle
    }
}

impl Drop for StringTensor {
    fn drop(&mut self) {
        unsafe { api().release_value()(self.value) }
    }
}
unsafe impl Send for StringTensor {}
unsafe impl Sync for StringTensor {}

// ─── engine-owned output value (tensor / sequence / map) ────────────────────

/// An ORT value whose backing memory is engine-owned (freed on drop), e.g. a run
/// output. Its value-kind (`OnnxType`) and — for tensors — element type/count are
/// stamped from the session's cached type-info, so tensor reads need no per-call
/// introspection. Sequence/map values expose `value_count` / `get_value` instead.
pub struct OwnedValue {
    pub(crate) value: *mut sys::ValueHandle,
    pub(crate) onnx_type: sys::OnnxType,
    pub(crate) elem_type: sys::ElementType,
    pub(crate) count: usize,
}

impl OwnedValue {
    /// Build from a raw owning handle by introspecting its kind (used for the children of
    /// a sequence/map value, whose type is only known at run time).
    pub(crate) fn from_introspect(value: *mut sys::ValueHandle) -> Result<Self> {
        let result = (|| {
            let mut value_kind = sys::OnnxType::Unknown;
            check(unsafe {
                api().get_value_type()(value as *const sys::ValueHandle, &mut value_kind)
            })?;
            let (elem_type, count) = if value_kind == sys::OnnxType::Tensor {
                let tsi = tensor_type_and_shape(value as *const sys::ValueHandle)?;
                (tsi.element_type()?, tsi.element_count()?)
            } else {
                (sys::ElementType::Undefined, 0)
            };
            Ok(Self {
                value,
                onnx_type: value_kind,
                elem_type,
                count,
            })
        })();
        if result.is_err() && !value.is_null() {
            unsafe { api().release_value()(value) };
        }
        result
    }

    /// Convert a raw owning output-handle array into owned values. On error, releases the
    /// failed handle (via `from_introspect`) and every unwrapped remaining handle.
    pub(crate) fn collect_from_raw(handles: &[*mut sys::ValueHandle]) -> Result<Vec<OwnedValue>> {
        let mut values = Vec::with_capacity(handles.len());
        for (i, &handle) in handles.iter().enumerate() {
            match Self::from_introspect(handle) {
                Ok(value) => values.push(value),
                Err(err) => {
                    for &remaining in &handles[i + 1..] {
                        if !remaining.is_null() {
                            unsafe { api().release_value()(remaining) };
                        }
                    }
                    return Err(err);
                },
            }
        }
        Ok(values)
    }

    /// The cached value kind (Tensor / Sequence / Map / …).
    #[inline]
    pub fn onnx_type(&self) -> sys::OnnxType {
        self.onnx_type
    }
    /// The cached element type (only meaningful for Tensor values).
    #[inline]
    pub fn element_type(&self) -> sys::ElementType {
        self.elem_type
    }
    /// The cached element count (product of dims; only meaningful for Tensor values).
    #[inline]
    pub fn element_count(&self) -> usize {
        self.count
    }
    /// Total numeric tensor backing-buffer size in bytes.
    ///
    /// Uses ORT's `GetTensorSizeInBytes`, so packed sub-byte tensor storage is measured by the
    /// engine. ORT returns an error for string tensors and non-tensor values.
    pub fn byte_len(&self) -> Result<usize> {
        if self.onnx_type != sys::OnnxType::Tensor {
            return Err(Error::new(
                -1,
                format!("zrt: byte_len on a non-tensor ({:?}) value", self.onnx_type),
            ));
        }
        tensor_value_byte_len(self.value as *const sys::ValueHandle)
    }

    /// Number of elements in a SEQUENCE value (always 2 for a MAP: index 0 = keys,
    /// index 1 = values). Errors for tensor values.
    pub fn value_count(&self) -> Result<usize> {
        let mut n: usize = 0;
        check(unsafe { api().get_value_count()(self.value as *const sys::ValueHandle, &mut n) })?;
        Ok(n)
    }

    /// For a SEQUENCE value: the `index`'th element. For a MAP value: index 0 = keys
    /// tensor, index 1 = values tensor. The returned value is engine-allocated and owned
    /// by the caller (released on drop). The child's kind is introspected automatically.
    pub fn get_value(&self, index: usize) -> Result<OwnedValue> {
        let count = self.value_count()?;
        if index >= count {
            return Err(Error::new(
                -1,
                format!("zrt: value index {index} out of range ({count} values)"),
            ));
        }
        let index = c_int::try_from(index)
            .map_err(|_| Error::new(-1, "zrt: value index overflows c_int"))?;
        let alloc = Allocator::get_default()?;
        let mut out: *mut sys::ValueHandle = ptr::null_mut();
        check(unsafe {
            api().get_value()(
                self.value as *const sys::ValueHandle,
                index,
                alloc.alloc,
                &mut out,
            )
        })?;
        OwnedValue::from_introspect(out)
    }

    /// Full type+shape introspection (tensor values only). Owns the returned handle.
    pub fn tensor_type_and_shape(&self) -> Result<TensorTypeAndShapeInfo> {
        tensor_type_and_shape(self.value as *const sys::ValueHandle)
    }

    /// Memory descriptor for this tensor's backing allocation.
    pub fn memory_info(&self) -> Result<crate::memory::MemoryInfoSnapshot> {
        tensor_memory_info(self.value as *const sys::ValueHandle)
    }

    /// Wrap this tensor output as a placement-aware value, forcing the caller to make explicit
    /// host/device copy decisions.
    pub fn into_device_value(self) -> Result<DeviceValue> {
        DeviceValue::from_owned(self)
    }

    /// Copy this value into an existing reusable tensor buffer via ORT `CopyTensors`.
    pub fn copy_to_tensor_buffer<T: TensorElement>(
        &self, session: &crate::Session, dst: &mut TensorBuffer<T>,
    ) -> Result<()> {
        session.copy_value_to_tensor_buffer(self, dst)
    }

    /// Copy this value into an existing ORT-allocated tensor via ORT `CopyTensors`.
    pub fn copy_to_allocated_tensor<T: TensorElement>(
        &self, session: &crate::Session, dst: &mut AllocatedTensor<T>,
    ) -> Result<()> {
        session.copy_value_to_allocated_tensor(self, dst)
    }

    /// ORT 1.27 memory-device descriptor for this tensor's backing allocation.
    #[cfg(feature = "model-editor")]
    pub fn memory_device(&self) -> Result<crate::memory::MemoryDeviceSnapshot> {
        let ep = crate::model_editor::ep_api()
            .ok_or_else(|| crate::Error::new(-1, "EpApi unavailable"))?;
        let device = unsafe {
            ep.Value_GetMemoryDevice
                .ok_or_else(|| crate::Error::new(-1, "Value_GetMemoryDevice unavailable"))?(
                self.value as *const sys::ValueHandle,
            )
        };
        crate::memory::memory_device_snapshot_from_ptr(device)
    }

    /// Zero-copy read of the engine-owned backing buffer as a typed slice
    /// (`GetTensorMutableData`). One FFI call, no allocation, no introspection.
    pub fn as_slice<T: TensorElement>(&self) -> Result<&[T]> {
        if self.onnx_type != sys::OnnxType::Tensor {
            return Err(Error::new(
                -1,
                format!("zrt: as_slice on a non-tensor ({:?}) value", self.onnx_type),
            ));
        }
        if self.elem_type as i32 != T::ELEM as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: as_slice<{}> on a {:?} tensor",
                    std::any::type_name::<T>(),
                    self.elem_type
                ),
            ));
        }
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut T, self.count, "tensor data")?;
        // SAFETY: data is a contiguous, aligned buffer of `self.count` elements of T,
        // owned by the value and live for at least 'self.
        Ok(unsafe { std::slice::from_raw_parts(data as *const T, self.count) })
    }

    /// Zero-copy read as raw bytes (no element-type assertion). For packed 2-bit/4-bit tensors
    /// this returns the packed backing storage.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        let n = self.byte_len()?;
        ensure_value_host_accessible(self.value as *const sys::ValueHandle)?;
        let mut data: *mut c_void = ptr::null_mut();
        check(unsafe { api().get_tensor_mutable_data()(self.value, &mut data) })?;
        let data = crate::slice_data_ptr(data as *mut u8, n, "tensor data")?;
        Ok(unsafe { std::slice::from_raw_parts(data as *const u8, n) })
    }

    /// Decode a string-tensor output to owned Rust strings. One bulk read
    /// (`GetStringTensorDataLength` + `GetStringTensorContent`) plus an offsets array —
    /// no per-element FFI. `count` (the cached element count) is the number of strings.
    pub fn as_strings(&self) -> Result<Vec<String>> {
        if self.onnx_type != sys::OnnxType::Tensor {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: as_strings on a non-tensor ({:?}) value",
                    self.onnx_type
                ),
            ));
        }
        if self.elem_type as i32 != sys::ElementType::String as i32 {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: as_strings on a {:?} tensor (expected String)",
                    self.elem_type
                ),
            ));
        }
        read_string_tensor(self.value as *const sys::ValueHandle, self.count)
    }
}

/// Bulk-read `count` strings from a string-tensor value (`GetStringTensorDataLength` +
/// `GetStringTensorContent`). Borrows the value — does not take ownership, does not release.
pub(crate) fn read_string_tensor(
    value: *const sys::ValueHandle, count: usize,
) -> Result<Vec<String>> {
    let mut total: usize = 0;
    check(unsafe { api().get_string_tensor_data_length()(value, &mut total) })?;
    let mut buf = vec![0u8; total];
    let mut offsets = vec![0usize; count];
    check(unsafe {
        api().get_string_tensor_content()(
            value,
            buf.as_mut_ptr() as *mut c_void,
            total,
            offsets.as_mut_ptr(),
            count,
        )
    })?;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = offsets[i];
        let end = if i + 1 < count { offsets[i + 1] } else { total };
        if start > end || end > total {
            return Err(Error::new(
                -1,
                format!("zrt: invalid string tensor offsets at index {i}"),
            ));
        }
        let s = std::str::from_utf8(&buf[start..end])
            .map_err(|_| Error::new(-1, "zrt: string tensor data is not valid UTF-8"))?;
        out.push(s.to_owned());
    }
    Ok(out)
}

unsafe impl Send for OwnedValue {}
unsafe impl Sync for OwnedValue {}

/// An owned ORT value intended for explicit device/placement-aware handling.
///
/// This is a thin ergonomic wrapper around [`OwnedValue`]. It does not change ownership: the
/// wrapped `OrtValue` is still released on drop. Use this when an output may live on CUDA or
/// another EP device and callers should make an explicit copy decision instead of trying to read
/// it as a host slice.
pub struct DeviceValue {
    value: OwnedValue,
}

impl DeviceValue {
    /// Wrap an engine-owned tensor value for explicit placement-aware handling.
    pub fn from_owned(value: OwnedValue) -> Result<Self> {
        if value.onnx_type != sys::OnnxType::Tensor {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: DeviceValue requires a tensor, got {:?}",
                    value.onnx_type
                ),
            ));
        }
        Ok(Self { value })
    }

    #[inline]
    pub fn as_owned(&self) -> &OwnedValue {
        &self.value
    }

    #[inline]
    pub fn into_owned(self) -> OwnedValue {
        self.value
    }

    #[inline]
    pub fn element_type(&self) -> sys::ElementType {
        self.value.element_type()
    }

    #[inline]
    pub fn element_count(&self) -> usize {
        self.value.element_count()
    }

    #[inline]
    pub fn byte_len(&self) -> Result<usize> {
        self.value.byte_len()
    }

    #[inline]
    pub fn memory_info(&self) -> Result<crate::memory::MemoryInfoSnapshot> {
        self.value.memory_info()
    }

    #[cfg(feature = "model-editor")]
    #[inline]
    pub fn memory_device(&self) -> Result<crate::memory::MemoryDeviceSnapshot> {
        self.value.memory_device()
    }

    /// Copy this value into an existing reusable tensor buffer via ORT `CopyTensors`.
    pub fn copy_to_tensor_buffer<T: TensorElement>(
        &self, session: &crate::Session, dst: &mut TensorBuffer<T>,
    ) -> Result<()> {
        session.copy_value_to_tensor_buffer(&self.value, dst)
    }

    /// Copy this value into an existing ORT-allocated tensor via ORT `CopyTensors`.
    pub fn copy_to_allocated_tensor<T: TensorElement>(
        &self, session: &crate::Session, dst: &mut AllocatedTensor<T>,
    ) -> Result<()> {
        session.copy_value_to_allocated_tensor(&self.value, dst)
    }
}

impl Drop for OwnedValue {
    fn drop(&mut self) {
        unsafe { api().release_value()(self.value) }
    }
}

#[cfg(test)]
mod tests {
    //! Engine-backed unit tests for the tensor surface that need no model:
    //!  - string-tensor write→read round-trip (FillStringTensor + GetStringTensor*),
    //!  - per-type element-type mapping through the real engine.
    use super::*;

    fn round_trip<T: TensorElement>(buf: &[T], expected: sys::ElementType, mem: &MemoryInfo) {
        let v = Tensor::from_buffer(buf, &[buf.len() as i64], mem).unwrap();
        let tsi = v.tensor_type_and_shape().unwrap();
        assert_eq!(
            tsi.element_type().unwrap(),
            expected,
            "element-type mapping"
        );
        assert_eq!(tsi.element_count().unwrap(), buf.len(), "element count");
    }

    /// Engine-owned tensor (`copy_from_slice`) read back via the `TensorView` accessors —
    /// exercises `as_slice`/`dims`/`element_type`/`element_count` on an engine-allocated
    /// buffer (the same path a custom-op kernel uses to read its inputs).
    #[test]
    fn tensor_view_read_accessors() {
        let buf = [1.0f32, 2.0, 3.0, 4.0];
        let v = Tensor::copy_from_slice(&buf, &[2, 2]).unwrap();
        assert_eq!(v.element_type().unwrap(), sys::ElementType::Float);
        assert_eq!(v.element_count().unwrap(), 4);
        assert_eq!(v.dims().unwrap(), vec![2, 2]);
        assert_eq!(v.byte_len().unwrap(), std::mem::size_of_val(&buf));
        assert_eq!(v.as_bytes().unwrap().len(), std::mem::size_of_val(&buf));
        assert_eq!(v.as_slice::<f32>().unwrap(), &buf[..]);
    }

    #[test]
    fn packed_sub_byte_tensor_wraps_and_reads_raw_bytes() {
        let mem = MemoryInfo::cpu().unwrap();
        let bytes = [0x21, 0x43];
        let v = Tensor::from_packed_bytes(&bytes, &[4], sys::ElementType::Uint4, &mem).unwrap();
        assert_eq!(v.element_type().unwrap(), sys::ElementType::Uint4);
        assert_eq!(v.element_count().unwrap(), 4);
        assert_eq!(v.byte_len().unwrap(), bytes.len());
        assert_eq!(v.as_bytes().unwrap(), &bytes);
        assert!(v.as_slice::<u8>().is_err());
    }

    #[test]
    fn packed_sub_byte_tensor_validates_storage_length() {
        let mem = MemoryInfo::cpu().unwrap();
        assert!(Tensor::from_packed_bytes(&[0], &[4], sys::ElementType::Uint4, &mem).is_err());
        assert!(validate_packed_bytes_len(&[4], sys::ElementType::Uint2, 1).is_ok());
        assert!(Tensor::from_packed_bytes(&[0], &[1], sys::ElementType::Uint8, &mem).is_err());
    }

    #[test]
    fn tensor_constructors_reject_dynamic_and_mismatched_shapes() {
        let mem = MemoryInfo::cpu().unwrap();
        let buf = [0.0f32; 4];
        assert!(Tensor::from_buffer(&buf, &[-1, 4], &mem).is_err());
        assert!(Tensor::from_buffer(&buf, &[5], &mem).is_err());
        assert!(Tensor::from_buffer(&buf, &[2, 2], &mem).is_ok());
        assert!(Tensor::copy_from_slice(&buf, &[-1, 4]).is_err());
        assert!(Tensor::copy_from_slice(&buf, &[5]).is_err());
        assert!(Tensor::copy_from_slice(&buf, &[2, 2]).is_ok());
    }

    #[test]
    fn rust_slice_tensors_reject_cuda_device_memory() {
        let mem = MemoryInfo::cuda(0).unwrap();
        let buf = [0.0f32; 4];
        assert!(Tensor::from_buffer(&buf, &[2, 2], &mem).is_err());
        assert!(TensorBuffer::from_vec(buf.to_vec(), &[2, 2], &mem).is_err());
    }

    #[test]
    fn string_tensor_rejects_dynamic_and_mismatched_shapes() {
        let strings = ["a", "b"];
        assert!(StringTensor::new(&strings, &[-1, 2]).is_err());
        assert!(StringTensor::new(&strings, &[3]).is_err());
        assert!(StringTensor::new(&strings, &[2]).is_ok());
    }

    #[test]
    fn allocated_tensor_cpu_round_trip_and_memory_info() {
        let model_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bench")
            .join("models")
            .join("mnist.onnx");
        if !model_path.exists() {
            eprintln!("skipping allocated tensor allocator test — mnist.onnx absent");
            return;
        }

        let env = crate::Environment::new().unwrap();
        let mem = MemoryInfo::cpu().unwrap();
        let sess = crate::Session::new(
            &env,
            model_path.to_str().unwrap(),
            crate::SessionOptions::new(),
        )
        .unwrap();
        let alloc = Allocator::create(&sess, &mem).unwrap();
        let mut tensor =
            AllocatedTensor::<f32>::copy_from_slice(alloc, &[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        tensor.as_mut_slice().unwrap()[0] = 9.0;
        assert_eq!(tensor.as_slice().unwrap()[0], 9.0);

        let info = tensor.memory_info().unwrap();
        assert_eq!(info.name, "Cpu");
        assert!(info.is_host_accessible());
    }

    #[test]
    fn sparse_tensor_coo_copy_round_trip() {
        let mem = MemoryInfo::cpu().unwrap();
        let values = [1.0f32, 2.0, 3.0];
        let indices = [0i64, 1, 1, 0, 1, 2];
        let sparse =
            SparseTensor::copy_coo(&values, &[2, 3], &[3], &indices, &mem).expect("coo sparse");

        assert!(sparse.is_sparse().unwrap());
        assert_eq!(sparse.format().unwrap(), sys::SparseFormat::Coo);
        assert_eq!(sparse.values_as_slice().unwrap(), &values);
        assert_eq!(
            sparse
                .values_type_and_shape()
                .unwrap()
                .element_type()
                .unwrap(),
            sys::ElementType::Float
        );
        assert_eq!(
            sparse
                .indices_type_and_shape(sys::SparseIndicesFormat::Coo)
                .unwrap()
                .element_type()
                .unwrap(),
            sys::ElementType::Int64
        );
        assert_eq!(
            sparse.indices_i64(sys::SparseIndicesFormat::Coo).unwrap(),
            &indices
        );
    }

    #[test]
    fn sparse_tensor_coo_buffer_is_zero_copy() {
        let mem = MemoryInfo::cpu().unwrap();
        let mut values = [1.0f32, 2.0, 3.0];
        let values_ptr = values.as_ptr();
        let mut indices = [0i64, 1, 1, 0, 1, 2];
        let indices_ptr = indices.as_ptr();
        let sparse = SparseTensor::from_coo_buffer(&mut values, &[2, 3], &[3], &mut indices, &mem)
            .expect("coo sparse buffer");

        assert_eq!(sparse.format().unwrap(), sys::SparseFormat::Coo);
        assert_eq!(sparse.values_data_ptr().unwrap(), values_ptr);
        assert_eq!(
            sparse
                .indices_i64(sys::SparseIndicesFormat::Coo)
                .unwrap()
                .as_ptr(),
            indices_ptr
        );
        assert_eq!(sparse.values_as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn sparse_tensor_csr_copy_round_trip() {
        let mem = MemoryInfo::cpu().unwrap();
        let values = [1.0f32, 2.0, 3.0];
        let inner = [1i64, 0, 2];
        let outer = [0i64, 1, 3];
        let sparse =
            SparseTensor::copy_csr(&values, &[2, 3], &[3], &inner, &outer, &mem).expect("csr");

        assert_eq!(sparse.format().unwrap(), sys::SparseFormat::Csrc);
        assert_eq!(sparse.values_as_slice().unwrap(), &values);
        assert_eq!(
            sparse
                .indices_i64(sys::SparseIndicesFormat::CsrInner)
                .unwrap(),
            &inner
        );
        assert_eq!(
            sparse
                .indices_i64(sys::SparseIndicesFormat::CsrOuter)
                .unwrap(),
            &outer
        );
    }

    #[test]
    fn sparse_tensor_block_sparse_copy_round_trip() {
        let mem = MemoryInfo::cpu().unwrap();
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let indices = [0i32, 0, 1, 0];
        let sparse =
            SparseTensor::copy_block_sparse(&values, &[2, 2], &[2, 1, 2], &[2, 2], &indices, &mem)
                .expect("block sparse");

        assert_eq!(sparse.format().unwrap(), sys::SparseFormat::BlockSparse);
        assert_eq!(sparse.values_as_slice().unwrap(), &values);
        assert_eq!(sparse.block_sparse_indices().unwrap(), &indices);
    }

    #[test]
    fn sparse_tensor_block_sparse_buffer_is_zero_copy() {
        let mem = MemoryInfo::cpu().unwrap();
        let mut values = [1.0f32, 2.0, 3.0, 4.0];
        let values_ptr = values.as_ptr();
        let mut indices = [0i32, 0, 1, 0];
        let indices_ptr = indices.as_ptr();
        let sparse = SparseTensor::from_block_sparse_buffer(
            &mut values,
            &[2, 2],
            &[2, 1, 2],
            &[2, 2],
            &mut indices,
            &mem,
        )
        .expect("block sparse buffer");

        assert_eq!(sparse.format().unwrap(), sys::SparseFormat::BlockSparse);
        assert_eq!(sparse.values_data_ptr().unwrap(), values_ptr);
        assert_eq!(sparse.block_sparse_indices().unwrap().as_ptr(), indices_ptr);
    }

    #[test]
    fn numeric_element_types() {
        let mem = MemoryInfo::cpu().unwrap();
        round_trip::<f32>(&[0.0; 4], sys::ElementType::Float, &mem);
        round_trip::<f64>(&[0.0; 2], sys::ElementType::Double, &mem);
        round_trip::<i8>(&[0; 3], sys::ElementType::Int8, &mem);
        round_trip::<i16>(&[0; 3], sys::ElementType::Int16, &mem);
        round_trip::<i32>(&[0; 3], sys::ElementType::Int32, &mem);
        round_trip::<i64>(&[0; 3], sys::ElementType::Int64, &mem);
        round_trip::<u8>(&[0; 3], sys::ElementType::Uint8, &mem);
        round_trip::<u16>(&[0; 3], sys::ElementType::Uint16, &mem);
        round_trip::<u32>(&[0; 3], sys::ElementType::Uint32, &mem);
        round_trip::<u64>(&[0; 3], sys::ElementType::Uint64, &mem);
        round_trip::<bool>(&[false, true], sys::ElementType::Bool, &mem);
    }

    #[cfg(feature = "half")]
    #[test]
    fn half_element_types() {
        let mem = MemoryInfo::cpu().unwrap();
        round_trip::<half::f16>(&[half::f16::ZERO; 3], sys::ElementType::Float16, &mem);
        round_trip::<half::bf16>(&[half::bf16::ZERO; 3], sys::ElementType::Bfloat16, &mem);
    }

    #[test]
    fn string_tensor_round_trip() {
        // Empty, ASCII, and multibyte UTF-8 — exercises the offsets-based bulk read.
        let words = ["hello", "", "world", "héllo", "wörld"];
        let st = StringTensor::new(&words, &[words.len() as i64]).unwrap();
        // Read back through the engine via the borrowed bulk-read helper (no ownership
        // transfer — `st` remains the sole owner and releases the value on drop).
        let got = read_string_tensor(st.value as *const sys::ValueHandle, words.len()).unwrap();
        assert_eq!(got, words);
        assert!(tensor_value_byte_len(st.value as *const sys::ValueHandle).is_err());
    }
}
