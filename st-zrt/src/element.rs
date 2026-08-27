//! Sealed mapping from Rust scalar types to ONNX tensor element types.
use crate::sys;

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for bool {}
    #[cfg(feature = "half")]
    impl Sealed for half::f16 {}
    #[cfg(feature = "half")]
    impl Sealed for half::bf16 {}
}

/// A Rust scalar that maps 1:1 to an ONNX tensor element type. Sealed — downstream
/// code cannot add new element types (the set mirrors the engine's POD types).
pub trait TensorElement: Copy + private::Sealed {
    /// The matching `ONNXTensorElementDataType`.
    const ELEM: sys::ElementType;
}

impl TensorElement for f32 {
    const ELEM: sys::ElementType = sys::ElementType::Float;
}
impl TensorElement for f64 {
    const ELEM: sys::ElementType = sys::ElementType::Double;
}
impl TensorElement for i8 {
    const ELEM: sys::ElementType = sys::ElementType::Int8;
}
impl TensorElement for i16 {
    const ELEM: sys::ElementType = sys::ElementType::Int16;
}
impl TensorElement for i32 {
    const ELEM: sys::ElementType = sys::ElementType::Int32;
}
impl TensorElement for i64 {
    const ELEM: sys::ElementType = sys::ElementType::Int64;
}
impl TensorElement for u8 {
    const ELEM: sys::ElementType = sys::ElementType::Uint8;
}
impl TensorElement for u16 {
    const ELEM: sys::ElementType = sys::ElementType::Uint16;
}
impl TensorElement for u32 {
    const ELEM: sys::ElementType = sys::ElementType::Uint32;
}
impl TensorElement for u64 {
    const ELEM: sys::ElementType = sys::ElementType::Uint64;
}
impl TensorElement for bool {
    const ELEM: sys::ElementType = sys::ElementType::Bool;
}

// f16 / bf16 are behind the `half` feature (optional `half` crate dep).
#[cfg(feature = "half")]
impl TensorElement for half::f16 {
    const ELEM: sys::ElementType = sys::ElementType::Float16;
}
#[cfg(feature = "half")]
impl TensorElement for half::bf16 {
    const ELEM: sys::ElementType = sys::ElementType::Bfloat16;
}
