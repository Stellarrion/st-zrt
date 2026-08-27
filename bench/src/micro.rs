//! Micro-benchmarks isolating single anti-patterns (task #7).
//!
//! `copy_tensor_f32` isolates **O3** — the cost of `Tensor::from_array` (copy
//! into ORT's allocator) as a function of tensor size. No model required.
use ort::value::Tensor;

/// O3 in isolation: create a 1-D f32 tensor of `n_floats` elements via the
/// owned (copying) path that variant A pays every call. Returns the tensor so
/// the work isn't elided.
#[inline]
pub fn copy_tensor_f32(n_floats: usize) -> ort::Result<Tensor<f32>> {
    // Non-zero fill: forces real, resident pages so the copy is measured honestly
    // (an all-zero source hits OS zero-page shortcuts at large sizes — see RESULTS.md anomaly).
    Tensor::<f32>::from_array((vec![n_floats as i64], vec![1.0; n_floats]))
}
