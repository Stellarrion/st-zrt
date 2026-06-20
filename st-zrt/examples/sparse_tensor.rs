//! Sparse tensor construction and readback.
//!
//! ```text
//! cargo run --example sparse_tensor
//! ```
//!
//! This example does not need a model. It exercises the ORT sparse tensor value APIs directly:
//! copied COO construction, zero-copy COO construction, and sparse value/index readback.

use st_zrt::{MemoryInfo, SparseIndicesFormat, SparseTensor};

fn main() -> st_zrt::Result<()> {
    let mem = MemoryInfo::cpu()?;

    let values = [1.0_f32, 2.0, 3.0];
    let indices = [0_i64, 1, 1, 0, 1, 2];
    let copied = SparseTensor::copy_coo(&values, &[2, 3], &[3], &indices, &mem)?;

    println!("copied sparse format: {:?}", copied.format()?);
    println!("copied values: {:?}", copied.values_as_slice()?);
    println!(
        "copied COO indices: {:?}",
        copied.indices_i64(SparseIndicesFormat::Coo)?
    );

    let mut values = [4.0_f32, 5.0, 6.0];
    let mut indices = [0_i64, 0, 0, 2, 1, 1];
    let caller_values = values.as_ptr();
    let zero_copy = SparseTensor::from_coo_buffer(&mut values, &[2, 3], &[3], &mut indices, &mem)?;

    println!(
        "zero-copy values pointer: {:p}",
        zero_copy.values_data_ptr()?
    );
    println!("caller values pointer:   {:p}", caller_values);
    Ok(())
}
