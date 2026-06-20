//! Dense mmap-backed initializer.
//!
//! This demonstrates the cold-load/RSS-oriented path for external dense weights:
//! write or ship dense typed bytes in a sidecar file, map them with
//! `TensorBuffer::from_mmap_file`, then pass the buffer through `OwnedInitializer::tensor`.
//!
//! This is not compressed-weight decode. ORT sees ordinary dense `f32` bytes and dereferences
//! them directly.
//!
//! ```text
//! cargo run --example mmap_initializer
//! ```
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedInitializer, OwnedValue, Session,
    SessionOptions, Tensor, TensorBuffer,
};
use std::error::Error;

const N: usize = 65_536;

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn main() -> Result<(), Box<dyn Error>> {
    let model = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../bench/models/relay_256k.onnx"
    );
    if !std::path::Path::new(model).exists() {
        eprintln!("skipping: relay_256k.onnx fixture is not present");
        return Ok(());
    }

    let weights_path = std::env::temp_dir().join(format!(
        "st-zrt-example-mmap-initializer-{}.bin",
        std::process::id()
    ));
    let weights = vec![2.0_f32; N];
    std::fs::write(&weights_path, f32_as_bytes(&weights))?;

    let env = Environment::new()?;
    let mem = MemoryInfo::cpu()?;
    let c = TensorBuffer::<f32>::from_mmap_file(&weights_path, &[1, N as i64], &mem)?;
    let initializer = OwnedInitializer::tensor("C", c)?;

    let session = Session::new_with_owned_initializers(
        &env,
        model,
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::All)
            .with_intra_threads(1),
        vec![initializer],
    )?;

    let _ = std::fs::remove_file(&weights_path);

    let x = vec![3.0_f32; N];
    let input = Tensor::from_buffer(&x, &[1, N as i64], &mem)?;
    let mut outputs: Vec<Option<OwnedValue>> = (0..session.output_count()).map(|_| None).collect();
    session.run(&[&input], &mut outputs)?;

    let Some(output0) = outputs[0].as_ref() else {
        eprintln!("model produced no output[0]");
        return Ok(());
    };
    let y = output0.as_slice::<f32>()?;
    println!(
        "mmap initializer override: y[0]={}, y[last]={}, len={}",
        y[0],
        y[N - 1],
        y.len()
    );

    Ok(())
}
