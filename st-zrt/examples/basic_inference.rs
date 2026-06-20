//! Minimal end-to-end inference with st-zrt.
//!
//! Loads an ONNX model, runs one inference over a zeroed input, and prints the output
//! shape and first few values. Demonstrates the zero-copy happy path:
//! `Environment` → `Session` → `Tensor::from_buffer` (zero-copy input) → `run` → `OwnedValue`.
//!
//! ```text
//! cargo run --example basic_inference -- [path/to/model.onnx]
//! ```
//! Defaults to the bundled MNIST fixture (`../bench/models/mnist.onnx` when run from
//! `st-zrt/`).
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn main() -> st_zrt::Result<()> {
    // Default to the bundled MNIST fixture, resolved from the crate root so it works
    // regardless of the directory `cargo run --example` was invoked from.
    let default_model = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/models/mnist.onnx");
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_model.to_string());

    let env = Environment::new()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, &model, opts)?;
    let mem = MemoryInfo::cpu()?;

    println!(
        "loaded {} — {} input(s), {} output(s)",
        model,
        sess.input_count(),
        sess.output_count()
    );

    // Zero-copy input: wrap a caller-owned buffer; the engine reads it in place.
    // MNIST is one f32 tensor of shape [1, 1, 28, 28] (784 values).
    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem)?;

    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)?;

    let Some(output0) = out.first().and_then(Option::as_ref) else {
        eprintln!("model produced no output[0]");
        return Ok(());
    };
    let logits = output0.as_slice::<f32>()?;
    println!(
        "output[0]: {} logits; first 3 = {:?}",
        logits.len(),
        &logits[..3.min(logits.len())]
    );
    Ok(())
}
