//! Fixed-shape serving lane with explicit warm-up priming.
//!
//! ```text
//! cargo run --example primed_lane -- [path/to/model.onnx]
//! ```
//!
//! Defaults to the bundled MNIST fixture. Real services should prepare one lane per serving
//! worker, fill representative inputs, call `prime`, and then keep reusing that lane.
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, Session, SessionOptions};

fn main() -> st_zrt::Result<()> {
    let default_model = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/models/mnist.onnx");
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_model.to_string());

    let env = Environment::new()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, &model, opts)?;
    let mem = MemoryInfo::cpu()?;

    let mut lane = sess.prepare_tensor_io_lane::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]])?;

    lane.input_mut(0)?.fill(0.0);
    lane.prime(8)?;

    lane.input_mut(0)?.fill(0.0);
    lane.run()?;
    let logits = lane.output(0)?;
    println!(
        "output[0]: {} logits; first 3 = {:?}",
        logits.len(),
        &logits[..3.min(logits.len())]
    );

    Ok(())
}
