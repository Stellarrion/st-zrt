//! Strict CUDA inference example.
//!
//! ```text
//! cargo run --example cuda_inference --features cuda -- [path/to/model.onnx]
//! ```
//!
//! Unlike `ep_config`, this requires the CUDA ONNX Runtime build and a working CUDA host.

use st_zrt::{
    CudaArenaExtendStrategy, CudaCudnnConvAlgoSearch, CudaProviderOptions, Environment,
    GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn main() -> st_zrt::Result<()> {
    let default_model = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/models/mnist.onnx");
    let model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_model.to_string());

    let env = Environment::new()?;
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda_options(
            CudaProviderOptions::new()
                .device_id(0)
                .arena_extend_strategy(CudaArenaExtendStrategy::NextPowerOfTwo)
                .cudnn_conv_algo_search(CudaCudnnConvAlgoSearch::Exhaustive)
                .do_copy_in_default_stream(true)
                .use_tf32(true),
        )?;
    let sess = Session::new(&env, &model, opts)?;

    let mem = MemoryInfo::cpu()?;
    let input_data = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&input_data, &[1, 1, 28, 28], &mem)?;

    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)?;
    let logits = out[0].as_ref().expect("output[0]").as_slice::<f32>()?;

    println!(
        "CUDA output: {} logits; first 3 = {:?}",
        logits.len(),
        &logits[..3]
    );
    Ok(())
}
