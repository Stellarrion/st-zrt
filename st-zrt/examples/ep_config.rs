//! Execution-provider configuration example.
//!
//! ```text
//! cargo run --example ep_config --features ep
//! ```
//!
//! `ep` exposes provider option builders and discovery/attach APIs. It does not by itself switch
//! the crate to a GPU ONNX Runtime binary. Use `cuda` when the program must actually run CUDA
//! inference.

use st_zrt::{
    CudaArenaExtendStrategy, CudaCudnnConvAlgoSearch, CudaProviderOptions, EpProvider,
    SessionOptions,
};

fn main() -> st_zrt::Result<()> {
    let cuda = CudaProviderOptions::new()
        .device_id(0)
        .arena_extend_strategy(CudaArenaExtendStrategy::NextPowerOfTwo)
        .cudnn_conv_algo_search(CudaCudnnConvAlgoSearch::Heuristic)
        .do_copy_in_default_stream(true)
        .use_tf32(true);

    let _opts = SessionOptions::new()
        .with_cuda_options(cuda)?
        .with_execution_provider(EpProvider::OpenVinoV2, &[("device_type", "CPU")])?;

    println!("queued CUDA and OpenVINO provider configuration");
    println!("feature `ep` configured providers; feature `cuda` is the strict GPU runtime gate");
    Ok(())
}
