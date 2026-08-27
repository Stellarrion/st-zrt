//! Execution-provider configuration example.
//!
//! ```text
//! cargo run --example ep_config --features ep
//! ```
//!
//! `ep` exposes provider option builders and discovery/attach APIs. It does not by itself switch
//! the crate to a GPU ONNX Runtime binary. Use `cuda` when the program must actually run CUDA
//! inference.

use st_zrt::{CudaConfig, DeviceInputPolicy, EpProvider, SessionOptions};

fn main() -> st_zrt::Result<()> {
    let cuda =
        CudaConfig::performance(0).with_device_input_policy(DeviceInputPolicy::UnifiedStream)?;

    let _opts = SessionOptions::new()
        .with_cuda(cuda)?
        .with_execution_provider(EpProvider::OpenVinoV2, &[("device_type", "CPU")])?;

    println!("queued CUDA and OpenVINO provider configuration");
    println!("feature `ep` configured providers; feature `cuda` is the strict GPU runtime gate");
    Ok(())
}
