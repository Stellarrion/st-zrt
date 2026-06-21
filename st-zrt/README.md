# st-zrt

Safe, zero-overhead Rust runtime bindings for ONNX Runtime 1.26.

`st-zrt` keeps ONNX Runtime as the kernel engine and focuses on the Rust boundary: zero-copy caller
buffers, prepared fixed-shape I/O, explicit lane-based serving, sparse tensors, IoBinding,
profiling, threading options, async runs, custom ops, CUDA/provider configuration, mmap-backed
dense initializers, session logging, and model-editor access behind feature gates.

```rust
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn main() -> st_zrt::Result<()> {
    let env = Environment::new()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let session = Session::new(&env, "model.onnx", opts)?;
    let mem = MemoryInfo::cpu()?;

    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &mem)?;

    let mut outputs: Vec<Option<OwnedValue>> = (0..session.output_count()).map(|_| None).collect();
    session.run(&[&input], &mut outputs)?;

    let logits = outputs[0].as_ref().unwrap().as_slice::<f32>()?;
    println!("{:?}", &logits[..3.min(logits.len())]);
    Ok(())
}
```

Feature flags:

- `half`: `f16` / `bf16` tensor element types.
- `serde`: serializable `SessionOptions` and provider config types.
- `ep`: execution-provider option builders and device discovery.
- `cuda`: CUDA ONNX Runtime build and strict GPU inference tests; implies `ep`.
- `custom-ops`: safe Rust custom operator authoring.
- `model-editor`: graph/model editing, attributed nodes, AOT compile, EP registry and interop wrappers.

Examples:

```bash
cargo run --example basic_inference
cargo run --example primed_lane
cargo run --example mmap_initializer
cargo run --example sparse_tensor
cargo run --example ep_config --features ep
cargo run --example cuda_inference --features cuda
cargo run --example bert_cuda_probe --features cuda -- /path/to/model_cuda.onnx
```

Reusable lanes bind inputs once by default for the CPU zero-allocation hot path. CUDA/TensorRT
callers that mutate reusable CPU input buffers can opt into per-run input rebinding with
`StaticIoLane::set_rebind_inputs_each_run(true)` or
`DynamicIoOptions::with_rebind_inputs_each_run(true)`.

The raw generated FFI lives in `st-zrt-sys`.

License: `Apache-2.0`.
