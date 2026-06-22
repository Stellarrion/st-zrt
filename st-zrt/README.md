# st-zrt

Safe, zero-overhead Rust runtime bindings for ONNX Runtime 1.27.

`st-zrt` keeps ONNX Runtime as the kernel engine and focuses on the Rust boundary: zero-copy caller
buffers, prepared fixed-shape I/O, explicit lane-based serving, sparse tensors, IoBinding,
profiling, threading options, async runs, custom ops, CUDA/provider configuration, mmap-backed
dense initializers, packed sub-byte raw byte access, session logging, placement diagnostics,
explicit tensor copies, and model-editor access behind feature gates.

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

Use `Session::io_placement()` to inspect ORT's planned input/output memory descriptors and
assigned EP devices. Device-bound outputs can be wrapped in `DeviceValue` and copied explicitly to
`TensorBuffer` or `AllocatedTensor` destinations through ORT `CopyTensors`.
Placement and audit APIs are setup/preflight tools and may allocate; keep them outside the
measured serving loop. Use `Session::run_array` for fixed-arity regular runs with stack-backed
handle arrays, and prepared IoBinding or lane APIs for hard zero-copy output.

For fixed dynamic-shape plans, prebuild and warm buckets before serving:
`dynamic.prebuild_buckets([ShapeSpec::new([input_shape], [output_shape])])?;` followed by
`dynamic.prime_cached_buckets(runs)?;`. For CUDA outputs that should stay device-resident, prefer
`prepare_allocated_output_tensor_io_lane` with `MemoryInfo::cuda(device_id)` so stable
`AllocatedTensor` outputs are bound once; `bind_output_device` plus `output_values()` remains the
flexible inspection path, not the hard zero-allocation serving path.

Latest local quick benchmarks, June 22, 2026, on the repository MNIST/relay models:
`static_io_bind_once` p50 18.581 us / p99 22.671 us, `runtime/static_io_direct` 18.953 us,
`runtime/dynamic_cached_run_on` 19.368 us, `C_lane` 19.107 us, 4 MiB relay lane 102.33 us, and
16 MiB relay lane 706.89 us. Criterion reported no serving-path regressions after the 0.2.1
caller-owned output, device-output, bucket prebuild, and warmup API additions. Unsynchronized runs
remain opt-in: they improved some tail samples locally but did not beat synchronized bind-once
median latency.

The raw generated FFI lives in `st-zrt-sys`.

Packed 2-bit/4-bit tensors are byte-oriented in the safe API. `as_slice::<T>()` remains reserved
for 1:1 Rust scalar element types; use `as_bytes()` or `Tensor::from_packed_bytes` for already
packed storage, subject to ORT accepting that packed element type.

License: `Apache-2.0`.
