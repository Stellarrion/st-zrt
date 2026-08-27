# st-zrt

Safe Rust runtime layer over ONNX Runtime 1.27. `st-zrt` keeps ONNX Runtime as the
execution engine and focuses on the Rust boundary: zero-copy caller buffers, prepared
fixed-shape I/O, serving lanes, dynamic shape buckets, and optional CUDA graphs —
without per-request setup, marshaling, or reallocation on the hot path.

## Status

`st-zrt` 0.3 targets ONNX Runtime 1.27 (API 27). Linux x86_64 is the reference
platform; MSRV is Rust 1.85 (edition 2024). See the repository
[CHANGELOG](https://github.com/Stellarrion/st-zrt/blob/main/CHANGELOG.md) for release details.

## Install

```toml
[dependencies]
st-zrt = "0.3.0"
```

The build downloads a SHA-256-pinned ONNX Runtime 1.27.0 archive, or use
`ST_ZRT_ORT_PATH` with a pre-extracted distribution (required on Windows). The sys
crate's [README](https://github.com/Stellarrion/st-zrt/blob/main/st-zrt-sys/README.md)
explains loader-path setup for downstream binaries.

## Minimal CPU inference

```rust,no_run
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn main() -> st_zrt::Result<()> {
    let env = Environment::new()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let session = Session::new(&env, "model.onnx", opts)?;
    let mem = MemoryInfo::cpu()?;

    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem)?;

    let mut outputs: Vec<Option<OwnedValue>> = (0..session.output_count()).map(|_| None).collect();
    session.run(&[&input], &mut outputs)?;

    let logits = outputs[0].as_ref().unwrap().as_slice::<f32>()?;
    println!("{:?}", &logits[..3.min(logits.len())]);
    Ok(())
}
```

## Choosing an API

| Need | Use |
|---|---|
| One-off or flexible inference | `Session::run` |
| Reused fixed shape, one lane | `Session::prepare_tensor_io_lane` → `TensorIoLane` |
| Fixed shape, N concurrent lanes | `StaticIoRuntime::shared_session` |
| Bounded dynamic shapes | `DynamicIoRuntime` + `DynamicIoOptions` |
| CUDA graph hot path | `DynamicIoRuntime` + `CudaConfig::graph_replay` + device inputs |

A *lane* binds caller-owned input/output buffers once; serving then mutates the same
buffers and runs. The repository's benchmark report records the prepared CPU lane at zero
Rust allocations per run after warmup; native ONNX Runtime allocation is measured separately.
CUDA/TensorRT callers that mutate reusable CPU input buffers can
opt into per-run rebinding with `ServingLane::set_rebind_inputs_each_run(true)` or
`DynamicIoOptions::with_rebind_inputs_each_run(true)`. Reusable-buffer placement is
configured with composable `BufferSpec` policies (`AUTO`, `LATENCY`,
`THROUGHPUT_LARGE`, `PINNED_HOST`, `CUDA_PINNED`, or explicit builders such as
`BufferSpec::aligned(4096).prefault()`).

## Features

- `ep` (default): execution-provider option builders (CUDA, TensorRT, ROCm, CANN, DNNL,
  OpenVINO, VitisAI, MIGraphX) and EP device discovery.
- `cuda`: GPU ONNX Runtime build (CUDA 13) linking a system CUDA 13 toolkit + cuDNN 9;
  implies `ep`.
- `half`: `f16`/`bf16` tensor element types.
- `serde`: serializable `SessionOptions` and provider config types.
- `custom-ops`: safe Rust custom-operator authoring.
- `model-editor`: graph/model editing, AOT compile, interop wrappers, and custom-EP
  authoring (`EpAuthor`/`EpFactoryAuthor`, `#[custom_ep]`, in-process registration).

## Platform support

| Target | Status |
|---|---|
| Linux x86_64 | reference platform; only one with automated native link/test coverage |
| Linux aarch64 | archive download supported; CI compile-only |
| macOS arm64 | archive download supported; no automated coverage |
| Windows x64 | no automatic ORT acquisition (set `ST_ZRT_ORT_PATH`); CI compile-only |
| macOS x86_64 | not supported by the ORT 1.27.0 archive set |

## Limitations

- Not a model server: no built-in scheduling, batching, or pool policy — bring your own.
- CUDA graphs require device-resident lane inputs; host-input graph configurations are
  rejected at construction (they replay stale inputs).
- Sequence/map values are read-only (value construction was removed from the ORT C API).
- `training` APIs are not exposed.

## Documentation

- [API docs (docs.rs)](https://docs.rs/st-zrt)
- [Architecture](https://github.com/Stellarrion/st-zrt/blob/main/docs/architecture.md)
- [CUDA graph paths](https://github.com/Stellarrion/st-zrt/blob/main/docs/cuda-graph-paths.md)
- [Benchmark results](https://github.com/Stellarrion/st-zrt/blob/main/docs/v0.3-benchmark-results.md)
- [CHANGELOG](https://github.com/Stellarrion/st-zrt/blob/main/CHANGELOG.md)

The raw generated FFI lives in the
[`st-zrt-sys`](https://github.com/Stellarrion/st-zrt/blob/main/st-zrt-sys/README.md)
crate.

## License

Apache-2.0.
