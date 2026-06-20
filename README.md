# st-zrt

[![CI](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml/badge.svg)](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/st-zrt.svg)](https://crates.io/crates/st-zrt)
[![docs.rs](https://docs.rs/st-zrt/badge.svg)](https://docs.rs/st-zrt)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

**A production-minded Rust runtime layer over ONNX Runtime.**

`st-zrt` keeps ONNX Runtime where it is strongest - kernels, graph optimization, and execution
providers - and makes the Rust boundary explicit: zero-copy tensors, prepared fixed-shape runs,
exclusive serving lanes, allocator policy control, and a generated no-bindgen FFI.

It is not a model server and it does not hide scheduling behind a pool. Bring your own Axum, Tokio,
thread-per-core loop, or service framework; `st-zrt` gives you the session, memory, and I/O
primitives to wire inference without wrapper overhead.

## Highlights

- **Zero-copy tensor I/O**: wrap caller-owned buffers with `Tensor::from_buffer`.
- **Prepared hot paths**: pre-marshaled names, shapes, bindings, run options, and lane buffers.
- **Explicit concurrency**: `Runtime` exposes lanes directly instead of locking a
  shared session pool.
- **Configurable output memory**: caller-owned buffers, ORT-allocator-owned outputs, aligned and
  prefaulted lane buffers, hugepage hints, optional `mlock`, CPU or device outputs.
- **Broad ONNX data surface**: numeric tensors, strings, sparse tensors, sequence/map reads,
  metadata, IoBinding, owned initializers, mmap-backed dense weights.
- **Advanced ORT surfaces**: custom ops, provider config/discovery, CUDA builds, async runs,
  prepacked weights, profiling, threading, graph/model editing, AOT compile, and interop wrappers.
- **Generated FFI**: `st-zrt-sys` mirrors ONNX Runtime 1.26.0 with a zrt-namespaced raw table and no
  `bindgen`.

## Measured Integration Result

`st-zrt` replaced the Rust `ort` wrapper in a real downstream embedding service benchmark
(`rs-celer`, BGE small ONNX model, CPU, batch 8 x sequence 64):

```text
cargo bench --features cpu --bench hot_path -- --warm-up-time 1 --measurement-time 2 --sample-size 10

Rust ort wrapper: [43.396 ms 49.779 ms 57.252 ms]
st-zrt:           [34.501 ms 35.155 ms 35.624 ms]
```

That run was **29.8% faster on the mean** and had a much tighter latency spread. This compares the
Rust wrapper layer and session/input/output path around ONNX Runtime; ONNX Runtime itself still
provides the kernels and graph execution.

## Install

```toml
[dependencies]
st-zrt = "0.1"
```

The default feature set covers CPU inference. Optional surfaces are explicit:

```toml
# Execution-provider option builders and EP device discovery.
st-zrt = { version = "0.1", features = ["ep"] }

# CUDA ONNX Runtime build and strict CUDA inference tests. Implies `ep`.
st-zrt = { version = "0.1", features = ["cuda"] }

# Safe Rust custom operator authoring.
st-zrt = { version = "0.1", features = ["custom-ops"] }

# Graph/model editing and AOT compile wrappers.
st-zrt = { version = "0.1", features = ["model-editor"] }
```

On first build, `st-zrt-sys` downloads and SHA-256 verifies the matching ONNX Runtime archive. To
use a local ORT distribution:

```bash
ST_ZRT_ORT_PATH=/path/to/onnxruntime cargo build
```

The directory must contain `include/` and `lib/`.

## Quick Start

```rust
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn main() -> st_zrt::Result<()> {
    let env = Environment::new()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let session = Session::new(&env, "model.onnx", opts)?;
    let memory = MemoryInfo::cpu()?;

    let input_buf = vec![0.0_f32; 1 * 1 * 28 * 28];
    let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &memory)?;

    let mut outputs: Vec<Option<OwnedValue>> = (0..session.output_count()).map(|_| None).collect();
    session.run(&[&input], &mut outputs)?;

    let output = outputs[0].as_ref().expect("output 0");
    let logits = output.as_slice::<f32>()?;
    println!("{:?}", &logits[..3.min(logits.len())]);
    Ok(())
}
```

Examples:

```bash
cargo run -p st-zrt --example basic_inference -- path/to/model.onnx
cargo run -p st-zrt --example primed_lane -- path/to/model.onnx
cargo run -p st-zrt --example sparse_tensor
cargo run -p st-zrt --example ep_config --features ep
cargo run -p st-zrt --example cuda_inference --features cuda -- path/to/model.onnx
```

Some examples default to local benchmark models when present. Those models are intentionally not
committed; pass a model path explicitly or generate/download benchmark models locally.

## Serving Lanes

The normal `run` API is useful for flexible inference. For a fixed-shape service, prepare the I/O
once and reuse it:

```rust
let mut lane = session.prepare_tensor_io_lane::<f32>(
    &MemoryInfo::cpu()?,
    &[&[1, 1, 28, 28]],
    &[&[1, 10]],
)?;

lane.input_mut(0)?.copy_from_slice(&input_buf);
lane.prime(8)?;
lane.run()?;

let logits = lane.output(0)?;
```

This is the API shape `st-zrt` is built around:

- allocate lane buffers once;
- prime ORT memory-pattern/cache behavior before serving;
- route requests to exclusive lanes from your own scheduler;
- choose caller-owned or ORT-owned output allocation policy.

`LaneBufferPolicy::Auto` keeps small tensors on plain `Vec` storage and uses aligned, prefaulted
storage for large static tensors. Very large buffers receive a best-effort Linux hugepage hint.
Explicit policies are available when you need predictable memory behavior.

## Feature Matrix

| Feature | Surface |
|---|---|
| default | CPU inference, tensors, strings, sparse tensors, metadata, IoBinding, prepared runs, lanes, profiling, threading, async run |
| `half` | `f16` and `bf16` tensor element types |
| `serde` | `SessionOptions` and execution-provider config serialization |
| `ep` | CUDA, TensorRT, ROCm, CANN, DNNL, OpenVINO, VitisAI, MIGraphX option builders plus EP device discovery |
| `cuda` | GPU ONNX Runtime build with CUDA 12 runtime libraries; implies `ep` |
| `custom-ops` | Safe Rust custom operator registration and kernel callbacks |
| `model-editor` | Graph/model editing, model serialization, AOT compile, EP registry gateway, external-memory interop |
| `training` | Reserved for ONNX Runtime training packages |

CUDA is currently Linux x86_64 focused. The `cuda` feature downloads the GPU ORT package and CUDA
12 runtime libraries. cuDNN 9 must be available on the host. Override CUDA runtime discovery with
`ST_ZRT_CUDA12_PATH`.

## Platform Support

| Target | Status |
|---|---|
| Linux x86_64 | reference platform |
| Linux aarch64 | build-supported with SHA-pinned ORT archive |
| macOS arm64 | build-supported with SHA-pinned ORT archive |
| Windows x64 | build-supported; `onnxruntime.dll` must be on `PATH` at runtime |
| macOS x86_64 | not supported by the ORT 1.26.0 release archive set |

MSRV: Rust **1.85**.

## Crates

- `st-zrt`: safe runtime API.
- `st-zrt-sys`: generated raw FFI and ORT acquisition/linking logic.
- `st-zrt-sys-codegen`: dev-time generator for the checked-in FFI table.

The benchmark harnesses are standalone crates because the incumbent `ort` crate and `st-zrt-sys`
both link `onnxruntime`.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo check -p st-zrt --all-features
```

Release checks are tag-bound:

```bash
git tag -a st-zrt-v0.1.0 -m "st-zrt v0.1.0"
scripts/release-check.sh pre-sys-publish
```

`scripts/release-check.sh` requires a clean worktree and the expected release tag at `HEAD`
(`st-zrt-v<workspace-version>` by default).

## License

Licensed under [Apache-2.0](LICENSE).
