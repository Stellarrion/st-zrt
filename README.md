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
  metadata, IoBinding, owned initializers, mmap-backed dense weights, and raw-byte access for
  packed sub-byte tensor storage.
- **Advanced ORT surfaces**: custom ops, provider config/discovery, CUDA builds, async runs,
  prepacked weights, profiling, threading, session logging, graph/model editing, AOT compile, and
  interop wrappers.
- **Provider-aware serving controls**: IoBinding synchronization and opt-in per-run input rebinding
  for reusable CUDA/TensorRT lanes that need stricter input freshness.
- **Placement diagnostics and explicit copies**: inspect session input/output memory plans,
  EP-device assignment, tensor memory devices, and copy values into reusable buffers via ORT
  `CopyTensors` with a host-accessible fallback.
- **Generated FFI**: `st-zrt-sys` mirrors ONNX Runtime 1.27.0 with a zrt-namespaced raw table and no
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

## Local Hot-Path Benchmarks

Latest local quick run, June 22, 2026, against the repository MNIST/relay benchmark models:

```bash
ST_ZRT_TAIL_ITERS=10000 cargo bench --manifest-path bench-c/Cargo.toml --bench tail_latency
cargo bench --manifest-path bench-c/Cargo.toml --bench runtime_shapes -- --quick
cargo bench --manifest-path bench-c/Cargo.toml --bench inference -- --quick
cargo bench --manifest-path bench-c/Cargo.toml --bench large -- --quick
```

Representative serving-path results:

| Path | Result |
| --- | ---: |
| static IoBinding lane, bind once | p50 18.581 us, p99 22.671 us |
| static IoBinding lane, rebind each run | p50 18.840 us, p99 23.000 us |
| static IoBinding lane, rebind + unsynchronized | p50 18.831 us, p99 22.460 us |
| `runtime/static_io_direct` | 18.953 us |
| `runtime/dynamic_cached_run_on` | 19.368 us |
| `runtime/dynamic_cached_rebind_run_on` | 19.552 us |
| `C_lane` MNIST inference | 19.107 us |
| 4 MiB relay lane | 102.33 us |
| 16 MiB relay lane | 706.89 us |

The unsynchronized run APIs are intentionally opt-in: in these local runs they improve some tail
samples, especially rebind p999, but do not beat synchronized bind-once median latency. Criterion
reported no serving-path regressions after the 0.2.1 caller-owned output, device-output, bucket
prebuild, and warmup API additions.

## Install

```toml
[dependencies]
st-zrt = "0.2.1"
```

The default feature set covers CPU inference. Optional surfaces are explicit:

```toml
# Execution-provider option builders and EP device discovery.
st-zrt = { version = "0.2.1", features = ["ep"] }

# CUDA ONNX Runtime build and strict CUDA inference tests. Implies `ep`.
st-zrt = { version = "0.2.1", features = ["cuda"] }

# Safe Rust custom operator authoring.
st-zrt = { version = "0.2.1", features = ["custom-ops"] }

# Graph/model editing and AOT compile wrappers.
st-zrt = { version = "0.2.1", features = ["model-editor"] }
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
cargo run -p st-zrt --example bert_cuda_probe --features cuda -- path/to/model_cuda.onnx
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
Explicit policies are available when you need predictable memory behavior. Named presets
(`balanced`, `latency`, `throughput_large`, `pinned_host_candidate`) keep the common choices
readable without changing the underlying enum.

CUDA and TensorRT callers that mutate reusable CPU input buffers can opt into stricter binding
freshness:

```rust
lane.set_rebind_inputs_each_run(true);
```

For dynamic shape-bucketed runtimes, use
`DynamicIoOptions::new(max_buckets).with_rebind_inputs_each_run(true)`. The default remains
bind-once because it preserves the zero-allocation CPU serving contract.

Fixed shape plans should be admitted before traffic and warmed once:

```rust
let spec = ShapeSpec::new([&[1, 1, 28, 28]], [&[1, 10]]);
dynamic.prebuild_buckets([spec])?;
dynamic.prime_cached_buckets(8)?;
```

This keeps bucket allocation, IoBinding setup, and first-run ORT cache work out of the request
path. `warm_buckets([spec], runs)` combines prebuild and warmup when startup code already has the
full shape plan.

## Diagnostics

Session-level ORT logging is available through `SessionOptions`:

```rust
let opts = SessionOptions::new()
    .with_log_id("placement-debug")?
    .with_log_severity(st_zrt::LoggingLevel::Verbose)
    .with_log_verbosity(1);
```

This is useful for execution-provider placement, graph optimization, and CUDA Memcpy diagnostics.
The `bert_cuda_probe` example is intentionally diagnostic: it compares normal `Session::run` with
reusable static I/O on BERT-style text encoder ONNX graphs.

For programmatic placement checks, use `Session::io_placement()` or the focused
`input_memory_info` / `output_memory_info` / `input_ep_device` / `output_ep_device` accessors.
Run outputs expose `OwnedValue::memory_info()` and, with `model-editor`, `memory_device()`.
Wrap a tensor output in `DeviceValue` when the caller should make an explicit copy decision:

```rust
let value = DeviceValue::from_owned(outputs[0].take().expect("output"))?;
let mut host = TensorBuffer::<f32>::zeros(&[1, 10], &MemoryInfo::cpu()?)?;
value.copy_to_tensor_buffer(&session, &mut host)?;
```

For CUDA hot paths that must keep outputs device-resident, prefer preallocated outputs:
`prepare_allocated_output_tensor_io_lane(..., &MemoryInfo::cuda(0)?, output_shapes)` binds stable
`AllocatedTensor`s once and avoids `GetBoundOutputValues` in the serving loop. Device-output
bindings that call `output_values()` are useful for inspection and flexible routing, but they are
not the hard zero-allocation output path.

These placement and audit APIs are setup/preflight tools and may allocate. Keep them outside the
measured serving loop. The hot path is: build sessions, construct and audit lanes, prime them,
then repeatedly mutate existing input buffers and call `run`.

For fixed-arity regular runs that still need ORT-owned outputs, `Session::run_array` uses
compile-time input/output counts and stack-backed handle arrays. For hard zero-copy output, prefer
prepared IoBinding or lane APIs with caller-owned `TensorBuffer`s.

## Packed Sub-Byte Tensors

ONNX Runtime 1.27 can report newer packed metadata types such as `UINT4`, `INT4`,
`FLOAT4E2M1`, `UINT2`, and `INT2`. These are not exposed through `as_slice::<T>()`, because one
Rust scalar is not one logical tensor element. Use `as_bytes()` to inspect host-accessible packed
storage, or `Tensor::from_packed_bytes` to wrap an already-packed caller buffer when ORT accepts
that element type through `CreateTensorWithDataAsOrtValue`.

Creation is intentionally fallible per ORT behavior: the runtime may report a packed metadata type
but still reject constructing that type through the C value-creation API.

## Feature Matrix

| Feature | Surface |
|---|---|
| default | CPU inference, tensors, strings, sparse tensors, metadata, IoBinding, prepared runs, lanes, profiling, threading, async run |
| `half` | `f16` and `bf16` tensor element types |
| `serde` | `SessionOptions` and execution-provider config serialization |
| `ep` | CUDA, TensorRT, ROCm, CANN, DNNL, OpenVINO, VitisAI, MIGraphX option builders plus EP device discovery |
| `cuda` | GPU ONNX Runtime build (CUDA 13); links a system CUDA 13 toolkit; implies `ep` |
| `custom-ops` | Safe Rust custom operator registration and kernel callbacks |
| `model-editor` | Graph/model editing, attributed nodes, model serialization, AOT compile, EP registry gateway, external-memory interop |
| `training` | Reserved for ONNX Runtime training packages |

CUDA is currently Linux x86_64 focused. The `cuda` feature downloads the GPU ORT package and links
a system CUDA 13.x toolkit (the `nvidia-*-cu13` wheels are not yet on PyPI). The CUDA 13 runtime
libs are resolved from `ST_ZRT_CUDA13_PATH` → `CUDA_PATH` → `/opt/cuda`, and cuDNN 9 must also be
present on the host.

## Platform Support

| Target | Status |
|---|---|
| Linux x86_64 | reference platform |
| Linux aarch64 | build-supported with SHA-pinned ORT archive |
| macOS arm64 | build-supported with SHA-pinned ORT archive |
| Windows x64 | GPU archives only in ORT 1.27; CPU build via `ST_ZRT_ORT_PATH` or the NuGet package |
| macOS x86_64 | not supported by the ORT 1.27.0 release archive set |

MSRV: Rust **1.85** (edition 2024).

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
git tag -a st-zrt-v0.2.1 -m "st-zrt v0.2.1"
scripts/release-check.sh pre-sys-publish
```

`scripts/release-check.sh` requires a clean worktree and the expected release tag at `HEAD`
(`st-zrt-v<workspace-version>` by default).

## License

Licensed under [Apache-2.0](LICENSE).
