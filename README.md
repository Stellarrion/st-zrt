# st-zrt

[![CI](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml/badge.svg)](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/st-zrt.svg)](https://crates.io/crates/st-zrt)
[![docs.rs](https://docs.rs/st-zrt/badge.svg)](https://docs.rs/st-zrt)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

`st-zrt` is a safe Rust runtime layer over ONNX Runtime 1.27: it removes the repeated
Rust-side setup, marshaling, and copies that a plain wrapper pays on every prepared
inference call. It is not a model server or scheduler, and it does not replace ONNX
Runtime kernels, graph optimization, or execution providers.

## Status

- Latest **published** wrapper: `st-zrt` **0.2.1** (crates.io).
- This checkout is the **unpublished 0.3.0 release candidate**; the changelog and the
  local `docs/v0.3-release-checklist.md` describe it.
- `st-zrt-sys` candidate: **1.27.1** (binding revision over the unchanged ONNX Runtime
  1.27.0 / API 27 ABI).
- Supported ONNX Runtime lines for this release: **1.27** (bundled by `st-zrt-sys`) and
  **1.28** (bring-your-own runtime via `ST_ZRT_ORT_PATH`; the append-only C API keeps the
  API-27 table valid there). `Environment` rejects every other line at creation.
- Linux x86_64 is the reference platform (the only one with automated native link and
  test coverage). MSRV: Rust 1.85, edition 2024.
- Support levels per platform: [`SUPPORT.md`](SUPPORT.md).

## Install and quick start

After 0.3.0 is published:

```toml
[dependencies]
st-zrt = "0.3.0"
```

Until then, run against this checkout:

```bash
cargo run -p st-zrt --example basic_inference -- path/to/model.onnx
```

Model fixtures are intentionally not committed; pass your own ONNX model path. Other focused
examples cover `primed_lane`, `sparse_tensor`, `mmap_initializer`, `custom_op`, `ep_config`,
`cuda_inference`, and `bert_cuda_probe` under [`st-zrt/examples/`](st-zrt/examples/).

Minimal CPU inference (matches `st-zrt/examples/basic_inference.rs`):

```rust
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

    let logits = outputs[0].as_ref().expect("output 0").as_slice::<f32>()?;
    println!("{:?}", &logits[..3.min(logits.len())]);
    Ok(())
}
```

On first build `st-zrt-sys` downloads and SHA-256-verifies the ONNX Runtime archive
(Linux x86_64/aarch64, macOS arm64) or uses `ST_ZRT_ORT_PATH` pointing at an extracted
distribution (required on Windows). The library is dynamic; a downstream binary must
arrange its loader path (`LD_LIBRARY_PATH`, rpath via `DEP_ONNXRUNTIME_LIBDIR`, or DLL
colocation). Details: [`st-zrt-sys/README.md`](st-zrt-sys/README.md).

## What you get

- **Safe typed tensors** — zero-copy input over caller buffers (`Tensor::from_buffer`),
  typed output reads (`OwnedValue::as_slice`), strings, sparse tensors, packed sub-byte
  access via `as_bytes()`.
- **Prepared fixed-shape runs** — bind names, shapes, and buffers once; then mutate the
  same buffers and run without per-call marshaling or reallocation.
- **Serving lanes** — a *lane* is an owned buffer + binding pair for one fixed shape;
  route requests to lanes from your own scheduler (no built-in pool policy).
- **Dynamic shape buckets** — bounded caches of per-shape lanes with O(1)
  generation-checked submission (`DynamicIoRuntime`, `PreparedBucketId`).
- **Execution-provider (EP) configuration** — an *EP* is ORT's backend selector
  (CUDA, TensorRT, ROCm, DNNL, ...); option builders and device discovery are behind
  the default `ep` feature.
- **CUDA graphs with correct inputs** — device-resident inputs/outputs, nonblocking
  exact-stream completion, and GPU-to-GPU chaining; invalid configurations fail closed.
- **Beyond inference** — custom ops, in-process custom-EP authoring, model editing, AOT
  compile, profiling, threading, and async runs under explicit feature flags.

## Choose an API

| Need | Use |
|---|---|
| One-off or flexible inference | `Session::run` |
| Reused fixed shape, single lane | `Session::prepare_tensor_io_lane` → `TensorIoLane` |
| Fixed shape, N concurrent lanes | `StaticIoRuntime::shared_session` |
| Bounded dynamic shapes | `DynamicIoRuntime` + `DynamicIoOptions` (`ServingLane` per bucket) |
| Hot CUDA graph path | `DynamicIoRuntime` + `CudaConfig::graph_replay` + device inputs |

```rust
// fixed shape, prepared once:
let mut lane = session.prepare_tensor_io_lane::<f32>(
    &MemoryInfo::cpu()?,
    &[&[1, 1, 28, 28]],
    &[&[1, 10]],
)?;
lane.input_mut(0)?.copy_from_slice(&input_buf);
lane.prime(8)?;          // warm ORT caches before serving
lane.run()?;
let logits = lane.output(0)?;
```

CUDA/TensorRT callers that mutate reusable CPU input buffers can opt into stricter
binding freshness with `ServingLane::set_rebind_inputs_each_run(true)` or
`DynamicIoOptions::with_rebind_inputs_each_run(true)`. Lane allocation policy uses
composable `BufferSpec` values (`AUTO`, `LATENCY`, `THROUGHPUT_LARGE`, `PINNED_HOST`,
`CUDA_PINNED`, or `BufferSpec::aligned(4096).prefault()`).

Deep dives: `docs/architecture.md` and `docs/cuda-graph-paths.md` (local-only).

## Performance (scoped)

All numbers below are local measurements from `docs/v0.3-benchmark-results.md`
(local-only; 2026-08-13; AMD
Ryzen 9 7900, RTX 4090; characterization, not cross-machine guarantees). They compare
the Rust wrapper/session/I/O path around ONNX Runtime; ORT still executes the graph.

- CPU: on the small relay fixture a prepared lane (~19.0 µs), prepared IoBinding
  (~18.8 µs), and a direct one-thread run (~19.6 µs) are within about 1 µs of each
  other — wrapper cost is already near the noise floor once prepared. The linked report
  retains the ResNet-50 A/B and counting-allocator evidence (0 Rust allocations/run for
  the prepared lane); the measured benefit is hot-path preparation, not faster kernels.
- CUDA graph replay (thenlper/gte-small, batch 1, seq 128, shared GPU):

| Path | Median | vs baseline | Correctness |
|---|---:|---:|---|
| baseline (rebind, no graph) | 750.5 µs | 1.00× | reference |
| host-input graph | 586.2 µs | 1.28× | **stale** (max abs diff 2.9210) |
| device-input graph | 639.5 µs | 1.17× | correct (max abs diff 0.0035) |

- The correct device-input graph gains about 1.2× (1.19× in a lower-contention rerun);
  the aspirational 1.5× target was **not met** on this fixture. The faster host-input
  number is invalid: replays read stale device memory, which is why that configuration
  is rejected at construction.
- Reproduce with `scripts/benchmark-zrt.sh` and the `bench`/`bench-c` harnesses.

## CUDA (advanced, optional)

The `cuda` feature links the GPU ONNX Runtime package (CUDA 13) plus a system CUDA 13
toolkit and cuDNN 9, on Linux x86_64 only. CUDA graphs require device-resident lane
inputs refreshed on a retained user stream; capture is device-wide serialized.
Start with `docs/cuda-graph-paths.md` (local-only) and the
`cuda_inference` / `bert_cuda_probe` examples.

## Features, limits, support

| Feature | Surface |
|---|---|
| default (`ep`) | CPU inference, tensors, prepared lanes, dynamic buckets, EP config/discovery |
| `half` / `serde` | `f16`/`bf16` elements; serializable session/provider config |
| `cuda` | GPU ORT build (CUDA 13); implies `ep` |
| `custom-ops` | safe Rust custom-operator authoring |
| `model-editor` | graph editing, AOT compile, interop, custom-EP authoring |

Known platform and acquisition limits are listed in [`SUPPORT.md`](SUPPORT.md);
CUDA-graph lease semantics and path-selection limits are documented in
`docs/cuda-graph-paths.md` (local-only).

## Project

- `st-zrt` — safe runtime API ([crate README](st-zrt/README.md)).
- `st-zrt-sys` — generated raw FFI and ORT acquisition/linking.
- `st-zrt-sys-codegen` — dev-time generator for the checked-in FFI table.
- `bench/`, `bench-c/`, `bench-cpp/` — standalone A/B/C benchmark crates and the C++
  expert baseline (kept out of the workspace because `ort-sys` and `st-zrt-sys` both link
  `onnxruntime`).

Tracked docs: [CHANGELOG](CHANGELOG.md) · [SUPPORT](SUPPORT.md) ·
[CONTRIBUTING](CONTRIBUTING.md). Deep-dive documents live in `docs/` locally and are
not published.

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Security: [`SECURITY.md`](SECURITY.md).

## License

Licensed under [Apache-2.0](LICENSE). Third-party components — including the ONNX
Runtime binaries downloaded at build time — are summarized in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
