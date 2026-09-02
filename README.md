# st-zrt

[![CI](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml/badge.svg)](https://github.com/Stellarrion/st-zrt/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/st-zrt.svg)](https://crates.io/crates/st-zrt)
[![docs.rs](https://docs.rs/st-zrt/badge.svg)](https://docs.rs/st-zrt)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

`st-zrt` is a safe Rust runtime layer over ONNX Runtime 1.29 that turns prepared
inference into a zero-ceremony hot path: bind tensors once, then run — no per-call
setup, marshaling, allocation, or copying. It keeps ONNX Runtime's kernels, graph
optimization, and execution providers, and leaves scheduling to you.

## Status

- Latest **published** wrapper: `st-zrt` **0.3.0** (crates.io).
- This checkout is the **unpublished 0.4.0 release candidate**; the changelog describes it.
- `st-zrt-sys` candidate: **1.29.0**, mirroring the bundled **ONNX Runtime 1.29.0 / API 29**
  exactly.
- Supported ONNX Runtime line for this release: **1.29** only (bundled by `st-zrt-sys`;
  another 1.29.x runtime may be supplied via `ST_ZRT_ORT_PATH`). `Environment` rejects every
  other line at creation — including 1.27/1.28, which remain supported by the 0.3.x line.
- Linux x86_64 is the reference platform (the only one with automated native link and
  test coverage). MSRV: Rust 1.85, edition 2024.
- Support levels per platform: [`SUPPORT.md`](SUPPORT.md).

Wrapper ↔ ONNX Runtime mapping (`st-zrt-sys` bundles the pinned ORT; "BYO" = another
runtime of that line supplied via `ST_ZRT_ORT_PATH`):

| `st-zrt` | `st-zrt-sys` (bundled ORT) | Accepted ORT runtimes |
|---|---|---|
| 0.1.x | 1.26.0 (1.26) | 1.26 |
| 0.2.x | 1.27.0 (1.27) | 1.27 |
| 0.3.x | 1.27.1 (1.27.0) | 1.27 · 1.28 (BYO) |
| 0.4.x | 1.29.0 (1.29.0) | 1.29 |

## Install and quick start

```toml
[dependencies]
st-zrt = "0.4.0"
```

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

## Benchmarks (local characterization)

All numbers are local characterizations on Linux x86_64, single-threaded, each crate
using its own pinned ONNX Runtime — not cross-machine guarantees. Reproduce with the
in-tree harnesses: `cargo run --release --manifest-path bench-c/Cargo.toml --example
wrapper_floor` (and the `bench/` counterpart) for the floor table, and
`cargo bench --manifest-path bench-c/Cargo.toml --bench inference` (plus the `bench/`
counterparts) for the end-to-end medians.

**Wrapper overhead without kernels** — a single-`Identity` model (kernel ≈ no-op),
per-run time and per-run Rust allocations:

| Identity 1×65536 | `ort` naive | `ort` expert (IoBinding) | `st-zrt` naive | `st-zrt` prepared lane |
|---|---:|---:|---:|---:|
| µs / run | 8.0 | 4.3 | 4.6 | **4.0** |
| allocs / run | 7 | 3 | 1 | **0** |

**End-to-end (kernels included)** — Criterion medians, 10 samples:

| Workload | `ort` naive | `ort` expert | `st-zrt` naive | `st-zrt` lane |
|---|---:|---:|---:|---:|
| MNIST 1×1×28×28 | 20.4 µs · 7 allocs | 19.8 µs · 3 | 18.4 µs · 1 | **18.1 µs · 0** |
| Relay, 4 MiB input | 160.4 µs · 7 | **92.5 µs · 3** | 109.3 µs · 1 | 99.5 µs · 0 |
| ResNet-50, batch 1 | 50.88 ms | 50.61 ms | — | **50.30 ms** |

Reading it honestly: the prepared lane is the only zero-allocation-per-run path, and it
edges every alternative on tiny and kernel-free workloads. Against the *expert* `ort`
IoBinding path the lane is roughly at parity on small models and ~8% behind on the
4 MiB copy-heavy relay — and on a 16 MiB variant the expert path pulls further ahead
(~15%): once kernels and memory traffic dominate, wrapper choice stops being the
bottleneck. On ResNet-50 every path converges to the same ORT kernels (~1% apart) —
the wrapper's job is to add nothing, and neither crate does at that scale. The big
naive-to-lane gaps (−39% on 4 MiB, −50% on Identity) are what eliminating per-run
copies and allocations buys, not faster kernels.

## CUDA (advanced, optional)

The `cuda` feature links the GPU ONNX Runtime package (CUDA 13) plus a system CUDA 13
toolkit and cuDNN 9, on Linux x86_64 only. CUDA graphs require device-resident lane
inputs refreshed on a retained user stream; capture is device-wide serialized.
See the `cuda_inference` / `bert_cuda_probe` examples.

GPU-architecture and privacy notes for the bundled 1.29.0 GPU package: it ships SASS for
sm_75 through sm_90a but **no sm_100a SASS and no PTX** (upstream packaging change), so
Blackwell GPUs (B100/B200/GB200) are unsupported with no forward-JIT fallback — use the
0.3.x line there. The GPU package is also built with POSIX telemetry compiled in;
`st-zrt` disables it by default: every `Environment` constructor sets
`ORT_DISABLE_TELEMETRY=1` before initialization **unless the variable is already
present**, so exporting `ORT_DISABLE_TELEMETRY=0` explicitly keeps telemetry on.

## Features, limits, support

| Feature | Surface |
|---|---|
| default (`ep`) | CPU inference, tensors, prepared lanes, dynamic buckets, EP config/discovery |
| `half` / `serde` | `f16`/`bf16` elements; serializable session/provider config |
| `cuda` | GPU ORT build (CUDA 13); implies `ep` |
| `custom-ops` | safe Rust custom-operator authoring |
| `model-editor` | graph editing, AOT compile, interop, custom-EP authoring |

Known platform and acquisition limits are listed in [`SUPPORT.md`](SUPPORT.md).

## Project

- `st-zrt` — safe runtime API ([crate README](st-zrt/README.md)).
- `st-zrt-sys` — generated raw FFI and ORT acquisition/linking.
- `st-zrt-sys-codegen` — dev-time generator for the checked-in FFI table.
- `bench/`, `bench-c/`, `bench-cpp/` — standalone A/B/C benchmark crates and the C++
  expert baseline (kept out of the workspace because `ort-sys` and `st-zrt-sys` both link
  `onnxruntime`).

Docs: [CHANGELOG](CHANGELOG.md) · [SUPPORT](SUPPORT.md) ·
[CONTRIBUTING](CONTRIBUTING.md).

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Security: [`SECURITY.md`](SECURITY.md).

## License

Licensed under [Apache-2.0](LICENSE). Third-party components — including the ONNX
Runtime binaries downloaded at build time — are summarized in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
