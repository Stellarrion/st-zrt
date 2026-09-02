# Changelog

All notable changes to st-zrt are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] — 2026-08-23

The serving/CUDA-graph release over the published 0.2.1 line: prepared serving lanes with
owned in-flight state, dynamic-shape bucket runtimes, device-resident CUDA graph I/O with
nonblocking completion, in-process execution-provider authoring, and a hardened
`st-zrt-sys 1.27.1` binding revision. All of this is the net delta from published 0.2.1;
nothing was published in between. An earlier unpublished worker-owned lane façade
was consolidated into the canonical `ServingLane`/`DynamicIoRuntime` design before release.

### Packaging
- `st-zrt-sys 1.27.1` targets the unchanged ONNX Runtime 1.27.0/API 27 ABI and revisions the Rust binding surface because crates.io already contains an earlier `st-zrt-sys 1.27.0` without the finalized API-27 profiling declarations. `st-zrt 0.3.0` pins it exactly (`=1.27.1`) so a later bindings release cannot silently change the native ABI contract.
- The earlier sys crate incorrectly generated `OrtProfilingEventCategory` as an opaque pointer-sized handle. The binding revision models its real `repr(i32)` enum ABI, compile-time pins its `c_int` size/alignment and four discriminants, and intentionally removes that unsound raw type; callers must rebuild against 1.27.1 rather than mix generated tables.
- Publish `st-zrt-sys 1.27.1` first, then verify/publish `st-zrt 0.3.0`; a wrapper dry-run cannot resolve the new sys version before that registry publication.
- Both crate tarballs include the full Apache-2.0 license text rather than relying only on the SPDX manifest field.

### Added — ONNX Runtime 1.28 runtime compatibility
- `st-zrt 0.3` now supports two ORT release lines: **1.27** (the version `st-zrt-sys 1.27.1` acquires and links by default) and **1.28** (a bring-your-own runtime bound via `ST_ZRT_ORT_PATH`). The ORT C API is append-only — API 27 → 28 adds `KernelContext_GetSyncStream` and removes nothing — so the API-27 function table remains valid against 1.28.x; the bindings, tests, and `GetApi(27)` dispatch are unchanged.
- New public surface: `st_zrt_sys::ORT_VERSION` (the exact bundled pin, injected by `build.rs` from the same constant that drives the download/sha256 table), `st_zrt_sys::SUPPORTED_RUNTIME_LINES` (`["1.27", "1.28"]`), `runtime_version_string()` (the loaded runtime's `GetVersionString()`), and `runtime_version_supported()`; `st-zrt` re-exports `ORT_VERSION` and `SUPPORTED_RUNTIME_LINES`.
- Every `Environment` constructor now enforces the supported-line policy before touching ORT: a wrong-line `libonnxruntime` (older, newer, or a different major) fails with an error naming the loaded version, the supported lines, and the bundled pin, instead of loading and misbehaving far from the cause.
- Compatibility evidence (CPU): the full default-features suite (199 tests, 10 binaries) and the all-non-CUDA feature matrix (`half,serde,ep,custom-ops,model-editor`; 480 tests incl. custom-op and model-editor integration tests) pass against **ONNX Runtime 1.28.1** (official GitHub `linux-x64` tgz) via `ST_ZRT_ORT_PATH`, including the new loaded-runtime assertions; the same suites pass against the bundled 1.27.0, and the loaded library is verified by `ldd` to be the 1.28.1 build.
- Compatibility evidence (CUDA): the `cuda_ep` suite (30 tests: graph capture/replay, device-resident I/O, GPU chaining, error/panic fencing, prebuilt buckets, stream identity) and the `cuda_inference`, `primed_lane`, `custom_op`, and `ep_config` examples pass against the **ONNX Runtime 1.28.1 `gpu_cuda13` package** on an RTX 4090 with the CUDA 13.3 toolkit and system cuDNN 9; linkage again verified via `ldd`. The legacy-fmha template pruning (Ampere trait count 6108 → 2925) is confirmed to land in 1.28 with no capability loss (`onnxruntime::flash` still covers fp16 and bf16, never fp32).
- Operational note: running this repository's own test binaries against an `ST_ZRT_ORT_PATH` override additionally needs `LD_LIBRARY_PATH=<ort>/lib` — Cargo injects only build-script out dirs into the test-runner loader path, and the override dir lives outside `target/`. Published crates follow the documented downstream loader setup unchanged.

### Added — serving and CUDA graphs
- Device-resident dynamic outputs, nonblocking exact-stream completion, and owned GPU-to-GPU chaining through reusable CUDA events.
- O(1), generation-checked `PreparedBucketId` submission and a bounded, preallocated lock-free dropped-token recovery stack.
- `CudaCompletionPoller` for one-device-validation batch event queries without embedding a busy-spin policy.
- Reproducible device-output and completion-poller benchmarks, Compute Sanitizer gate, and lifecycle-soak script.
- `bounded_spsc` / `bounded_spsc_with_spins`: a std-only bounded SPSC queue for router-to-worker-lane request paths (`std::sync::mpsc` roundtrip is microsecond-scale; the SPSC queue is sub-microsecond in the router benchmark).
- `DynamicIoOptions::with_strict_shape_cache(true)`: reject unexpected shapes after prewarm instead of allocating/capturing on the request path. With `cuda_graph=true`, `max_buckets` is hard capacity (ORT's legacy CUDA EP keeps captured graphs session-scoped), so eviction no longer silently releases live graph resources.
- CUDA-graph gate benchmark (`bench-c/benches/cuda_graph_gte.rs`): a three-config harness with changing inputs — baseline rebind, host-input graph, device-input graph — reporting latency, speedup, and correctness vs a direct `Session::run` reference.

### Added — EP authoring (feature `model-editor`)
- Implementor vtables (`st-zrt-sys/src/ep_vtables.rs`): the three `repr(C)` tables an execution provider implements (`EpVTable` 24 callbacks, `EpFactoryVTable` 19, `NodeComputeInfoVTable` 3), transcribed field-for-field from `onnxruntime_ep_c_api.h`; layout pinned via `size_of`/`offset_of` asserts.
- `OrtEpApi` helper wrappers (`st-zrt/src/ep_authoring.rs`): `KernelDefBuilder`/`KernelDef`, `OpSchema`/`OpSchemaTypeConstraint`, `ProfilingEvent`.
- `EpAuthor` + `EpFactoryAuthor` traits to implement a custom EP in safe Rust, with `catch_unwind` trampolines that keep panics off the FFI boundary, plus the `#[custom_ep]` macro emitting the cdylib symbols ORT `dlopen`s.
- In-process registration: `OwnedHardwareDevice` + `OwnedEpDevice` + `SessionOptions::append_ep_device`, validated end-to-end against real ORT (a stub EP with CPU fallback runs a session and the factory callbacks fire at init/teardown).

### Added — audit-driven safety and polish
- Primary opaque/runtime/value types gained manual `Debug` implementations.
- Lane pools recover poisoned mutexes instead of bricking on a panic.
- Custom-op APIs no longer expose raw owning FFI handles through safe accessors.
- Tensor metadata and host-accessibility checks are cached on owning wrappers (previously ~11 FFI calls + 2 allocations per slice read).
- `CudaStream::drop` best-effort synchronizes before destroying the stream.

### Changed
- `ServingLane` is the single canonical prepared lane core. Its state is owned behind the handle, so an unfenced in-flight run cannot be freed underneath the provider; `ServingLanePool` checks lanes out of the same owned storage. All lane/bucket machinery was consolidated into `ServingLane` + `DynamicIoRuntime` (an earlier unpublished façade was removed before any release).
- CUDA completion resources retain exact streams and sessions until downstream device completion; failure paths fence or leak rather than risk provider-visible use-after-free.
- Release checks compile every workspace target and feature with Rust 1.85.1, matching the declared MSRV; serde-skipped opaque pointers deserialize explicitly to null on that toolchain.
- CI and local release gates enforce cargo-deny advisory/license/source policy; all-feature CI uses the docs.rs native-link guard and serializes ORT environment tests.
- Release/CI gates enforce locked resolution, independent wrapper/sys feature rustdoc checks, package license inventories, cross-target checks, and byte-for-byte regeneration of the API 27 bindings from the extracted upstream header.
- The default feature set now includes `ep` (a plain build produces a usable CPU crate with EP config/discovery APIs). Downstream builds that pin `default-features = false` are unaffected.

### Migration — from the published 0.2.1 API
- **`StaticIoLane` → `ServingLane`.** The 0.2.1 `StaticIoLane`/`StaticIoLanePool` are renamed and
  re-shaped: `ServingLane` (with `ServingLanePool`) is the canonical prepared lane, and its state
  is owned behind the handle so an unfenced in-flight run cannot be freed underneath the
  provider. Migrate mechanical renames; note `input_mut`/`inputs_mut` now return `Result` and
  reject mutation while a run is in flight (`run_enqueued`/owned tokens).
- **`LaneBufferPolicy` → `BufferSpec`/`BufferStorage`.** The 0.2.1 Cartesian
  `LaneBufferPolicy::AlignedPrefaulted { .. }`-style enum is replaced by composable builder
  methods: `BufferSpec::aligned(4096).prefault()`, plus presets `AUTO`/`LATENCY`/
  `THROUGHPUT_LARGE`/`PINNED_HOST`/`CUDA_PINNED`, over the `BufferStorage` (`Vec`/`Aligned`/
  `CudaPinned`) backing.
- **`CudaPreset` → `CudaConfig` + `DeviceInputPolicy`.** 0.2.1's `CudaPreset::Performance`/
  `CudaPreset::CudaGraph { device_id }`/`CudaPreset::LowMemory { .. }` presets become typed
  `CudaConfig::performance(device_id)` / `CudaConfig::graph_replay(device_id, &stream)` /
  `CudaConfig::low_memory(device_id, gpu_mem_limit)`, with input placement/sequencing expressed
  explicitly by `DeviceInputPolicy` (`DefaultStream`/`UnifiedStream`/`UserStream`).
- **The wrapper `training` feature was removed.** `st-zrt-sys` still carries the deferred
  `training` declarations, but `st-zrt` no longer exposes a `training` feature; `OrtTrainingApi`
  remains unwrapped (the CPU release headers do not include it).
- **CUDA invalid configurations now fail closed at construction.** Host-input `cuda_graph` lanes,
  shared-session multi-lane CUDA-graph buckets, `cuda_graph` + `rebind_inputs_each_run`, and lazy
  CUDA-graph bucket creation while any lane is in flight are rejected instead of silently serving
  wrong results or crashing at replay.

### Removed — breaking cleanup
- The ten legacy `TensorBuffer::zeros_*` constructor wrappers (`zeros_prefaulted`, `zeros_cuda_pinned`, `zeros_aligned`, `zeros_aligned_prefaulted`, `zeros_aligned_mlocked`, `zeros_aligned_mlocked_prefaulted`, `zeros_aligned_hugepage`, `zeros_aligned_hugepage_prefaulted`, `zeros_aligned_hugepage_mlocked`, `zeros_aligned_hugepage_mlocked_prefaulted`). Compose the same behavior through `TensorBuffer::zeros_with(shape, mem, BufferSpec::...)`.
- The `StaticIoRuntime::from_shared_session`/`from_shared_session_with_buffer_policy` aliases; `shared_session`/`shared_session_with_buffer_policy` are the canonical constructor names.
- `RunOptions::plain()`; `RunOptions::new()` is the canonical empty configuration.
- The dead `PinnedBuffer::len_bytes`/`PinnedBuffer::copy_to_device_async` pair; pass `PinnedBuffer::as_ptr()` with `len() * size_of::<T>()` bytes to `memcpy_async_h2d` instead.
- `DynamicIoRuntime::prime_cached_buckets_enqueued`; use `prime_cached_buckets`/`warm_buckets`.

### Fixed — prebuilt CUDA-graph buckets captured on first serve
- `DynamicIoRuntime::prebuild_buckets` created CUDA-graph buckets without capturing; capture
  happened on each lane's first run, and only bucket *creation* was guarded by the runtime-wide
  idleness check. The public sequence `prebuild A+B → enqueue A → first run B` therefore
  attempted B's capture (a plain cache hit, unguarded) while prebuilt A was replaying — capture is
  device-wide serialized and must not overlap a live replay. `get_or_create_bucket_inner` now
  eagerly captures every fresh lane while that idleness guard is proven, so the first served run
  of any prebuilt bucket is a pure replay. Regression:
  `cuda_graph_prebuilt_bucket_first_run_is_replay_not_capture_while_sibling_replays`.
- New `ServingLane::graph_captured()` reports whether the capture run for the current
  `gpu_graph_id` has completed, making eager capture observable at the st-zrt level.
- `StaticIoRuntime::set_gpu_graph_id` validates every lane (idle + device-input fail-closed
  checks) before assigning to any, so a mid-set rejection can no longer leave earlier lanes
  annotated and later lanes not.
- Documented: a deliberately leaked unfenced lane pins its captured-graph lease forever, so a
  later `Session::release_captured_graph` with the same id blocks indefinitely (runtime bucket
  teardown deliberately skips release for buckets it leaks); concurrent capture across separate
  runtimes/session clones is not serialized by st-zrt and must be externally serialized.
- CUDA graph capture remains device-wide serial (a replay during another lane's capture raises
  CUDA errors 900/901); the capturing tests serialize through a shared capture lock and the full
  `cuda_ep` suite is green under the default parallel harness.

### Fixed — build-script acquisition hardening
- `st-zrt-sys` downloads stream to `<archive>.part` and atomically rename only after a complete
  flushed+fsynced body; stale `.part` files are removed and never renamed. Bounded policy for the
  ~200 MiB GPU archive: 30 s connect, 120 s per-read body, 45 min global timeout, 3 attempts with
  backoff and clear diagnostics. A cached archive failing SHA-256 is removed and fetched fresh
  (bounded) instead of poisoning OUT_DIR; verification still runs before extraction, and the
  extraction marker no longer hides a wiped `onnxruntime/` directory.
- `rerun-if-env-changed=DOCS_RS` is emitted unconditionally before the DOCS_RS early return.
- Windows facts corrected: upstream does publish a win-x64 CPU ZIP, but this build script
  extracts `.tgz` only, so automatic Windows acquisition is documented as not implemented;
  Windows users must supply a pre-extracted `ST_ZRT_ORT_PATH` (release ZIP or NuGet package).
- Transitive-rpath overstatement removed: `rustc-link-arg` from a lib-only sys crate does not
  propagate to downstream final binaries. The script now documents downstream loader setup
  (LD_LIBRARY_PATH/DYLD_LIBRARY_PATH, DLL colocation + PATH, or consumer-side rpath) and exports
  `DEP_ONNXRUNTIME_ROOT`/`DEP_ONNXRUNTIME_LIBDIR`/`DEP_ONNXRUNTIME_INCLUDE` Cargo metadata so a
  consumer's build script can emit its own final-binary rpath.

### Known limitation — host-resident lane inputs + cuda-graph
- With host-resident lane inputs, captured CUDA graphs bake device pointers that ORT never
  repopulates on replay, so host-input `cuda_graph` lanes are rejected at construction; the
  supported path is device-resident lane inputs refreshed on the retained user stream
  (`with_device_inputs`/`with_device_input_streams`). Separately, the bind-once host lane can
  serve stale data on CUDA even without a graph for some models; use per-run rebinding
  (`set_rebind_inputs_each_run`/`DynamicIoOptions::with_rebind_inputs_each_run`) or a
  non-graph `CudaConfig` for host-input serving.

### Verified
- Post-hardening validation on the final tree is recorded in
  the local `docs/v0.3-release-checklist.md` (not tracked): default-features test
  suite (serialized ORT environments) and the `cuda_ep` suite including the eager-capture
  regression all green; clippy `-D warnings` and `cargo fmt --all --check` clean; Compute
  Sanitizer memcheck/racecheck/initcheck/synccheck zero errors; 10,000-iteration device-output
  and GPU-chain soak plus a 100-cycle lifecycle soak passed.

## [0.2.0] — 2026-06-22

Tracks ONNX Runtime 1.27.0 (`st-zrt-sys` is now version-mirrored to `1.27.0`, API version 27) and
moves the workspace to Rust edition 2024.

### Changed
- **ONNX Runtime 1.27.0**: regenerated the FFI table (422 `OrtApi` fields, +3 API-27 functions
  including EP-selection policy + delegate). `API_VERSION` is now 27.
- **CUDA 13 track**: the `cuda` feature now uses the ORT `linux-x64-gpu_cuda13` package and links a
  system CUDA 13.x toolkit (resolved from `ST_ZRT_CUDA13_PATH` → `CUDA_PATH` → `/opt/cuda`). ORT
  1.27 deprecated the CUDA 12 packages; `nvidia-*-cu13` wheels are not yet on PyPI, so the CUDA 13
  runtime libs + cuDNN 9 are expected on the host. The cu12 wheel-fetch machinery
  (`CUDA12_WHEELS`, `fetch_cuda12`, `extract_wheel_libs`, the `zip` build-dependency) was removed.
- **Edition 2024** across the workspace and bench crates (`extern "C"` blocks are now
  `unsafe extern`, `unsafe` ops in `unsafe fn`s are explicitly wrapped). MSRV unchanged at 1.85.
- `win-x64` CPU: ORT 1.27 dropped the GitHub win-x64 CPU archive; the build now points Windows-x64
  CPU users to `ST_ZRT_ORT_PATH` or the NuGet `Microsoft.ML.OnnxRuntime` package.
- Pinned SHA-256 refreshed for the 1.27 archives (linux-x64, linux-aarch64, osx-arm64,
  linux-x64-gpu_cuda13).

### Added
- ONNX element-metadata coverage for `Float8E8M0`, `Uint2`, and `Int2` (ONNX 1.21): `element_size`
  reports them (Float8E8M0 = 1 byte; packed 2-bit = opaque, like int4/uint4/float4).

### Fixed (upstream, picked up by the engine bump)
- Double-free in `OrtModelEditorApi` ownership transfer (ORT #28123).
- Session use-after-free when `UserLoggingFunction` is used (ORT #28314).
- Plugin EP provider-library load refcount leak (ORT #28396, #28430).

### Breaking
- `st-zrt-sys` `1.26.0` → `1.27.0`; `st-zrt` now depends on `st-zrt-sys = "1.27.0"`.
- `cuda` requires a host CUDA 13 toolkit (no longer auto-fetches cu12 wheels).

## [0.1.1] — 2026-06-21

Patch release focused on CUDA diagnostics, model-editor graph augmentation, and newer ONNX
metadata coverage.

### Added
- `SessionOptions::with_log_id`, `with_log_severity`, and `with_log_verbosity` for ORT session
  placement/debug logging. This is useful when diagnosing CUDA execution-provider partitioning and
  inserted Memcpy nodes.
- `StaticIoLane::set_rebind_inputs_each_run` and
  `DynamicIoOptions::with_rebind_inputs_each_run`. The default remains bind-once for the CPU
  zero-allocation contract; CUDA/TensorRT callers can opt into per-run input rebinding when ORT
  reusable CPU input bindings would otherwise observe stale mutated inputs.
- `model-editor` `NodeAttr` and `Node::with_attributes`, covering scalar/array ONNX node
  attributes without enabling the `custom-ops` feature.
- Model-editor session finalization now refreshes cached input/output metadata after graph edits,
  so sessions augmented with `apply_model` expose the finalized output names/shapes.
- Metadata coverage for newer ONNX tensor element types: complex, float8, int4/uint4, and
  float4. Packed 4-bit values remain opaque for typed Rust slices.
- `bert_cuda_probe` CUDA diagnostic example for comparing direct `Session::run` with reusable
  `StaticIoLane` on BERT-style encoder graphs.

### Changed
- `Session::run_binding` and `run_binding_with` now synchronize bound inputs before execution and
  outputs after execution. This makes IoBinding behavior safer for CUDA/provider-backed runs while
  preserving the existing API shape.

### Verified
- Full default `st-zrt` test suite passed locally.
- CUDA example compile check passed with `--features cuda`.
- Downstream `rs-celer` CUDA release build passed against this local runtime after enabling
  per-run input rebinding only for CUDA/TensorRT text lanes.

## [0.1.0] — 2026-06-20

First tagged release: the full CPU inference surface over libonnxruntime 1.26.0.

### `st-zrt-sys` — generated FFI
- 419 `OrtApi` functions generated from the preprocessed header (`gcc -E -P`, own parser — no
  `bindgen`, no `Ort*` names). ABI-proven (`api_table_loads`, `generated_indices_functionally_validated`).
- Supply-chain: libonnxruntime 1.26.0 downloaded and **SHA-256 verified** at build time.
- Off-by-default feature gates: `ep`, `custom-ops`, `training`, `model-editor`.

### `st-zrt` — safe layer
- `Environment` (Arc-shared — the Env outlives every `Session`; eliminates the use-after-free that
  releasing the Env first would cause), `SessionOptions` (pure-value config), `MemoryInfo` (+ named
  constructor + getters), `Allocator` (+ create/allocate/RAII), `ArenaCfg`.
- `Session` — `new` / `from_bytes` (in-memory, no temp file) / `run` / `run_with` / `run_binding` /
  `run_binding_with` / `metadata`; pre-marshaled I/O names, reused `RunOptions`, cached output type/shape.
- Zero-copy I/O: `Tensor` (owning numeric input — releases its `OrtValue` on drop) +
  `TensorView` (borrowed read view — what a custom-op kernel receives from
  `KernelContext::input`; never released), `StringTensor`, `OutputValue` (output via
  `IoBinding`), `OwnedValue` (tensor/sequence/map read + strings), `TensorTypeAndShapeInfo`.
- `RunOptions` (log level/tag/config + terminate), `ModelMetadata`.
- Typed `OrtErrorCode` (all 15 codes, verified against `onnxruntime_c_api.h:257-273`);
  `Error::ort_code()` recovers it and `Display` names the code. ABI-proven
  (`ort_error_code_round_trips` round-trips `CreateStatus`→`GetErrorCode` through the engine).
- All tensor element types incl. `f16`/`bf16` (`half` feature).

### Build system (cross-platform, pure-Rust)
- `st-zrt-sys/build.rs` rewritten in pure Rust — `ureq` + rustls download, `sha2` verify,
  `flate2`/`tar` + `zip` extract — no `curl`/`tar`/`sha256sum` shell-outs. Reproducible on
  Linux/macOS/Windows runners; target is detected at runtime from the `TARGET` triple
  (not `#[cfg]`, so it is correct under cross-compilation).
- SHA-256 pinned for all four CPU targets: linux-x64, linux-aarch64, osx-arm64, win-x64.
- macOS x86_64 is unsupported (ORT 1.26.0 ships no Intel-mac build).

### Verified
- Zero overhead: st-zrt at 1 intra-op thread = `ort`'s 1-thread floor (19.78 µs vs 19.70–19.88 µs).
- IoBinding zero-copy output validated bit-for-bit against the regular `run()` path.
- Regression test `session_outlives_env_drop` proves the Env-use-after-free is gone.

### Extension surfaces (off-by-default features)
- `ep` — execution-provider option builders (CUDA / TensorRT / ROCm / CANN / DNNL / OpenVINO
  V2 / VitisAI / MIGraphX / OpenVINO v1). The options-handle providers (CUDA/TRT/ROCm/CANN/DNNL),
  OpenVINO V2, and VitisAI use `with_execution_provider` (key/value); MIGraphX and the deprecated
  OpenVINO v1 use flat `#[repr(C)]` config structs (`MigraphxOptions` + `with_migraphx`,
  `OpenvinoOptions` + `with_openvino` — prefer V2 over v1). The lifecycle is exercisable on any
  host (a GPU/accelerator is needed only to *run* the session); every append is FFI-verified on
  CPU (returns "EP not available"), and the two flat structs' layouts are pinned to the header
  via C probes (MIGraphX sizeof=88, OpenVINO v1 sizeof=56). EP coverage is now complete.
- `custom-ops` — custom-operator authoring end-to-end: `CustomOp` trait + `custom_op!` macro
  (emits the `OrtCustomOp` vtable with sound `catch_unwind`→`ORT_FAIL` trampolines), the
  kernel-time API (`KernelInfo` / `KernelContext` / `Op` / `OpAttr`), zero-copy kernel I/O
  (`TensorView::as_slice` read + `KernelContext::output_mut` write), and `CustomOpDomain`
  registration (`add_op`). The whole path — register → `create` → `compute` → `destroy` — is
  **runtime-verified on CPU** via a bundled `com.example::MyRelu` fixture
  (`tests/fixtures/custom_relu.onnx`, regenerated by `gen_custom_relu.py`); the OpAttr
  round-trip + the domain lifecycle are unit-tested too. Shape inference is wired too:
  `CustomOp::infer_shapes` + `ShapeInferContext` (read input type+shape, set output type+shape)
  + a `TensorTypeAndShapeInfo` builder (`new` / `set_element_type` / `set_dimensions`),
  runtime-verified via an unshaped-output fixture that loads only if `infer_shapes` fires.
  (Empirical: `SetOutputTypeShape` *takes ownership* of the info — releasing it double-frees,
  despite its `const` C annotation.)
- `model-editor` — typed `#[repr(C)]` vtables + safe gateway accessors for the four deref-style
  sub-APIs (`ModelEditorApi` / `CompileApi` / `EpApi` / `InteropApi` = 121 functions, where
  `EpApi` is the modern DirectML/QNN/CoreML surface); per-function safe wrappers are added on demand.

### Known limits (honest)
- Sequence/map **value construction** is impossible from the C API (removed in 1.26.0) — read path only.
- Custom-op `compute` / `create` / `destroy` **are runtime-exercised** (`tests/custom_op_run.rs`
  loads a `com.example::MyRelu` fixture end-to-end and asserts the ReLU output). The earlier
  "compile-verified only" caveat is resolved; the schema callbacks, registration lifecycle,
  OpAttr round-trip, and `TensorView::as_slice` are likewise runtime-verified on CPU.
- `training` is deferred: `OrtTrainingApi` is not in the CPU release headers; needs the
  `onnxruntime-training` package.
