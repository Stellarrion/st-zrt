# st-zrt Architecture

`st-zrt` is a safe Rust layer over ONNX Runtime 1.27. The crate keeps ORT as the graph execution
engine and optimizes the Rust/ORT boundary: tensor ownership, prepared I/O binding, dynamic-shape
bucket caching, CUDA provider configuration, and serving lanes.

## Layers

```text
application
  |
  |  Session / Runtime / lanes / CUDA graph lanes
st-zrt
  |
  |  generated ORT FFI, version-pinned to ORT 1.27
st-zrt-sys
  |
  |  libonnxruntime
ONNX Runtime
```

- `st-zrt-sys` owns raw generated bindings and ORT dynamic/static library discovery.
- `st-zrt` owns safe handles, lifetime contracts, allocation policy, and serving abstractions.
- `bench-c` is a separate benchmark crate for profiling the st-zrt path. It is not part of the
  library API.

## Core Runtime Shapes

### Regular Session Runs

`Session::run` is the simplest API. Inputs are passed each call and outputs are returned as
`OwnedValue`s. Use this for setup, one-shot inference, and correctness baselines.

### Prepared Runs and IoBinding

Prepared runs and `IoBinding` reduce repeated ORT setup. `Session::run_binding` and
`Session::run_binding_with` intentionally call:

1. `SynchronizeBoundInputs`
2. `RunWithBinding`
3. `SynchronizeBoundOutputs`

That is the safe default for host-readable outputs and provider-managed memory movement. Internal
unsynchronized helpers exist only for carefully-audited lanes and profiling.

### Run configuration

`RunOptions` is a cloneable pure-value configuration and performs no FFI while it is composed.
`RunOptions::freeze` validates its strings and produces a reusable `MaterializedRunOptions` handle.
Sessions, bindings, and lanes accept only the materialized type, so configuration errors occur during
setup rather than on the serving path. The typed `graph_replay` and `enqueued` presets compose
`gpu_graph_id` with execution-provider synchronization policy without string mutation. Frozen handles
also own every `Arc<LoraAdapter>` referenced by ORT. `SyncStream` and both configuration types retain `Arc` ownership guards, so stream attachment is
safe and the originating environment outlives every ORT reference. Sessions and lanes reject
cross-environment streams before running. General on-stream `CopyTensors` remains `unsafe` because
the asynchronous operation can outlive its borrowed buffers; serving lanes fence their owned
buffers before reuse.

### Static Lanes

`ServingLane` owns fixed-shape input/output buffers and a prepared binding. This is the CPU
zero-allocation serving path. CUDA/TensorRT callers that mutate stable CPU input buffers can opt into
per-run input rebinding when required.

### Dynamic Runtime

`DynamicIoRuntime` caches shape buckets containing direct `Vec<ServingLane>` storage. Non-CUDA buckets use bounded LRU
eviction. CUDA graph buckets are different: ORT's legacy CUDA EP keeps captured graphs alive for the
session lifetime, so `cuda_graph=true` treats `max_buckets` as hard capacity and returns an overflow
error for unexpected shapes.

For production CUDA graph serving, prebuild or warm the expected shape set and enable
`DynamicIoOptions::with_strict_shape_cache(true)` so unknown shapes fail before capture/allocation
appears on the request path. Lazy bucket creation is additionally refused while any lane of the
runtime is still in flight (detached into an owned run, or left enqueued by the legacy
`run_enqueued`), because the new bucket's first run captures and capture must not overlap a live
replay — the failure directs callers to prebuild/warm before traffic.

## CUDA Graph Serving

st-zrt exposes two CUDA graph serving topologies. See
[`cuda-graph-paths.md`](cuda-graph-paths.md) for the operational decision matrix.

### Shared Session / Shared Stream

`DynamicIoRuntime` and `ServingLane` can use a shared session, a shared `Arc`-owned CUDA stream, and
per-shape `gpu_graph_id`s — **with one lane per shape bucket**. A bucket mints a single
`gpu_graph_id`, and with a shared session every lane of that bucket replays the same captured graph,
which baked whichever lane captured it; more lanes would silently cross buffers, so construction
rejects `cuda_graph` + shared session + `lane_count > 1`. Host-input lanes cannot take graph ids at
all (replays would read stale inputs); the graph path requires device-resident inputs on the
retained user stream. This is the lowest-friction CUDA graph path for single-stream or
low-concurrency serving; concurrent lanes need the replicated topology below.

### Replicated Worker-Owned Lanes

`DynamicIoRuntime` supports one replicated ORT session and exact retained CUDA stream per lane. Build
and warm it on the owning worker before serving. Capture all lanes serially, wait at a barrier so no
replay overlaps another lane's capture, and only then start concurrent replay. An earlier unpublished lane façade duplicated this
canonical machinery without supporting owned asynchronous runs or device-output chaining; it was
consolidated into `ServingLane`/`DynamicIoRuntime` before any release.

Request distribution remains caller-owned; `bounded_spsc` is provided for the common router-to-worker
case.

## SPSC Queue

`bounded_spsc` is a std-only bounded single-producer/single-consumer queue. It exists because
`std::sync::mpsc` roundtrip overhead is microsecond-scale on the hot request path, while the SPSC
queue is sub-microsecond in the router benchmark.

Use it only where the topology is truly SPSC. The sender/receiver halves are not cloneable by
design.

## Memory placement and reusable buffers

`MemoryInfo` classifies immutable placement once as `MemoryClass`; repeated host-access checks are
plain Rust matches rather than provider-name FFI. Engine-owned tensor values cache the same class on
first access. CPU and CUDA pinned/shared classes permit host slices; CUDA device and arbitrary EP
device classes do not.

`BufferSpec` is the bounded reusable-buffer policy. `BufferSpec::AUTO` preserves the established
size thresholds, while builders such as `BufferSpec::aligned(4096).prefault()` compose alignment,
hugepage, prefault, and mlock behavior without a Cartesian enum. Device-input lanes resolve `AUTO`
to `CUDA_PINNED` staging.

## Ownership and safety boundaries

`Session` is a cheap-clone `Arc<SessionInner>` handle. The final inner owner releases the native
session before its initializer, prepacked-weight, and environment guards. Session-scoped allocators
transfer the same guard into `AllocatedTensor`, and each `IoBinding` owns a guard for its originating
session. Cross-session binding execution is rejected before entering ORT.

- ORT handles are wrapped in owning or borrowed Rust types with explicit drop order.
- Tensor wrappers cache immutable type/count/placement metadata so hot accessors avoid repeated ORT
  introspection. `OutputValue` also caches its caller-buffer pointer, so repeated typed reads perform
  no ORT calls.
- CUDA stream ownership is explicit. `CudaStream::drop` best-effort synchronizes before destroy.
- Pointer-valued CUDA options, such as `user_compute_stream`, are not serialized by `serde`; reattach
  live pointers after deserializing provider options.

## Nonblocking CUDA device-output path

For CUDA graphs, host-visible outputs can force the CUDA EP to perform device-to-host work before
`RunWithBinding` returns. The lowest-host-overhead path therefore uses device-resident inputs **and**
outputs, disables EP end synchronization, and records a reusable event on the exact retained user
stream. `OwnedDynamicIoRun::try_complete` queries that event without blocking; resources and graph
leases remain token-owned while pending. A downstream CUDA stream should wait on the event rather
than involving the host. Host synchronization remains only at an actual CPU-read boundary or as a
teardown/error fallback.

For GPU-to-GPU continuation, `OwnedDynamicIoRun::chain_on_stream` queues
`cudaStreamWaitEvent` on an owned downstream stream, gives a synchronous enqueue closure temporary
access to stable device tensors, then records a second reusable lane event. The returned
`GpuChainedDynamicIoRun` owns the original lane, both streams, graph lease, buffers, and session until
the downstream event completes. Dropping it blocks as a teardown fallback; if event recording fails,
it fences the retained downstream stream and leaks the provider-visible ownership chain only when no
completion proof can be obtained. A callback error is surfaced only after its possibly-partial
downstream enqueue has been fenced.

A sealed/prebuilt bucket can be resolved once with `prepared_bucket_id` and submitted through
`enqueue_prepared`. The handle is an O(1) slot+generation identity: removal/eviction increments the
generation, so stale handles cannot alias recycled slots. Dropped-token recovery uses unique, preallocated lane cells plus a bounded Treiber ready-index
stack rather than a per-request MPSC sender clone. Producers and the sole runtime consumer allocate
nothing and take no locks; stale/double publication, out-of-range notification, or missing fixed-slot
accounting leaks without panicking rather than replacing or destroying provider-visible state.
The consumer uses Acquire ordering both for its initial head observation and failed-CAS retries, so a
retry cannot read a newly published node's relaxed next link before publication becomes visible.
A temporary Loom model confirmed the distinction: the Acquire-failure version exhaustively passed,
while the otherwise-identical Relaxed-failure version admitted a stale sentinel link.

`CudaCompletionPoller` queries multiple same-device events after one calling-thread device
validation. It performs one nonblocking pass only; timeout, sleep, and adaptive-backoff policy remains
with the service reactor. GPU consumers should prefer `chain_on_stream` and avoid host polling.

## Performance Decision Rules

- Avoid request queues on the measured hot path when direct lane ownership is available.
- If a queue is needed, use per-lane SPSC rather than a shared MPSC queue.
- For CUDA graph dynamic serving, prewarm shapes and use strict cache mode.
- Treat host-readable CUDA outputs as a synchronization boundary. If a downstream stage can consume
  device-resident outputs, benchmark a device-output path separately.
- Use `ServingLane::run_event_profiled` and the device-output benchmark before attributing CUDA
  latency to "launch overhead"; ORT synchronization and `RunWithBinding` can dominate.
