# CUDA graph serving path

st-zrt 0.3 has one canonical CUDA graph path: `ServingLane` and `DynamicIoRuntime`.
An earlier unpublished typestate façade duplicated shape lookup, graph-id allocation, capture,
stream, and replay logic while lacking owned in-flight runs, device outputs, and GPU chaining; it
was consolidated into this canonical path before any release.

## Canonical setup

1. Create one retained `Arc<CudaStream>` per replicated session/lane.
2. Configure each session with `CudaConfig::graph_replay(device_id, &stream)`.
3. Build `DynamicIoRuntime` with `DynamicIoOptions::with_cuda_graph(true)` and either
   `with_device_inputs` or `with_device_input_streams`.
4. Install a finite `ServingShapePlan`, prebuild every bucket, and warm/capture every lane before
   serving. Capture calls remain device-wide serial and must not overlap live replay.
5. Resolve each `PreparedBucketId` once. Its slot+generation lookup is O(1), allocation-free, and
   rejects stale handles after removal or eviction.
6. Serve with `enqueue_prepared`. Use `try_complete`/`CudaCompletionPoller` for CPU scheduling or
   `chain_on_stream` for GPU consumers. Never expose/reuse outputs before completion.

## Fail-closed configurations

These configurations used to build and silently serve wrong results; construction now rejects them
before any provider work:

- **Host-input `cuda_graph`.** ORT captures the device buffers it is handed and never repopulates
  them from host bindings on replay, so replays read stale or never-initialized inputs (the
  earlier `cuda_graph_host_input_replay_reads_stale` limitation). The only supported graph input
  path is device-resident lane inputs refreshed on the retained user stream
  (`with_device_inputs`/`with_device_input_streams`). Rejected in `DynamicIoOptions::validate` and
  in `ServingLane::set_gpu_graph_id` (host-input lanes cannot take a graph id).
- **Shared session + `cuda_graph` + more than one lane per bucket.** One bucket mints one
  `gpu_graph_id`, and with a shared session every lane of that bucket replays the same captured
  graph — which baked whichever lane ran first, so the other lanes silently read and write that
  lane's buffers. Use replicated sessions (one session + exact stream per lane) or a single lane.
- **Lazy bucket creation under traffic.** A new CUDA-graph bucket captures eagerly at creation
  time, and capture must not overlap a live replay. `get_or_create_bucket`/`enqueue_owned` refuse
  to create a bucket while any lane of the runtime is detached into an owned run or still in
  flight after a legacy `run_enqueued`; the error directs callers to prebuild/warm the whole
  planned shape set before serving. Eager capture at creation also closes the
  prebuild-then-serve window: buckets returned by `prebuild_buckets` are already captured, so the
  first served run of any prebuilt bucket (a plain cache hit) is a replay and can never capture
  while a sibling bucket is in flight.

## Capture constraints

- CUDA graph shapes are finite, canonical, prewarmed, and sealed.
- Pointers must remain stable across replay; lane-owned staging/device tensors provide that stability.
- Capture across test binaries or lanes must be serialized. Replay may be concurrent only after all
  captures complete.
- The idleness guard in `DynamicIoRuntime` is per-runtime. Concurrent capture across separate
  `DynamicIoRuntime`s (including runtimes built from cloned sessions), or between a runtime and a
  directly owned `ServingLane` capturing its first run, is not serialized by st-zrt: callers must
  serialize it externally (the test suite does this with `cuda_graph_capture_lock`).
- A deliberately leaked unfenced `ServingLane` keeps its captured-graph lease forever. A later
  `Session::release_captured_graph` with the same annotation id then blocks indefinitely waiting
  for that lease to drain — owners that leak lanes must not release those ids. The runtime's
  internal bucket teardown deliberately skips release for buckets it leaks instead of blocking.
- Explicit bucket removal (`remove_bucket`) releases the retired bucket's captured graph on a
  background thread. On a replicated-session runtime that release runs once per session and can
  still be in flight while the caller creates a replacement bucket, whose eager capture starts
  immediately. st-zrt does not serialize that release against the new capture; correctness relies
  on ORT's device-wide capture serialization. This is an accepted residual risk: avoid
  remove/recreate churn on live shape buckets and prebuild the full planned shape set instead.
- Every graph id comes from the checked runtime/session allocator and is never reused.
- Replicated sessions with one exact retained stream per lane are the safest concurrent topology;
  a shared session is single-lane-per-bucket only (see fail-closed configurations above).

## Completion

Host outputs force a D2H synchronization boundary. Device outputs plus disabled EP end-sync allow ORT
submission to return in a few microseconds. `OwnedDynamicIoRun::chain_on_stream` installs a GPU-side
wait and retains resources through a downstream event. `CudaCompletionPoller` validates the device
once per batch query and implements no busy-spin policy; callers own deadline/backoff scheduling.

Use `bench-c/examples/device_output_no_sync.rs` and `docs/v0.3-benchmark-results.md` for current
evidence. CUDA graph tests must use the shared capture lock and the serialized release gate.
