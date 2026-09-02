//! CUDA execution-provider inference coverage.
//!
//! With only `ep`, this file compiles and skips if CUDA is unavailable. With the `cuda` feature,
//! CUDA availability is part of the release gate: session creation and runs must succeed.

#![cfg(feature = "ep")]

mod common;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use st_zrt::{AllocatedTensor, BufferSpec, ServingLane, ServingLanePool};
use st_zrt::{
    CudaConfig, DynamicIoOptions, DynamicIoRuntime, Environment, GraphOptimizationLevel, IoBinding,
    MemoryInfo, OutputValue, OwnedValue, PreparedRun, Runtime, Session, SessionOptions, Tensor,
};
#[cfg(feature = "cuda")]
use st_zrt::{OutputPolicy, ServingShapePlan, ShapeSpec};

fn mnist_path() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx");
    if path.exists() {
        Some(path)
    } else if cfg!(feature = "cuda") {
        panic!("cuda release gate requires bench/models/mnist.onnx");
    } else {
        eprintln!("skip — mnist.onnx not cached");
        None
    }
}

fn cpu_session(env: &Environment, path: &std::path::Path) -> Session {
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    Session::new(env, path.to_str().unwrap(), opts).expect("cpu session")
}

fn cuda_session(env: &Environment, path: &std::path::Path) -> Option<Session> {
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::performance(0))
        .expect("append CUDA options");
    match Session::new(env, path.to_str().unwrap(), opts) {
        Ok(s) => Some(s),
        Err(e) if cfg!(feature = "cuda") => panic!("CUDA EP unavailable in cuda build: {e}"),
        Err(e) => {
            eprintln!("CUDA EP unavailable on this build/host — skipping ({e})");
            None
        },
    }
}

fn zero_input<'a>(mem: &MemoryInfo, buf: &'a [f32]) -> Tensor<'a> {
    Tensor::from_buffer(buf, &[1, 1, 28, 28], mem).expect("zero-copy input")
}

fn run_cpu_reference(sess: &Session, input: &Tensor<'_>) -> Vec<f32> {
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[input], &mut out).expect("cpu run");
    out[0]
        .as_ref()
        .expect("cpu output")
        .as_slice()
        .expect("cpu output read")
        .to_vec()
}

fn assert_logits_close(cpu: &[f32], got: &[f32]) {
    assert_eq!(got.len(), 10, "MNIST output is 10 logits");
    for (a, b) in cpu.iter().zip(got.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "CUDA vs CPU logit mismatch: cpu={a} gpu={b}"
        );
    }
}

#[test]
fn cuda_ep_regular_run_matches_cpu() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);
    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut out: Vec<Option<OwnedValue>> = (0..cuda.output_count()).map(|_| None).collect();
    cuda.run(&[&input], &mut out).expect("cuda run");
    let got = out[0]
        .as_ref()
        .expect("cuda output")
        .as_slice::<f32>()
        .expect("cuda output read");
    assert_logits_close(&cpu_logits, got);
}

#[test]
fn cuda_prepared_run_matches_cpu() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut run: PreparedRun<'_, '_> = cuda.prepare_run(&[&input]).expect("prepare run");
    run.run().expect("prepared cuda run");
    let got = run
        .output(0)
        .expect("prepared output index")
        .expect("prepared output")
        .as_slice::<f32>()
        .expect("prepared output read");
    assert_logits_close(&cpu_logits, got);
}

#[test]
fn cuda_iobinding_cpu_output_matches_cpu() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut out_buf = vec![0.0_f32; 10];
    let out = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("cpu output value");
    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output(cuda.output_name(0).expect("output name"), &out)
        .expect("bind output");
    cuda.run_binding(&binding).expect("cuda iobinding run");
    binding.synchronize_outputs().expect("sync outputs");
    assert_logits_close(&cpu_logits, out.as_slice::<f32>().expect("read cpu output"));
}

#[test]
fn cuda_iobinding_device_output_reports_cuda_memory() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let cpu_mem = MemoryInfo::cpu().expect("cpu mem");
    let cuda_mem = MemoryInfo::cuda(0).expect("cuda memory info");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&cpu_mem, &buf);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_device(cuda.output_name(0).expect("output name"), &cuda_mem)
        .expect("bind cuda output device");
    cuda.run_binding(&binding).expect("cuda device-output run");
    binding.synchronize_outputs().expect("sync cuda outputs");

    let vals = binding.output_values().expect("output values");
    assert_eq!(vals.len(), 1);
    let info = vals[0].memory_info().expect("output memory info");
    assert_eq!(info.name, "Cuda");
    assert_eq!(info.device_id, 0);
    assert!(
        vals[0].as_slice::<f32>().is_err(),
        "device-resident output must not expose a Rust slice"
    );
    let mut host = st_zrt::TensorBuffer::<f32>::zeros(&[1, 10], &cpu_mem).expect("host output");
    let copy = vals[0].copy_to_tensor_buffer(&cuda, &mut host);
    assert!(
        copy.is_err(),
        "CUDA-to-host copy must be explicit and must not be silently emulated"
    );
}

fn dynamic_batch_path() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dynamic_batch.onnx");
    if path.exists() {
        Some(path)
    } else if cfg!(feature = "cuda") {
        panic!(
            "cuda release gate requires tests/fixtures/dynamic_batch.onnx \
             (regenerate: python3 st-zrt/tests/fixtures/gen_dynamic_batch.py)"
        );
    } else {
        eprintln!("skip — dynamic_batch.onnx not cached");
        None
    }
}

/// A CUDA session with CUDA-graph capture enabled (`enable_cuda_graph=true`).
#[cfg(feature = "cuda")]
fn cuda_graph_session(env: &Environment, path: &std::path::Path) -> Option<Session> {
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::performance(0).with_cuda_graph(true))
        .expect("append CUDA graph options");
    match Session::new(env, path.to_str().unwrap(), opts) {
        Ok(s) => Some(s),
        Err(e) if cfg!(feature = "cuda") => {
            panic!("CUDA graph capture unavailable in cuda build: {e}")
        },
        Err(e) => {
            eprintln!("CUDA graph capture unavailable on this host — skipping ({e})");
            None
        },
    }
}

/// Like [`cuda_graph_session`] but pins ORT to a caller-owned CUDA `stream` via
/// `user_compute_stream` — ORT then replays the captured graph on that stream, so a device-input
/// lane can refresh its bound device buffer on the same stream (race-free ordering).
#[cfg(feature = "cuda")]
fn cuda_graph_session_streamed(
    env: &Environment, path: &std::path::Path, stream: &Arc<st_zrt::CudaStream>,
) -> Option<Session> {
    let cuda = CudaConfig::graph_replay(0, stream).expect("CUDA graph config");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(cuda)
        .expect("append CUDA graph+stream options");
    match Session::new(env, path.to_str().unwrap(), opts) {
        Ok(s) => Some(s),
        Err(e) if cfg!(feature = "cuda") => {
            panic!("CUDA graph+stream capture unavailable in cuda build: {e}")
        },
        Err(e) => {
            eprintln!("CUDA graph+stream capture unavailable on this host — skipping ({e})");
            None
        },
    }
}

/// CUDA-graph capture/replay mechanics on the **device-input** path (the deterministic cuda-graph
/// lane mode). With `enable_cuda_graph=true` + `user_compute_stream` + device-resident lane inputs
/// refreshed each run, the CUDA EP captures a graph keyed by the `gpu_graph_id` run-config entry on
/// the first run with that id and replays it after. The runtime mints a distinct `gpu_graph_id` per
/// shape bucket, so batch=1 and batch=2 become two independently captured graphs on one session.
/// This proves capture, replay, multi-graph coexistence, clean explicit release, and
/// release-on-eviction — all on the device-input path, so outputs are deterministic (the inverse of
/// the host-input `cuda_graph_host_input_configuration_is_rejected` fail-closed guard). (cuda.)
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_captures_replays_and_releases_multiple_batches() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    // CUDA graph capture is device-wide serial (see `cuda_graph_capture_lock`); hold it for the
    // whole test so concurrent capturing tests cannot interleave capture with this one.
    let _capture_guard = common::cuda_graph_capture_lock();
    // Caller-owned CUDA stream: ORT replays the captured graph on it (user_compute_stream); the
    // device-input lanes refresh their bound device buffers on the same stream each run.
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let Some(sess) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    let session = sess;

    // --- Session 1: capture/replay + multi-graph coexistence + explicit release ---
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session.clone(),
        MemoryInfo::cpu().expect("cpu input mem"),
        MemoryInfo::cpu().expect("cpu output mem"),
        1, // one lane per bucket
        DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device input options"),
    )
    .expect("cuda-graph device-input runtime");
    let mut plan_builder = ServingShapePlan::builder();
    plan_builder
        .add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::HostBuffer)
        .add_shape([vec![2, 32]], [vec![2, 4]], OutputPolicy::HostBuffer);
    runtime
        .install_shape_plan(Arc::new(plan_builder.build().expect("shape plan")))
        .expect("install shape plan");

    // batch=1: the bucket's graph was eagerly captured at creation, so every served run replays.
    // The host staging buffer is held at 1.0 across replays, so a correct replay reproduces the
    // same output exactly.
    let cap1_run = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(1.0);
            Ok(())
        })
        .expect("batch=1 enqueue capture");
    let cap1 = runtime
        .complete_owned(cap1_run, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("batch=1 capture");
    assert_eq!(cap1.len(), 4);
    assert!(
        cap1.iter().all(|v| v.is_finite()),
        "capture produced finite output"
    );
    let rep1_run = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |_| Ok(()))
        .expect("batch=1 enqueue replay");
    let rep1 = runtime
        .complete_owned(rep1_run, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("batch=1 replay");
    assert_eq!(
        rep1, cap1,
        "cuda-graph replay must reproduce the captured output"
    );

    // batch=2: a new shape bucket -> a new gpu_graph_id -> a second captured graph coexisting
    // with the first on the same session.
    let cap2_run = runtime
        .enqueue_owned([&[2, 32]], [&[2, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(2.0);
            Ok(())
        })
        .expect("batch=2 enqueue capture");
    let cap2 = runtime
        .complete_owned(cap2_run, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("batch=2 capture");
    assert_eq!(cap2.len(), 8);
    assert_eq!(
        runtime.bucket_count(),
        2,
        "two distinct shapes -> two captured graphs"
    );

    // Releasing a real captured graph (id 1 = the batch=1 bucket) must succeed.
    session
        .release_captured_graph(1)
        .expect("release a real captured graph");

    // --- Session 2: CUDA graph buckets are no-evict on legacy CUDA ---
    // A fresh session keeps its gpu_graph_ids from colliding with session 1's captured-graph map.
    // ORT's legacy CUDA EP keeps captured graphs session-scoped, so with max_buckets=1 a second
    // shape must fail instead of evicting and accumulating an unreclaimable captured graph.
    let Some(sess2) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    let session2 = sess2;
    let mut churn = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session2.clone(),
        MemoryInfo::cpu().expect("churn input mem"),
        MemoryInfo::cpu().expect("churn output mem"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device input options"),
    )
    .expect("cuda-graph device-input churn runtime");
    let mut churn_plan = ServingShapePlan::builder();
    churn_plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::HostBuffer);
    churn
        .install_shape_plan(Arc::new(churn_plan.build().expect("churn shape plan")))
        .expect("install churn shape plan");
    let _ = churn
        .run_on([&[1, 32]], [&[1, 4]], 0, |lane| {
            lane.input_mut_at::<0>()?.fill(1.0);
            lane.run()?;
            Ok(lane.output_at::<0>()?.to_vec())
        })
        .expect("churn batch=1 (capture graph 1)");
    let err = churn
        .run_on([&[2, 32]], [&[2, 4]], 0, |lane| {
            lane.input_mut_at::<0>()?.fill(2.0);
            lane.run()?;
            Ok(lane.output_at::<0>()?.to_vec())
        })
        .expect_err("churn batch=2 should be rejected by the sealed plan");
    let msg = err.to_string();
    assert!(msg.contains("sealed plan"), "unexpected error: {msg}");
    assert_eq!(churn.bucket_count(), 1, "batch=1 bucket remains cached");
    assert!(
        churn.bucket([&[1, 32]], [&[1, 4]]).is_some(),
        "batch=1 bucket should not be evicted"
    );
}

/// Regression: a prebuilt CUDA-graph bucket must never *capture* on its first served run.
/// CUDA-graph capture is device-wide serialized and must not overlap a live replay. Buckets used
/// to be created-but-not-captured by `prebuild_buckets` (capture happened on each lane's first
/// run), and only *creation* was guarded by the runtime-wide idleness check — a plain cache HIT
/// was not. The public sequence `prebuild A+B -> enqueue A (in flight) -> first run B` therefore
/// captured B while prebuilt A was already replaying, which ORT rejects as an illegal capture.
/// `get_or_create_bucket_inner` now eagerly captures every fresh lane while that idleness guard is
/// proven, so the first served run of any prebuilt bucket is a pure replay. (cuda.)
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_prebuilt_bucket_first_run_is_replay_not_capture_while_sibling_replays() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let session = cuda_graph_session_streamed(&env, &path, &stream).expect("session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("output mem"),
        1,
        DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device input options"),
    )
    .expect("cuda-graph runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::HostBuffer)
        .add_shape([vec![2, 32]], [vec![2, 4]], OutputPolicy::HostBuffer);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");

    // Prebuild both buckets. With eager capture this captures BOTH graphs now, while the runtime
    // is provably idle; before the fix both buckets were merely created, uncaptured.
    // `graph_captured` is the deterministic st-zrt-level proof: it flips exactly when a run
    // completes with a fresh `gpu_graph_id`, so the fix is observable without racing ORT
    // internals — assert capture already happened BEFORE any served run.
    let prebuilt = runtime
        .prebuild_buckets([
            ShapeSpec::new([&[1, 32][..]], [&[1, 4][..]]),
            ShapeSpec::new([&[2, 32][..]], [&[2, 4][..]]),
        ])
        .expect("prebuild both buckets (eagerly captures both graphs)");
    assert_eq!(prebuilt, 2);
    assert_eq!(runtime.bucket_count(), 2);
    for bucket in runtime.buckets() {
        for lane in bucket.lanes() {
            assert!(
                lane.graph_captured(),
                "prebuild_buckets must capture every lane's graph eagerly, not on first serve"
            );
        }
    }

    // Enqueue bucket A and leave the token alive: the replay may still be executing on the
    // retained stream (in flight).
    let run_a = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(7.0);
            Ok(())
        })
        .expect("enqueue A (replay of eagerly captured graph)");
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 1);

    // Bucket B's FIRST served run is a plain cache hit: it must replay the graph captured at
    // prebuild. Before the eager-capture fix this lane still reported `graph_captured == false`
    // here (nothing had captured it), and the run attempted a capture while A's replay was in
    // flight.
    let run_b = runtime
        .enqueue_owned([&[2, 32]], [&[2, 4]], |lane| {
            assert!(
                lane.graph_captured(),
                "first served run of a prebuilt bucket must be a replay, never a capture"
            );
            lane.input_mut_at::<0>()?.fill(9.0);
            Ok(())
        })
        .expect("first served run of B must replay, not capture, while A is in flight");

    let out_a = runtime
        .complete_owned(run_a, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("complete A");
    let out_b = runtime
        .complete_owned(run_b, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("complete B");
    assert_eq!(out_a.len(), 4);
    assert_eq!(out_b.len(), 8);
    assert!(
        out_a.iter().chain(out_b.iter()).all(|v| v.is_finite()),
        "both replays produced finite output"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_device_output_stays_device_resident_and_owned() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("stream"));
    let session = cuda_graph_session_streamed(&env, &path, &stream).expect("session");
    let mut lane = ServingLane::<f32, f32, 1, 1>::with_device_io(
        session,
        &MemoryInfo::cpu().expect("input mem"),
        [&[1, 32]],
        [&[1, 4]],
        BufferSpec::AUTO,
        0,
        &stream,
    )
    .expect("device I/O lane");
    lane.set_gpu_graph_id(1).expect("graph id");
    lane.input_mut_at::<0>().expect("input").fill(3.0);
    lane.run().expect("capture/run");
    lane.run().expect("replay");
    let output = lane.device_output(0).expect("device output");
    let memory = output.memory_info().expect("memory info");
    assert_eq!(memory.name, "Cuda");
    assert_eq!(memory.device_id, 0);
    assert!(
        output.as_slice().is_err(),
        "device output must reject host access"
    );
    assert_eq!(output.shape(), &[1, 4]);
    assert!(
        lane.output_at::<0>().is_err(),
        "host lane accessor must reject device output"
    );
    assert!(lane.output_mut(0).is_err());
    assert!(lane.outputs().is_err());
    assert!(lane.outputs_mut().is_err());
    assert!(lane.output_buffer(0).is_err());
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_dynamic_device_output_completes_without_host_blocking() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("stream"));
    let session = cuda_graph_session_streamed(&env, &path, &stream).expect("session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused host output descriptor"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("device-I/O runtime");
    let mut builder = ServingShapePlan::builder();
    builder.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(builder.build().expect("plan")))
        .expect("install device-output plan");

    // Capture once using the blocking setup path. Live replay below uses only event queries.
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture graph");
    let mut run = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(5.0);
            Ok(())
        })
        .expect("enqueue device-output replay");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match run.try_complete().expect("query completion") {
            st_zrt::CompletionStatus::Ready => break,
            st_zrt::CompletionStatus::Pending => {
                assert!(
                    std::time::Instant::now() < deadline,
                    "CUDA event did not complete"
                );
                std::hint::spin_loop();
            },
        }
    }
    let output = run
        .lane()
        .expect("completed lane")
        .device_output(0)
        .expect("device output");
    assert_eq!(output.shape(), &[1, 4]);
    assert_eq!(output.memory_info().expect("memory").name, "Cuda");
    assert!(output.as_slice().is_err());
    runtime
        .complete_owned(run, |lane| {
            assert_eq!(lane.device_output(0)?.shape(), &[1, 4]);
            Ok(())
        })
        .expect("return ready lane");
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_non_graph_device_input_does_not_claim_exact_event_completion() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("stream"));
    let options = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(
            CudaConfig::performance(0)
                .with_user_stream(&stream)
                .expect("user stream"),
        )
        .expect("CUDA provider");
    let session = Session::new(&env, path.to_str().expect("path"), options).expect("session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused host output descriptor"),
        1,
        DynamicIoOptions::new(1)
            .with_device_inputs(0, &stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("device-I/O runtime");
    let mut run = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(5.0);
            Ok(())
        })
        .expect("enqueue eager CUDA run");
    let error = run
        .try_complete()
        .expect_err("non-graph CUDA execution has no proven exact-stream event fence");
    assert!(
        error
            .to_string()
            .contains("no nonblocking completion event")
    );
    runtime
        .complete_owned(run, |lane| {
            assert_eq!(lane.device_output(0)?.shape(), &[1, 4]);
            Ok(())
        })
        .expect("blocking IoBinding completion fallback");
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_device_output_gpu_chain_owns_lane_until_downstream_completion() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let source_stream = Arc::new(st_zrt::CudaStream::new(0).expect("source stream"));
    let downstream_stream = Arc::new(st_zrt::CudaStream::new(0).expect("downstream stream"));
    let session = cuda_graph_session_streamed(&env, &path, &source_stream).expect("source session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused output mem"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &source_stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture");
    let bucket = runtime
        .prepared_bucket_id([&[1, 32]], [&[1, 4]])
        .expect("bucket");
    let run = runtime
        .enqueue_prepared(bucket, |lane| {
            lane.input_mut_at::<0>()?.fill(9.0);
            Ok(())
        })
        .expect("enqueue");
    let mut chained = run
        .chain_on_stream(&downstream_stream, |outputs, stream| {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].shape(), &[1, 4]);
            assert_eq!(stream.device_id(), 0);
            // A real consumer enqueues kernels/copies here. ZRT records downstream completion only
            // after this closure returns.
            Ok(())
        })
        .map_err(|failure| failure.error)
        .expect("chain");
    drop(downstream_stream);
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while chained.try_complete().expect("query") == st_zrt::CompletionStatus::Pending {
        assert!(std::time::Instant::now() < deadline);
        std::hint::spin_loop();
    }
    let run = chained.synchronize().expect("take completed ORT run");
    runtime
        .complete_owned(run, |lane| {
            assert_eq!(lane.device_output(0)?.shape(), &[1, 4]);
            Ok(())
        })
        .expect("return lane");
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_gpu_chain_rejects_cross_device_stream_before_exposing_outputs() {
    if st_zrt::device_count().expect("device count") < 2 {
        eprintln!("skipping cross-device chain test on a single-GPU host");
        return;
    }
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let source_stream = Arc::new(st_zrt::CudaStream::new(0).expect("source stream"));
    let downstream_stream =
        Arc::new(st_zrt::CudaStream::new(1).expect("device-1 downstream stream"));
    let session = cuda_graph_session_streamed(&env, &path, &source_stream).expect("source session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused output mem"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &source_stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture");
    let bucket = runtime
        .prepared_bucket_id([&[1, 32]], [&[1, 4]])
        .expect("bucket");
    let run = runtime
        .enqueue_prepared(bucket, |lane| {
            lane.input_mut_at::<0>()?.fill(9.0);
            Ok(())
        })
        .expect("enqueue");
    let failure = run
        .chain_on_stream(&downstream_stream, |_outputs, _stream| {
            panic!("cross-device rejection must happen before output exposure")
        })
        .expect_err("cross-device GPU chain must be rejected");
    assert!(failure.error.to_string().contains("same CUDA device"));
    runtime
        .complete_owned(failure.run, |_| Ok(()))
        .expect("return source lane after rejection");
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_gpu_chain_enqueue_error_fences_before_lane_recovery() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let source_stream = Arc::new(st_zrt::CudaStream::new(0).expect("source stream"));
    let downstream_stream = Arc::new(st_zrt::CudaStream::new(0).expect("downstream stream"));
    let session = cuda_graph_session_streamed(&env, &path, &source_stream).expect("source session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused output mem"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &source_stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture");
    let bucket = runtime
        .prepared_bucket_id([&[1, 32]], [&[1, 4]])
        .expect("bucket");
    let run = runtime
        .enqueue_prepared(bucket, |_| Ok(()))
        .expect("enqueue");
    let chained = run
        .chain_on_stream(&downstream_stream, |_outputs, _stream| {
            Err(st_zrt::Error::local("injected downstream enqueue failure"))
        })
        .expect("a post-wait callback failure remains an owned chain");
    let error = chained
        .synchronize()
        .expect_err("callback failure must surface after downstream fence");
    assert!(error.to_string().contains("injected downstream"));
    assert_eq!(runtime.reclaim_dropped_runs(), 1);
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_gpu_chain_panic_fences_before_unwind_recovery() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let source_stream = Arc::new(st_zrt::CudaStream::new(0).expect("source stream"));
    let downstream_stream = Arc::new(st_zrt::CudaStream::new(0).expect("downstream stream"));
    let session = cuda_graph_session_streamed(&env, &path, &source_stream).expect("source session");
    let destination = AllocatedTensor::<f32>::cuda(&session, 0, &[1, 4]).expect("destination");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused output mem"),
        1,
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_inputs(0, &source_stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture");
    let bucket = runtime
        .prepared_bucket_id([&[1, 32]], [&[1, 4]])
        .expect("bucket");
    let run = runtime
        .enqueue_prepared(bucket, |_| Ok(()))
        .expect("enqueue");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run.chain_on_stream(&downstream_stream, |outputs, stream| {
            unsafe {
                st_zrt::memcpy_async_d2d(
                    destination.raw_mut_ptr()?,
                    outputs[0].raw_mut_ptr()? as *const _,
                    outputs[0].byte_len()?,
                    stream,
                )?;
            }
            panic!("injected downstream panic")
        });
    }));
    assert!(panic.is_err());
    assert_eq!(runtime.reclaim_dropped_runs(), 1);
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_completion_poller_queries_multiple_owned_runs_in_one_pass() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream_a = Arc::new(st_zrt::CudaStream::new(0).expect("stream a"));
    let stream_b = Arc::new(st_zrt::CudaStream::new(0).expect("stream b"));
    let session_a = cuda_graph_session_streamed(&env, &path, &stream_a).expect("session a");
    let session_b = cuda_graph_session_streamed(&env, &path, &stream_b).expect("session b");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::from_sessions_with_options(
        vec![session_a, session_b],
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("unused output mem"),
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_input_streams(0, vec![stream_a, stream_b])
            .expect("device streams")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::DeviceResident);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture");
    let bucket = runtime
        .prepared_bucket_id([&[1, 32]], [&[1, 4]])
        .expect("bucket");
    let run_a = runtime.enqueue_prepared(bucket, |_| Ok(())).expect("run a");
    let run_b = runtime.enqueue_prepared(bucket, |_| Ok(())).expect("run b");
    let poller = st_zrt::CudaCompletionPoller::new(0).expect("poller");
    let mut runs = [run_a, run_b];
    let mut too_short = [st_zrt::CompletionStatus::Pending; 1];
    let error = runtime
        .poll_owned_runs(poller, &mut runs, &mut too_short)
        .expect_err("short result buffer must fail before polling");
    assert_eq!(
        error.message,
        "zrt: owned-run completion result buffer is smaller than the run batch"
    );
    assert!(
        runs.iter().all(|run| run.completion_event().is_some()),
        "preflight failure must not change either owned run"
    );
    let mut statuses = [st_zrt::CompletionStatus::Pending; 2];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !statuses
        .iter()
        .all(|&status| status == st_zrt::CompletionStatus::Ready)
    {
        runtime
            .poll_owned_runs(poller, &mut runs, &mut statuses)
            .expect("batch query");
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    let [run_a, run_b] = runs;
    assert!(
        run_a.completion_event().is_none(),
        "ready run event was discharged"
    );
    runtime
        .complete_owned(run_b, |_| Ok(()))
        .expect("complete b");
    runtime
        .complete_owned(run_a, |_| Ok(()))
        .expect("complete a");
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_nonzero_device_stream_event_survives_moved_thread() {
    let _capture_guard = common::cuda_graph_capture_lock();
    if st_zrt::device_count().expect("device count") < 2 {
        eprintln!("skipping nonzero-device test on a single-GPU host");
        return;
    }
    let worker = std::thread::spawn(|| {
        let stream = st_zrt::CudaStream::new(1).expect("device-1 stream");
        let event = st_zrt::CudaEvent::new(1).expect("device-1 event");
        event.record(&stream).expect("record");
        event.synchronize().expect("complete");
        assert!(event.is_complete().expect("query"));
    });
    worker.join().expect("device-1 worker");
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_owned_lanes_pipeline_before_completion() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let stream_a = Arc::new(st_zrt::CudaStream::new(0).expect("stream a"));
    let stream_b = Arc::new(st_zrt::CudaStream::new(0).expect("stream b"));
    let session_a = cuda_graph_session_streamed(&env, &path, &stream_a).expect("session a");
    let session_b = cuda_graph_session_streamed(&env, &path, &stream_b).expect("session b");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::from_sessions_with_options(
        vec![session_a, session_b],
        MemoryInfo::cpu().expect("input mem"),
        MemoryInfo::cpu().expect("output mem"),
        DynamicIoOptions::new(1)
            .with_cuda_graph(true)
            .with_device_input_streams(0, vec![stream_a, stream_b])
            .expect("device streams"),
    )
    .expect("two-lane runtime");
    let mut builder = ServingShapePlan::builder();
    builder.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::HostBuffer);
    runtime
        .install_shape_plan(Arc::new(builder.build().expect("plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 2)
        .expect("capture both lane graphs");

    let run_a = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(1.0);
            Ok(())
        })
        .expect("enqueue a");
    let run_b = runtime
        .enqueue_owned([&[1, 32]], [&[1, 4]], |lane| {
            lane.input_mut_at::<0>()?.fill(7.0);
            Ok(())
        })
        .expect("enqueue b before completing a");
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 2);

    let out_b = runtime
        .complete_owned(run_b, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("complete b");
    let out_a = runtime
        .complete_owned(run_a, |lane| Ok(lane.output_at::<0>()?.to_vec()))
        .expect("complete a");
    assert!(
        max_abs_diff(&out_a, &out_b) > 1.0,
        "pipelined lanes must preserve their distinct refreshed inputs"
    );
    assert_eq!(runtime.buckets()[0].detached_lane_count(), 0);
}

/// Max absolute element-wise difference of two f32 slices (0 for empty).
#[cfg(feature = "cuda")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut m = 0.0_f64;
    for i in 0..n {
        m = m.max((a[i] as f64 - b[i] as f64).abs());
    }
    m
}

/// FAIL-CLOSED (host-input cuda-graph): the configuration that `cuda_graph_host_input_replay_reads_stale`
/// used to reproduce as a stale-output limitation is now rejected at construction. With host-resident
/// lane inputs, a captured CUDA graph bakes a device buffer ORT never repopulates from the host
/// binding on replay, so changing the host input has no effect (stale/never-initialized reads). Even
/// against a real CUDA-graph session, a host-input `DynamicIoRuntime` must refuse to build; the
/// supported path is device-resident inputs on the retained user stream
/// (`cuda_graph_device_input_replay_reads_fresh`).
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_host_input_configuration_is_rejected() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    // A real CUDA-graph session proves this is a topology rejection, not an EP-availability skip.
    let Some(cg) = cuda_graph_session(&env, &path) else {
        return;
    };
    let err = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        cg,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(4).with_cuda_graph(true),
    )
    .expect_err("host-input cuda_graph must be rejected even with a CUDA-graph session");
    let msg = err.to_string();
    assert!(
        msg.contains("device-resident inputs"),
        "guard should name the supported input path, got: {msg}"
    );
}

/// FAIL-CLOSED (shared-session multi-lane cuda-graph): one bucket mints one `gpu_graph_id`, and with
/// a shared session every lane of that bucket replays the same ORT-captured graph — which baked
/// whichever lane ran first, so the other lanes would silently read and write that lane's buffers.
/// Construction must reject `cuda_graph` + shared session + `lanes > 1` even with device inputs on
/// the retained stream; the remedy is replicated sessions (one session + stream per lane) or one lane.
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_shared_session_multi_lane_is_rejected() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let Some(sess) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    let err = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        2,
        DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device input options"),
    )
    .expect_err("cuda_graph + shared session + 2 lanes must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("replicated sessions"),
        "guard should direct callers to replicated sessions or one lane, got: {msg}"
    );

    // The sound single-lane shared-session topology with device inputs still builds.
    let stream_one = Arc::new(st_zrt::CudaStream::new(0).expect("single-lane stream"));
    let Some(sess_one) = cuda_graph_session_streamed(&env, &path, &stream_one) else {
        return;
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess_one,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(2)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream_one)
            .expect("device input options"),
    )
    .expect("single-lane shared-session graph runtime builds");

    // A graph runtime without a sealed plan still refuses bucket creation before any capture.
    runtime
        .get_or_create_bucket([&[1, 32]], [&[1, 4]])
        .expect_err("cuda graph buckets require a sealed shape plan");
}

/// FAIL-CLOSED (lazy graph bucket creation under traffic): a new CUDA-graph bucket captures on its
/// first run, and capture must not overlap a live replay on another lane of the runtime. While a
/// lane is detached into an owned run, lazy creation of a second bucket must fail with an error
/// directing callers to prebuild/warm before traffic; after completion, creation succeeds.
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_bucket_creation_refuses_inflight_lanes() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let Some(sess) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    // Single lane per bucket: the sound shared-session graph topology (the multi-lane variant is
    // rejected by `cuda_graph_shared_session_multi_lane_is_rejected`).
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device input options"),
    )
    .expect("cuda-graph device-input runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape([vec![1, 32]], [vec![1, 4]], OutputPolicy::HostBuffer)
        .add_shape([vec![2, 32]], [vec![2, 4]], OutputPolicy::HostBuffer);
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("shape plan")))
        .expect("install shape plan");

    // Warm the batch=1 bucket (captures) while every lane is idle.
    runtime
        .prime_bucket_enqueued([&[1, 32]], [&[1, 4]], 1)
        .expect("warm batch=1");

    // Hold one lane in flight, then attempt a lazy bucket creation for batch=2.
    let run = runtime
        .enqueue_prepared(
            runtime
                .prepared_bucket_id([&[1, 32]], [&[1, 4]])
                .expect("bucket"),
            |lane| {
                lane.input_mut_at::<0>()?.fill(3.0);
                Ok(())
            },
        )
        .expect("enqueue batch=1 replay");
    let err = runtime
        .get_or_create_bucket([&[2, 32]], [&[2, 4]])
        .expect_err("lazy graph bucket creation must be refused under an in-flight lane");
    let msg = err.to_string();
    assert!(
        msg.contains("prebuild or warm"),
        "guard should direct callers to prebuild/warm before traffic, got: {msg}"
    );
    assert_eq!(runtime.bucket_count(), 1, "no second bucket was created");

    // Completing the run fences the lane; lazy creation proceeds again.
    runtime
        .complete_owned(run, |_| Ok(()))
        .expect("complete in-flight run");
    runtime
        .get_or_create_bucket([&[2, 32]], [&[2, 4]])
        .expect("bucket creation after every lane is fenced");
    assert_eq!(runtime.bucket_count(), 2);
}

/// CORRECTNESS (device-input cuda-graph path, the fix for the host-input limitation above): with
/// **device-resident** lane inputs (`ServingLane::with_device_inputs`) + `enable_cuda_graph`, the
/// per-run host→device refresh (on the caller-owned `user_compute_stream`) makes the replay read the
/// FRESH input — the inverse of the rejected host-input configuration. Reference from a CPU
/// session; the device-input lane captures on A then replays on B; the replay-of-B output must match
/// the B reference.
///
/// ORT's `CopyTensors` has no CPU↔CUDA route (the built-in CUDA EP is a legacy provider), so the
/// refresh uses the CUDA runtime directly (`cudaMemcpyAsync` on the same stream ORT replays the graph
/// on — race-free by stream ordering). See `cuda_rt.rs`.
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_device_input_replay_reads_fresh() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let cpu = MemoryInfo::cpu().expect("cpu");

    let ref_sess = cpu_session(&env, &path);
    let ref_out = |x: f32| -> Vec<f32> {
        let buf = [x; 32];
        let v = Tensor::from_buffer(&buf, &[1, 32], &cpu).expect("X");
        let inputs: [&dyn st_zrt::RunInput; 1] = [&v];
        let mut out: Vec<Option<OwnedValue>> = (0..ref_sess.output_count()).map(|_| None).collect();
        ref_sess.run(&inputs, &mut out).expect("ref run");
        out[0]
            .as_ref()
            .expect("out")
            .as_slice::<f32>()
            .expect("read")
            .to_vec()
    };
    let ref_a = ref_out(1.0);
    let ref_b = ref_out(7.0);
    assert!(
        max_abs_diff(&ref_a, &ref_b) > 5.0,
        "distinct inputs must produce distinct reference outputs"
    );

    // Capture is device-wide serial (see `cuda_graph_capture_lock`); hold it across the capture run.
    let _capture_guard = common::cuda_graph_capture_lock();
    // Caller-owned CUDA stream: ORT replays the captured graph on it (user_compute_stream); the
    // device-input lane refreshes its bound device buffer on the same stream each run.
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let Some(cg) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    let session = cg;
    let mut lane = st_zrt::ServingLane::<f32, f32, 1, 1>::with_device_inputs(
        session,
        &cpu,
        &cpu,
        [&[1, 32]],
        [&[1, 4]],
        st_zrt::BufferSpec::AUTO,
        st_zrt::BufferSpec::AUTO,
        0,
        &stream,
    )
    .expect("device-input lane");
    lane.set_gpu_graph_id(1).expect("gpu_graph_id");

    // Capture with input A (host staging → device, capture).
    lane.input_mut_at::<0>().expect("input A").fill(1.0);
    lane.run().expect("capture A");
    // Replay with input B (host staging refreshed → device, replay).
    lane.input_mut_at::<0>().expect("input B").fill(7.0);
    lane.run().expect("replay B");
    let cap_b = lane.output_at::<0>().expect("output").to_vec();

    let diff_correct = max_abs_diff(&cap_b, &ref_b);
    let diff_stale = max_abs_diff(&cap_b, &ref_a);
    eprintln!(
        "device-input replay-of-B: max|cap_B-ref_B|={diff_correct:.4}  max|cap_B-ref_A|={diff_stale:.4}"
    );
    assert!(
        diff_correct < 5.0,
        "device-input cuda-graph replay must read the fresh input B \
         (max|cap_B-ref_B|={diff_correct:.4})"
    );
}

/// Concurrent device-input cuda-graph **replay** is race-free and correct. Builds a
/// [`ServingLanePool`] of device-input cuda-graph lanes (one per worker, each with a distinct
/// `gpu_graph_id` over its own device buffers) on a shared session + shared caller-owned CUDA
/// stream. CUDA graph capture is device-wide serial (errors 900/901 under concurrent capture — see
/// `cuda_graph_capture_lock`), so each lane's graph is captured **sequentially** at warmup; then N
/// threads each check out a lane, fill a DISTINCT changing input, run (replay), and assert the
/// output matches the CPU reference for its own input. Proves concurrent host-side enqueue of the
/// per-run H2D refresh + graph replay on a shared stream is race-free and that every replay reads
/// its own fresh input (a cross-lane race would read another lane's buffer → a large diff). (cuda.)
///
/// Caveat: ORT's CUDA-graph docs state multi-threaded `Run()` on the *same* `InferenceSession` is
/// unsupported with cuda-graphs. This test shares one session across the N concurrent replays and
/// passes on this ORT/CUDA build, but that is an undocumented dependency — the supported production
/// shape is a replicated session per lane (see `docs/cuda-graph-paths.md`).
#[test]
#[cfg(feature = "cuda")]
fn cuda_graph_device_input_concurrent_replay_is_race_free() {
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let cpu = MemoryInfo::cpu().expect("cpu");
    // Hold the device-wide capture lock for the whole test: warmup captures here, and this keeps
    // other capturing tests from interleaving capture with our replay window.
    let _capture_guard = common::cuda_graph_capture_lock();

    // CPU reference: output for a given scalar fill of the [1,32] input.
    let cpu_sess = cpu_session(&env, &path);
    let cpu_ref = |x: f32| -> Vec<f32> {
        let buf = [x; 32];
        let v = Tensor::from_buffer(&buf, &[1, 32], &cpu).expect("X");
        let inputs: [&dyn st_zrt::RunInput; 1] = [&v];
        let mut out: Vec<Option<OwnedValue>> = (0..cpu_sess.output_count()).map(|_| None).collect();
        cpu_sess.run(&inputs, &mut out).expect("cpu ref run");
        out[0]
            .as_ref()
            .expect("out")
            .as_slice::<f32>()
            .expect("read")
            .to_vec()
    };
    assert!(
        max_abs_diff(&cpu_ref(1.0), &cpu_ref(2.0)) > 1e-3,
        "distinct inputs must produce distinct reference outputs"
    );

    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let Some(sess) = cuda_graph_session_streamed(&env, &path, &stream) else {
        return;
    };
    let session = sess;

    const N: usize = 4;
    // Each lane gets a distinct gpu_graph_id so each captures its own graph over its OWN device
    // buffer; the shared stream sequences per-run refresh + replay. Capture happens sequentially in
    // this loop (warmup), not concurrently.
    let lanes: Vec<ServingLane<f32, f32, 1, 1>> = (0..N)
        .map(|i| {
            let mut lane = ServingLane::with_device_inputs(
                session.clone(),
                &cpu,
                &cpu,
                [&[1, 32]],
                [&[1, 4]],
                BufferSpec::AUTO,
                BufferSpec::AUTO,
                0,
                &stream,
            )
            .expect("device-input lane");
            lane.set_gpu_graph_id((i as i32) + 1)
                .expect("distinct gpu_graph_id");
            lane.input_mut_at::<0>()
                .expect("warmup input")
                .fill((i as f32) + 1.0);
            lane.run().expect("warmup capture (first run)");
            lane
        })
        .collect();
    let pool = Arc::new(ServingLanePool::from_lanes(lanes).expect("pool"));

    // One distinct changing input per worker; precompute each one's CPU reference.
    let inputs: Vec<f32> = (1..=N).map(|t| t as f32 + 0.5).collect();
    let refs: Vec<Vec<f32>> = inputs.iter().map(|&x| cpu_ref(x)).collect();

    // N threads, N lanes — every checkout succeeds immediately. Each replays its own graph with its
    // own fresh input and must match its own CPU reference.
    let oks: Vec<bool> = std::thread::scope(|s| {
        (0..N)
            .map(|t| {
                let pool = pool.clone();
                let want = refs[t].clone();
                let x = inputs[t];
                s.spawn(move || {
                    let mut guard = pool.checkout();
                    guard.input_mut_at::<0>().expect("input").fill(x);
                    guard.run().expect("concurrent replay run");
                    let got = guard.output_at::<0>().expect("output").to_vec();
                    max_abs_diff(&got, &want) < 5.0
                })
            })
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    assert!(
        oks.iter().all(|&ok| ok),
        "every worker's concurrent replay must match its own CPU reference (a cross-lane race would \
         read another lane's buffer and diverge)"
    );
    assert_eq!(pool.idle_count(), N, "all lanes returned to the pool");
}

/// The rebind+cudagraph guard: `with_cuda_graph(true).with_rebind_inputs_each_run(true)` must
/// reject construction — per-run input rebinding tears down the pointers a captured CUDA graph
/// bakes, crashing at replay. Runs on CPU (no GPU needed): the guard fires in
/// `DynamicIoOptions::validate`, before any EP is touched.
#[test]
fn cudagraph_and_rebind_are_mutually_exclusive() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let env = Environment::new().expect("env");
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let sess = cpu_session(&env, &path);
    let rt = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .with_rebind_inputs_each_run(true),
    );
    let err = match rt {
        Ok(_) => panic!("cuda_graph + rebind_inputs_each_run must be rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("mutually exclusive"),
        "guard should explain the conflict, got: {msg}"
    );
}

/// The device-input + rebind guard: `with_device_inputs(..).with_rebind_inputs_each_run(true)` must
/// reject construction — per-run input rebinding tears down the device pointers a captured CUDA graph
/// bakes (same hazard as `cuda_graph` + rebind). The guard fires in
/// `DynamicIoOptions::validate` before session/provider execution.
#[test]
#[cfg(feature = "cuda")]
fn device_input_and_rebind_are_mutually_exclusive() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let env = Environment::new().expect("env");
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let sess = cpu_session(&env, &path);
    let dummy_stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let rt = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(4)
            .with_device_inputs(0, &dummy_stream)
            .expect("device input options")
            .with_rebind_inputs_each_run(true),
    );
    let err = match rt {
        Ok(_) => panic!("device_inputs + rebind_inputs_each_run must be rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("mutually exclusive"),
        "guard should explain the conflict, got: {msg}"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn device_input_options_reject_a_stream_from_another_device() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let err = DynamicIoOptions::new(1)
        .with_device_inputs(1, &stream)
        .expect_err("device-input options must reject a stream created on another device");
    assert!(
        err.to_string().contains("different device"),
        "unexpected wrong-device error: {err}"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn serving_lane_requires_exact_session_cuda_stream_identity() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let env = Environment::new().expect("env");
    let session_stream = Arc::new(st_zrt::CudaStream::new(0).expect("session stream"));
    let other_stream = Arc::new(st_zrt::CudaStream::new(0).expect("other stream"));
    let session =
        cuda_graph_session_streamed(&env, &path, &session_stream).expect("CUDA graph session");

    let lane = ServingLane::<f32, f32, 1, 1>::with_device_inputs(
        session,
        &MemoryInfo::cpu().expect("input mem"),
        &MemoryInfo::cpu().expect("output mem"),
        [&[1, 32]],
        [&[1, 4]],
        BufferSpec::AUTO,
        BufferSpec::AUTO,
        0,
        &other_stream,
    );
    let err = match lane {
        Ok(_) => panic!("same-device but distinct stream identity must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("exact owned CUDA stream configured on its Session"),
        "unexpected stream-identity error: {err}"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn per_lane_device_streams_require_replicated_sessions() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let env = Environment::new().expect("env");
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let sess = cpu_session(&env, &path);
    let dummy_stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let rt = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        1,
        DynamicIoOptions::new(4)
            .with_device_input_streams(0, vec![dummy_stream])
            .expect("device streams"),
    );
    let err = match rt {
        Ok(_) => panic!("per-lane streams with a shared session must be rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("replicated sessions"),
        "guard should explain replicated-session requirement, got: {msg}"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn per_lane_device_stream_count_must_match_replicated_sessions() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let env = Environment::new().expect("env");
    let Some(path) = dynamic_batch_path() else {
        return;
    };
    let sessions = vec![cpu_session(&env, &path), cpu_session(&env, &path)];
    let dummy_stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));
    let rt = DynamicIoRuntime::<f32, f32, 1, 1>::from_sessions_with_options(
        sessions,
        MemoryInfo::cpu().expect("in mem"),
        MemoryInfo::cpu().expect("out mem"),
        DynamicIoOptions::new(4)
            .with_device_input_streams(0, vec![dummy_stream])
            .expect("device streams"),
    );
    let err = match rt {
        Ok(_) => panic!("per-lane stream count mismatch must be rejected"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("expected 2 per-lane CUDA streams, got 1"),
        "guard should explain stream count mismatch, got: {msg}"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_allocated_output_tensor_binds_and_reports_cuda_memory() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let cpu_mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&cpu_mem, &buf);

    let cuda = cuda_session(&env, &path).expect("cuda session");
    let out = AllocatedTensor::<f32>::cuda(&cuda, 0, &[1, 10]).expect("cuda output tensor");
    assert_eq!(out.memory_info().expect("allocated mem").name, "Cuda");

    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_allocated(cuda.output_name(0).expect("output name"), &out)
        .expect("bind allocated cuda output");
    cuda.run_binding(&binding)
        .expect("allocated cuda output run");
    binding
        .synchronize_outputs()
        .expect("sync allocated output");
    assert!(!out.raw_typed_ptr().expect("device pointer").is_null());
    assert!(out.as_slice().is_err());

    let cuda_mem = MemoryInfo::cuda(0).expect("cuda memory info");
    let mut lane = cuda
        .prepare_allocated_output_tensor_io_lane::<f32>(
            &cpu_mem,
            &cuda_mem,
            &[&[1, 1, 28, 28]],
            &[&[1, 10]],
        )
        .expect("allocated-output lane");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("allocated-output lane run");
    let lane_out_info = lane
        .output_tensor(0)
        .expect("lane output")
        .memory_info()
        .expect("lane output memory");
    assert_eq!(lane_out_info.name, "Cuda");
    assert_eq!(lane_out_info.device_id, 0);
    assert!(lane.output(0).is_err());
}

#[test]
fn cuda_runtime_shared_session_matches_cpu() {
    let _capture_guard = common::cuda_graph_capture_lock();
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut runtime =
        Runtime::<f32>::shared_session(cuda, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 1)
            .expect("cuda runtime");
    let got = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").to_vec())
        })
        .expect("runtime run");
    assert_logits_close(&cpu_logits, &got);
}

/// LoRA positive path on CUDA. The base model's `lora_param_a`/`lora_param_b` initializers are
/// zero (`Y == X`); attaching an adapter (non-zero `a`/`b`) via `RunOptions` must change the
/// output. This also exercises the device-allocator path: ORT's lora load rejects the CPU
/// allocator, so the adapter is loaded against a CUDA allocator created from the session
/// (`LoraAdapter::from_array_with_allocator`). Fixtures: `fixtures/lora_base.onnx` +
/// `fixtures/lora_adapter.onnx_adapter` (regenerate via `fixtures/gen_lora.py`).
#[cfg(feature = "cuda")]
#[test]
fn lora_adapter_changes_output_on_cuda() {
    let base =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/lora_base.onnx");
    assert!(base.exists(), "missing fixture {}", base.display());

    let env = Environment::new().expect("env");
    let cuda_mem = st_zrt::MemoryInfo::cuda(0).expect("cuda mem");
    let sess = cuda_session(&env, &base).expect("cuda session for lora base");
    // Serialize against cuda-graph capture in sibling tests — capture is device-wide serial and
    // this test's CUDA inference must not overlap another test's capture phase.
    let _capture_guard = common::cuda_graph_capture_lock();

    // ORT's lora load needs a non-CPU allocator — create one from the CUDA session.
    let dev_alloc = st_zrt::Allocator::create(&sess, &cuda_mem).expect("cuda allocator");

    let cpu_mem = st_zrt::MemoryInfo::cpu().expect("cpu mem");
    let xbuf = vec![1.0_f32; 16];
    let x = st_zrt::Tensor::from_buffer(&xbuf, &[4, 4], &cpu_mem).expect("input");

    // Base run (no adapter): a,b are zero ⇒ Y == X (all ones).
    let mut base_out: Vec<Option<st_zrt::OwnedValue>> =
        (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&x], &mut base_out).expect("base run");
    let base_y = base_out[0]
        .as_ref()
        .expect("base output")
        .as_slice::<f32>()
        .expect("read base output")
        .to_vec();
    assert!(
        base_y.iter().all(|&v| (v - 1.0).abs() < 1e-4),
        "base run (a,b=0) should yield Y == X; got {base_y:?}"
    );

    // Adapter loaded against the CUDA allocator.
    let adapter_bytes = include_bytes!("fixtures/lora_adapter.onnx_adapter");
    let adapter = Arc::new(
        st_zrt::LoraAdapter::from_array_with_allocator(adapter_bytes, &dev_alloc)
            .expect("load lora adapter with cuda allocator"),
    );
    let opts = st_zrt::RunOptions::new()
        .with_lora_adapter(&adapter)
        .freeze()
        .expect("attach adapter");
    drop(adapter); // MaterializedRunOptions owns the adapter guard required by ORT.

    let mut tuned: Vec<Option<st_zrt::OwnedValue>> =
        (0..sess.output_count()).map(|_| None).collect();
    sess.run_with(&[&x], &mut tuned, &opts)
        .expect("adapter run");
    let tuned_y = tuned[0]
        .as_ref()
        .expect("tuned output")
        .as_slice::<f32>()
        .expect("read tuned output");

    assert!(
        tuned_y
            .iter()
            .zip(&base_y)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "active adapter must change the output; base={:?}, tuned={:?}",
        &base_y[..4],
        &tuned_y[..4]
    );
}
