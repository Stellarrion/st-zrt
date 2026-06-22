//! End-to-end smoke tests for the variant-C safe layer against the real engine.
//!
//! These load `bench/models/mnist.onnx` (ort's hosted MNIST test model: one float
//! input [1,1,28,28] → one float output [1,10]) and exercise the full path:
//! Environment → SessionOptions → MemoryInfo → Session (pre-marshaled names) →
//! Tensor::from_buffer (zero-copy input) → run → OwnedValue::as_slice (zero-copy output).

use st_zrt::{
    Allocator, AllocatorType, ArenaCfg, ArenaExtendStrategy, DynamicIoOptions, DynamicIoRuntime,
    Environment, GraphOptimizationLevel, IoBinding, LaneBufferPolicy, LoggingLevel, MemType,
    MemoryInfo, ModelMetadata, OutputValue, OwnedInitializer, OwnedValue,
    PrepackedWeightsContainer, RunOptions, Runtime, RuntimeMode, Session, SessionOptions,
    StaticIoLane, StaticIoRuntime, Tensor, TensorBuffer, sys,
};
use std::sync::Arc;

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn mnist_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx")
}

fn relay_path(label: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join(format!("relay_{label}.onnx"))
}

/// Load the MNIST session at opt-level All, or skip the caller (returns None) if the model
/// isn't cached. `env` is owned by the caller and must outlive the returned Session (ORT
/// sessions reference the Env's thread pools/allocator; releasing the Env first is a UAF).
fn mnist_session(env: &Environment) -> Option<(MemoryInfo, Session)> {
    let path = mnist_path();
    if !path.exists() {
        return None;
    }
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(env, path.to_str().unwrap(), opts).expect("session");
    Some((mem, sess))
}

#[test]
fn mnist_end_to_end() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping mnist_end_to_end — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem info");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");

    assert_eq!(sess.input_count(), 1, "MNIST has 1 input");
    assert_eq!(sess.output_count(), 1, "MNIST has 1 output");
    assert_eq!(
        sess.input_meta(0).expect("input meta"),
        (
            sys::OnnxType::Tensor,
            st_zrt::ElementType::Float,
            Some(28 * 28)
        )
    );
    assert_eq!(
        sess.output_meta(0).expect("output meta"),
        (sys::OnnxType::Tensor, st_zrt::ElementType::Float, Some(10))
    );
    eprintln!("input[0]  = {}", sess.input_name(0).expect("input name"));
    eprintln!("output[0] = {}", sess.output_name(0).expect("output name"));

    // Zero-copy input: wrap a caller-owned buffer; the engine reads it in place.
    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("zero-copy input");

    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");

    let out = outputs[0].as_ref().expect("output 0 present");
    let logits: &[f32] = out.as_slice().expect("zero-copy output read");
    assert_eq!(logits.len(), 10, "MNIST output should be 10 logits");
    assert!(
        out.get_value(0).is_err(),
        "tensor output has no child values"
    );
    eprintln!("logits: {:?}", logits);
}

#[test]
fn public_misuse_returns_errors_not_panics() {
    let Some((mem, sess)) = mnist_session(&Environment::new().expect("env")) else {
        eprintln!("skipping misuse test — mnist.onnx absent");
        return;
    };
    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("zero-copy input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();

    assert!(sess.run(&[], &mut outputs).is_err());
    assert!(sess.prepare_run(&[]).is_err());
    assert!(sess.run(&[&input], &mut []).is_err());
    assert!(
        sess.prepare_tensor_io_lane::<f32>(&mem, &[], &[&[1, 10]])
            .is_err()
    );
    assert!(sess.input_name(sess.input_count()).is_err());
    assert!(sess.output_name(sess.output_count()).is_err());
    assert!(sess.input_meta(sess.input_count()).is_err());
    assert!(sess.output_meta(sess.output_count()).is_err());
    assert!(sess.input_shape(sess.input_count()).is_err());
    assert!(sess.output_shape(sess.output_count()).is_err());
    assert!(sess.input_symbolic_dims(sess.input_count()).is_err());
    assert!(sess.output_symbolic_dims(sess.output_count()).is_err());
}

#[test]
fn run_is_shared_reentrant() {
    // run(&self) must be safe to call concurrently from multiple threads — ORT's Run
    // is thread-safe on a session, and our safe layer must not introduce shared state.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping reentrancy test — mnist.onnx absent");
        return;
    }

    let env = Arc::new(Environment::new().unwrap());
    let sess = {
        let opts = SessionOptions::new();
        Arc::new(Session::new(&env, path.to_str().unwrap(), opts).unwrap())
    };

    fn run_once(sess: &Session) -> usize {
        let mem = MemoryInfo::cpu().unwrap();
        let buf = vec![0.0_f32; 784];
        let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).unwrap();
        let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len()
    }

    let s2 = sess.clone();
    let h = std::thread::spawn(move || run_once(&s2));
    let main_n = run_once(&sess);
    let thread_n = h.join().unwrap();
    assert_eq!(main_n, 10);
    assert_eq!(thread_n, 10);
    eprintln!("concurrent shared-ref runs OK (both returned 10 logits)");
}

#[test]
fn session_outlives_env_drop() {
    // The Env is dropped right after Session construction — the exact pattern behind the
    // historical ">4MB segfault": ORT sessions reference the Env's thread pools/allocator, so
    // running a Session after its Env was freed corrupted the heap (a use-after-free). Session
    // now holds an `Arc` ref to the Env, keeping it alive for the Session's whole lifetime, so
    // this must run cleanly. Reverting that fix would make this a UAF again. (The large-tensor,
    // sustained-load variant of the same invariant is guarded by bench-c/benches/crash_repro.rs.)
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }
    let sess = {
        let env = Environment::new().expect("env");
        Session::new(&env, path.to_str().unwrap(), SessionOptions::new()).expect("session")
        // `env` drops here; the Env survives via the Session's Arc ref.
    };
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)
        .expect("run after Env dropped");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
    eprintln!("Session outlives Env drop OK — Arc keeps the Env alive (UAF gone)");
}

#[test]
fn iobinding_zero_copy_output() {
    // Bind the output to a CALLER-OWNED buffer (zero-copy: ORT writes logits straight into
    // out_buf). Result must match the regular run() path bit-for-bit.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    // Reference logits from the regular path.
    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let ref_logits: Vec<f32> = out[0].as_ref().unwrap().as_slice::<f32>().unwrap().to_vec();

    // IoBinding: bind input + a preallocated [1,10] output buffer, then run_binding.
    let in2 = vec![0.0_f32; 784];
    let input2 = Tensor::from_buffer(&in2, &[1, 1, 28, 28], &mem).expect("input2");
    let mut out_buf = vec![0.0_f32; 10];
    let out_val = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("output value");
    let mut binding = IoBinding::new(&sess).expect("binding");
    binding
        .bind_input(sess.input_name(0).expect("input name"), &input2)
        .expect("bind input");
    binding
        .bind_output(sess.output_name(0).expect("output name"), &out_val)
        .expect("bind output");
    sess.run_binding(&binding).expect("run_binding");

    let got: &[f32] = out_val.as_slice::<f32>().expect("zero-copy output read");
    assert_eq!(got.len(), 10, "MNIST output is 10 logits");
    assert_eq!(
        got,
        ref_logits.as_slice(),
        "zero-copy output must match the regular path"
    );
    eprintln!(
        "IoBinding zero-copy output OK ({} logits match regular path)",
        got.len()
    );
}

#[test]
fn prepared_run_reuses_hot_path_state() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut run = sess.prepare_run(&[&input]).expect("prepare_run");

    for _ in 0..3 {
        run.run().expect("prepared run");
        let logits = run
            .output(0)
            .expect("output index")
            .unwrap()
            .as_slice::<f32>()
            .unwrap();
        assert_eq!(logits.len(), 10);
    }
    assert!(run.output(sess.output_count()).is_err());
    eprintln!("PreparedRun reused handles and returned 10 logits");
}

#[test]
fn prepared_iobinding_zero_copy_output() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out_buf = vec![0.0_f32; 10];
    let out_val = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("output value");
    let mut prepared = sess
        .prepare_io_binding(&[&input], &[&out_val])
        .expect("prepare_io_binding");

    prepared.run().expect("prepared iobinding run");
    let logits = out_val.as_slice::<f32>().expect("zero-copy output read");
    assert_eq!(logits.len(), 10);
    eprintln!("PreparedIoBinding wrote 10 logits into caller output");
}

#[test]
fn session_with_prepacked_weights_keeps_cache_alive() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let sess = {
        let cache = PrepackedWeightsContainer::new().expect("prepacked cache");
        Session::new_with_prepacked_weights(
            &env,
            path.to_str().unwrap(),
            SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
            &cache,
        )
        .expect("session with prepacked weights")
    };

    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)
        .expect("run after external cache handle dropped");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
}

#[test]
fn session_with_owned_initializer_overrides_model_weight() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let c = TensorBuffer::from_vec(vec![2.0_f32; n], &[1, n as i64], &mem).expect("C buffer");
    let init = OwnedInitializer::tensor("C", c).expect("initializer");
    let sess = Session::new_with_owned_initializers(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::All)
            .with_intra_threads(1),
        vec![init],
    )
    .expect("session with owned initializer");

    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y.len(), n);
    assert_eq!(y[0], 5.0);
    assert_eq!(y[n - 1], 5.0);
}

#[test]
fn session_with_mmap_owned_initializer_overrides_model_weight() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let mmap_path = std::env::temp_dir().join(format!(
        "st-zrt-relay-c-mmap-{}-{}.bin",
        std::process::id(),
        n
    ));
    let c_values = vec![2.0_f32; n];
    std::fs::write(&mmap_path, f32_as_bytes(&c_values)).expect("write mmap initializer");
    let c = TensorBuffer::<f32>::from_mmap_file(&mmap_path, &[1, n as i64], &mem)
        .expect("mmap C buffer");
    let init = OwnedInitializer::tensor("C", c).expect("initializer");
    let sess = Session::new_with_owned_initializers(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::All)
            .with_intra_threads(1),
        vec![init],
    )
    .expect("session with mmap owned initializer");
    let _ = std::fs::remove_file(&mmap_path);

    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y.len(), n);
    assert_eq!(y[0], 5.0);
    assert_eq!(y[n - 1], 5.0);
}

#[test]
fn allocator_stats_snapshot_is_available_when_ort_supports_it() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let allocator = Allocator::create(&sess, &mem).expect("allocator");
    match allocator.stats() {
        Ok(stats) => {
            let delta = stats.diff(&stats);
            assert!(delta.entries().is_empty());
            assert!(delta.get("missing").is_none());
            eprintln!("allocator stats: {:?}", stats.entries());
        },
        Err(err) => eprintln!("allocator stats unsupported by this allocator: {err}"),
    }
}

#[test]
fn tensor_io_lane_reuses_owned_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("lane");
    assert!(lane.input(1).is_err());
    assert!(lane.input_mut(1).is_err());
    assert!(lane.output(1).is_err());
    assert!(lane.output_mut(1).is_err());
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_buffer(1).is_err());
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    lane.run().expect("second lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    eprintln!("TensorIoLane reused owned input/output buffers");
}

#[test]
fn static_tensor_io_lane_reuses_owned_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_static_tensor_io_lane::<f32, 1, 1>(&mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static lane");
    assert_eq!(lane.inputs().len(), 1);
    assert_eq!(lane.outputs().len(), 1);
    assert!(lane.input(1).is_err());
    assert!(lane.output(1).is_err());
    lane.inputs_mut()[0].as_mut_slice().fill(0.0);
    lane.run().expect("static lane run");
    assert_eq!(lane.outputs()[0].as_slice().len(), 10);

    let allocator = Allocator::create(&sess, &mem).expect("allocator");
    match lane.run_with_allocator_stats(&allocator) {
        Ok(stats) => {
            let _ = stats.delta();
        },
        Err(err) => eprintln!("allocator stats unsupported by this allocator: {err}"),
    }
}

#[test]
fn tensor_io_lane_buffer_policy_controls_alignment() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&[1, 1, 28, 28]],
            &[&[1, 10]],
            LaneBufferPolicy::AlignedPrefaulted { alignment: 64 },
        )
        .expect("aligned lane");
    assert_eq!((lane.input(0).expect("input").as_ptr() as usize) % 64, 0);
    assert_eq!((lane.output(0).expect("output").as_ptr() as usize) % 64, 0);
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.prime(2).expect("aligned lane prime");
    lane.run().expect("aligned lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);

    let locked_lane = sess
        .prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&[1, 1, 28, 28]],
            &[&[1, 10]],
            LaneBufferPolicy::AlignedMlockedPrefaulted { alignment: 4096 },
        )
        .expect("mlocked lane");
    assert_eq!(
        (locked_lane.input(0).expect("input").as_ptr() as usize) % 4096,
        0
    );
    assert_eq!(
        (locked_lane.output(0).expect("output").as_ptr() as usize) % 4096,
        0
    );
}

#[test]
fn tensor_io_lane_auto_policy_aligns_large_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let large_shape = [1, 1_i64 << 19]; // 2 MiB of f32, the Auto hugepage threshold.
    let lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&large_shape], &[&large_shape])
        .expect("auto large lane");
    assert_eq!(
        (lane.input(0).expect("input").as_ptr() as usize) % (2 << 20),
        0
    );
    assert_eq!(
        (lane.output(0).expect("output").as_ptr() as usize) % (2 << 20),
        0
    );
}

#[test]
fn allocated_output_tensor_io_lane_runs() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_allocated_output_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("allocated-output lane");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("allocated-output lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_tensor(1).is_err());

    let mut allocated_lane = sess
        .prepare_allocated_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("allocated tensor lane");
    allocated_lane
        .input_mut(0)
        .expect("allocated lane input")
        .fill(0.0);
    allocated_lane.run().expect("allocated tensor lane run");
    assert_eq!(
        allocated_lane
            .output(0)
            .expect("allocated lane output")
            .len(),
        10
    );
    assert!(allocated_lane.input_tensor(1).is_err());
    assert!(allocated_lane.output_tensor(1).is_err());

    let mut device_lane = sess
        .prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]])
        .expect("device-output lane");
    device_lane
        .input_mut(0)
        .expect("device lane input")
        .fill(0.0);
    let outputs = device_lane.run().expect("device-output lane run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        device_lane
            .output(0)
            .expect("device lane output")
            .as_slice::<f32>()
            .expect("device lane output slice")
            .len(),
        10
    );
    assert!(device_lane.input_buffer(1).is_err());
    assert!(device_lane.output(1).is_err());
}

#[test]
fn tensor_io_lanes_run_independent_bindings() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lanes = sess
        .prepare_tensor_io_lanes::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
        .expect("lanes");
    for (i, lane) in lanes.iter_mut().enumerate() {
        lane.input_mut(0).expect("lane input").fill(i as f32);
        lane.run().expect("lane run");
        assert_eq!(lane.output(0).expect("lane output").len(), 10);
    }
    eprintln!("TensorIoLane set ran independent bindings");
}

#[test]
fn runtime_shared_session_runs_exclusive_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut runtime =
        Runtime::<f32>::shared_session(Arc::new(sess), &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
            .expect("runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let output_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").len())
        })
        .expect("runtime run");
    assert_eq!(output_len, 10);

    let lane = runtime.lane_mut(1).expect("lane");
    assert!(lane.input(1).is_err());
    assert!(lane.input_mut(1).is_err());
    assert!(lane.output(1).is_err());
    assert!(lane.output_mut(1).is_err());
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_buffer(1).is_err());
    lane.input_mut(0).expect("lane input").fill(1.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
}

#[test]
fn runtime_runs_without_checkout() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lanes =
        Runtime::<f32>::shared_session(Arc::new(sess), &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
            .expect("lane set");
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes.session_mode(), RuntimeMode::SharedSession);
    assert!(lanes.lane(2).is_err());
    assert!(lanes.lane_mut(2).is_err());

    let lane = lanes.lane_mut(0).expect("lane 0");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
}

#[test]
fn runtime_converts_into_lane_set() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let runtime =
        Runtime::<f32>::shared_session(Arc::new(sess), &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 1)
            .expect("runtime");
    let mut lanes = runtime.into_lane_set();
    assert_eq!(lanes.len(), 1);
    let lane = lanes.lane_mut(0).expect("lane");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
}

#[test]
fn runtime_replicated_sessions_run_independent_lanes() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let mut runtime = Runtime::<f32>::replicated_sessions(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
        &mem,
        &[&[1, 1, 28, 28]],
        &[&[1, 10]],
        2,
    )
    .expect("runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::ReplicatedSessions);
    assert!(runtime.lane(2).is_err());
    assert!(runtime.lane_mut(2).is_err());

    let output_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").len())
        })
        .expect("runtime run");
    assert_eq!(output_len, 10);
}

#[test]
fn runtime_session_factory_supports_owned_initializers() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let path = path.to_str().unwrap().to_owned();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let mut runtime =
        Runtime::<f32>::from_session_factory(2, &mem, &[&[1, n as i64]], &[&[1, n as i64]], |_| {
            let c = TensorBuffer::from_vec(vec![2.0_f32; n], &[1, n as i64], &mem)?;
            let init = OwnedInitializer::tensor("C", c)?;
            Session::new_with_owned_initializers(&env, &path, opts.clone(), vec![init])
        })
        .expect("runtime with owned initializers");

    let y_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(3.0);
            lane.run()?;
            let y = lane.output(0).expect("lane output");
            assert_eq!(y[0], 5.0);
            assert_eq!(y[n - 1], 5.0);
            Ok(y.len())
        })
        .expect("runtime run");
    assert_eq!(y_len, n);
}

#[test]
fn iobinding_device_output() {
    // Bind the output to device memory (ORT allocates) and read it back via
    // GetBoundOutputValues — the path for dynamic-shape outputs.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut binding = IoBinding::new(&sess).expect("binding");
    binding
        .bind_input(sess.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_device(sess.output_name(0).expect("output name"), &mem)
        .expect("bind output device");
    sess.run_binding(&binding).expect("run_binding");

    let vals = binding.output_values().expect("output_values");
    assert_eq!(vals.len(), 1, "one output value");
    let logits = vals[0].as_slice::<f32>().expect("device output read");
    assert_eq!(logits.len(), 10, "MNIST output is 10 logits");
    eprintln!(
        "IoBinding device output OK ({} logits via GetBoundOutputValues)",
        logits.len()
    );
}

#[test]
fn static_io_lane_runs_static_typed_io() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = Arc::new(sess);
    let mut lane =
        StaticIoLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
            .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);
    lane.run().expect("run");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
}

#[test]
fn static_io_runtime_runs_shared_typed_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = Arc::new(sess);
    let mut runtime = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        session,
        &mem,
        [&[1, 1, 28, 28]],
        [&[1, 10]],
        2,
    )
    .expect("static I/O runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let len = runtime
        .run_on(1, |lane| {
            lane.input_mut(0)?.fill(0.0);
            lane.run()?;
            Ok(lane.output(0)?.len())
        })
        .expect("run lane");
    assert_eq!(len, 10);
}

#[test]
fn dynamic_io_runtime_caches_and_runs_shape_bucket() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = Arc::new(sess);
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(session, mem, 2)
        .expect("dynamic I/O runtime");
    assert_eq!(runtime.bucket_count(), 0);
    assert_eq!(runtime.lane_count(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let len = runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 1, |lane| {
            lane.input_mut_at::<0>()?.fill(0.0);
            lane.run()?;
            Ok(lane.output_at::<0>()?.len())
        })
        .expect("dynamic run");
    assert_eq!(len, 10);
    assert_eq!(runtime.bucket_count(), 1);

    runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| {
            lane.input_mut(0)?.fill(0.0);
            lane.run()
        })
        .expect("dynamic cached run");
    assert_eq!(runtime.bucket_count(), 1);
    assert_eq!(
        runtime.buckets()[0].key().input_shape(0),
        Some(&[1, 1, 28, 28][..])
    );

    runtime
        .prime_bucket([&[1, 1, 28, 28]], [&[1, 10]], 1)
        .expect("prime cached bucket");
    assert!(runtime.remove_bucket([&[1, 1, 28, 28]], [&[1, 10]]));
    assert_eq!(runtime.bucket_count(), 0);

    runtime
        .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("recreate bucket");
    assert_eq!(runtime.bucket_count(), 1);
    runtime.clear_buckets();
    assert_eq!(runtime.bucket_count(), 0);
}

#[test]
fn dynamic_io_runtime_bounds_shape_buckets() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let session = Arc::new(sess);
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(1),
    )
    .expect("dynamic I/O runtime");

    runtime
        .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("first bucket");
    assert_eq!(runtime.bucket_count(), 1);
    assert!(runtime.bucket([&[1, 1, 28, 28]], [&[1, 10]]).is_some());

    runtime
        .get_or_create_bucket([&[1, 1, 28, 29]], [&[1, 10]])
        .expect("second bucket");
    assert_eq!(runtime.bucket_count(), 1);
    assert!(runtime.bucket([&[1, 1, 28, 28]], [&[1, 10]]).is_none());
    assert!(runtime.bucket([&[1, 1, 28, 29]], [&[1, 10]]).is_some());
}

#[test]
fn session_from_bytes_and_metadata() {
    // Load from an in-memory byte buffer (no temp file) and read the model metadata.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }
    let bytes = std::fs::read(&path).expect("read mnist.onnx");

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::from_bytes(&env, &bytes, opts).expect("from_bytes");

    // Inference through the from_bytes session works identically to the file path.
    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );

    // Metadata: producer name + version are readable; producer is Some for a real model.
    let md: ModelMetadata = sess.metadata().expect("metadata");
    let producer = md.producer_name().expect("producer name");
    let version = md.version().expect("version");
    eprintln!("from_bytes OK; producer={:?} version={}", producer, version);
    assert!(producer.is_some(), "a real model has a producer name");
    // version() must not error and must be non-negative for a valid model.
    assert!(version >= 0, "model version is non-negative");
}

#[test]
fn run_with_options_config() {
    // Build a caller RunOptions (log severity + config entry) and pass it to run_with.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut opts = RunOptions::new().expect("run options");
    opts.set_log_severity(LoggingLevel::Fatal)
        .expect("severity")
        .set_run_tag("zrt-smoke")
        .expect("run tag");

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run_with(&[&input], &mut out, &opts).expect("run_with");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
    eprintln!("run_with (caller RunOptions) OK");
}

#[test]
fn run_options_terminate_cancels() {
    // Pre-terminate the RunOptions; the subsequent run must return an error (ORT checks the
    // terminate flag and aborts). Proves SetTerminate takes effect.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let opts = RunOptions::new().expect("run options");
    opts.terminate().expect("terminate");

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    let res = sess.run_with(&[&input], &mut out, &opts);
    assert!(
        res.is_err(),
        "a run with a pre-terminated RunOptions must error"
    );
    eprintln!(
        "terminate → run_with returned Err (cancelled): {:?}",
        res.err()
    );
}

#[test]
fn memory_info_named() {
    // General named MemoryInfo + introspection getters (round-trip through the engine).
    let mem = MemoryInfo::new_named("Cpu", AllocatorType::Device, 0, MemType::Default)
        .expect("new_named");
    assert_eq!(mem.name().unwrap(), "Cpu");
    assert_eq!(mem.device_id().unwrap(), 0);
    assert_eq!(mem.alloc_type().unwrap(), AllocatorType::Device);
    assert_eq!(mem.mem_type().unwrap(), MemType::Default);
    eprintln!(
        "MemoryInfo::new_named round-trips (name={:?})",
        mem.name().unwrap()
    );
}

#[test]
fn arena_cfg_construct() {
    // Both ArenaCfg constructors succeed on CPU; register_allocator is best-effort here
    // (its CPU support is ORT-version-specific, so we log rather than hard-assert).
    assert_eq!(ArenaExtendStrategy::NextPowerOfTwo as i32, 0);
    assert_eq!(ArenaExtendStrategy::SameAsRequested as i32, 1);
    let cfg =
        ArenaCfg::new(usize::MAX, ArenaExtendStrategy::NextPowerOfTwo, -1, -1).expect("arena cfg");
    let cfg_v2 = ArenaCfg::with_entries(&[(
        "arena_extend_strategy",
        ArenaExtendStrategy::SameAsRequested as usize,
    )])
    .expect("arena cfg v2");
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let reg = env.register_allocator(&mem, &cfg);
    eprintln!(
        "ArenaCfg OK (v1 + v2); register_allocator(CPU) -> {:?}",
        reg.is_ok()
    );
    drop(cfg_v2);
}

#[test]
fn allocator_create_and_allocate() {
    // Create a session-scoped allocator, allocate through it, let RAII free on drop.
    let env = Environment::new().expect("env");
    let (_mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let alloc = Allocator::create(&sess, &mem).expect("create allocator");
    let buf = alloc.allocate(128).expect("allocate 128 bytes");
    assert!(!buf.as_ptr().is_null(), "allocated buffer is non-null");
    // `buf` frees on drop (AllocatorFree); `alloc` releases on drop (ReleaseAllocator).
    eprintln!("Allocator::create + allocate(128) OK (ptr non-null, RAII frees)");
}
