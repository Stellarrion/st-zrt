//! `RunAsync` → generic `Future`: runs MNIST asynchronously and asserts the result matches
//! the synchronous `run()`. Uses a tiny std-only `block_on` (no async runtime) — the whole
//! point is that `RunFuture` is pollable by *any* executor.

use st_zrt::{
    Allocator, ElementType, Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Runtime,
    ServingLane, Session, SessionOptions, Tensor,
};

fn mnist_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx")
}

/// A minimal `block_on`: pin the future on the stack and poll it, yielding the thread each
/// spin so the ORT worker thread (which fires `RunAsync`'s callback + wakes) can make
/// progress. The waker is a no-op — we rely on the yield-spin to observe completion.
fn block_on<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake};

    struct NoopWake;
    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }
    let waker = Arc::new(NoopWake).into();
    let mut cx = Context::from_waker(&waker);
    // SAFETY: `fut` stays pinned on this stack frame for the whole poll loop.
    let mut pinned = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
    loop {
        if let Poll::Ready(v) = pinned.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

#[test]
fn run_async_matches_sync() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skip — mnist.onnx not cached");
        return;
    }
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");

    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");

    // Synchronous reference run.
    let mut sync_out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut sync_out).expect("sync run");
    let sync_logits: Vec<f32> = sync_out[0]
        .as_ref()
        .expect("sync output")
        .as_slice()
        .expect("sync output read")
        .to_vec();

    // Asynchronous run — the future borrows `sess` + the inputs for `'a`, so bind the input
    // slice to a local (not a temporary) that outlives `block_on`.
    let inputs: [&dyn st_zrt::RunInput; 1] = [&input];
    let fut = sess.run_async(&inputs).expect("start async");
    let async_out = block_on(fut).expect("async run completed");
    let async_logits: &[f32] = async_out[0].as_slice().expect("async output read");

    eprintln!("sync  logits: {sync_logits:?}");
    eprintln!("async logits: {async_logits:?}");
    assert_eq!(async_logits.len(), 10, "MNIST output is 10 logits");
    for (a, b) in sync_logits.iter().zip(async_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "async vs sync logit mismatch: sync={a} async={b}"
        );
    }
    eprintln!("RunAsync output matches the sync run within 1e-6 ✓");
}

#[test]
fn run_async_owned_inputs_matches_sync() {
    // The owned-input async variant: inputs are MOVED into the run state (no borrow hazard, no
    // `unsafe`). The future borrows only the session. We build an `OwnedValue` input via the
    // default allocator and compare its async output to the sync path.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skip — mnist.onnx not cached");
        return;
    }
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");

    // Synchronous reference (borrowed input).
    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut sync_out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut sync_out).expect("sync run");
    let sync_logits: Vec<f32> = sync_out[0]
        .as_ref()
        .expect("sync output")
        .as_slice()
        .expect("sync output read")
        .to_vec();

    // Owned-input async: build an OwnedValue with the same zero data and move it in. The default
    // CPU allocator may not zero-init, so write zeros explicitly to match the sync reference.
    let alloc = Allocator::get_default().expect("default alloc");
    let bytes = 28 * 28 * std::mem::size_of::<f32>();
    let allocation = alloc.allocate(bytes).expect("allocate");
    unsafe {
        std::ptr::write_bytes(allocation.as_mut_ptr() as *mut u8, 0, bytes);
    }
    let owned_input = OwnedValue::from_allocated(allocation, &[1, 1, 28, 28], ElementType::Float)
        .expect("owned input");
    let fut = sess
        .run_async_owned_inputs(vec![owned_input])
        .expect("start async");
    let async_out = block_on(fut).expect("async run completed");
    let async_logits: &[f32] = async_out[0].as_slice().expect("async output read");

    assert_eq!(async_logits.len(), 10, "MNIST output is 10 logits");
    for (a, b) in sync_logits.iter().zip(async_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "owned-async vs sync logit mismatch: sync={a} async={b}"
        );
    }
    eprintln!("Session::run_async_owned_inputs matches the sync run within 1e-6 ✓");
}

/// Build a MNIST session directly (no `mnist_session` helper in this file).
fn mnist_session_direct() -> Option<(MemoryInfo, Session)> {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skip — mnist.onnx not cached");
        return None;
    }
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    Some((mem, sess))
}

#[test]
fn static_io_lane_run_async_matches_sync() {
    let (mem, sess) = match mnist_session_direct() {
        Some(v) => v,
        None => return,
    };
    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);

    // Sync reference: outputs land in the lane's caller-owned buffer.
    lane.run().expect("sync run");
    let sync_out: Vec<f32> = lane.output_at::<0>().expect("output").to_vec();

    // Async: RunAsync produces ORT-owned outputs — there is no asynchronous IoBinding
    // run in the ORT C API, so the async lane cannot fill the caller-owned output buffers.
    let fut = lane.run_async().expect("start async");
    let async_out = block_on(fut).expect("async run completed");
    let async_logits: &[f32] = async_out[0].as_slice().expect("async output read");

    assert_eq!(async_logits.len(), sync_out.len(), "output length mismatch");
    for (a, b) in sync_out.iter().zip(async_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "async vs sync mismatch: sync={a} async={b}"
        );
    }
    eprintln!("ServingLane::run_async matches the sync run within 1e-6 ✓");
}

#[test]
fn runtime_lane_run_async_matches_sync() {
    // `Lane<T>` is the `Session`-owned serving lane held by `Runtime`; it is not built
    // directly (`Lane::new` is crate-private), so reach it through a one-lane `Runtime`.
    let (mem, sess) = match mnist_session_direct() {
        Some(v) => v,
        None => return,
    };
    let mut runtime =
        Runtime::<f32>::shared_session(sess, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 1)
            .expect("runtime");
    let lane = runtime.lane_mut(0).expect("lane");
    lane.input_mut(0).expect("input").fill(0.0);

    // Sync reference: outputs land in the lane's caller-owned buffer.
    lane.run().expect("sync run");
    let sync_out: Vec<f32> = lane.output(0).expect("output").to_vec();

    let fut = lane.run_async().expect("start async");
    let async_out = block_on(fut).expect("async run completed");
    let async_logits: &[f32] = async_out[0].as_slice().expect("async output read");

    assert_eq!(async_logits.len(), sync_out.len(), "output length mismatch");
    for (a, b) in sync_out.iter().zip(async_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "async vs sync mismatch: sync={a} async={b}"
        );
    }
    eprintln!("Lane::run_async matches the sync run within 1e-6 ✓");
}
