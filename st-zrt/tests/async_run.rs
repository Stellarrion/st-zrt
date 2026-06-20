//! `RunAsync` → generic `Future`: runs MNIST asynchronously and asserts the result matches
//! the synchronous `run()`. Uses a tiny std-only `block_on` (no async runtime) — the whole
//! point is that `RunFuture` is pollable by *any* executor.

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
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
