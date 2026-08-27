//! Model-loader + large-tensor validation: loads a synthetic relay model (Y = X + C,
//! C a constant) of a controlled size through st-zrt, runs it via both the regular path
//! and IoBinding, and validates the result. This unblocks blocker #3 (no large-tensor
//! model): the relay models exercise the O3 large-input crossover and the E2 IoBinding
//! zero-copy-output path that MNIST (3 KB / 40 B) cannot.
use st_zrt::{
    Environment, GraphOptimizationLevel, IoBinding, MemoryInfo, OutputValue, Session,
    SessionOptions, Tensor,
};
use st_zrt_bench_c::models;

fn relay_session(env: &Environment, label: &str) -> (MemoryInfo, Session) {
    // Caller owns `env` for the session's lifetime — the Env MUST outlive the Session (ORT
    // sessions reference the Env's thread pools/allocator; releasing the Env first is a
    // use-after-free that surfaces as heap corruption under sustained runs).
    let path = models::ensure_relay(label).expect("ensure_relay (needs `pip install onnx`)");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = Session::new(env, path.to_str().unwrap(), opts).expect("session");
    (mem, sess)
}

#[test]
fn relay_loads_runs_and_reports_metadata() {
    // 4 MB of f32 I/O: input X / output Y both [1, 1<<20]. Y = X + 1.
    let n = 1usize << 20;
    let env = Environment::new().expect("env");
    let (mem, sess) = relay_session(&env, "4m");
    assert_eq!(sess.input_count(), 1);
    assert_eq!(sess.output_count(), 1);
    assert_eq!(sess.output_shape(0).expect("output shape"), &[1, n as i64]);

    let x: Vec<f32> = vec![3.0; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("input");

    let mut out: Vec<Option<st_zrt::OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().expect("output");
    assert_eq!(y.len(), n, "output element count");
    assert!(y.iter().all(|&v| v == 4.0), "Y = X + 1 (3.0 -> 4.0)");

    let md = sess.metadata().expect("metadata");
    assert_eq!(md.producer_name().unwrap().as_deref(), Some("st-zrt-bench"));
    eprintln!(
        "relay_4m OK: {} elems, producer={:?} version={}",
        n,
        md.producer_name().unwrap(),
        md.version().unwrap()
    );
}

#[test]
fn relay_iobinding_zero_copy_output() {
    // Bind a caller-owned output buffer; ORT must write Y = X + 1 straight into it (zero-copy).
    let n = 1usize << 20;
    let env = Environment::new().expect("env");
    let (mem, sess) = relay_session(&env, "4m");

    let x: Vec<f32> = vec![3.0; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("input");
    let mut y_buf = vec![0.0_f32; n];
    let out_val = OutputValue::from_buffer(&mut y_buf, &[1, n as i64], &mem).expect("output value");

    let mut binding = IoBinding::new(&sess).expect("binding");
    binding
        .bind_input(sess.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output(sess.output_name(0).expect("output name"), &out_val)
        .expect("bind output");
    sess.run_binding(&binding).expect("run_binding");

    let y = out_val.as_slice::<f32>().expect("zero-copy output");
    assert_eq!(y.len(), n);
    assert!(y.iter().all(|&v| v == 4.0), "zero-copy output Y = X + 1");
    eprintln!(
        "relay_4m IoBinding zero-copy output OK ({} elems, all 4.0)",
        n
    );
}

#[test]
fn relay_runs_at_16mb() {
    // Above the ~5 MB O3 crossover: a 16 MB I/O relay. Arena ON (ORT default) — zero-copy
    // large inputs are stable with it. The env is held in this test scope so the session is
    // never used after the env drops (the use-after-free that once crashed this path).
    let n = 1usize << 22; // 4,194,304 elems = 16 MiB f32
    let path = models::ensure_relay("16m").expect("ensure_relay");
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    assert_eq!(sess.output_shape(0).expect("output shape"), &[1, n as i64]);

    let x: Vec<f32> = vec![1.5; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("input");
    let mut out: Vec<Option<st_zrt::OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().expect("output");
    assert_eq!(y.len(), n);
    assert!(
        y.iter().all(|&v| (v - 2.5).abs() < 1e-6),
        "Y = X + 1 (1.5 -> 2.5)"
    );
    eprintln!(
        "relay_16m OK: {} elems (16 MiB I/O) end-to-end, arena on",
        n
    );
}
