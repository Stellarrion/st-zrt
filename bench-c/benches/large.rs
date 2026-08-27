//! Large-tensor benches on the synthetic relay model (Y = X + C).
//!
//! Arena note (corrected 2026-06-19, see RESULTS.md §8): the zero-copy input path
//! (`CreateTensorWithDataAsOrtValue` over a caller buffer) is STABLE with the BFCArena
//! ENABLED (ORT's default) — verified across arena×pattern×size (4m/16m), under both
//! criterion's measurement loop and a tight plain loop. An earlier st-zrt build crashed
//! here and the failure was misattributed to an ORT BFCArena bug; re-investigation showed
//! the crash does NOT reproduce with the current code, the st-zrt FFI matches the
//! documented `CreateTensorWithDataAsOrtValue` contract, and libonnxruntime is pinned to
//! v1.27.0 (SHA-256-verified) so the ORT binary is byte-identical — i.e. the crash was a
//! st-zrt lifetime/handle bug, now fixed. The arena stays ON (the realistic default);
//! `crash_repro.rs` and `examples/zc_repro.rs` lock that in as a regression guard.
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OutputValue, SessionOptions, Tensor,
};
use st_zrt_bench_c::models;

fn relay_session(env: &Environment, label: &str) -> (st_zrt::Session, MemoryInfo) {
    let path = models::ensure_relay(label).expect("ensure_relay");
    let mem = MemoryInfo::cpu().unwrap();
    // Arena ON (ORT default). The caller owns `env` for the Session's whole lifetime — the
    // Env MUST outlive every Session created from it: ORT sessions reference the Env's thread
    // pools/allocator, and releasing the Env first is a use-after-free that surfaces as heap
    // corruption (SIGSEGV) under sustained runs.
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(env, path.to_str().unwrap(), opts).unwrap();
    (sess, mem)
}

fn bench_relay_run_4m(c: &mut Criterion) {
    // Regular path: per-iteration zero-copy input + malloc-backed 4 MiB output.
    let n = 1usize << 20;
    let env = Environment::new().unwrap();
    let (sess, mem) = relay_session(&env, "4m");
    let x = vec![3.0_f32; n];
    for _ in 0..16 {
        let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
        let mut out: Vec<Option<st_zrt::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
    }
    c.bench_function("C_relay_run_4m", |b| {
        b.iter(|| {
            let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
            let mut out: Vec<Option<st_zrt::OwnedValue>> =
                (0..sess.output_count()).map(|_| None).collect();
            sess.run(&[&input], &mut out).unwrap();
            black_box(out[0].as_ref().unwrap().as_slice::<f32>().unwrap());
        });
    });
}

fn bench_relay_iobinding_4m(c: &mut Criterion) {
    // IoBinding: input + a caller 4 MiB output buffer bound once; no per-run alloc.
    let n = 1usize << 20;
    let env = Environment::new().unwrap();
    let (sess, mem) = relay_session(&env, "4m");
    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
    let mut y_buf = vec![0.0_f32; n];
    let out_val = OutputValue::from_buffer(&mut y_buf, &[1, n as i64], &mem).unwrap();
    let mut prepared = sess.prepare_io_binding(&[&input], &[&out_val]).unwrap();
    for _ in 0..16 {
        prepared.run().unwrap();
        black_box(out_val.as_slice::<f32>().unwrap());
    }
    c.bench_function("C_relay_iobinding_4m", |b| {
        b.iter(|| {
            prepared.run().unwrap();
            black_box(out_val.as_slice::<f32>().unwrap());
        });
    });
}

fn bench_relay_lane_4m(c: &mut Criterion) {
    // Lane-local path: owns stable input/output buffers + prepared binding.
    let n = 1usize << 20;
    let env = Environment::new().unwrap();
    let (sess, mem) = relay_session(&env, "4m");
    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, n as i64]], &[&[1, n as i64]])
        .unwrap();
    lane.input_mut(0).expect("lane input").fill(3.0);
    for _ in 0..16 {
        lane.run().unwrap();
        black_box(lane.output(0).expect("lane output"));
    }
    c.bench_function("C_relay_lane_4m", |b| {
        b.iter(|| {
            lane.run().unwrap();
            black_box(lane.output(0).expect("lane output"));
        });
    });
}

fn bench_relay_iobinding_16m(c: &mut Criterion) {
    // IoBinding at 16 MiB I/O — above the ~5 MB crossover.
    let n = 1usize << 22;
    let env = Environment::new().unwrap();
    let (sess, mem) = relay_session(&env, "16m");
    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
    let mut y_buf = vec![0.0_f32; n];
    let out_val = OutputValue::from_buffer(&mut y_buf, &[1, n as i64], &mem).unwrap();
    let mut prepared = sess.prepare_io_binding(&[&input], &[&out_val]).unwrap();
    for _ in 0..16 {
        prepared.run().unwrap();
        black_box(out_val.as_slice::<f32>().unwrap());
    }
    c.bench_function("C_relay_iobinding_16m", |b| {
        b.iter(|| {
            prepared.run().unwrap();
            black_box(out_val.as_slice::<f32>().unwrap());
        });
    });
}

fn bench_relay_lane_16m(c: &mut Criterion) {
    let n = 1usize << 22;
    let env = Environment::new().unwrap();
    let (sess, mem) = relay_session(&env, "16m");
    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, n as i64]], &[&[1, n as i64]])
        .unwrap();
    lane.input_mut(0).expect("lane input").fill(3.0);
    for _ in 0..16 {
        lane.run().unwrap();
        black_box(lane.output(0).expect("lane output"));
    }
    c.bench_function("C_relay_lane_16m", |b| {
        b.iter(|| {
            lane.run().unwrap();
            black_box(lane.output(0).expect("lane output"));
        });
    });
}

fn bench_relay_run_4m_copied_arena(c: &mut Criterion) {
    // Copied input (engine-owned) + arena ENABLED — a latency comparison against the
    // zero-copy `C_relay_run_4m` (also arena-on). The ~60 µs delta IS the copy cost the
    // zero-copy path avoids. (Was once a "discriminator" for a now-disproven crash; kept
    // as a useful copied-vs-zero-copy comparison.)
    let n = 1usize << 20;
    let path = models::ensure_relay("4m").unwrap();
    let env = Environment::new().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1); // arena ON (ort's default)
    let sess = st_zrt::Session::new(&env, path.to_str().unwrap(), opts).unwrap();
    let x = vec![3.0_f32; n];
    for _ in 0..16 {
        let input = Tensor::copy_from_slice(&x, &[1, n as i64]).unwrap();
        let mut out: Vec<Option<st_zrt::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
    }
    c.bench_function("C_relay_run_4m_copied_arena", |b| {
        b.iter(|| {
            let input = Tensor::copy_from_slice(&x, &[1, n as i64]).unwrap();
            let mut out: Vec<Option<st_zrt::OwnedValue>> =
                (0..sess.output_count()).map(|_| None).collect();
            sess.run(&[&input], &mut out).unwrap();
            black_box(out[0].as_ref().unwrap().as_slice::<f32>().unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_relay_run_4m,
    bench_relay_iobinding_4m,
    bench_relay_lane_4m,
    bench_relay_iobinding_16m,
    bench_relay_lane_16m,
    bench_relay_run_4m_copied_arena
);
criterion_main!(benches);
