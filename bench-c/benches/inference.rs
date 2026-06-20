//! Variant C criterion entry point + phase bisection.
//!
//! `C_full` is the intended ZRT default fast path: prepare once, run many. The
//! diagnostic benches isolate per-phase cost: `C_legacy_full` keeps the old per-call
//! wrapper path, `C_prepared` pre-builds the zero-copy input and reuses the output
//! slots, and `C_1thread` pins intra-op threads to 1 (tiny models are often
//! latency-bound by thread-pool scheduling rather than compute).
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{GraphOptimizationLevel, OutputValue, SessionOptions, Tensor};
use st_zrt_bench_c::{models, variant_c};

const SHAPE: [i64; 4] = [1, 1, 28, 28];

fn load() -> variant_c::VariantC {
    let model = models::ensure_mnist().expect("failed to resolve mnist.onnx");
    let path = model.to_str().unwrap();
    variant_c::VariantC::new(path).expect("failed to load session (C)")
}

fn warmup(v: &mut variant_c::VariantC) {
    for _ in 0..32 {
        v.run_once().expect("warmup C failed");
    }
}

fn bench_c_full(c: &mut Criterion) {
    // Intended default path: precompute stable input handles and reusable output slots.
    let variant = load();
    let sess = variant.session();
    let mem = variant.mem();
    let input = Tensor::from_buffer(variant.input_buf(), &SHAPE, mem).expect("input");
    let mut run = sess.prepare_run(&[&input]).expect("prepare_run");
    for _ in 0..32 {
        run.run().expect("warmup C prepared failed");
    }
    c.bench_function("C_full", |b| {
        b.iter(|| {
            run.run().expect("run C prepared failed");
            black_box(
                run.output(0)
                    .expect("output index")
                    .unwrap()
                    .as_slice::<f32>()
                    .expect("out"),
            );
        });
    });
}

fn bench_c_legacy_full(c: &mut Criterion) {
    let mut variant = load();
    warmup(&mut variant);
    c.bench_function("C_legacy_full", |b| {
        b.iter(|| variant.run_once().expect("run C legacy failed"));
    });
}

fn bench_c_prepared(c: &mut Criterion) {
    // Pre-build the zero-copy input once; reuse the output buffer across iterations.
    // This measures only Session::run + as_slice, isolating input-view/output-alloc cost.
    let mut variant = load();
    warmup(&mut variant);
    let sess = variant.session();
    let mem = st_zrt::MemoryInfo::cpu().unwrap();
    let buf = variant.input_buf();
    let input = Tensor::from_buffer(buf, &SHAPE, &mem).expect("input");
    let mut outputs: Vec<Option<st_zrt::OwnedValue>> =
        (0..sess.output_count()).map(|_| None).collect();

    c.bench_function("C_prepared", |b| {
        b.iter(|| {
            sess.run(&[&input], &mut outputs).expect("run");
            black_box(outputs[0].as_ref().unwrap().as_slice::<f32>().expect("out"));
        });
    });
}

fn bench_c_1thread(c: &mut Criterion) {
    // Same as C_full but intra-op threads pinned to 1.
    let model = models::ensure_mnist().expect("mnist");
    let path = model.to_str().unwrap();
    let env = st_zrt::Environment::new().unwrap();
    let mem = st_zrt::MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(&env, path, opts).unwrap();
    let input_buf = vec![0.0_f32; 784];
    let mut outputs: Vec<Option<st_zrt::OwnedValue>> =
        (0..sess.output_count()).map(|_| None).collect();
    // warmup
    for _ in 0..32 {
        let input = Tensor::from_buffer(&input_buf, &SHAPE, &mem).unwrap();
        sess.run(&[&input], &mut outputs).unwrap();
    }

    c.bench_function("C_1thread", |b| {
        b.iter(|| {
            let input = Tensor::from_buffer(&input_buf, &SHAPE, &mem).unwrap();
            sess.run(&[&input], &mut outputs).unwrap();
            black_box(outputs[0].as_ref().unwrap().as_slice::<f32>().unwrap());
        });
    });
}

fn bench_c_iobinding(c: &mut Criterion) {
    // IoBinding, bind-once-mutate-in-place, ZERO-COPY OUTPUT into a caller buffer, at the
    // 1-thread floor (directly comparable to C_1thread). The binding is built once: input
    // view + a preallocated [1,10] output buffer are bound by name and reused — no per-run
    // name marshaling, no per-run input/output value allocation. On MNIST the output is
    // 40 bytes, so the E2 (no-output-alloc) win is negligible here; the large-output win
    // is gated on a bigger model (see progress.md blocker #3).
    let model = models::ensure_mnist().expect("mnist");
    let path = model.to_str().unwrap();
    let env = st_zrt::Environment::new().unwrap();
    let mem = st_zrt::MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(&env, path, opts).unwrap();

    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &SHAPE, &mem).unwrap();
    let mut out_buf = vec![0.0_f32; 10];
    let out_val = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).unwrap();
    let mut prepared = sess.prepare_io_binding(&[&input], &[&out_val]).unwrap();

    // warmup
    for _ in 0..32 {
        prepared.run().unwrap();
        black_box(out_val.as_slice::<f32>().unwrap());
    }

    c.bench_function("C_iobinding", |b| {
        b.iter(|| {
            prepared.run().expect("run");
            black_box(out_val.as_slice::<f32>().expect("out"));
        });
    });
}

fn bench_c_lane(c: &mut Criterion) {
    // Lane-local path: owns reusable input/output buffers + a prepared binding.
    let model = models::ensure_mnist().expect("mnist");
    let path = model.to_str().unwrap();
    let env = st_zrt::Environment::new().unwrap();
    let mem = st_zrt::MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(&env, path, opts).unwrap();
    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&SHAPE], &[&[1, 10]])
        .unwrap();

    for _ in 0..32 {
        lane.run().unwrap();
        black_box(lane.output(0).expect("lane output"));
    }

    c.bench_function("C_lane", |b| {
        b.iter(|| {
            lane.run().expect("run");
            black_box(lane.output(0).expect("lane output"));
        });
    });
}

criterion_group!(
    benches,
    bench_c_full,
    bench_c_legacy_full,
    bench_c_prepared,
    bench_c_1thread,
    bench_c_iobinding,
    bench_c_lane
);
criterion_main!(benches);
