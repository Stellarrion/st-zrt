//! Static/dynamic runtime dispatch overhead benches.
//!
//! These use MNIST because the model is tiny enough that wrapper dispatch, shape lookup, and
//! binding choices are visible. The cached dynamic path should be close to static; the cold path
//! intentionally includes bucket allocation and binding setup.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{
    DynamicIoOptions, DynamicIoRuntime, Environment, GraphOptimizationLevel, MemoryInfo,
    Runtime, Session, SessionOptions, StaticIoRuntime,
};
use st_zrt_bench_c::models;
use std::sync::Arc;

const INPUT: [i64; 4] = [1, 1, 28, 28];
const OUTPUT: [i64; 2] = [1, 10];

fn session(env: &Environment) -> (Session, MemoryInfo) {
    let model = models::ensure_mnist().expect("mnist");
    let mem = MemoryInfo::cpu().expect("cpu memory");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = Session::new(env, model.to_str().unwrap(), opts).expect("session");
    (sess, mem)
}

fn bench_homogeneous_runtime_direct(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime =
        Runtime::<f32>::shared_session(Arc::new(sess), &mem, &[&INPUT], &[&OUTPUT], 1)
            .expect("runtime");
    runtime.prime(32).expect("prime");

    c.bench_function("runtime/homogeneous_direct", |b| {
        b.iter(|| {
            let lane = runtime.lane_mut(0).expect("lane");
            lane.run().expect("run");
            black_box(lane.output(0).expect("output"));
        });
    });
}

fn bench_static_io_direct(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime =
        StaticIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), &mem, [&INPUT], [&OUTPUT], 1)
            .expect("static runtime");
    runtime.prime(32).expect("prime");

    c.bench_function("runtime/static_io_direct", |b| {
        b.iter(|| {
            let lane = runtime.lane_mut(0).expect("lane");
            lane.run().expect("run");
            black_box(lane.output_at::<0>().expect("output"));
        });
    });
}

fn bench_static_io_run_on(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime =
        StaticIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), &mem, [&INPUT], [&OUTPUT], 1)
            .expect("static runtime");
    runtime.prime(32).expect("prime");

    c.bench_function("runtime/static_io_run_on", |b| {
        b.iter(|| {
            runtime
                .run_on(0, |lane| {
                    lane.run()?;
                    black_box(lane.output_at::<0>()?);
                    Ok(())
                })
                .expect("run_on");
        });
    });
}

fn bench_static_io_dispatch_only(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        Arc::new(sess),
        &mem,
        [&INPUT],
        [&OUTPUT],
        1,
    )
    .expect("static runtime");

    c.bench_function("runtime/static_io_dispatch_only", |b| {
        b.iter(|| {
            runtime
                .run_on(0, |lane| {
                    black_box(lane.output_at::<0>()?.as_ptr());
                    Ok(())
                })
                .expect("dispatch");
        });
    });
}

fn bench_dynamic_cached_run_on(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime =
        DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), mem, 1)
            .expect("dynamic runtime");
    runtime
        .prime_bucket([&INPUT], [&OUTPUT], 32)
        .expect("prime bucket");

    c.bench_function("runtime/dynamic_cached_run_on", |b| {
        b.iter(|| {
            runtime
                .run_on([&INPUT], [&OUTPUT], 0, |lane| {
                    lane.run()?;
                    black_box(lane.output_at::<0>()?);
                    Ok(())
                })
                .expect("dynamic run_on");
        });
    });
}

fn bench_dynamic_cached_dispatch_only(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), mem, 1)
        .expect("dynamic runtime");
    runtime
        .prime_bucket([&INPUT], [&OUTPUT], 1)
        .expect("prime bucket");

    c.bench_function("runtime/dynamic_cached_dispatch_only", |b| {
        b.iter(|| {
            runtime
                .run_on([&INPUT], [&OUTPUT], 0, |lane| {
                    black_box(lane.output_at::<0>()?.as_ptr());
                    Ok(())
                })
                .expect("dynamic dispatch");
        });
    });
}

fn bench_dynamic_cached_bucket_direct(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime =
        DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), mem, 1)
            .expect("dynamic runtime");
    runtime
        .prime_bucket([&INPUT], [&OUTPUT], 32)
        .expect("prime bucket");

    c.bench_function("runtime/dynamic_cached_bucket_direct", |b| {
        b.iter(|| {
            let bucket = &mut runtime.buckets_mut()[0];
            let lane = bucket.lane_mut(0).expect("lane");
            lane.run().expect("run");
            black_box(lane.output_at::<0>().expect("output"));
        });
    });
}

fn bench_dynamic_cached_bucket_dispatch_only(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(Arc::new(sess), mem, 1)
        .expect("dynamic runtime");
    runtime
        .prime_bucket([&INPUT], [&OUTPUT], 1)
        .expect("prime bucket");

    c.bench_function("runtime/dynamic_cached_bucket_dispatch_only", |b| {
        b.iter(|| {
            let bucket = &mut runtime.buckets_mut()[0];
            let lane = bucket.lane_mut(0).expect("lane");
            black_box(lane.output_at::<0>().expect("output").as_ptr());
        });
    });
}

fn bench_dynamic_lookup_16_buckets(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let output_mem = mem.try_clone_descriptor().expect("output memory");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        Arc::new(sess),
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(16),
    )
    .expect("dynamic runtime");
    let shapes = (0..16)
        .map(|i| [1, 1, 28, 13 + i])
        .collect::<Vec<[i64; 4]>>();
    for shape in &shapes {
        runtime
            .get_or_create_bucket([shape.as_slice()], [&OUTPUT])
            .expect("bucket");
    }
    let last = shapes.last().expect("last shape");

    c.bench_function("runtime/dynamic_lookup_16_buckets", |b| {
        b.iter(|| {
            let bucket = runtime
                .bucket_mut([last.as_slice()], [&OUTPUT])
                .expect("bucket");
            black_box(bucket.key().input_shape(0).expect("shape").as_ptr());
        });
    });
}

fn bench_dynamic_cold_create_and_run(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let output_mem = mem.try_clone_descriptor().expect("output memory");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        Arc::new(sess),
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(1),
    )
    .expect("dynamic runtime");

    c.bench_function("runtime/dynamic_cold_create_run", |b| {
        b.iter(|| {
            runtime.clear_buckets();
            runtime
                .run_on([&INPUT], [&OUTPUT], 0, |lane| {
                    lane.run()?;
                    black_box(lane.output_at::<0>()?);
                    Ok(())
                })
                .expect("dynamic cold run");
        });
    });
}

fn bench_dynamic_cold_create_only(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let output_mem = mem.try_clone_descriptor().expect("output memory");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        Arc::new(sess),
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(1),
    )
    .expect("dynamic runtime");

    c.bench_function("runtime/dynamic_cold_create_only", |b| {
        b.iter(|| {
            runtime.clear_buckets();
            let bucket = runtime
                .get_or_create_bucket([&INPUT], [&OUTPUT])
                .expect("bucket");
            black_box(bucket.lane(0).expect("lane").outputs()[0].as_slice().as_ptr());
        });
    });
}

criterion_group!(
    benches,
    bench_homogeneous_runtime_direct,
    bench_static_io_direct,
    bench_static_io_run_on,
    bench_static_io_dispatch_only,
    bench_dynamic_cached_run_on,
    bench_dynamic_cached_dispatch_only,
    bench_dynamic_cached_bucket_direct,
    bench_dynamic_cached_bucket_dispatch_only,
    bench_dynamic_lookup_16_buckets,
    bench_dynamic_cold_create_and_run,
    bench_dynamic_cold_create_only
);
criterion_main!(benches);
