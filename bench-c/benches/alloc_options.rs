//! Arena/mem-pattern matrix for the lane-local zero-copy path.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions};
use st_zrt_bench_c::models;

fn opts(disable_arena: bool, disable_mem_pattern: bool) -> SessionOptions {
    let mut opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    if disable_arena {
        opts = opts.disable_cpu_mem_arena();
    }
    if disable_mem_pattern {
        opts = opts.disable_mem_pattern();
    }
    opts
}

fn bench_lane(
    c: &mut Criterion,
    name: &str,
    model_path: &str,
    input_shape: &[i64],
    output_shape: &[i64],
    opts: SessionOptions,
) {
    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let sess = st_zrt::Session::new(&env, model_path, opts).unwrap();
    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[input_shape], &[output_shape])
        .unwrap();
    lane.input_mut(0).expect("lane input").fill(3.0);

    for _ in 0..32 {
        lane.run().unwrap();
        black_box(lane.output(0).expect("lane output"));
    }

    c.bench_function(name, |b| {
        b.iter(|| {
            lane.run().unwrap();
            black_box(lane.output(0).expect("lane output"));
        });
    });
}

fn bench_alloc_options(c: &mut Criterion) {
    let mnist = models::ensure_mnist().expect("mnist");
    let mnist = mnist.to_str().unwrap();
    let relay = models::ensure_relay("4m").expect("relay 4m");
    let relay = relay.to_str().unwrap();

    let cases = [
        ("default", false, false),
        ("noarena", true, false),
        ("nomempattern", false, true),
        ("noarena_nomempattern", true, true),
    ];

    for (suffix, no_arena, no_mem_pattern) in cases {
        bench_lane(
            c,
            &format!("C_lane_mnist_{suffix}"),
            mnist,
            &[1, 1, 28, 28],
            &[1, 10],
            opts(no_arena, no_mem_pattern),
        );
    }

    for (suffix, no_arena, no_mem_pattern) in cases {
        bench_lane(
            c,
            &format!("C_lane_relay4m_{suffix}"),
            relay,
            &[1, 1 << 20],
            &[1, 1 << 20],
            opts(no_arena, no_mem_pattern),
        );
    }
}

criterion_group!(benches, bench_alloc_options);
criterion_main!(benches);
