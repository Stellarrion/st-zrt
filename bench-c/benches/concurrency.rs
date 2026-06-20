//! Lane-set concurrency benchmark.
//!
//! One shared session, N independent `TensorIoLane` lanes, N scoped threads. Each thread
//! runs its lane repeatedly. This measures the serving architecture we care about: no
//! per-request allocation/copy/rebind in ZRT, and no shared mutable tensor buffers.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions};
use st_zrt_bench_c::models;

const MNIST_IN: [i64; 4] = [1, 1, 28, 28];
const MNIST_OUT: [i64; 2] = [1, 10];
const RUNS_PER_LANE: usize = 128;

fn bench_mnist_lanes(c: &mut Criterion, lane_count: usize) {
    let model = models::ensure_mnist().expect("mnist");
    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(&env, model.to_str().unwrap(), opts).unwrap();
    let mut lanes = sess
        .prepare_tensor_io_lanes::<f32>(&mem, &[&MNIST_IN], &[&MNIST_OUT], lane_count)
        .unwrap();
    for (i, lane) in lanes.iter_mut().enumerate() {
        lane.input_mut(0).expect("lane input").fill(i as f32);
        lane.run().unwrap();
    }

    c.bench_function(&format!("C_lanes_mnist_{lane_count}"), |b| {
        b.iter(|| {
            std::thread::scope(|scope| {
                for lane in lanes.iter_mut() {
                    scope.spawn(move || {
                        for _ in 0..RUNS_PER_LANE {
                            lane.run().unwrap();
                            black_box(lane.output(0).expect("lane output"));
                        }
                    });
                }
            });
        });
    });
}

fn bench_lanes_1(c: &mut Criterion) {
    bench_mnist_lanes(c, 1);
}

fn bench_lanes_2(c: &mut Criterion) {
    bench_mnist_lanes(c, 2);
}

fn bench_lanes_4(c: &mut Criterion) {
    bench_mnist_lanes(c, 4);
}

fn bench_lanes_8(c: &mut Criterion) {
    bench_mnist_lanes(c, 8);
}

criterion_group!(
    benches,
    bench_lanes_1,
    bench_lanes_2,
    bench_lanes_4,
    bench_lanes_8
);
criterion_main!(benches);
