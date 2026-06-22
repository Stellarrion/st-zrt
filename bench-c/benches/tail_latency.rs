//! Tail-latency probe for the ZRT serving hot path.
//!
//! Criterion is still the primary regression tool. This harness records per-iteration elapsed
//! time and reports min/p50/p90/p99/p999/max so tail shifts are visible.
use std::sync::Arc;
use std::time::{Duration, Instant};

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, Session, SessionOptions, StaticIoRuntime,
};
use st_zrt_bench_c::models;

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

#[derive(Debug)]
struct TailStats {
    min: Duration,
    p50: Duration,
    p90: Duration,
    p99: Duration,
    p999: Duration,
    max: Duration,
}

fn percentile(sorted: &[Duration], permille: usize) -> Duration {
    let len = sorted.len();
    let idx = ((len - 1) * permille).div_ceil(1000);
    sorted[idx]
}

fn stats(mut samples: Vec<Duration>) -> TailStats {
    samples.sort_unstable();
    TailStats {
        min: samples[0],
        p50: percentile(&samples, 500),
        p90: percentile(&samples, 900),
        p99: percentile(&samples, 990),
        p999: percentile(&samples, 999),
        max: samples[samples.len() - 1],
    }
}

fn measure(label: &str, iters: usize, mut f: impl FnMut()) {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed());
    }
    let s = stats(samples);
    println!(
        "{label:34} min={:>8.3}us p50={:>8.3}us p90={:>8.3}us p99={:>8.3}us p999={:>8.3}us max={:>8.3}us",
        s.min.as_secs_f64() * 1e6,
        s.p50.as_secs_f64() * 1e6,
        s.p90.as_secs_f64() * 1e6,
        s.p99.as_secs_f64() * 1e6,
        s.p999.as_secs_f64() * 1e6,
        s.max.as_secs_f64() * 1e6,
    );
}

fn main() {
    let iters = std::env::var("ST_ZRT_TAIL_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10_000)
        .max(1);

    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let mut bind_once = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        Arc::new(sess),
        &mem,
        [&INPUT],
        [&OUTPUT],
        1,
    )
    .expect("static runtime");
    bind_once.prime(64).expect("prime");

    let (sess, mem) = session(&env);
    let mut rebind = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        Arc::new(sess),
        &mem,
        [&INPUT],
        [&OUTPUT],
        1,
    )
    .expect("static runtime");
    rebind.set_rebind_inputs_each_run(true);
    rebind.prime(64).expect("prime");

    println!("tail_latency iters={iters}");
    measure("static_io_bind_once", iters, || {
        let lane = bind_once.lane_mut(0).expect("lane");
        lane.run().expect("run");
        std::hint::black_box(lane.output_at::<0>().expect("output"));
    });
    measure("static_io_bind_once_unsync", iters, || {
        let lane = bind_once.lane_mut(0).expect("lane");
        lane.run_unsynchronized().expect("run");
        std::hint::black_box(lane.output_at::<0>().expect("output"));
    });
    measure("static_io_rebind_each_run", iters, || {
        let lane = rebind.lane_mut(0).expect("lane");
        lane.run().expect("run");
        std::hint::black_box(lane.output_at::<0>().expect("output"));
    });
    measure("static_io_rebind_unsync", iters, || {
        let lane = rebind.lane_mut(0).expect("lane");
        lane.run_unsynchronized().expect("run");
        std::hint::black_box(lane.output_at::<0>().expect("output"));
    });
}
