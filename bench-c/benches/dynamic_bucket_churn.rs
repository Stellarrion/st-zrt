//! Dynamic-shape bucket churn tail benchmark.
//!
//! The cached dynamic path is covered by `runtime_shapes.rs`; this harness stresses misses and LRU
//! eviction on CPU by cycling more valid shapes than `max_buckets`. With `--features cuda` and
//! `ST_ZRT_CHURN_CUDA=1`, CUDA graph mode only runs when the shape set fits in the bucket cache:
//! ORT's legacy CUDA EP keeps captured graphs for the session lifetime, so CUDA graph buckets are
//! intentionally no-evict.
//!
//! Run:
//!   cargo bench --bench dynamic_bucket_churn
//!   ST_ZRT_CHURN_CUDA=1 cargo bench --features cuda --bench dynamic_bucket_churn

use st_zrt::{
    DynamicIoOptions, DynamicIoRuntime, Environment, GraphOptimizationLevel, MemoryInfo, Session,
    SessionOptions, ShapeSpec,
};
use std::time::{Duration, Instant};

fn main() {
    let iters = env_usize("ST_ZRT_CHURN_ITERS", 1_000).max(1);
    let max_buckets = env_usize("ST_ZRT_CHURN_MAX_BUCKETS", 4).max(1);
    let shape_count = env_usize_opt("ST_ZRT_CHURN_SHAPES")
        .unwrap_or(max_buckets + 1)
        .max(1);
    let warm_runs = env_usize("ST_ZRT_CHURN_WARM_RUNS", 0);
    let top_n = env_usize("ST_ZRT_CHURN_TOP", 0);
    let device_output = env_bool("ST_ZRT_CHURN_DEVICE_OUTPUT");
    let pinned_output = env_bool("ST_ZRT_CHURN_PINNED_OUTPUT");
    let profile_run = env_bool("ST_ZRT_CHURN_PROFILE");
    let cfg = ChurnConfig {
        iters,
        max_buckets,
        shape_count,
        warm_runs,
        device_output,
        pinned_output,
        profile_run,
    };

    let Some(path) = dynamic_batch_path() else {
        eprintln!("dynamic_bucket_churn: dynamic_batch.onnx not found; skipping");
        return;
    };

    let env = Environment::new().expect("env");
    println!(
        "dynamic_bucket_churn model={} iters={iters} max_buckets={max_buckets} shapes={shape_count} warm_runs={warm_runs} output_mem={} profile={profile_run}",
        path.display(),
        output_mem_label(device_output, pinned_output)
    );

    let cpu = run_cpu(&env, &path, cfg);
    print_churn_stats("cpu/create_run_evict", &cpu, top_n);

    #[cfg(feature = "cuda")]
    if matches!(
        std::env::var("ST_ZRT_CHURN_CUDA").as_deref(),
        Ok("1" | "true" | "yes")
    ) {
        if shape_count > max_buckets {
            println!(
                "cuda_graph/device_input_churn      skipped: shape_count={shape_count} exceeds max_buckets={max_buckets}; cuda_graph buckets are no-evict because ORT keeps captured graphs for the session lifetime"
            );
        } else {
            match run_cuda_graph(&env, &path, cfg) {
                Ok(cuda) => print_churn_stats("cuda_graph/device_input_churn", &cuda, top_n),
                Err(err) => {
                    eprintln!("cuda_graph/device_input_churn      skipped: {err}");
                },
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    if std::env::var("ST_ZRT_CHURN_CUDA").is_ok() {
        eprintln!(
            "dynamic_bucket_churn: ST_ZRT_CHURN_CUDA requested but bench was built without --features cuda"
        );
    }
}

#[derive(Clone, Copy)]
struct ChurnConfig {
    iters: usize,
    max_buckets: usize,
    shape_count: usize,
    warm_runs: usize,
    device_output: bool,
    pinned_output: bool,
    profile_run: bool,
}

fn run_cpu(env: &Environment, path: &std::path::Path, cfg: ChurnConfig) -> ChurnSamples {
    let sess = cpu_session(env, path);
    let mem = MemoryInfo::cpu().expect("cpu memory");
    let out_mem = MemoryInfo::cpu().expect("cpu output");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        mem,
        out_mem,
        1,
        DynamicIoOptions::new(cfg.max_buckets),
    )
    .expect("dynamic runtime");
    warm_runtime(&mut runtime, cfg.shape_count, cfg.warm_runs).expect("warm dynamic buckets");
    measure_churn(
        &mut runtime,
        cfg.iters,
        cfg.shape_count,
        true,
        cfg.profile_run,
    )
}

#[cfg(feature = "cuda")]
fn run_cuda_graph(
    env: &Environment, path: &std::path::Path, cfg: ChurnConfig,
) -> st_zrt::Result<ChurnSamples> {
    use st_zrt::{CudaConfig, CudaStream};
    use std::sync::Arc;

    let stream = Arc::new(CudaStream::new(0)?);
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::graph_replay(0, &stream)?)?;
    let sess = Session::new(env, path.to_str().unwrap(), opts)?;
    let mem = MemoryInfo::cpu()?;
    let out_mem = if cfg.device_output {
        MemoryInfo::cuda(0)?
    } else if cfg.pinned_output {
        MemoryInfo::cuda_pinned(0)?
    } else {
        MemoryInfo::cpu()?
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        mem,
        out_mem,
        1,
        DynamicIoOptions::new(cfg.max_buckets)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)?,
    )?;
    warm_runtime(&mut runtime, cfg.shape_count, cfg.warm_runs)?;
    Ok(measure_churn(
        &mut runtime,
        cfg.iters,
        cfg.shape_count,
        !cfg.device_output,
        cfg.profile_run,
    ))
}

fn warm_runtime(
    runtime: &mut DynamicIoRuntime<f32, f32, 1, 1>, shape_count: usize, warm_runs: usize,
) -> st_zrt::Result<()> {
    if warm_runs == 0 {
        return Ok(());
    }
    let shapes = shape_plan(shape_count);
    let specs = shapes.iter().map(|(input, output)| ShapeSpec {
        input_shapes: [input.as_slice()],
        output_shapes: [output.as_slice()],
    });
    runtime.warm_buckets(specs, warm_runs)?;
    Ok(())
}

fn measure_churn(
    runtime: &mut DynamicIoRuntime<f32, f32, 1, 1>, iters: usize, shape_count: usize,
    inspect_host_output: bool, profile_run: bool,
) -> ChurnSamples {
    let mut samples = Vec::with_capacity(iters);
    for iter in 0..iters {
        let batch_idx = iter % shape_count;
        let input = [1 + batch_idx as i64, 32];
        let output = [1 + batch_idx as i64, 4];
        let fill = 1.0 + (iter as f32 * 0.001);
        let total_start = Instant::now();
        let mut fill_duration = Duration::ZERO;
        let mut run_duration = Duration::ZERO;
        let mut rebind_duration = Duration::ZERO;
        let mut refresh_duration = Duration::ZERO;
        let mut ort_run_duration = Duration::ZERO;
        let mut sync_inputs_duration = Duration::ZERO;
        let mut run_with_binding_duration = Duration::ZERO;
        let mut sync_outputs_duration = Duration::ZERO;
        let mut output_duration = Duration::ZERO;
        runtime
            .run_on([&input], [&output], 0, |lane| {
                let fill_start = Instant::now();
                lane.input_mut_at::<0>()?.fill(fill);
                fill_duration = fill_start.elapsed();

                if profile_run {
                    let timings = lane.run_profiled()?;
                    run_duration = timings.total;
                    rebind_duration = timings.rebind_inputs;
                    refresh_duration = timings.device_input_refresh;
                    ort_run_duration = timings.ort_run;
                    sync_inputs_duration = timings.bound_input_sync;
                    run_with_binding_duration = timings.run_with_binding;
                    sync_outputs_duration = timings.bound_output_sync;
                } else {
                    let run_start = Instant::now();
                    lane.run()?;
                    run_duration = run_start.elapsed();
                }

                let output_start = Instant::now();
                if inspect_host_output {
                    std::hint::black_box(lane.output_at::<0>()?.as_ptr());
                } else {
                    std::hint::black_box(lane.output_buffer(0)?.engine_data_ptr()?);
                }
                output_duration = output_start.elapsed();
                Ok(())
            })
            .expect("dynamic churn run");
        let total = total_start.elapsed();
        let accounted = fill_duration + run_duration + output_duration;
        let runtime_overhead = total.checked_sub(accounted).unwrap_or(Duration::ZERO);
        samples.push(ChurnSample {
            iter,
            shape_idx: batch_idx,
            total,
            fill: fill_duration,
            run: run_duration,
            rebind: rebind_duration,
            refresh: refresh_duration,
            ort_run: ort_run_duration,
            sync_inputs: sync_inputs_duration,
            run_with_binding: run_with_binding_duration,
            sync_outputs: sync_outputs_duration,
            output: output_duration,
            runtime_overhead,
        });
    }
    ChurnSamples {
        samples,
        profiled: profile_run,
    }
}

fn shape_plan(shape_count: usize) -> Vec<([i64; 2], [i64; 2])> {
    (0..shape_count)
        .map(|i| {
            let batch = 1 + i as i64;
            ([batch, 32], [batch, 4])
        })
        .collect()
}

fn cpu_session(env: &Environment, path: &std::path::Path) -> Session {
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    Session::new(env, path.to_str().unwrap(), opts).expect("cpu session")
}

fn dynamic_batch_path() -> Option<std::path::PathBuf> {
    [
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../st-zrt/tests/fixtures/dynamic_batch.onnx"),
        std::path::PathBuf::from("../st-zrt/tests/fixtures/dynamic_batch.onnx"),
        std::path::PathBuf::from("st-zrt/tests/fixtures/dynamic_batch.onnx"),
    ]
    .into_iter()
    .find(|c| c.exists())
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

#[derive(Debug)]
struct ChurnSamples {
    samples: Vec<ChurnSample>,
    profiled: bool,
}

#[derive(Debug)]
struct ChurnSample {
    iter: usize,
    shape_idx: usize,
    total: Duration,
    fill: Duration,
    run: Duration,
    rebind: Duration,
    refresh: Duration,
    ort_run: Duration,
    sync_inputs: Duration,
    run_with_binding: Duration,
    sync_outputs: Duration,
    output: Duration,
    runtime_overhead: Duration,
}

fn print_churn_stats(label: &str, samples: &ChurnSamples, top_n: usize) {
    print_stats(
        label,
        &samples
            .samples
            .iter()
            .map(|sample| sample.total)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/fill"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.fill)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.run)
            .collect::<Vec<_>>(),
    );
    if !samples.profiled {
        print_stats(
            &format!("{label}/output"),
            &samples
                .samples
                .iter()
                .map(|sample| sample.output)
                .collect::<Vec<_>>(),
        );
        print_stats(
            &format!("{label}/runtime_overhead"),
            &samples
                .samples
                .iter()
                .map(|sample| sample.runtime_overhead)
                .collect::<Vec<_>>(),
        );
        print_slowest_samples(label, samples, top_n);
        return;
    }
    print_stats(
        &format!("{label}/run/rebind"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.rebind)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run/refresh"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.refresh)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run/ort"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.ort_run)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run/sync_inputs"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.sync_inputs)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run/run_binding"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.run_with_binding)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/run/sync_outputs"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.sync_outputs)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/output"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.output)
            .collect::<Vec<_>>(),
    );
    print_stats(
        &format!("{label}/runtime_overhead"),
        &samples
            .samples
            .iter()
            .map(|sample| sample.runtime_overhead)
            .collect::<Vec<_>>(),
    );
    print_slowest_samples(label, samples, top_n);
}

fn print_slowest_samples(label: &str, samples: &ChurnSamples, top_n: usize) {
    if top_n == 0 {
        return;
    }
    let mut slowest = samples.samples.iter().collect::<Vec<_>>();
    slowest.sort_unstable_by_key(|sample| std::cmp::Reverse(sample.total));
    println!("{label:34} slowest samples:");
    if !samples.profiled {
        println!(
            "{:>8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "iter", "shape", "total us", "fill us", "run us", "out us", "rt+lookup"
        );
        for sample in slowest.into_iter().take(top_n) {
            println!(
                "{:>8} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>10.3}",
                sample.iter,
                sample.shape_idx,
                us(sample.total),
                us(sample.fill),
                us(sample.run),
                us(sample.output),
                us(sample.runtime_overhead),
            );
        }
        return;
    }
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10}",
        "iter",
        "shape",
        "total us",
        "fill us",
        "run us",
        "refresh",
        "in_sync",
        "binding",
        "out_sync",
        "out us",
        "rt+lookup"
    );
    for sample in slowest.into_iter().take(top_n) {
        println!(
            "{:>8} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>8.3} {:>10.3}",
            sample.iter,
            sample.shape_idx,
            us(sample.total),
            us(sample.fill),
            us(sample.run),
            us(sample.refresh),
            us(sample.sync_inputs),
            us(sample.run_with_binding),
            us(sample.sync_outputs),
            us(sample.output),
            us(sample.runtime_overhead),
        );
    }
}

fn print_stats(label: &str, samples: &[Duration]) {
    let s = stats(samples.to_vec());
    println!(
        "{label:34} min={:>8.3}us p50={:>8.3}us p90={:>8.3}us p99={:>8.3}us p999={:>8.3}us max={:>8.3}us",
        us(s.min),
        us(s.p50),
        us(s.p90),
        us(s.p99),
        us(s.p999),
        us(s.max),
    );
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

fn percentile(sorted: &[Duration], permille: usize) -> Duration {
    let idx = ((sorted.len() - 1) * permille).div_ceil(1000);
    sorted[idx]
}

fn us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn env_usize(name: &str, default: usize) -> usize {
    env_usize_opt(name).unwrap_or(default)
}

fn env_usize_opt(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn env_bool(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1" | "true" | "yes" | "on")
    )
}

fn output_mem_label(device_output: bool, pinned_output: bool) -> &'static str {
    if device_output {
        "cuda"
    } else if pinned_output {
        "cuda_pinned"
    } else {
        "cpu"
    }
}
