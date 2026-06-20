use std::error::Error;
use std::fs;
use std::time::{Duration, Instant};

use st_zrt::{
    Environment, GraphOptimizationLevel, LaneBufferPolicy, MemoryInfo, Session, SessionOptions,
};
use st_zrt_bench_c::models;

const MNIST_IN: [i64; 4] = [1, 1, 28, 28];
const MNIST_OUT: [i64; 2] = [1, 10];
const RESNET_IN: [i64; 4] = [1, 3, 224, 224];
const RESNET_OUT: [i64; 2] = [1, 1000];

#[derive(Clone, Copy)]
struct Rss {
    rss_kb: u64,
    hwm_kb: u64,
}

fn rss() -> std::io::Result<Rss> {
    let status = fs::read_to_string("/proc/self/status")?;
    let mut rss_kb = 0;
    let mut hwm_kb = 0;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            rss_kb = parse_status_kb(rest);
        } else if let Some(rest) = line.strip_prefix("VmHWM:") {
            hwm_kb = parse_status_kb(rest);
        }
    }
    Ok(Rss { rss_kb, hwm_kb })
}

fn parse_status_kb(rest: &str) -> u64 {
    rest.split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn checksum(values: &[f32]) -> f64 {
    let step = (values.len() / 32).max(1);
    values.iter().step_by(step).map(|v| *v as f64).sum::<f64>()
}

fn image_input() -> Vec<f32> {
    (0..(3 * 224 * 224))
        .map(|i| ((i % 251) as f32 - 125.0) / 128.0)
        .collect()
}

fn session(env: &Environment, path: &str) -> st_zrt::Result<Session> {
    let mut opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    if std::env::var("ZRT_DISABLE_ARENA").ok().as_deref() == Some("1") {
        opts = opts.disable_cpu_mem_arena();
    }
    if std::env::var("ZRT_DISABLE_MEM_PATTERN").ok().as_deref() == Some("1") {
        opts = opts.disable_mem_pattern();
    }
    Session::new(env, path, opts)
}

fn lane_policy() -> LaneBufferPolicy {
    let alignment = std::env::var("ZRT_LANE_ALIGNMENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    match std::env::var("ZRT_LANE_POLICY")
        .unwrap_or_else(|_| "auto".to_string())
        .as_str()
    {
        "vec" => LaneBufferPolicy::Vec,
        "prefaulted" => LaneBufferPolicy::Prefaulted,
        "aligned" => LaneBufferPolicy::Aligned { alignment },
        "aligned-prefaulted" => LaneBufferPolicy::AlignedPrefaulted { alignment },
        "hugepage" => LaneBufferPolicy::HugePage,
        "hugepage-prefaulted" => LaneBufferPolicy::HugePagePrefaulted,
        "aligned-hugepage-prefaulted" => LaneBufferPolicy::AlignedHugePagePrefaulted { alignment },
        "aligned-mlocked" => LaneBufferPolicy::AlignedMlocked { alignment },
        "aligned-mlocked-prefaulted" => LaneBufferPolicy::AlignedMlockedPrefaulted { alignment },
        "hugepage-mlocked" => LaneBufferPolicy::HugePageMlocked,
        "hugepage-mlocked-prefaulted" => LaneBufferPolicy::HugePageMlockedPrefaulted,
        "aligned-hugepage-mlocked-prefaulted" => {
            LaneBufferPolicy::AlignedHugePageMlockedPrefaulted { alignment }
        }
        _ => LaneBufferPolicy::Auto,
    }
}

fn timed_load<F, T>(load: F) -> Result<(T, Duration), Box<dyn Error>>
where
    F: FnOnce() -> Result<T, Box<dyn Error>>,
{
    let start = Instant::now();
    let value = load()?;
    Ok((value, start.elapsed()))
}

fn bench_mnist(warmups: usize, iters: usize, mode: &str) -> Result<Probe, Box<dyn Error>> {
    let start_rss = rss()?;
    let ((_env, mem, sess), load_time) = timed_load(|| {
        let model = models::ensure_mnist()?;
        let env = Environment::new()?;
        let mem = MemoryInfo::cpu()?;
        let sess = session(&env, model.to_str().unwrap())?;
        Ok((env, mem, sess))
    })?;
    let loaded_rss = rss()?;

    let (mode, warmup_time, run_time, sum) = if mode == "allocated-output" {
        let mut lane = sess.prepare_allocated_output_tensor_io_lane::<f32>(
            &mem,
            &mem,
            &[&MNIST_IN],
            &[&MNIST_OUT],
        )?;
        lane.input_mut(0).expect("mnist input").fill(0.0);
        let (warmup_time, run_time, sum) = run_allocated_lane(&mut lane, warmups, iters)?;
        ("allocated_output", warmup_time, run_time, sum)
    } else if mode == "allocated-io" {
        let mut lane =
            sess.prepare_allocated_tensor_io_lane::<f32>(&mem, &mem, &[&MNIST_IN], &[&MNIST_OUT])?;
        lane.input_mut(0).expect("mnist input").fill(0.0);
        let (warmup_time, run_time, sum) = run_allocated_io_lane(&mut lane, warmups, iters)?;
        ("allocated_io", warmup_time, run_time, sum)
    } else if mode == "device-output" {
        let mut lane =
            sess.prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&MNIST_IN])?;
        lane.input_mut(0).expect("mnist input").fill(0.0);
        let (warmup_time, run_time, sum) = run_device_output_lane(&mut lane, warmups, iters)?;
        ("device_output", warmup_time, run_time, sum)
    } else {
        let mut lane = sess.prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&MNIST_IN],
            &[&MNIST_OUT],
            lane_policy(),
        )?;
        lane.input_mut(0).expect("mnist input").fill(0.0);
        let (warmup_time, run_time, sum) = run_lane(&mut lane, warmups, iters)?;
        ("lane_auto", warmup_time, run_time, sum)
    };
    Ok(Probe::new(
        "zrt",
        "mnist",
        mode,
        warmups,
        iters,
        load_time,
        warmup_time,
        run_time,
        start_rss,
        loaded_rss,
        sum,
    ))
}

fn bench_relay(
    label: &str,
    warmups: usize,
    iters: usize,
    mode: &str,
) -> Result<Probe, Box<dyn Error>> {
    let n = match label {
        "relay4m" => 1usize << 20,
        "relay16m" => 1usize << 22,
        _ => return Err(format!("unsupported relay model: {label}").into()),
    };
    let relay_label = label.trim_start_matches("relay");
    let shape = [1, n as i64];
    let start_rss = rss()?;
    let ((_env, mem, sess), load_time) = timed_load(|| {
        let model = models::ensure_relay(relay_label)?;
        let env = Environment::new()?;
        let mem = MemoryInfo::cpu()?;
        let sess = session(&env, model.to_str().unwrap())?;
        Ok((env, mem, sess))
    })?;
    let loaded_rss = rss()?;

    let (mode, warmup_time, run_time, sum) = if mode == "allocated-output" {
        let mut lane =
            sess.prepare_allocated_output_tensor_io_lane::<f32>(&mem, &mem, &[&shape], &[&shape])?;
        lane.input_mut(0).expect("relay input").fill(3.0);
        let (warmup_time, run_time, sum) = run_allocated_lane(&mut lane, warmups, iters)?;
        ("allocated_output", warmup_time, run_time, sum)
    } else if mode == "allocated-io" {
        let mut lane =
            sess.prepare_allocated_tensor_io_lane::<f32>(&mem, &mem, &[&shape], &[&shape])?;
        lane.input_mut(0).expect("relay input").fill(3.0);
        let (warmup_time, run_time, sum) = run_allocated_io_lane(&mut lane, warmups, iters)?;
        ("allocated_io", warmup_time, run_time, sum)
    } else if mode == "device-output" {
        let mut lane = sess.prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&shape])?;
        lane.input_mut(0).expect("relay input").fill(3.0);
        let (warmup_time, run_time, sum) = run_device_output_lane(&mut lane, warmups, iters)?;
        ("device_output", warmup_time, run_time, sum)
    } else {
        let mut lane = sess.prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&shape],
            &[&shape],
            lane_policy(),
        )?;
        lane.input_mut(0).expect("relay input").fill(3.0);
        let (warmup_time, run_time, sum) = run_lane(&mut lane, warmups, iters)?;
        ("lane_auto", warmup_time, run_time, sum)
    };
    Ok(Probe::new(
        "zrt",
        label,
        mode,
        warmups,
        iters,
        load_time,
        warmup_time,
        run_time,
        start_rss,
        loaded_rss,
        sum,
    ))
}

fn bench_resnet(warmups: usize, iters: usize, mode: &str) -> Result<Probe, Box<dyn Error>> {
    let start_rss = rss()?;
    let ((_env, mem, sess), load_time) = timed_load(|| {
        let model = models::ensure_hf_resnet50()?;
        let env = Environment::new()?;
        let mem = MemoryInfo::cpu()?;
        let sess = session(&env, model.to_str().unwrap())?;
        Ok((env, mem, sess))
    })?;
    let loaded_rss = rss()?;

    let input = image_input();
    let (mode, warmup_time, run_time, sum) = if mode == "allocated-output" {
        let mut lane = sess.prepare_allocated_output_tensor_io_lane::<f32>(
            &mem,
            &mem,
            &[&RESNET_IN],
            &[&RESNET_OUT],
        )?;
        lane.input_mut(0)
            .expect("resnet input")
            .copy_from_slice(&input);
        let (warmup_time, run_time, sum) = run_allocated_lane(&mut lane, warmups, iters)?;
        ("allocated_output", warmup_time, run_time, sum)
    } else if mode == "allocated-io" {
        let mut lane = sess.prepare_allocated_tensor_io_lane::<f32>(
            &mem,
            &mem,
            &[&RESNET_IN],
            &[&RESNET_OUT],
        )?;
        lane.input_mut(0)
            .expect("resnet input")
            .copy_from_slice(&input);
        let (warmup_time, run_time, sum) = run_allocated_io_lane(&mut lane, warmups, iters)?;
        ("allocated_io", warmup_time, run_time, sum)
    } else if mode == "device-output" {
        let mut lane =
            sess.prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&RESNET_IN])?;
        lane.input_mut(0)
            .expect("resnet input")
            .copy_from_slice(&input);
        let (warmup_time, run_time, sum) = run_device_output_lane(&mut lane, warmups, iters)?;
        ("device_output", warmup_time, run_time, sum)
    } else {
        let mut lane = sess.prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&RESNET_IN],
            &[&RESNET_OUT],
            lane_policy(),
        )?;
        lane.input_mut(0)
            .expect("resnet input")
            .copy_from_slice(&input);
        let (warmup_time, run_time, sum) = run_lane(&mut lane, warmups, iters)?;
        ("lane_auto", warmup_time, run_time, sum)
    };
    Ok(Probe::new(
        "zrt",
        "hf_resnet50",
        mode,
        warmups,
        iters,
        load_time,
        warmup_time,
        run_time,
        start_rss,
        loaded_rss,
        sum,
    ))
}

fn run_lane(
    lane: &mut st_zrt::TensorIoLane<f32>,
    warmups: usize,
    iters: usize,
) -> st_zrt::Result<(Duration, Duration, f64)> {
    let warmup_start = Instant::now();
    for _ in 0..warmups {
        lane.run()?;
        std::hint::black_box(lane.output(0).expect("warmup output"));
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        lane.run()?;
        let out = lane.output(0).expect("run output");
        sum += checksum(out);
        std::hint::black_box(out);
    }
    Ok((warmup_time, run_start.elapsed(), sum))
}

fn run_allocated_lane(
    lane: &mut st_zrt::AllocatedOutputTensorIoLane<f32>,
    warmups: usize,
    iters: usize,
) -> st_zrt::Result<(Duration, Duration, f64)> {
    let warmup_start = Instant::now();
    for _ in 0..warmups {
        lane.run()?;
        std::hint::black_box(lane.output(0).expect("warmup output"));
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        lane.run()?;
        let out = lane.output(0).expect("run output");
        sum += checksum(out);
        std::hint::black_box(out);
    }
    Ok((warmup_time, run_start.elapsed(), sum))
}

fn run_allocated_io_lane(
    lane: &mut st_zrt::AllocatedTensorIoLane<f32>,
    warmups: usize,
    iters: usize,
) -> st_zrt::Result<(Duration, Duration, f64)> {
    let warmup_start = Instant::now();
    for _ in 0..warmups {
        lane.run()?;
        std::hint::black_box(lane.output(0).expect("warmup output"));
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        lane.run()?;
        let out = lane.output(0).expect("run output");
        sum += checksum(out);
        std::hint::black_box(out);
    }
    Ok((warmup_time, run_start.elapsed(), sum))
}

fn run_device_output_lane(
    lane: &mut st_zrt::DeviceOutputTensorIoLane<f32>,
    warmups: usize,
    iters: usize,
) -> st_zrt::Result<(Duration, Duration, f64)> {
    let warmup_start = Instant::now();
    for _ in 0..warmups {
        lane.run()?;
        std::hint::black_box(lane.output(0).expect("warmup output"));
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        lane.run()?;
        let out = lane.output(0).expect("run output").as_slice::<f32>()?;
        sum += checksum(out);
        std::hint::black_box(out);
    }
    Ok((warmup_time, run_start.elapsed(), sum))
}

struct Probe {
    runtime: &'static str,
    model: String,
    mode: &'static str,
    warmups: usize,
    iters: usize,
    load_ms: f64,
    warmup_ms: f64,
    avg_us: f64,
    rss_start_kb: u64,
    rss_loaded_kb: u64,
    rss_done_kb: u64,
    hwm_kb: u64,
    checksum: f64,
}

impl Probe {
    #[allow(clippy::too_many_arguments)]
    fn new(
        runtime: &'static str,
        model: &str,
        mode: &'static str,
        warmups: usize,
        iters: usize,
        load_time: Duration,
        warmup_time: Duration,
        run_time: Duration,
        start_rss: Rss,
        loaded_rss: Rss,
        checksum: f64,
    ) -> Self {
        let done = rss().unwrap_or(loaded_rss);
        Self {
            runtime,
            model: model.to_string(),
            mode,
            warmups,
            iters,
            load_ms: load_time.as_secs_f64() * 1_000.0,
            warmup_ms: warmup_time.as_secs_f64() * 1_000.0,
            avg_us: run_time.as_secs_f64() * 1_000_000.0 / iters as f64,
            rss_start_kb: start_rss.rss_kb,
            rss_loaded_kb: loaded_rss.rss_kb,
            rss_done_kb: done.rss_kb,
            hwm_kb: done.hwm_kb.max(loaded_rss.hwm_kb).max(start_rss.hwm_kb),
            checksum,
        }
    }

    fn print(&self) {
        println!(
            "{},{},{},{},{},{:.3},{:.3},{:.3},{},{},{},{},{:.6}",
            self.runtime,
            self.model,
            self.mode,
            self.warmups,
            self.iters,
            self.load_ms,
            self.warmup_ms,
            self.avg_us,
            self.rss_start_kb,
            self.rss_loaded_kb,
            self.rss_done_kb,
            self.hwm_kb,
            self.checksum
        );
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "mnist".to_string());
    let iters = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        std::env::var("ZRT_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30)
    });
    let mode = args.next().unwrap_or_else(|| "lane".to_string());
    let warmups = std::env::var("ZRT_BENCH_WARMUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let probe = match model.as_str() {
        "mnist" => bench_mnist(warmups, iters, &mode)?,
        "relay4m" | "relay16m" => bench_relay(&model, warmups, iters, &mode)?,
        "hf_resnet50" => bench_resnet(warmups, iters, &mode)?,
        _ => {
            return Err(
                "usage: cargo run --release --example mem_probe -- [mnist|relay4m|relay16m|hf_resnet50] [iters] [lane|allocated-output|allocated-io|device-output]"
                    .into(),
            )
        }
    };
    probe.print();
    Ok(())
}
