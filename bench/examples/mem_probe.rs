use std::error::Error;
use std::fs;
use std::time::{Duration, Instant};

use ort::memory::MemoryInfo;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::Tensor;
use st_zrt_bench::models;

const MNIST_IN: [i64; 4] = [1, 1, 28, 28];
const RESNET_IN: [i64; 4] = [1, 3, 224, 224];

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

fn session(path: &str) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(1)?
        .commit_from_file(path)
}

fn timed_load<F, T>(load: F) -> Result<(T, Duration), Box<dyn Error>>
where
    F: FnOnce() -> Result<T, Box<dyn Error>>,
{
    let start = Instant::now();
    let value = load()?;
    Ok((value, start.elapsed()))
}

fn bind(session: &Session, input: &Tensor<f32>) -> ort::Result<IoBinding> {
    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();
    let mut binding = session.create_binding()?;
    binding.bind_input(in_name, input)?;
    binding.bind_output_to_device(out_name, &MemoryInfo::default())?;
    Ok(binding)
}

fn bench_mnist(warmups: usize, iters: usize) -> Result<Probe, Box<dyn Error>> {
    let start_rss = rss()?;
    let ((mut session, input, binding), load_time) = timed_load(|| {
        let model = models::ensure_mnist()?;
        let session = session(model.to_str().unwrap())?;
        let input = Tensor::<f32>::from_array((MNIST_IN.to_vec(), vec![0.0; 28 * 28]))?;
        let binding = bind(&session, &input)?;
        Ok((session, input, binding))
    })?;
    let loaded_rss = rss()?;
    let (warmup_time, run_time, sum) =
        run_binding(&mut session, &binding, warmups, iters, "mnist")?;
    std::hint::black_box(&input);
    Ok(Probe::new(
        "ort",
        "mnist",
        "iobinding",
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

fn bench_relay(label: &str, warmups: usize, iters: usize) -> Result<Probe, Box<dyn Error>> {
    let n = match label {
        "relay4m" => 1usize << 20,
        "relay16m" => 1usize << 22,
        _ => return Err(format!("unsupported relay model: {label}").into()),
    };
    let relay_label = label.trim_start_matches("relay");
    let shape = vec![1, n as i64];
    let start_rss = rss()?;
    let ((mut session, input, binding), load_time) = timed_load(|| {
        let model = models::ensure_relay(relay_label)?;
        let session = session(model.to_str().unwrap())?;
        let input = Tensor::<f32>::from_array((shape, vec![3.0; n]))?;
        let binding = bind(&session, &input)?;
        Ok((session, input, binding))
    })?;
    let loaded_rss = rss()?;
    let (warmup_time, run_time, sum) = run_binding(&mut session, &binding, warmups, iters, label)?;
    std::hint::black_box(&input);
    Ok(Probe::new(
        "ort",
        label,
        "iobinding",
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

fn bench_resnet(warmups: usize, iters: usize) -> Result<Probe, Box<dyn Error>> {
    let start_rss = rss()?;
    let ((mut session, input, binding), load_time) = timed_load(|| {
        let model = models::ensure_hf_resnet50()?;
        let session = session(model.to_str().unwrap())?;
        let input = Tensor::<f32>::from_array((RESNET_IN.to_vec(), image_input()))?;
        let binding = bind(&session, &input)?;
        Ok((session, input, binding))
    })?;
    let loaded_rss = rss()?;
    let (warmup_time, run_time, sum) =
        run_binding(&mut session, &binding, warmups, iters, "hf_resnet50")?;
    std::hint::black_box(&input);
    Ok(Probe::new(
        "ort",
        "hf_resnet50",
        "iobinding",
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

fn run_binding(
    session: &mut Session,
    binding: &IoBinding,
    warmups: usize,
    iters: usize,
    _model: &str,
) -> ort::Result<(Duration, Duration, f64)> {
    let warmup_start = Instant::now();
    for _ in 0..warmups {
        let outputs = session.run_binding(binding)?;
        let view = outputs[0].try_extract_array::<f32>()?;
        std::hint::black_box(&view);
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        let outputs = session.run_binding(binding)?;
        let view = outputs[0].try_extract_array::<f32>()?;
        sum += checksum(view.as_slice().unwrap_or(&[]));
        std::hint::black_box(&view);
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
    let warmups = std::env::var("ZRT_BENCH_WARMUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    let probe = match model.as_str() {
        "mnist" => bench_mnist(warmups, iters)?,
        "relay4m" | "relay16m" => bench_relay(&model, warmups, iters)?,
        "hf_resnet50" => bench_resnet(warmups, iters)?,
        _ => {
            return Err(
                "usage: cargo run --release --example mem_probe -- [mnist|relay4m|relay16m|hf_resnet50] [iters]"
                    .into(),
            )
        }
    };
    probe.print();
    Ok(())
}
