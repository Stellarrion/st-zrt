use std::error::Error;
use std::fs;
use std::time::{Duration, Instant};

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, MmapTensorOptions, OwnedInitializer,
    OwnedValue, Session, SessionOptions, Tensor, TensorBuffer,
};
use st_zrt_bench_c::models;

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

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn checksum(values: &[f32]) -> f64 {
    let step = (values.len() / 32).max(1);
    values.iter().step_by(step).map(|v| *v as f64).sum::<f64>()
}

fn timed_load<F, T>(load: F) -> Result<(T, Duration), Box<dyn Error>>
where
    F: FnOnce() -> Result<T, Box<dyn Error>>,
{
    let start = Instant::now();
    let value = load()?;
    Ok((value, start.elapsed()))
}

fn session_options() -> SessionOptions {
    SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1)
}

fn mmap_options() -> MmapTensorOptions {
    MmapTensorOptions {
        byte_offset: 0,
        sequential: std::env::var("ZRT_MMAP_SEQUENTIAL")
            .ok()
            .as_deref()
            != Some("0"),
        hugepage: std::env::var("ZRT_MMAP_HUGEPAGE").ok().as_deref() == Some("1"),
        locked: std::env::var("ZRT_MMAP_LOCKED").ok().as_deref() == Some("1"),
    }
}

fn run_session(
    sess: &Session,
    mem: &MemoryInfo,
    n: usize,
    warmups: usize,
    iters: usize,
) -> st_zrt::Result<(Duration, Duration, f64)> {
    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], mem)?;
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();

    let warmup_start = Instant::now();
    for _ in 0..warmups {
        sess.run(&[&input], &mut outputs)?;
        let out = outputs[0].as_ref().expect("warmup output").as_slice::<f32>()?;
        std::hint::black_box(out);
    }
    let warmup_time = warmup_start.elapsed();

    let run_start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..iters {
        sess.run(&[&input], &mut outputs)?;
        let out = outputs[0].as_ref().expect("run output").as_slice::<f32>()?;
        sum += checksum(out);
        std::hint::black_box(out);
    }
    Ok((warmup_time, run_start.elapsed(), sum))
}

fn prepare_sidecar(n: usize) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let path = std::env::temp_dir().join(format!(
        "st-zrt-mmap-initializer-probe-{}-{}.bin",
        std::process::id(),
        n
    ));
    let values = vec![2.0_f32; n];
    fs::write(&path, f32_as_bytes(&values))?;
    Ok(path)
}

struct Probe {
    mode: &'static str,
    model: String,
    warmups: usize,
    iters: usize,
    sidecar_mb: f64,
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
        mode: &'static str,
        model: &str,
        warmups: usize,
        iters: usize,
        sidecar_bytes: usize,
        load_time: Duration,
        warmup_time: Duration,
        run_time: Duration,
        start_rss: Rss,
        loaded_rss: Rss,
        checksum: f64,
    ) -> Self {
        let done = rss().unwrap_or(loaded_rss);
        Self {
            mode,
            model: model.to_string(),
            warmups,
            iters,
            sidecar_mb: sidecar_bytes as f64 / (1024.0 * 1024.0),
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

    fn print_header() {
        println!(
            "mode,model,warmups,iters,sidecar_mb,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
        );
    }

    fn print(&self) {
        println!(
            "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{:.6}",
            self.mode,
            self.model,
            self.warmups,
            self.iters,
            self.sidecar_mb,
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

fn bench(mode: &str, label: &str, warmups: usize, iters: usize) -> Result<Probe, Box<dyn Error>> {
    let n = match label {
        "256k" => 65_536usize,
        "4m" => 1usize << 20,
        "16m" => 1usize << 22,
        _ => return Err(format!("unsupported relay label: {label}").into()),
    };
    let model = if mode == "embedded" {
        models::ensure_relay(label)?
    } else {
        models::ensure_relay_external(label)?
    };
    let sidecar = if mode == "mmap" {
        Some(prepare_sidecar(n)?)
    } else {
        None
    };
    let sidecar_bytes = sidecar
        .as_ref()
        .and_then(|path| path.metadata().ok())
        .map(|meta| meta.len() as usize)
        .unwrap_or(0);

    let start_rss = rss()?;
    let ((_env, mem, sess), load_time) = timed_load(|| {
        let env = Environment::new()?;
        let mem = MemoryInfo::cpu()?;
        let sess = match mode {
            "embedded" | "external" => {
                Session::new(&env, model.to_str().unwrap(), session_options())?
            }
            "vec" => {
                let c = TensorBuffer::from_vec(vec![2.0_f32; n], &[1, n as i64], &mem)?;
                let init = OwnedInitializer::tensor("C", c)?;
                Session::new_with_owned_initializers(
                    &env,
                    model.to_str().unwrap(),
                    session_options(),
                    vec![init],
                )?
            }
            "mmap" => {
                let c = TensorBuffer::<f32>::from_mmap_file_with_options(
                    sidecar.as_ref().expect("sidecar"),
                    &[1, n as i64],
                    &mem,
                    mmap_options(),
                )?;
                let init = OwnedInitializer::tensor("C", c)?;
                Session::new_with_owned_initializers(
                    &env,
                    model.to_str().unwrap(),
                    session_options(),
                    vec![init],
                )?
            }
            other => return Err(format!("unsupported mode: {other}").into()),
        };
        Ok((env, mem, sess))
    })?;
    let loaded_rss = rss()?;
    let (warmup_time, run_time, sum) = run_session(&sess, &mem, n, warmups, iters)?;
    if let Some(path) = sidecar {
        let _ = fs::remove_file(path);
    }
    Ok(Probe::new(
        match mode {
            "embedded" => "embedded",
            "external" => "external_data",
            "vec" => "vec_initializer",
            "mmap" => "mmap_initializer",
            _ => unreachable!(),
        },
        &format!("relay{label}"),
        warmups,
        iters,
        sidecar_bytes,
        load_time,
        warmup_time,
        run_time,
        start_rss,
        loaded_rss,
        sum,
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let label = args.next().unwrap_or_else(|| "4m".to_string());
    let mode = args.next().unwrap_or_else(|| "all".to_string());
    let iters = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        std::env::var("ZRT_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100)
    });
    let warmups = std::env::var("ZRT_BENCH_WARMUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);

    Probe::print_header();
    if mode == "all" {
        for mode in ["embedded", "external", "vec", "mmap"] {
            bench(mode, &label, warmups, iters)?.print();
        }
    } else {
        bench(&mode, &label, warmups, iters)?.print();
    }
    Ok(())
}
