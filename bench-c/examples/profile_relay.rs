use std::error::Error;

use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions};
use st_zrt_bench_c::models;

fn main() -> Result<(), Box<dyn Error>> {
    let label = std::env::args().nth(1).unwrap_or_else(|| "16m".to_string());
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "lane".to_string());
    let iters: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let n = match label.as_str() {
        "4m" => 1usize << 20,
        "16m" => 1usize << 22,
        other => return Err(format!("unsupported relay label: {other}").into()),
    };
    let shape = [1, n as i64];
    let prefix = std::env::var("ZRT_PROFILE_PREFIX")
        .unwrap_or_else(|_| format!("/tmp/zrt-relay-{label}-{mode}"));

    let model = models::ensure_relay(&label)?;
    let env = Environment::new()?;
    let mem = MemoryInfo::cpu()?;
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1)
        .enable_profiling(&prefix)?;
    let session = st_zrt::Session::new(&env, model.to_str().unwrap(), opts)?;

    match mode.as_str() {
        "lane" | "lane-auto" => {
            let mut lane = session.prepare_tensor_io_lane::<f32>(&mem, &[&shape], &[&shape])?;
            lane.input_mut(0)?.fill(3.0);
            for _ in 0..iters {
                lane.run()?;
                std::hint::black_box(lane.output(0)?);
            }
        }
        "allocated-output" => {
            let mut lane =
                session.prepare_allocated_output_tensor_io_lane::<f32>(&mem, &mem, &[&shape], &[&shape])?;
            lane.input_mut(0)?.fill(3.0);
            for _ in 0..iters {
                lane.run()?;
                std::hint::black_box(lane.output(0)?);
            }
        }
        "device-output" => {
            let mut lane = session.prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&shape])?;
            lane.input_mut(0)?.fill(3.0);
            for _ in 0..iters {
                lane.run()?;
                std::hint::black_box(lane.output(0)?);
            }
        }
        other => {
            return Err(format!(
                "usage: profile_relay [4m|16m] [lane|allocated-output|device-output] [iters], got mode {other}"
            )
            .into())
        }
    }

    println!(
        "profiling_start_time_ns={}",
        session.profiling_start_time_ns()?
    );
    println!("profile_path={}", session.end_profiling()?);
    Ok(())
}
