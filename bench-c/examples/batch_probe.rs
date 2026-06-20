use std::error::Error;
use std::time::Instant;

#[cfg(feature = "ep")]
use st_zrt::{CudaPreset, EpProvider};
use st_zrt::{
    Environment, ExecutionMode, GraphOptimizationLevel, LoggingLevel, MemoryInfo, SessionOptions,
    ThreadingOptions,
};
use st_zrt_bench_c::models;

#[derive(Clone)]
struct ProbeConfig {
    ep: String,
    intra_threads: Option<i32>,
    inter_threads: Option<i32>,
    execution_mode: Option<ExecutionMode>,
    execution_mode_label: String,
    free_dim_names: Vec<String>,
    free_dim_denotations: Vec<String>,
    intra_op_spinning: Option<bool>,
    inter_op_spinning: Option<bool>,
    use_global_thread_pool: bool,
    global_intra_affinity: Option<String>,
}

fn resolve_model(name: &str) -> Result<std::path::PathBuf, Box<dyn Error>> {
    match name {
        "mnist" => Ok(models::ensure_mnist()?),
        "hf_resnet50" | "resnet50" => Ok(models::ensure_hf_resnet50()?),
        other => Err(format!("unsupported model: {other}; use mnist or hf_resnet50").into()),
    }
}

fn input_fill(model: &str, buf: &mut [f32]) {
    match model {
        "hf_resnet50" | "resnet50" => {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = ((i % 251) as f32 - 125.0) / 128.0;
            }
        }
        _ => buf.fill(0.0),
    }
}

fn element_count(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        if dim < 0 {
            None
        } else {
            acc.checked_mul(dim as usize)
        }
    })
}

fn batch_shape(base: &[i64], batch: i64) -> Option<Vec<i64>> {
    let (&first, rest) = base.split_first()?;
    if first != -1 && first != batch {
        return None;
    }
    let mut out = Vec::with_capacity(base.len());
    out.push(batch);
    out.extend_from_slice(rest);
    Some(out)
}

fn known_batch_shapes(model: &str, batch: i64) -> Option<(Vec<i64>, Vec<i64>)> {
    match model {
        "hf_resnet50" | "resnet50" => Some((vec![batch, 3, 224, 224], vec![batch, 1000])),
        _ => None,
    }
}

fn checksum(values: &[f32]) -> f64 {
    let step = (values.len() / 32).max(1);
    values.iter().step_by(step).map(|v| *v as f64).sum::<f64>()
}

fn parse_i32_env(name: &str) -> Option<i32> {
    std::env::var(name).ok().and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            trimmed.parse().ok()
        }
    })
}

fn parse_bool_env(name: &str) -> Result<Option<bool>, Box<dyn Error>> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(None);
    };
    let parsed = match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        other => return Err(format!("{name} must be bool-like, got {other:?}").into()),
    };
    Ok(Some(parsed))
}

fn split_list_env(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .into_iter()
        .flat_map(|value| {
            value
                .split([',', ' ', ';'])
                .filter_map(|s| {
                    let s = s.trim();
                    if s.is_empty() {
                        None
                    } else {
                        Some(s.to_string())
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn probe_config() -> Result<ProbeConfig, Box<dyn Error>> {
    let execution_mode_label =
        std::env::var("ZRT_EXECUTION_MODE").unwrap_or_else(|_| "default".to_string());
    let execution_mode = match execution_mode_label.trim().to_ascii_lowercase().as_str() {
        "" | "default" => None,
        "sequential" | "seq" => Some(ExecutionMode::Sequential),
        "parallel" | "par" => Some(ExecutionMode::Parallel),
        other => return Err(format!("unsupported ZRT_EXECUTION_MODE={other:?}").into()),
    };

    let mut free_dim_names = split_list_env("ZRT_FREE_DIM_NAMES");
    free_dim_names.extend(split_list_env("ZRT_FREE_DIM_NAME"));
    free_dim_names.sort();
    free_dim_names.dedup();

    let mut free_dim_denotations = split_list_env("ZRT_FREE_DIM_DENOTATIONS");
    free_dim_denotations.extend(split_list_env("ZRT_FREE_DIM_DENOTATION"));
    free_dim_denotations.sort();
    free_dim_denotations.dedup();

    Ok(ProbeConfig {
        ep: std::env::var("ZRT_EP").unwrap_or_else(|_| "cpu".to_string()),
        intra_threads: parse_i32_env("ZRT_INTRA_THREADS"),
        inter_threads: parse_i32_env("ZRT_INTER_THREADS"),
        execution_mode,
        execution_mode_label,
        free_dim_names,
        free_dim_denotations,
        intra_op_spinning: parse_bool_env("ZRT_INTRA_OP_SPIN")?,
        inter_op_spinning: parse_bool_env("ZRT_INTER_OP_SPIN")?,
        use_global_thread_pool: parse_bool_env("ZRT_USE_GLOBAL_THREAD_POOL")?.unwrap_or(false),
        global_intra_affinity: std::env::var("ZRT_GLOBAL_INTRA_AFFINITY")
            .ok()
            .filter(|s| !s.trim().is_empty()),
    })
}

fn make_env(cfg: &ProbeConfig) -> Result<Environment, Box<dyn Error>> {
    if cfg.use_global_thread_pool || cfg.global_intra_affinity.is_some() {
        let mut threading = ThreadingOptions::new()?;
        if let Some(n) = cfg.intra_threads {
            threading = threading.with_intra_threads(n)?;
        }
        if let Some(n) = cfg.inter_threads {
            threading = threading.with_inter_threads(n)?;
        }
        if let Some(affinity) = &cfg.global_intra_affinity {
            threading = threading.with_intra_thread_affinity(affinity)?;
        }
        return Ok(Environment::new_with_global_thread_pools(
            LoggingLevel::Warning,
            "zrt-batch-probe",
            threading,
        )?);
    }
    Ok(Environment::new()?)
}

fn apply_ep(opts: SessionOptions, cfg: &ProbeConfig) -> Result<SessionOptions, Box<dyn Error>> {
    let ep = cfg.ep.trim().to_ascii_lowercase();
    if ep.is_empty() || ep == "cpu" {
        return Ok(opts);
    }

    #[cfg(feature = "ep")]
    {
        let device_id = std::env::var("ZRT_CUDA_DEVICE_ID")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let opts = match ep.as_str() {
            "dnnl" => opts.with_execution_provider(EpProvider::Dnnl, &[])?,
            "openvino" | "openvino_cpu" => {
                opts.with_execution_provider(EpProvider::OpenVinoV2, &[("device_type", "CPU")])?
            }
            "openvino_gpu" => {
                opts.with_execution_provider(EpProvider::OpenVinoV2, &[("device_type", "GPU")])?
            }
            "cuda" => opts.with_cuda_preset(CudaPreset::performance(device_id))?,
            "cuda_graph" => opts.with_cuda_preset(CudaPreset::cuda_graph(device_id))?,
            "tensorrt" | "trt" => opts.with_execution_provider(
                EpProvider::TensorRt,
                &[("device_id", &device_id.to_string())],
            )?,
            other => return Err(format!("unsupported ZRT_EP={other:?}").into()),
        };
        Ok(opts)
    }

    #[cfg(not(feature = "ep"))]
    {
        Err(format!("ZRT_EP={ep:?} requires bench-c feature `ep` or `cuda`").into())
    }
}

fn make_options(cfg: &ProbeConfig, batch: i64) -> Result<SessionOptions, Box<dyn Error>> {
    let mut opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    if let Some(n) = cfg.intra_threads {
        opts = opts.with_intra_threads(n);
    }
    if let Some(n) = cfg.inter_threads {
        opts = opts.with_inter_threads(n);
    }
    if let Some(mode) = cfg.execution_mode {
        opts = opts.with_execution_mode(mode);
    }
    if let Some(enabled) = cfg.intra_op_spinning {
        opts = opts.with_intra_op_spinning(enabled)?;
    }
    if let Some(enabled) = cfg.inter_op_spinning {
        opts = opts.with_inter_op_spinning(enabled)?;
    }
    for name in &cfg.free_dim_names {
        opts = opts.with_free_dimension_override_by_name(name, batch)?;
    }
    for denotation in &cfg.free_dim_denotations {
        opts = opts.with_free_dimension_override(denotation, batch)?;
    }
    if !cfg.use_global_thread_pool {
        opts = opts.use_per_session_threads();
    }
    apply_ep(opts, cfg)
}

fn free_dim_label(cfg: &ProbeConfig) -> String {
    let mut parts = Vec::new();
    if !cfg.free_dim_names.is_empty() {
        parts.push(format!("name:{}", cfg.free_dim_names.join("|")));
    }
    if !cfg.free_dim_denotations.is_empty() {
        parts.push(format!("denotation:{}", cfg.free_dim_denotations.join("|")));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("+")
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hf_resnet50".to_string());
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let batches: Vec<i64> = std::env::var("ZRT_BATCH_SIZES")
        .unwrap_or_else(|_| "1 2 4 8".to_string())
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();

    let cfg = probe_config()?;
    let env = make_env(&cfg)?;
    let mem = MemoryInfo::cpu()?;
    let model = resolve_model(&model_name)?;
    let model_path = model.to_str().ok_or("model path is not UTF-8")?;
    let base_session = st_zrt::Session::new(&env, model_path, make_options(&cfg, 1)?)?;
    let base_input = base_session.input_shape(0)?.to_vec();
    let base_output = base_session.output_shape(0)?.to_vec();
    drop(base_session);

    let intra = cfg
        .intra_threads
        .map(|v| v.to_string())
        .unwrap_or_else(|| "default".to_string());
    let inter = cfg
        .inter_threads
        .map(|v| v.to_string())
        .unwrap_or_else(|| "default".to_string());
    let free_dim = free_dim_label(&cfg);
    println!(
        "model,ep,intra_threads,inter_threads,execution_mode,free_dim,batch,status,avg_us,per_item_us,input_elems,output_elems,checksum"
    );
    for batch in batches {
        let opts = match make_options(&cfg, batch) {
            Ok(opts) => opts,
            Err(err) => {
                println!(
                    "{model_name},{},{intra},{inter},{},{free_dim},{batch},options_error,,,,,{}",
                    cfg.ep, cfg.execution_mode_label, err
                );
                continue;
            }
        };
        let session = match st_zrt::Session::new(&env, model_path, opts) {
            Ok(session) => session,
            Err(err) => {
                println!(
                    "{model_name},{},{intra},{inter},{},{free_dim},{batch},session_error,,,,,{}",
                    cfg.ep, cfg.execution_mode_label, err
                );
                continue;
            }
        };

        let known = known_batch_shapes(&model_name, batch);
        let input_shape = if let Some((input_shape, _)) = &known {
            input_shape.clone()
        } else if let Some(input_shape) = batch_shape(&base_input, batch) {
            input_shape
        } else {
            println!(
                "{model_name},{},{intra},{inter},{},{free_dim},{batch},unsupported_static_input,,,,,",
                cfg.ep, cfg.execution_mode_label
            );
            continue;
        };
        let output_shape = known
            .as_ref()
            .map(|(_, output_shape)| output_shape.clone())
            .or_else(|| batch_shape(&base_output, batch));
        if element_count(&input_shape).is_none() {
            println!(
                "{model_name},{},{intra},{inter},{},{free_dim},{batch},unsupported_dynamic_input,,,,,",
                cfg.ep, cfg.execution_mode_label
            );
            continue;
        }

        if let Some(output_shape) = output_shape.and_then(|s| element_count(&s).map(|_| s)) {
            let mut lane =
                session.prepare_tensor_io_lane::<f32>(&mem, &[&input_shape], &[&output_shape])?;
            input_fill(&model_name, lane.input_mut(0)?);
            for _ in 0..5 {
                lane.run()?;
                std::hint::black_box(lane.output(0)?);
            }

            let start = Instant::now();
            let mut sum = 0.0;
            for _ in 0..iters {
                lane.run()?;
                let out = lane.output(0)?;
                sum += checksum(out);
                std::hint::black_box(out);
            }
            let avg_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
            println!(
                "{model_name},{},{intra},{inter},{},{free_dim},{batch},ok,{avg_us:.3},{:.3},{},{},{sum:.6}",
                cfg.ep,
                cfg.execution_mode_label,
                avg_us / batch as f64,
                element_count(&input_shape).unwrap_or(0),
                element_count(&output_shape).unwrap_or(0)
            );
        } else {
            let mut lane =
                session.prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&input_shape])?;
            input_fill(&model_name, lane.input_mut(0)?);
            for _ in 0..5 {
                lane.run()?;
                std::hint::black_box(lane.output(0)?);
            }

            let start = Instant::now();
            let mut sum = 0.0;
            let mut output_elems = 0usize;
            for _ in 0..iters {
                lane.run()?;
                let out = lane.output(0)?.as_slice::<f32>()?;
                output_elems = out.len();
                sum += checksum(out);
                std::hint::black_box(out);
            }
            let avg_us = start.elapsed().as_secs_f64() * 1_000_000.0 / iters as f64;
            println!(
                "{model_name},{},{intra},{inter},{},{free_dim},{batch},ok_dynamic_output,{avg_us:.3},{:.3},{},{output_elems},{sum:.6}",
                cfg.ep,
                cfg.execution_mode_label,
                avg_us / batch as f64,
                element_count(&input_shape).unwrap_or(0)
            );
        }
    }

    Ok(())
}
