//! Probe HF ResNet session creation across raw ORT optimizer levels through ZRT.
//!
//! This is intentionally an example, not a Criterion bench: it diagnoses optimizer/session
//! creation failures before we decide what should be benchmarked.

use std::ffi::CString;
use std::ptr;

use st_zrt::{GraphOptimizationLevel, Session, SessionOptions};
use st_zrt_bench_c::models;

const INPUT_SHAPE: [i64; 4] = [1, 3, 224, 224];
const OUTPUT_SHAPE: [i64; 2] = [1, 1000];
const INPUT_LEN: usize = 3 * 224 * 224;

fn image_input() -> Vec<f32> {
    (0..INPUT_LEN)
        .map(|i| ((i % 251) as f32 - 125.0) / 128.0)
        .collect()
}

fn summarize(label: &str, values: &[f32]) {
    let sum: f32 = values.iter().sum();
    let (max_idx, max_val) = values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap_or((usize::MAX, f32::NAN));
    let first3: Vec<f32> = values.iter().take(3).copied().collect();
    println!(
        "{label}: len={} sum={sum:.6} max_idx={max_idx} max_val={max_val:.6} first3={first3:?}",
        values.len()
    );
}

fn zrt_version() -> String {
    unsafe {
        let base = st_zrt::sys::api_base();
        if base.is_null() {
            return "<OrtGetApiBase returned null>".to_string();
        }
        match (*base).version_string() {
            Some(v) => v.to_string_lossy().into_owned(),
            None => "<GetVersionString missing>".to_string(),
        }
    }
}

fn try_level(env: &st_zrt::Environment, model: &str, name: &str, level: GraphOptimizationLevel) {
    let opts = SessionOptions::new()
        .with_opt_level(level)
        .with_intra_threads(1);

    match Session::new(env, model, opts) {
        Ok(session) => {
            println!(
                "ZRT {name:<10} OK   inputs={} outputs={} input0={} output0={} output0_shape={:?}",
                session.input_count(),
                session.output_count(),
                session.input_name(0).expect("input name"),
                session.output_name(0).expect("output name"),
                session.output_shape(0).expect("output shape")
            );
        }
        Err(err) => {
            println!("ZRT {name:<10} FAIL {err}");
        }
    }
}

fn raw_status(api: &st_zrt::sys::Api, status: st_zrt::sys::StatusPtr) -> Result<(), String> {
    unsafe {
        st_zrt::sys::status_to_result(api, status).map_err(|(code, msg)| {
            format!("code={code} message={}", msg.to_string_lossy().trim_end())
        })
    }
}

fn try_raw_create(model: &str, name: &str, level: GraphOptimizationLevel) {
    unsafe {
        let api = &*st_zrt::sys::api();
        let mut env: *mut st_zrt::sys::EnvHandle = ptr::null_mut();
        let logid = CString::new(format!("zrt-raw-{name}")).expect("log id");
        if let Err(err) = raw_status(
            api,
            api.create_env()(st_zrt::LoggingLevel::Warning, logid.as_ptr(), &mut env),
        ) {
            println!("ZRT raw {name:<10} FAIL CreateEnv {err}");
            return;
        }

        let mut opts: *mut st_zrt::sys::SessionOptionsHandle = ptr::null_mut();
        let opts_status = raw_status(api, api.create_session_options()(&mut opts));
        if let Err(err) = opts_status {
            api.release_env()(env);
            println!("ZRT raw {name:<10} FAIL CreateSessionOptions {err}");
            return;
        }
        let opt_status = raw_status(api, api.set_session_graph_optimization_level()(opts, level));
        if let Err(err) = opt_status {
            api.release_session_options()(opts);
            api.release_env()(env);
            println!("ZRT raw {name:<10} FAIL SetGraphOptimizationLevel {err}");
            return;
        }

        let cpath = CString::new(model).expect("model path");
        let mut session: *mut st_zrt::sys::SessionHandle = ptr::null_mut();
        let create = raw_status(
            api,
            api.create_session()(env, cpath.as_ptr(), opts as *const _, &mut session),
        );
        match create {
            Ok(()) => {
                println!("ZRT raw {name:<10} OK");
                api.release_session()(session);
            }
            Err(err) => {
                println!("ZRT raw {name:<10} FAIL CreateSession {err}");
            }
        }

        api.release_session_options()(opts);
        api.release_env()(env);
    }
}

fn run_all_outputs(model: &str) {
    let env = st_zrt::Environment::new().expect("environment");
    let mem = st_zrt::MemoryInfo::cpu().expect("memory info");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let session = Session::new(&env, model, opts).expect("session");

    let input_buf = image_input();
    let input = st_zrt::Tensor::from_buffer(&input_buf, &INPUT_SHAPE, &mem).expect("input");
    let mut run = session.prepare_run(&[&input]).expect("prepared run");
    run.run().expect("run");
    let prepared = run
        .output(0)
        .expect("output index")
        .expect("output")
        .as_slice::<f32>()
        .expect("output slice");
    summarize("ZRT All prepared_run output", prepared);

    let mut lane = session
        .prepare_tensor_io_lane::<f32>(&mem, &[&INPUT_SHAPE], &[&OUTPUT_SHAPE])
        .expect("lane");
    lane.input_mut(0)
        .expect("lane input")
        .copy_from_slice(&input_buf);
    lane.run().expect("lane run");
    summarize("ZRT All lane output", lane.output(0).expect("lane output"));
}

fn main() {
    let model = models::ensure_hf_resnet50().expect("hf resnet50");
    let model = model.to_str().expect("utf-8 model path");
    let env = st_zrt::Environment::new().expect("environment");

    println!("ZRT ORT version: {}", zrt_version());
    println!("model: {model}");

    println!("-- raw CreateSession --");
    try_raw_create(model, "DisableAll", GraphOptimizationLevel::DisableAll);
    try_raw_create(model, "Basic", GraphOptimizationLevel::Basic);
    try_raw_create(model, "Extended", GraphOptimizationLevel::Extended);
    try_raw_create(model, "Layout", GraphOptimizationLevel::Layout);
    try_raw_create(model, "All", GraphOptimizationLevel::All);

    println!("-- safe Session::new --");
    try_level(
        &env,
        model,
        "DisableAll",
        GraphOptimizationLevel::DisableAll,
    );
    try_level(&env, model, "Basic", GraphOptimizationLevel::Basic);
    try_level(&env, model, "Extended", GraphOptimizationLevel::Extended);
    try_level(&env, model, "Layout", GraphOptimizationLevel::Layout);
    try_level(&env, model, "All", GraphOptimizationLevel::All);

    println!("-- output check --");
    run_all_outputs(model);
}
