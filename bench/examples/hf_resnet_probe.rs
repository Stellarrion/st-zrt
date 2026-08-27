//! Probe HF ResNet session creation across Rust `ort` optimizer levels.
//!
//! This verifies whether `ort`'s default is the same optimizer enum as ZRT's `All`.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use st_zrt_bench::models;

const INPUT_SHAPE: [i64; 4] = [1, 3, 224, 224];
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

fn try_level(model: &str, name: &str, level: Option<GraphOptimizationLevel>) {
    let mut builder = Session::builder()
        .expect("builder")
        .with_intra_threads(1)
        .expect("intra threads");

    if let Some(level) = level {
        builder = builder
            .with_optimization_level(level)
            .expect("optimization level");
    }

    match builder.commit_from_file(model) {
        Ok(session) => {
            println!(
                "ort {name:<10} OK   inputs={} outputs={} input0={:?} output0={:?}",
                session.inputs().len(),
                session.outputs().len(),
                (session.inputs()[0].name(), session.inputs()[0].dtype()),
                (session.outputs()[0].name(), session.outputs()[0].dtype())
            );
        }
        Err(err) => {
            println!("ort {name:<10} FAIL {err}");
        }
    }
}

fn run_all_output(model: &str) {
    let mut session = Session::builder()
        .expect("builder")
        .with_optimization_level(GraphOptimizationLevel::All)
        .expect("optimization level")
        .with_intra_threads(1)
        .expect("intra threads")
        .commit_from_file(model)
        .expect("session");

    let tensor = ort::value::Tensor::<f32>::from_array((INPUT_SHAPE.to_vec(), image_input()))
        .expect("input");
    let outputs = session.run(ort::inputs![tensor]).expect("run");
    let output = outputs[0].try_extract_array::<f32>().expect("output");
    let values: Vec<f32> = output.iter().copied().collect();
    summarize("ort All run output", &values);
}

fn main() {
    let model = models::ensure_hf_resnet50().expect("hf resnet50");
    let model = model.to_str().expect("utf-8 model path");

    println!("ort build info: {}", ort::info());
    println!("model: {model}");

    try_level(model, "Default", None);
    try_level(model, "Level1", Some(GraphOptimizationLevel::Level1));
    try_level(model, "Level2", Some(GraphOptimizationLevel::Level2));
    try_level(model, "Level3", Some(GraphOptimizationLevel::Level3));
    try_level(model, "All", Some(GraphOptimizationLevel::All));

    println!("-- output check --");
    run_all_output(model);
}
