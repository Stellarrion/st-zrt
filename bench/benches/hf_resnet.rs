//! Hugging Face real-model benchmark for the Rust `ort` wrapper.
//!
//! Model: `Xenova/resnet-50`, `onnx/model.onnx`, cached as `bench/models/hf_resnet50.onnx`.
//! Input is deterministic synthetic image data; preprocessing is intentionally outside the
//! benchmark so the measurement is runtime/session/tensor overhead plus model execution.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ort::memory::MemoryInfo;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use st_zrt_bench::models;

const INPUT_SHAPE: [i64; 4] = [1, 3, 224, 224];
const INPUT_LEN: usize = 3 * 224 * 224;

fn image_input() -> Vec<f32> {
    (0..INPUT_LEN)
        .map(|i| ((i % 251) as f32 - 125.0) / 128.0)
        .collect()
}

fn load_session() -> Session {
    let model = models::ensure_hf_resnet50().expect("hf resnet50");
    Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_file(model.to_str().unwrap())
        .unwrap()
}

fn bench_ort_hf_resnet50_run(c: &mut Criterion) {
    let mut session = load_session();
    let input = image_input();
    let shape = INPUT_SHAPE.to_vec();

    for _ in 0..8 {
        let tensor = Tensor::<f32>::from_array((shape.clone(), input.clone())).unwrap();
        let outputs = session.run(ort::inputs![tensor]).unwrap();
        black_box(outputs[0].try_extract_array::<f32>().unwrap());
    }

    c.bench_function("HF_resnet50_ort_all_run", |b| {
        b.iter(|| {
            let tensor = Tensor::<f32>::from_array((shape.clone(), input.clone())).unwrap();
            let outputs = session.run(ort::inputs![tensor]).unwrap();
            black_box(outputs[0].try_extract_array::<f32>().unwrap());
        });
    });
}

fn bench_ort_hf_resnet50_iobinding(c: &mut Criterion) {
    let mut session = load_session();
    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();
    let input = Tensor::<f32>::from_array((INPUT_SHAPE.to_vec(), image_input())).unwrap();
    let mut binding = session.create_binding().unwrap();
    binding.bind_input(in_name, &input).unwrap();
    binding
        .bind_output_to_device(out_name, &MemoryInfo::default())
        .unwrap();

    for _ in 0..8 {
        let outputs = session.run_binding(&binding).unwrap();
        black_box(outputs[0].try_extract_array::<f32>().unwrap());
    }

    c.bench_function("HF_resnet50_ort_all_iobinding", |b| {
        b.iter(|| {
            let outputs = session.run_binding(&binding).unwrap();
            black_box(outputs[0].try_extract_array::<f32>().unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_ort_hf_resnet50_run,
    bench_ort_hf_resnet50_iobinding
);
criterion_main!(benches);
