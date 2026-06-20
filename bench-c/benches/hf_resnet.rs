//! Hugging Face real-model benchmark for ZRT.
//!
//! Model: `Xenova/resnet-50`, `onnx/model.onnx`, cached as `bench/models/hf_resnet50.onnx`.
//! Input is deterministic synthetic image data; preprocessing is intentionally outside the
//! benchmark so the measurement is runtime/session/tensor overhead plus model execution.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{GraphOptimizationLevel, SessionOptions, Tensor};
use st_zrt_bench_c::models;

const INPUT_SHAPE: [i64; 4] = [1, 3, 224, 224];
const OUTPUT_SHAPE: [i64; 2] = [1, 1000];
const INPUT_LEN: usize = 3 * 224 * 224;

fn image_input() -> Vec<f32> {
    (0..INPUT_LEN)
        .map(|i| ((i % 251) as f32 - 125.0) / 128.0)
        .collect()
}

fn load_session() -> (st_zrt::Environment, st_zrt::MemoryInfo, st_zrt::Session) {
    let model = models::ensure_hf_resnet50().expect("hf resnet50");
    let env = st_zrt::Environment::new().unwrap();
    let mem = st_zrt::MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let session = st_zrt::Session::new(&env, model.to_str().unwrap(), opts).unwrap();
    (env, mem, session)
}

fn bench_zrt_hf_resnet50_prepared_run(c: &mut Criterion) {
    let (_env, mem, session) = load_session();
    let input_buf = image_input();
    let input = Tensor::from_buffer(&input_buf, &INPUT_SHAPE, &mem).unwrap();
    let mut run = session.prepare_run(&[&input]).unwrap();

    for _ in 0..8 {
        run.run().unwrap();
        black_box(
            run.output(0)
                .expect("output index")
                .unwrap()
                .as_slice::<f32>()
                .unwrap(),
        );
    }

    c.bench_function("HF_resnet50_zrt_all_prepared_run", |b| {
        b.iter(|| {
            run.run().unwrap();
            black_box(
                run.output(0)
                    .expect("output index")
                    .unwrap()
                    .as_slice::<f32>()
                    .unwrap(),
            );
        });
    });
}

fn bench_zrt_hf_resnet50_lane(c: &mut Criterion) {
    let (_env, mem, session) = load_session();
    let mut lane = session
        .prepare_tensor_io_lane::<f32>(&mem, &[&INPUT_SHAPE], &[&OUTPUT_SHAPE])
        .unwrap();
    lane.input_mut(0)
        .expect("lane input")
        .copy_from_slice(&image_input());

    for _ in 0..8 {
        lane.run().unwrap();
        black_box(lane.output(0).expect("lane output"));
    }

    c.bench_function("HF_resnet50_zrt_all_lane", |b| {
        b.iter(|| {
            lane.run().unwrap();
            black_box(lane.output(0).expect("lane output"));
        });
    });
}

criterion_group!(
    benches,
    bench_zrt_hf_resnet50_prepared_run,
    bench_zrt_hf_resnet50_lane
);
criterion_main!(benches);
