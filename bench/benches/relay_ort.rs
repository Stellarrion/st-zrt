//! Discriminator bench: run the SAME relay model through the `ort` crate (its own ort-sys
//! libonnxruntime) under criterion, with the arena ON (ort's default). If this crashes too,
//! the failure is ORT+platform+criterion, not st-zrt's FFI. If it survives, st-zrt's run()
//! path is the suspect.
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ort::{session::Session, value::Tensor};

fn load_relay(label: &str) -> (Session, usize) {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(format!("relay_{label}.onnx"));
    assert!(path.exists(), "model missing: {}", path.display());
    let n = match label {
        "4m" => 1usize << 20,
        "16m" => 1usize << 22,
        _ => 1usize << 20,
    };
    let session = Session::builder()
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_file(path.to_str().unwrap())
        .unwrap();
    (session, n)
}

fn bench_ort_relay_run_4m(c: &mut Criterion) {
    let (mut session, n) = load_relay("4m");
    let shape = vec![1, n as i64];
    let x = vec![3.0_f32; n];
    // warmup
    for _ in 0..16 {
        let tensor = Tensor::<f32>::from_array((shape.clone(), x.clone())).unwrap();
        let outputs = session.run(ort::inputs![tensor]).unwrap();
        black_box(outputs[0].try_extract_array::<f32>().unwrap());
    }
    c.bench_function("ort_relay_run_4m", |b| {
        b.iter(|| {
            let tensor = Tensor::<f32>::from_array((shape.clone(), x.clone())).unwrap();
            let outputs = session.run(ort::inputs![tensor]).unwrap();
            black_box(outputs[0].try_extract_array::<f32>().unwrap());
        });
    });
}

criterion_group!(benches, bench_ort_relay_run_4m);
criterion_main!(benches);
