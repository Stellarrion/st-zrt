//! Criterion entry point — 3-way inference A/B/C (M0 feasibility gate).
//! A = ort default (copying) · B = ort expert (IoBinding, bind-once-mutate) · C = ortx proto (task #6).
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt_bench::{models, ort_default, ort_expert};

fn bench_variant_a(c: &mut Criterion) {
    let model = models::ensure_mnist().expect("failed to download mnist.onnx");
    let path = model.to_str().unwrap();
    let mut variant = ort_default::VariantA::new(path).expect("failed to load session (A)");
    for _ in 0..32 {
        variant.run_once().expect("warmup A failed");
    }
    c.bench_function("A_ort_default", |b| {
        b.iter(|| black_box(variant.run_once().expect("run A failed")));
    });
}

fn bench_variant_b(c: &mut Criterion) {
    let model = models::ensure_mnist().expect("failed to download mnist.onnx");
    let path = model.to_str().unwrap();
    let mut variant = ort_expert::VariantB::new(path).expect("failed to load session (B)");
    for _ in 0..32 {
        variant.run_once().expect("warmup B failed");
    }
    c.bench_function("B_ort_expert", |b| {
        b.iter(|| black_box(variant.run_once().expect("run B failed")));
    });
}

fn bench_variant_a_1thread(c: &mut Criterion) {
    let model = models::ensure_mnist().expect("failed to download mnist.onnx");
    let path = model.to_str().unwrap();
    let mut variant = ort_default::VariantA::new_with_intra_threads(path, Some(1))
        .expect("failed to load session (A/1t)");
    for _ in 0..32 {
        variant.run_once().expect("warmup A/1t failed");
    }
    c.bench_function("A_ort_default_1thread", |b| {
        b.iter(|| black_box(variant.run_once().expect("run A/1t failed")));
    });
}

fn bench_variant_b_1thread(c: &mut Criterion) {
    let model = models::ensure_mnist().expect("failed to download mnist.onnx");
    let path = model.to_str().unwrap();
    let mut variant = ort_expert::VariantB::new_with_intra_threads(path, Some(1))
        .expect("failed to load session (B/1t)");
    for _ in 0..32 {
        variant.run_once().expect("warmup B/1t failed");
    }
    c.bench_function("B_ort_expert_1thread", |b| {
        b.iter(|| black_box(variant.run_once().expect("run B/1t failed")));
    });
}

criterion_group!(
    benches,
    bench_variant_a,
    bench_variant_b,
    bench_variant_a_1thread,
    bench_variant_b_1thread
);
criterion_main!(benches);
