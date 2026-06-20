//! O3 micro-benchmark — per-call copy cost vs tensor size.
//!
//! Combined with ORT's ~37 µs fixed per-run floor (from `inference.rs` A/B),
//! this reveals the crossover size above which the binding-layer copy dominates
//! and zero-copy (variant C) becomes a clear win. No model needed.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use st_zrt_bench::micro;

const SIZES_F32: &[usize] = &[
    1_000,      // ~4 KB
    16_000,     // ~64 KB
    65_000,     // ~256 KB
    262_000,    // ~1 MB
    1_000_000,  // ~4 MB
    4_000_000,  // ~16 MB
    16_000_000, // ~64 MB  (exceeds L3 -> DRAM bandwidth)
    32_000_000, // ~128 MB
    57_500_000, // ~230 MB (the question)
];

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("O3_copy_tensor");
    for &n in SIZES_F32 {
        let bytes = (n * 4) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &n, |b, &n| {
            b.iter(|| black_box(micro::copy_tensor_f32(n).unwrap()));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_copy);
criterion_main!(benches);
