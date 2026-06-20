//! Zero-copy + arena STABILITY guard (RESULTS.md §8, root cause re-investigated). Asserts that
//! the large-tensor zero-copy path (arena ON, `CreateTensorWithDataAsOrtValue` input) stays
//! crash-free under criterion's measurement loop. The historical ">4MB segfault" was NOT this
//! combination — it was a use-after-free of the OrtEnv in the bench harness (the Env was freed
//! while the Session was alive). This bench keeps the Env in scope, so a crash here would signal
//! a NEW, distinct regression rather than the old issue.
//!
//!   cargo bench --bench crash_repro                     # arena ON, pattern ON
//!   ZRT_PATTERN=off cargo bench --bench crash_repro     # pattern off
//!   ZRT_LABEL=16m cargo bench --bench crash_repro       # larger I/O
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions, Tensor};
use st_zrt_bench_c::models;

fn label() -> String {
    std::env::var("ZRT_LABEL").unwrap_or_else(|_| "4m".to_string())
}
fn pat_on() -> bool {
    !matches!(std::env::var("ZRT_PATTERN").as_deref(), Ok("off"))
}
fn n_for(label: &str) -> usize {
    match label {
        "1m" => 1 << 18,
        "4m" => 1 << 20,
        "16m" => 1 << 22,
        _ => 1 << 20,
    }
}

fn bench_crash(c: &mut Criterion) {
    let label = label();
    let n = n_for(&label);
    let path = models::ensure_relay(&label).expect("ensure_relay");
    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let mut opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    // Arena deliberately LEFT ON — this is the crashing combination.
    if !pat_on() {
        opts = opts.disable_mem_pattern();
    }
    let sess = st_zrt::Session::new(&env, path.to_str().unwrap(), opts).unwrap();
    let x = vec![3.0_f32; n];
    eprintln!(
        "crash_repro: label={label} n={n} arena=ON pattern={}",
        pat_on()
    );

    for _ in 0..16 {
        let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
        let mut out: Vec<Option<st_zrt::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
    }

    c.bench_function("crash_arena_on_zerocopy", |b| {
        b.iter(|| {
            let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
            let mut out: Vec<Option<st_zrt::OwnedValue>> =
                (0..sess.output_count()).map(|_| None).collect();
            sess.run(&[&input], &mut out).unwrap();
            black_box(out[0].as_ref().unwrap().as_slice::<f32>().unwrap());
        });
    });
}

criterion_group!(benches, bench_crash);
criterion_main!(benches);
