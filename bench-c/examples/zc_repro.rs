//! Zero-copy + arena stability sweep (no criterion). Runs the large-tensor zero-copy path
//! (arena ON, `CreateTensorWithDataAsOrtValue` input) under a tight PLAIN loop and sweeps
//! arena × mem-pattern × size — proving the path is crash-free outside criterion too. The
//! historical ">4MB segfault" was a use-after-free of the OrtEnv in the bench harness (Env
//! freed while the Session lived), NOT this zero-copy + arena combination; this example keeps
//! the Env in scope, so a crash here would signal a NEW, distinct regression. Optional
//! ZRT_MODEL loads an explicit model path (e.g. an Add(X,X) model) instead of the relay.
//!
//! Env (all optional):
//!   ZRT_MODEL   explicit model .onnx path                (default: ensure_relay(ZRT_LABEL))
//!   ZRT_LABEL   model label, one of 1m|4m|16m            (default 4m)
//!   ZRT_ITERS   iterations                                (default 200000)
//!   ZRT_ARENA   on|off                                    (default on)
//!   ZRT_PATTERN on|off                                    (default on)
//! Exit: 0 = clean, non-zero = a regression reproduced.
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions, Tensor};
use st_zrt_bench_c::models;

fn n_for(label: &str) -> usize {
    match label {
        "1m" => 1 << 18,  // 262144 f32 = 1 MiB
        "4m" => 1 << 20,  // 1048576 f32 = 4 MiB
        "16m" => 1 << 22, // 4194304 f32 = 16 MiB
        _ => panic!("unknown label {label}; use 1m|4m|16m"),
    }
}

fn flag(var: &str, default_on: bool) -> bool {
    match std::env::var(var) {
        Ok(v) => {
            !v.eq_ignore_ascii_case("off")
                && !v.eq_ignore_ascii_case("0")
                && !v.eq_ignore_ascii_case("false")
        }
        Err(_) => default_on,
    }
}

fn main() {
    let label = std::env::var("ZRT_LABEL").unwrap_or_else(|_| "4m".into());
    let iters: usize = std::env::var("ZRT_ITERS")
        .unwrap_or_else(|_| "200000".into())
        .parse()
        .unwrap();
    let arena_on = flag("ZRT_ARENA", true);
    let pat_on = flag("ZRT_PATTERN", true);
    let n = n_for(&label);

    let path: std::path::PathBuf = match std::env::var("ZRT_MODEL") {
        Ok(p) => p.into(),
        Err(_) => models::ensure_relay(&label).expect("ensure_relay"),
    };
    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let mut opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    if !arena_on {
        opts = opts.disable_cpu_mem_arena();
    }
    if !pat_on {
        opts = opts.disable_mem_pattern();
    }
    let sess = st_zrt::Session::new(&env, path.to_str().unwrap(), opts).unwrap();

    let x = vec![3.0_f32; n];
    eprintln!("zc_repro: label={label} n={n} iters={iters} arena={arena_on} pattern={pat_on}");

    let mut checksum: u64 = 0;
    for i in 0..iters {
        let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
        let mut out: Vec<Option<st_zrt::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
        let s = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
        // Touch the output so the write is observed (no dead-store elision), sampled to
        // keep the hot loop clean. Summing bit patterns makes the result order-dependent
        // and thus un-elidable.
        checksum = checksum.wrapping_add(s[i & (n - 1)].to_bits() as u64);
        if i > 0 && i % 20000 == 0 {
            eprintln!("zc_repro: iter {i}/{iters} (checksum so far {checksum:#x})");
        }
    }
    eprintln!("zc_repro: CLEAN after {iters} iters; checksum={checksum:#x}");
}
