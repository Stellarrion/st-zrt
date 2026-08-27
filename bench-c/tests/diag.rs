//! Single-process diagnostic for the >4 MB Run-allocated-output failure. Driven by env:
//!   ZRT_SIZE_MIB (e.g. 4,6,8,...,16)  ZRT_ARENA=on|off  ZRT_PATTERN=on|off  ZRT_ITERS=N
//! Loads relay_<size>m, does ZRT_ITERS run()s, prints OK or crashes (segfault). Each combo
//! runs in its own process so a crash is isolated.
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, Session, SessionOptions, Tensor};

#[test]
fn diag_one_run() {
    let mib: usize = std::env::var("ZRT_SIZE_MIB")
        .unwrap_or_else(|_| "8".to_string())
        .parse()
        .unwrap();
    let arena = std::env::var("ZRT_ARENA")
        .map(|v| v != "off")
        .unwrap_or(true);
    let pattern = std::env::var("ZRT_PATTERN")
        .map(|v| v != "off")
        .unwrap_or(true);
    let iters: usize = std::env::var("ZRT_ITERS")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap();
    let n = (mib << 20) / 4;
    eprintln!(
        "DIAG size={}MiB n={} arena={} pattern={} iters={}",
        mib, n, arena, pattern, iters
    );

    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join(format!("relay_{mib}m.onnx"));
    assert!(path.exists(), "model missing: {}", path.display());

    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let mut opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    if !arena {
        opts = opts.disable_cpu_mem_arena();
    }
    if !pattern {
        opts = opts.disable_mem_pattern();
    }
    let sess = Session::new(&env, path.to_str().unwrap(), opts).unwrap();
    let x: Vec<f32> = vec![3.0; n];
    let bb = std::env::var("ZRT_BB").map(|v| v == "1").unwrap_or(false);
    for i in 0..iters {
        let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).unwrap();
        let mut out: Vec<Option<st_zrt::OwnedValue>> =
            (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).expect("run");
        if bb {
            std::hint::black_box(out[0].as_ref().unwrap().as_slice::<f32>().unwrap());
        }
        if i == 0 {
            let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
            assert_eq!(y.len(), n);
            eprintln!("DIAG first-run OK Y[0]={}", y[0]);
        }
    }
    eprintln!(
        "DIAG DONE: size={}MiB arena={} pattern={} iters={} (no crash)",
        mib, arena, pattern, iters
    );
}
