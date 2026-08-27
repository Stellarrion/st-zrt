//! CUDA-graph GTE 3-config gate bench (Phase 1).
//!
//! Settles the Phase-1 CUDA-graph gate on the **correct** path. Measures THREE configs on
//! thenlper/gte-small (`model_cuda.onnx` — the CUDA-optimized export that partitions cleanly for
//! capture; the plain `model.onnx` export has CPU-fallback nodes ORT rejects for capture):
//!   1. **Baseline** (`CudaConfig::performance`, no graph) — the correct latency reference.
//!   2. **Host-input graph** (`CudaConfig::graph_replay`, host-resident lane inputs) — the naive
//!      fast-but-WRONG path: a captured graph bakes a device input pointer ORT does not refresh
//!      from host staging on replay, so changing the host input between runs has no effect (the bug
//!      the prior constant-input bench masked). Reported for completeness; its latency is the number
//!      the prior bench mistook for correct.
//!   3. **Device-input graph** (`CudaConfig::graph_replay` with an owned stream + device-resident
//!      lane inputs refreshed each run via `DynamicIoOptions::with_device_inputs`) — the CORRECT
//!      serving path: the per-run host→device refresh on the owned stream makes every replay
//!      read fresh data.
//!
//! Inputs CHANGE every iteration (a constant input masks the host-input bug). Reports min + median
//! latency, speedup vs the baseline (median), and correctness (max-abs-diff of each graph config's
//! output vs the baseline output on a shared final input). The 1.5× gate is settled on config 3
//! (correct): reported by default, hard-asserted only with `ZRT_GATE=strict`.
//!
//!   cargo bench --features cuda --bench cuda_graph_gte
//!   ZRT_GATE=strict cargo bench --features cuda --bench cuda_graph_gte   # enforce the 1.5× gate
//!   ZRT_GTE_MODEL=/path/to/model_cuda.onnx cargo bench --features cuda --bench cuda_graph_gte
//!   ZRT_SEQ=64 ZRT_BATCH=1 cargo bench --features cuda --bench cuda_graph_gte

// All helpers are used only by the CUDA `main`; the non-CUDA skip main is trivial, so silence the
// dead-code warnings in that build.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("cuda_graph_gte: requires --features cuda; skipping");
}

#[cfg(feature = "cuda")]
#[allow(non_snake_case)]
fn main() {
    use st_zrt::{
        CudaConfig, CudaStream, DynamicIoOptions, DynamicIoRuntime, Environment,
        GraphOptimizationLevel, MemoryInfo, OutputPolicy, OwnedValue, Result, RunInput,
        ServingLane, ServingShapePlan, Session, SessionOptions, Tensor,
    };
    use std::sync::Arc;
    use std::time::Instant;

    let BATCH: i64 = env_i64("ZRT_BATCH", 1);
    let SEQ: i64 = env_i64("ZRT_SEQ", 128);
    const WARMUP: usize = 30;
    const ITERS: usize = 300;
    // Offset baked into the shared "final" input so it differs from the iter-0 capture input — the
    // host-input graph's stale replay must then diverge from the baseline on that final input.
    const FINAL_OFFSET: i64 = 12345;

    /// Which input path + capture mode a measured config uses.
    #[derive(Clone, Copy, Debug)]
    enum Cfg {
        Baseline,
        HostInputGraph,
        DeviceInputGraph,
    }

    /// Per-run latency summary (µs): `min` is the best-case capability (least system noise);
    /// `median` is the typical run.
    struct Latency {
        min: f64,
        median: f64,
    }

    let Some(path) = gte_path() else {
        eprintln!(
            "cuda_graph_gte: GTE model not found; set ZRT_GTE_MODEL=<path> or populate \
             $HOME/.cache/models/thenlper--gte-small; skipping"
        );
        return;
    };
    let env = Environment::new().expect("env");
    // One owned stream for the device-input config (ORT replays the captured graph on it and
    // the lane refreshes its device buffer on the same stream). The baseline/host configs use ORT's
    // default stream.
    let stream = Arc::new(st_zrt::CudaStream::new(0).expect("cuda stream"));

    let base_sess =
        build_session(&env, &path, Cfg::Baseline, &stream).expect("baseline CUDA session");
    let hidden = base_sess.output_shape(0).expect("baseline output shape")[2];
    // Ground-truth correctness reference: a direct (no-lane) `Session::run`. The bind-once host lane
    // returns STALE output on CUDA for this model (a known reusable-IoBinding limitation), so it
    // cannot serve as the reference; `direct_run` is input-sensitive (verified below) and
    // unambiguous. The final input uses FINAL_OFFSET so it differs from the iter-0 capture input.
    let reference = direct_run(&base_sess, FINAL_OFFSET, BATCH, SEQ);
    let ref_sensitivity = max_abs_diff(
        &direct_run(&base_sess, 0, BATCH, SEQ),
        &direct_run(&base_sess, 50, BATCH, SEQ),
    );
    eprintln!(
        "reference: direct Session::run (input-sensitive: |direct(0)-direct(50)|={ref_sensitivity:.4})"
    );

    // Per-config latency + output. The baseline rebinds inputs each run (the correct no-graph,
    // fresh-input path — bind-once is stale on CUDA for this model). The host-input graph cannot
    // rebind (`cuda_graph` + rebind is rejected), so it is the naive stale path. The device-input
    // graph refreshes its device buffers each run on the owned stream.
    let (base_lat, base_out) = run_config(
        base_sess.clone(),
        Cfg::Baseline,
        &stream,
        hidden,
        BATCH,
        SEQ,
    );
    drop(base_sess);

    if matches!(
        std::env::var("ZRT_GTE_SKIP_HOST").as_deref(),
        Ok("1" | "true" | "yes")
    ) {
        let dev_sess = build_session(&env, &path, Cfg::DeviceInputGraph, &stream)
            .expect("device-input graph CUDA session");
        let (dev_lat, dev_out) = run_config(
            dev_sess,
            Cfg::DeviceInputGraph,
            &stream,
            hidden,
            BATCH,
            SEQ,
        );
        let base_diff = max_abs_diff(&base_out, &reference);
        let dev_diff = max_abs_diff(&dev_out, &reference);
        let dev_speedup = base_lat.median / dev_lat.median;
        println!();
        println!("=== CUDA-graph GTE device-input-only gate ===");
        println!("model  : {}", path.display());
        println!("bucket : batch={BATCH} seq={SEQ} hidden={hidden} (inputs CHANGE every iter)");
        println!(
            "baseline median µs={:.1} max|diff|={base_diff:.4}",
            base_lat.median
        );
        println!(
            "device-input graph min µs={:.1} median µs={:.1} speedup={dev_speedup:.2}× max|diff|={dev_diff:.4}",
            dev_lat.min, dev_lat.median
        );
        if dev_diff >= 1e-2 {
            panic!("device-input graph output diverged from reference: max|diff|={dev_diff:.4}");
        }
        return;
    }

    let host_sess = build_session(&env, &path, Cfg::HostInputGraph, &stream)
        .expect("host-input graph CUDA session");
    let (host_lat, host_out) = run_config(
        host_sess,
        Cfg::HostInputGraph,
        &stream,
        hidden,
        BATCH,
        SEQ,
    );
    let dev_sess = build_session(&env, &path, Cfg::DeviceInputGraph, &stream)
        .expect("device-input graph CUDA session");
    let (dev_lat, dev_out) = run_config(
        dev_sess,
        Cfg::DeviceInputGraph,
        &stream,
        hidden,
        BATCH,
        SEQ,
    );

    let base_diff = max_abs_diff(&base_out, &reference);
    let host_diff = max_abs_diff(&host_out, &reference);
    let dev_diff = max_abs_diff(&dev_out, &reference);
    let host_speedup = base_lat.median / host_lat.median;
    let dev_speedup = base_lat.median / dev_lat.median;

    println!();
    println!("=== CUDA-graph GTE 3-config gate ===");
    println!("model  : {}", path.display());
    println!("bucket : batch={BATCH} seq={SEQ} hidden={hidden} (inputs CHANGE every iter)");
    println!("reference: direct Session::run on the shared final input");
    println!(
        "{:<22}{:>9}{:>11}{:>17}{:>26}",
        "config", "min µs", "median µs", "speedup(median)", "max|diff| vs reference"
    );
    println!(
        "{:<22}{:>9.1}{:>11.1}{:>16.2}×{:>17.4} ({})",
        "1 baseline (rebind)",
        base_lat.min,
        base_lat.median,
        1.0,
        base_diff,
        if base_diff < 1e-2 { "correct" } else { "stale" }
    );
    println!(
        "{:<22}{:>9.1}{:>11.1}{:>16.2}×{:>17.4} ({})",
        "2 host-input graph",
        host_lat.min,
        host_lat.median,
        host_speedup,
        host_diff,
        if host_diff < 1e-2 { "correct" } else { "STALE" }
    );
    println!(
        "{:<22}{:>9.1}{:>11.1}{:>16.2}×{:>17.4} ({})",
        "3 device-input graph",
        dev_lat.min,
        dev_lat.median,
        dev_speedup,
        dev_diff,
        if dev_diff < 1e-2 { "correct" } else { "WRONG" }
    );

    let dev_correct = dev_diff < 1e-2;
    let host_stale = host_diff > 1.0;
    let base_correct = base_diff < 1e-2;
    let gate_met = dev_speedup >= 1.5;
    println!();
    println!(
        "host-input graph (naive): {} — max|diff|={host_diff:.4} (the stale-input limitation: a \
         captured graph bakes a device input pointer ORT does not refresh from host staging)",
        if host_stale { "STALE" } else { "fresh?" }
    );
    println!(
        "device-input graph: {} — max|diff|={dev_diff:.4} (the correct serving path: per-run H2D \
         refresh on the owned stream)",
        if dev_correct { "CORRECT" } else { "WRONG" }
    );
    println!(
        "gate: device-input correct-path speedup(median) = {dev_speedup:.2}× {} 1.5×",
        if gate_met { ">=" } else { "<" }
    );

    // The device-input path's correctness is invariant and must always hold; the 1.5× gate is
    // hardware/model/shape-dependent, so it is reported unless ZRT_GATE=strict opts into a hard fail.
    if !base_correct {
        eprintln!(
            "warning: baseline (rebind) did not match the reference (max|diff|={base_diff:.4}); the \
             no-graph reference is unreliable"
        );
    }
    if !dev_correct {
        panic!(
            "device-input graph output diverged from the reference: max|diff|={dev_diff:.4} >= 1e-2"
        );
    }
    if !host_stale {
        eprintln!(
            "warning: host-input graph did NOT read stale (max|diff|={host_diff:.4}); expected the \
             stale-input limitation"
        );
    }
    if matches!(std::env::var("ZRT_GATE").as_deref(), Ok("strict")) && !gate_met {
        panic!(
            "cuda-graph 1.5× gate NOT met on the correct (device-input) path: speedup={dev_speedup:.2}×"
        );
    }

    // ---- helpers ----

    /// Fill the 3 int64 GTE inputs (ids / attention_mask / token_type) with iter-varying token ids
    /// so the host-input bug is exposed (a constant input would mask it).
    fn fill_changing(lane: &mut ServingLane<i64, f32, 3, 1>, iter: usize) -> Result<()> {
        let off = iter as i64;
        for (i, v) in lane.input_mut_at::<0>()?.iter_mut().enumerate() {
            *v = ((i as i64 + off) % 30000) + 1;
        }
        for v in lane.input_mut_at::<1>()? {
            *v = 1; // attention_mask: all tokens valid
        }
        for v in lane.input_mut_at::<2>()? {
            *v = 0; // token_type_ids: single segment
        }
        Ok(())
    }

    /// Fill the shared "final" input (a fixed, iter-0-distinct pattern) used for the correctness
    /// comparison across all three configs.
    fn fill_final(lane: &mut ServingLane<i64, f32, 3, 1>) -> Result<()> {
        for (i, v) in lane.input_mut_at::<0>()?.iter_mut().enumerate() {
            *v = ((i as i64 + FINAL_OFFSET) % 30000) + 1;
        }
        for v in lane.input_mut_at::<1>()? {
            *v = 1;
        }
        for v in lane.input_mut_at::<2>()? {
            *v = 0;
        }
        Ok(())
    }

    fn run_config(
        session: Session, cfg: Cfg, stream: &Arc<CudaStream>, hidden: i64, batch: i64,
        seq: i64,
    ) -> (Latency, Vec<f32>) {
        let in_mem = MemoryInfo::cpu().expect("cpu in mem");
        let out_mem = MemoryInfo::cpu().expect("cpu out mem");
        let opts = match cfg {
            // Baseline: no graph, but rebind inputs each run — bind-once is stale on CUDA for this
            // model, so rebind is the correct no-graph fresh-input reference.
            Cfg::Baseline => DynamicIoOptions::new(4).with_rebind_inputs_each_run(true),
            Cfg::HostInputGraph => DynamicIoOptions::new(4).with_cuda_graph(true),
            Cfg::DeviceInputGraph => DynamicIoOptions::new(4)
                .with_cuda_graph(true)
                .with_device_inputs(0, &stream)
                .expect("device input options"),
        };
        let mut rt = DynamicIoRuntime::<i64, f32, 3, 1>::shared_session_with_options(
            session, in_mem, out_mem, 1, opts,
        )
        .expect("dynamic io runtime");
        let in_shapes: [&[i64]; 3] = [&[batch, seq], &[batch, seq], &[batch, seq]];
        let out_shapes: [&[i64]; 1] = [&[batch, seq, hidden]];
        if !matches!(cfg, Cfg::Baseline) {
            let mut plan = ServingShapePlan::builder();
            plan.add_shape(
                in_shapes.map(<[i64]>::to_vec),
                out_shapes.map(<[i64]>::to_vec),
                OutputPolicy::HostBuffer,
            );
            rt.install_shape_plan(Arc::new(plan.build().expect("shape plan")))
                .expect("install sealed graph shape plan");
        }

        // Warmup (also primes the bucket; for the graph configs the first run captures, the rest
        // replay).
        for iter in 0..WARMUP {
            rt.run_on(in_shapes, out_shapes, 0, |lane| {
                fill_changing(lane, iter)?;
                lane.run()
            })
            .unwrap_or_else(|err| panic!("warmup run cfg={cfg:?} iter={iter}: {err}"));
        }
        // Per-run timing with changing inputs: the baseline jitters from host launch/scheduling
        // variance while the captured paths are stable — min + median surface both the capability
        // and the stability win of capture.
        let mut times: Vec<f64> = Vec::with_capacity(ITERS);
        for iter in 0..ITERS {
            let t0 = Instant::now();
            rt.run_on(in_shapes, out_shapes, 0, |lane| {
                fill_changing(lane, iter)?;
                lane.run()
            })
            .unwrap_or_else(|err| panic!("timed run cfg={cfg:?} iter={iter}: {err}"));
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = times[0];
        let median = times[times.len() / 2];

        // Shared final input → output for the correctness comparison.
        let out = rt
            .run_on(in_shapes, out_shapes, 0, |lane| {
                fill_final(lane)?;
                lane.run()?;
                Ok(lane.output_at::<0>()?.to_vec())
            })
            .expect("final output run");
        (Latency { min, median }, out)
    }

    /// Direct (no-lane) `Session::run` ground-truth output for a given input_id offset.
    fn direct_run(sess: &Session, off: i64, batch: i64, seq: i64) -> Vec<f32> {
        let mem = MemoryInfo::cpu().expect("cpu mem");
        let n = (batch * seq) as usize;
        let ids: Vec<i64> = (0..n).map(|i| ((i as i64 + off) % 30000) + 1).collect();
        let mask = vec![1i64; n];
        let ttype = vec![0i64; n];
        let t_ids = Tensor::from_buffer(&ids, &[batch, seq], &mem).expect("ids tensor");
        let t_mask = Tensor::from_buffer(&mask, &[batch, seq], &mem).expect("mask tensor");
        let t_type = Tensor::from_buffer(&ttype, &[batch, seq], &mem).expect("ttype tensor");
        let inputs: [&dyn RunInput; 3] = [&t_ids, &t_mask, &t_type];
        let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
        sess.run(&inputs, &mut out).expect("direct run");
        out[0]
            .as_ref()
            .expect("direct output")
            .as_slice::<f32>()
            .expect("direct output read")
            .to_vec()
    }

    fn build_session(
        env: &Environment, path: &std::path::Path, cfg: Cfg, stream: &Arc<CudaStream>,
    ) -> Result<Session> {
        let base = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
        let opts = match cfg {
            Cfg::Baseline => base
                .with_cuda(CudaConfig::performance(0))
                .expect("append baseline CUDA config"),
            Cfg::HostInputGraph => base
                .with_cuda(CudaConfig::performance(0).with_cuda_graph(true))
                .expect("append host-input graph CUDA config"),
            Cfg::DeviceInputGraph => base
                .with_cuda(CudaConfig::graph_replay(0, stream).expect("graph config"))
                .expect("append device-input graph CUDA config"),
        };
        Session::new(env, path.to_str().unwrap(), opts)
    }
}

/// Max absolute element-wise difference of two f32 slices (0 for empty).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut m = 0.0_f64;
    for i in 0..n {
        m = m.max((a[i] as f64 - b[i] as f64).abs());
    }
    m
}

/// Read `name` as an i64, or `default` if unset/unparseable.
fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Resolve the GTE-small model: `ZRT_GTE_MODEL` env override, then the user's conventional
/// model cache (CUDA-optimized first, plain second), then a sibling `bench/models/gte-small.onnx`.
/// `None` when no candidate exists (bench skips). CUDA-graph capture requires every node on the
/// CUDA EP, so the CUDA-optimized export is preferred when both cache files exist.
fn gte_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    if let Ok(p) = std::env::var("ZRT_GTE_MODEL") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return Some(pb);
        }
    }

    let mut candidates = Vec::with_capacity(3);
    if let Some(home) = std::env::var_os("HOME") {
        let model_dir = PathBuf::from(home)
            .join(".cache")
            .join("models")
            .join("thenlper--gte-small");
        candidates.push(model_dir.join("model_cuda.onnx"));
        candidates.push(model_dir.join("model.onnx"));
    }
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bench")
            .join("models")
            .join("gte-small.onnx"),
    );
    candidates.into_iter().find(|p| p.exists())
}
