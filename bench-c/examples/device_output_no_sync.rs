//! Characterize host submission versus completion for device-resident CUDA graph outputs.
//! Run: cargo run --release --features cuda --example device_output_no_sync

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("device_output_no_sync requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use st_zrt::{
        BufferSpec, CompletionStatus, CudaConfig, CudaStream, DynamicIoOptions, DynamicIoRuntime,
        Environment, GraphOptimizationLevel, MemoryInfo, OutputPolicy, ServingShapePlan, Session,
        SessionOptions,
    };
    use std::{sync::Arc, time::Instant};

    let path = st_zrt_bench_c::models::ensure_relay("4m").expect("relay model");
    let env = Environment::new().expect("environment");
    let stream = Arc::new(CudaStream::new(0).expect("stream"));
    let options = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::graph_replay(0, &stream).expect("graph config"))
        .expect("CUDA provider");
    let session = Session::new(&env, path.to_str().expect("model path"), options).expect("session");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        MemoryInfo::cpu().expect("input memory"),
        MemoryInfo::cpu().expect("unused output descriptor"),
        1,
        DynamicIoOptions::new(1)
            .with_input_policy(BufferSpec::CUDA_PINNED)
            .with_cuda_graph(true)
            .with_device_inputs(0, &stream)
            .expect("device inputs")
            .with_device_outputs(true),
    )
    .expect("runtime");
    let mut plan = ServingShapePlan::builder();
    plan.add_shape(
        [vec![1, 1_048_576]],
        [vec![1, 1_048_576]],
        OutputPolicy::DeviceResident,
    );
    runtime
        .install_shape_plan(Arc::new(plan.build().expect("shape plan")))
        .expect("install plan");
    runtime
        .prime_bucket_enqueued([&[1, 1_048_576]], [&[1, 1_048_576]], 8)
        .expect("capture/warm graph");
    let bucket = runtime
        .prepared_bucket_id([&[1, 1_048_576]], [&[1, 1_048_576]])
        .expect("prepared bucket");

    let iterations = std::env::var("ST_ZRT_DEVICE_OUTPUT_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(200)
        .max(2);
    let mut issue_us = Vec::with_capacity(iterations);
    let mut completion_us = Vec::with_capacity(iterations);
    let mut queries = 0usize;
    for i in 0..iterations {
        let start = Instant::now();
        let mut issue_start = None;
        let mut run = runtime
            .enqueue_prepared(bucket, |lane| {
                lane.input_mut_at::<0>()?.fill(1.0 + i as f32 * 0.001);
                issue_start = Some(Instant::now());
                Ok(())
            })
            .expect("enqueue");
        issue_us.push(issue_start.expect("issue start").elapsed().as_secs_f64() * 1e6);
        let mut pending_queries = 0usize;
        loop {
            queries += 1;
            match run.try_complete().expect("completion query") {
                CompletionStatus::Ready => break,
                CompletionStatus::Pending => {
                    pending_queries += 1;
                    if pending_queries < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                },
            }
        }
        completion_us.push(start.elapsed().as_secs_f64() * 1e6);
        runtime
            .complete_owned(run, |lane| {
                std::hint::black_box(lane.device_output(0)?.shape());
                Ok(())
            })
            .expect("return lane");
    }
    issue_us.sort_by(f64::total_cmp);
    completion_us.sort_by(f64::total_cmp);
    println!(
        "device output: ORT issue p50={:.3}us p99={:.3}us; fill+complete p50={:.3}us p99={:.3}us; queries/run={:.1}",
        issue_us[iterations / 2],
        issue_us[iterations * 99 / 100],
        completion_us[iterations / 2],
        completion_us[iterations * 99 / 100],
        queries as f64 / iterations as f64,
    );

    let downstream = Arc::new(CudaStream::new(0).expect("downstream stream"));
    let mut chain_issue_us = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let mut issue_start = None;
        let run = runtime
            .enqueue_prepared(bucket, |lane| {
                lane.input_mut_at::<0>()?.fill(2.0 + i as f32 * 0.001);
                issue_start = Some(Instant::now());
                Ok(())
            })
            .expect("enqueue chained run");
        let chained = run
            .chain_on_stream(&downstream, |outputs, _stream| {
                std::hint::black_box(outputs[0].raw_typed_ptr()?);
                Ok(())
            })
            .map_err(|failure| failure.error)
            .expect("queue GPU chain");
        chain_issue_us.push(issue_start.expect("issue start").elapsed().as_secs_f64() * 1e6);
        let run = chained.synchronize().expect("complete GPU chain");
        runtime
            .complete_owned(run, |_| Ok(()))
            .expect("return chained lane");
    }
    chain_issue_us.sort_by(f64::total_cmp);
    println!(
        "GPU chain issue (ORT + stream wait + downstream event): p50={:.3}us p99={:.3}us",
        chain_issue_us[iterations / 2], chain_issue_us[iterations * 99 / 100],
    );
}
