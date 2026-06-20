//! CUDA execution-provider inference coverage.
//!
//! With only `ep`, this file compiles and skips if CUDA is unavailable. With the `cuda` feature,
//! CUDA availability is part of the release gate: session creation and runs must succeed.

#![cfg(feature = "ep")]

use std::sync::Arc;

#[cfg(feature = "cuda")]
use st_zrt::AllocatedTensor;
use st_zrt::{
    CudaArenaExtendStrategy, CudaCudnnConvAlgoSearch, CudaProviderOptions, Environment,
    GraphOptimizationLevel, IoBinding, MemoryInfo, OutputValue, OwnedValue, PreparedRun, Session,
    SessionOptions, Tensor, ZrtRuntime,
};

fn mnist_path() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx");
    if path.exists() {
        Some(path)
    } else if cfg!(feature = "cuda") {
        panic!("cuda release gate requires bench/models/mnist.onnx");
    } else {
        eprintln!("skip — mnist.onnx not cached");
        None
    }
}

fn cpu_session(env: &Environment, path: &std::path::Path) -> Session {
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    Session::new(env, path.to_str().unwrap(), opts).expect("cpu session")
}

fn cuda_session(env: &Environment, path: &std::path::Path) -> Option<Session> {
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda_options(
            CudaProviderOptions::new()
                .device_id(0)
                .arena_extend_strategy(CudaArenaExtendStrategy::NextPowerOfTwo)
                .cudnn_conv_algo_search(CudaCudnnConvAlgoSearch::Exhaustive)
                .do_copy_in_default_stream(true)
                .use_tf32(true),
        )
        .expect("append CUDA options");
    match Session::new(env, path.to_str().unwrap(), opts) {
        Ok(s) => Some(s),
        Err(e) if cfg!(feature = "cuda") => panic!("CUDA EP unavailable in cuda build: {e}"),
        Err(e) => {
            eprintln!("CUDA EP unavailable on this build/host — skipping ({e})");
            None
        },
    }
}

fn zero_input<'a>(mem: &MemoryInfo, buf: &'a [f32]) -> Tensor<'a> {
    Tensor::from_buffer(buf, &[1, 1, 28, 28], mem).expect("zero-copy input")
}

fn run_cpu_reference(sess: &Session, input: &Tensor<'_>) -> Vec<f32> {
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[input], &mut out).expect("cpu run");
    out[0]
        .as_ref()
        .expect("cpu output")
        .as_slice()
        .expect("cpu output read")
        .to_vec()
}

fn assert_logits_close(cpu: &[f32], got: &[f32]) {
    assert_eq!(got.len(), 10, "MNIST output is 10 logits");
    for (a, b) in cpu.iter().zip(got.iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "CUDA vs CPU logit mismatch: cpu={a} gpu={b}"
        );
    }
}

#[test]
fn cuda_ep_regular_run_matches_cpu() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut out: Vec<Option<OwnedValue>> = (0..cuda.output_count()).map(|_| None).collect();
    cuda.run(&[&input], &mut out).expect("cuda run");
    let got = out[0]
        .as_ref()
        .expect("cuda output")
        .as_slice::<f32>()
        .expect("cuda output read");
    assert_logits_close(&cpu_logits, got);
}

#[test]
fn cuda_prepared_run_matches_cpu() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut run: PreparedRun<'_, '_> = cuda.prepare_run(&[&input]).expect("prepare run");
    run.run().expect("prepared cuda run");
    let got = run
        .output(0)
        .expect("prepared output index")
        .expect("prepared output")
        .as_slice::<f32>()
        .expect("prepared output read");
    assert_logits_close(&cpu_logits, got);
}

#[test]
fn cuda_iobinding_cpu_output_matches_cpu() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut out_buf = vec![0.0_f32; 10];
    let out = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("cpu output value");
    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output(cuda.output_name(0).expect("output name"), &out)
        .expect("bind output");
    cuda.run_binding(&binding).expect("cuda iobinding run");
    binding.synchronize_outputs().expect("sync outputs");
    assert_logits_close(&cpu_logits, out.as_slice::<f32>().expect("read cpu output"));
}

#[test]
fn cuda_iobinding_device_output_reports_cuda_memory() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let cpu_mem = MemoryInfo::cpu().expect("cpu mem");
    let cuda_mem = MemoryInfo::cuda(0).expect("cuda memory info");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&cpu_mem, &buf);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_device(cuda.output_name(0).expect("output name"), &cuda_mem)
        .expect("bind cuda output device");
    cuda.run_binding(&binding).expect("cuda device-output run");
    binding.synchronize_outputs().expect("sync cuda outputs");

    let vals = binding.output_values().expect("output values");
    assert_eq!(vals.len(), 1);
    let info = vals[0].memory_info().expect("output memory info");
    assert_eq!(info.name, "Cuda");
    assert_eq!(info.device_id, 0);
    assert!(
        vals[0].as_slice::<f32>().is_err(),
        "device-resident output must not expose a Rust slice"
    );
}

#[test]
#[cfg(feature = "cuda")]
fn cuda_allocated_output_tensor_binds_and_reports_cuda_memory() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let cpu_mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&cpu_mem, &buf);

    let cuda = cuda_session(&env, &path).expect("cuda session");
    let out = AllocatedTensor::<f32>::cuda(&cuda, 0, &[1, 10]).expect("cuda output tensor");
    assert_eq!(out.memory_info().expect("allocated mem").name, "Cuda");

    let mut binding = IoBinding::new(&cuda).expect("binding");
    binding
        .bind_input(cuda.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_allocated(cuda.output_name(0).expect("output name"), &out)
        .expect("bind allocated cuda output");
    cuda.run_binding(&binding)
        .expect("allocated cuda output run");
    binding
        .synchronize_outputs()
        .expect("sync allocated output");
    assert!(!out.raw_typed_ptr().expect("device pointer").is_null());
    assert!(out.as_slice().is_err());
}

#[test]
fn cuda_zrt_runtime_shared_session_matches_cpu() {
    let Some(path) = mnist_path() else { return };
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = zero_input(&mem, &buf);

    let cpu = cpu_session(&env, &path);
    let cpu_logits = run_cpu_reference(&cpu, &input);

    let Some(cuda) = cuda_session(&env, &path) else {
        return;
    };
    let mut runtime =
        ZrtRuntime::<f32>::shared_session(Arc::new(cuda), &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 1)
            .expect("cuda runtime");
    let got = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").to_vec())
        })
        .expect("runtime run");
    assert_logits_close(&cpu_logits, &got);
}
