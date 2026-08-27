#![cfg(feature = "cuda")]

use st_zrt::{
    CudaConfig, Environment, GraphOptimizationLevel, Session, SessionOptions,
};

#[test]
fn cuda_stream_is_visible_from_bench_c() {
    let stream = st_zrt::CudaStream::new(0).expect("cuda stream");
    stream.synchronize().expect("stream synchronize");
}

fn mnist_path() -> std::path::PathBuf {
    st_zrt_bench_c::models::ensure_mnist().expect("mnist")
}

#[test]
fn cuda_explicit_session_is_visible_from_bench_c() {
    let env = Environment::new().expect("env");
    let model = mnist_path();

    let explicit = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::performance(0))
        .expect("typed CUDA config");
    Session::new(&env, model.to_str().unwrap(), explicit).expect("explicit cuda session");
}

#[test]
fn cuda_typed_session_is_visible_from_bench_c() {
    let env = Environment::new().expect("env");
    let model = mnist_path();

    let preset = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_cuda(CudaConfig::performance(0))
        .expect("typed cuda options");
    Session::new(&env, model.to_str().unwrap(), preset).expect("preset cuda session");
}
