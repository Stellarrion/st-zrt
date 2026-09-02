//! End-to-end smoke tests for the variant-C safe layer against the real engine.
//!
//! These load `bench/models/mnist.onnx` (ort's hosted MNIST test model: one float
//! input [1,1,28,28] → one float output [1,10]) and exercise the full path:
//! Environment → SessionOptions → MemoryInfo → Session (pre-marshaled names) →
//! Tensor::from_buffer (zero-copy input) → run → OwnedValue::as_slice (zero-copy output).

use st_zrt::{
    AllocatedTensor, Allocator, AllocatorType, ArenaCfg, ArenaExtendStrategy, BufferSpec,
    BufferStorage, DynamicIoOptions, DynamicIoRuntime, EnvCreationOptions, Environment,
    GraphOptimizationLevel, IoBinding, IoDirection, LogRecord, LoggingLevel, MemType, MemoryClass,
    MemoryInfo, ModelMetadata, OutputValue, OwnedInitializer, OwnedValue,
    PrepackedWeightsContainer, RunOptions, Runtime, RuntimeMode, ServingLane, Session,
    SessionOptions, ShapeSpec, StaticIoRuntime, Tensor, TensorBuffer, ThreadingOptions, sys,
};
use std::sync::{Arc, Mutex};

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn mnist_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx")
}

fn relay_path(label: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join(format!("relay_{label}.onnx"))
}

/// Load the MNIST session at opt-level All, or skip the caller (returns None) if the model
/// isn't cached. `env` is owned by the caller and must outlive the returned Session (ORT
/// sessions reference the Env's thread pools/allocator; releasing the Env first is a UAF).
fn mnist_session(env: &Environment) -> Option<(MemoryInfo, Session)> {
    let path = mnist_path();
    if !path.exists() {
        return None;
    }
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(env, path.to_str().unwrap(), opts).expect("session");
    Some((mem, sess))
}

#[test]
fn session_clone_keeps_native_session_and_environment_alive() {
    let env = Environment::new().expect("env");
    let Some((mem, session)) = mnist_session(&env) else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };
    let shared = session.clone();
    drop(session);
    drop(env);

    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs = [None];
    shared
        .run(&[&input], &mut outputs)
        .expect("run through surviving clone");
    assert_eq!(outputs[0].as_ref().unwrap().element_count(), 10);
}

#[test]
fn session_owned_allocator_tensor_outlives_public_session_handles() {
    let env = Environment::new().expect("env");
    let Some((mem, session)) = mnist_session(&env) else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };
    let mut tensor =
        AllocatedTensor::<f32>::for_session(&session, &mem, &[16]).expect("session-owned tensor");
    drop(session);
    drop(env);

    tensor
        .as_mut_slice()
        .expect("guarded tensor remains usable")
        .fill(7.0);
    assert_eq!(tensor.as_slice().expect("guarded tensor read"), &[7.0; 16]);
    drop(tensor);
}

#[test]
fn io_binding_keeps_its_session_alive_and_rejects_another_session() {
    let env = Environment::new().expect("env");
    let Some((_mem, session_a)) = mnist_session(&env) else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };
    let Some((_mem, session_b)) = mnist_session(&env) else {
        unreachable!("the model existed for the first session")
    };
    let binding = IoBinding::new(&session_a).expect("binding");
    // SAFETY: this call is expected to fail its session-identity check before entering ORT.
    let err = unsafe { session_b.run_binding_unsynchronized(&binding) }
        .expect_err("cross-session binding must be rejected before ORT");
    assert!(err.to_string().contains("different Session"));

    drop(session_a);
    drop(session_b);
    drop(env);
    assert!(
        binding
            .output_names()
            .expect("binding remains valid")
            .is_empty(),
        "fresh binding should have no output names"
    );
    drop(binding);
}

#[test]
fn lora_adapter_from_array_rejects_invalid_bytes() {
    // A LoRA adapter must be a valid ONNX adapter document; garbage bytes must surface as an
    // ORT error (never a panic), and the failed-construction path must drop cleanly (no handle
    // leaks — `create` returns `Err` before `Self` is ever built, so `Drop` does not run).
    let _env = Environment::new().expect("env");
    let bogus = b"not an onnx lora adapter";
    let err = st_zrt::LoraAdapter::from_array(bogus)
        .err()
        .expect("invalid lora adapter bytes must error, not panic");
    assert!(
        !err.to_string().is_empty(),
        "expected a descriptive ORT error message"
    );
}

#[test]
fn compatibility_info_preflight_before_load() {
    // The compat pre-flight answers "will this model run on EP X?" without loading it. For the
    // always-registered CPU EP it returns a non-empty info blob for a valid model and surfaces an
    // ORT error (never a panic) for unparsable bytes.
    let _env = Environment::new().expect("env");
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping compat pre-flight — mnist.onnx absent");
        return;
    }
    let bytes = std::fs::read(&path).expect("read mnist bytes");
    // MNIST is a standard (non-precompiled) ONNX model, so it carries no EP compatibility
    // metadata — the pre-flight returns Ok(None) (a null out-pointer), proving the parse +
    // allocator-free path works without needing a precompiled fixture.
    assert_eq!(
        st_zrt::compatibility_info_from_bytes(&bytes, "CPUExecutionProvider").expect("ok"),
        None,
        "standard model has no precompiled compat info"
    );
    assert_eq!(
        st_zrt::compatibility_info_from_path(path.to_str().unwrap(), "CPUExecutionProvider")
            .expect("ok"),
        None,
        "standard model (path) has no precompiled compat info"
    );
    // Garbage model bytes must surface as an ORT error, never a panic.
    assert!(
        st_zrt::compatibility_info_from_bytes(b"not an onnx model", "CPUExecutionProvider")
            .is_err()
    );
}

#[test]
fn external_initializer_info_round_trips() {
    // The info handle just describes a (path, offset, size) external-data region; the getters
    // must round-trip the construction values exactly.
    let info = st_zrt::ExternalInitializerInfo::new("/tmp/weights.bin", 4096, 1024)
        .expect("create external initializer info");
    assert_eq!(info.file_path().expect("path"), "/tmp/weights.bin");
    assert_eq!(info.file_offset(), 4096);
    assert_eq!(info.byte_size(), 1024);
}

#[test]
fn external_initializers_batch_replaces_external_backing() {
    // AddExternalInitializers replaces an initializer the model marks external-data with an
    // in-memory OrtValue (name/shape/dtype must match). external_add.onnx's C is external
    // (its .data holds [2,2,2,2]); we supply [7,7,7,7] instead — no .data file needed — and the
    // run resolves Y = X + C with the supplied value.
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let onnx = fixture_dir.join("external_add.onnx");
    if !onnx.exists() {
        eprintln!(
            "skipping — external_add fixture absent \
             (regenerate: python3 tests/fixtures/gen_external_data.py)"
        );
        return;
    }
    let tmp = std::env::temp_dir().join(format!("st-zrt-ext-batch-{}.onnx", std::process::id()));
    std::fs::copy(&onnx, &tmp).expect("copy onnx to temp (no .data sibling)");

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let c = TensorBuffer::from_vec(vec![7.0_f32; 4], &[4], &mem).expect("C buffer");
    let init = OwnedInitializer::tensor("C", c).expect("initializer");
    let sess = Session::new_with_external_initializers(
        &env,
        tmp.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
        vec![init],
    )
    .expect("session with batch external initializer");
    let x = vec![1.0_f32; 4];
    let input = Tensor::from_buffer(&x, &[4], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y, &[8.0_f32; 4], "1 + supplied C(=7)");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn external_initializer_files_in_memory() {
    // The model's C initializer lives in a sibling .data file. Copy the .onnx to a temp dir
    // WITHOUT the .data sibling, then supply C's bytes from memory — session creation succeeds
    // only via AddExternalInitializersFromFilesInMemory.
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");
    let onnx = fixture_dir.join("external_add.onnx");
    let data = fixture_dir.join("external_add.onnx.data");
    if !onnx.exists() || !data.exists() {
        eprintln!(
            "skipping — external_add fixture absent \
             (regenerate: python3 tests/fixtures/gen_external_data.py)"
        );
        return;
    }
    let data_bytes = std::fs::read(&data).expect("read .data");
    let tmp = std::env::temp_dir().join(format!("st-zrt-ext-add-{}.onnx", std::process::id()));
    std::fs::copy(&onnx, &tmp).expect("copy onnx to temp (no .data sibling)");

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let sess = Session::new_with_external_initializer_files(
        &env,
        tmp.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
        vec![("external_add.onnx.data".to_string(), data_bytes)],
    )
    .expect("session via in-memory external initializer file");
    let x = vec![1.0_f32; 4];
    let input = Tensor::from_buffer(&x, &[4], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y, &[3.0_f32; 4], "1 + externalized C(=2)");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn run_options_profiling_and_log_getters_round_trip() {
    // Pure configuration composes without FFI; freezing applies every field once.
    let config = RunOptions::new()
        .with_log_severity(LoggingLevel::Error)
        .with_log_verbosity(3)
        .with_profiling("st-zrt-profile");
    assert_eq!(
        config.log_severity().expect("config severity"),
        Some(LoggingLevel::Error)
    );
    assert_eq!(config.log_verbosity(), Some(3));
    assert_eq!(config.profiling_prefix(), Some("st-zrt-profile"));
    let opts = config.freeze().expect("freeze run opts");
    assert_eq!(
        opts.log_severity().expect("get severity"),
        LoggingLevel::Error
    );
    assert_eq!(opts.log_verbosity().expect("get verbosity"), 3);
}

#[test]
fn string_tensor_element_access_round_trips() {
    // Per-element read/write on a string tensor. The two write paths (NUL-terminated and raw
    // UTF-8 bytes) and the read path must all agree.
    let mut t =
        st_zrt::StringTensor::new(&["alpha", "beta", "gamma"], &[3]).expect("string tensor");
    assert_eq!(t.element(0).expect("elem 0"), "alpha");
    t.set_element(1, "BETA").expect("set elem 1");
    assert_eq!(t.element(1).expect("elem 1"), "BETA");
    // Raw UTF-8 bytes, no NUL terminator (GetResizedStringTensorElementBuffer path).
    t.set_element_utf8(2, b"GAMMA").expect("set elem 2 utf8");
    assert_eq!(t.element(2).expect("elem 2"), "GAMMA");
}

#[test]
fn hardware_enumeration_and_ep_incompatibility_details() {
    // A CPU host enumerates its hardware devices; the always-registered CPU EP yields
    // (compatible, empty) incompatibility details, while an EP with no registered factory
    // surfaces an ORT error. Together these exercise all 7 hardware/incompatibility accessors.
    let env = Environment::new().expect("env");
    let n = st_zrt::num_hardware_devices(&env).expect("num hardware devices");
    assert!(n >= 1, "a CPU host enumerates at least one hardware device");
    let devs = st_zrt::hardware_devices(&env).expect("hardware devices");
    assert_eq!(devs.len(), n);

    // CPU EP applies to a CPU device -> Some(details); all three getters must succeed.
    let det =
        st_zrt::hardware_device_ep_incompatibility_details(&env, "CPUExecutionProvider", &devs[0])
            .expect("incompat probe")
            .expect("CPU EP on a CPU device yields details");
    // reasons=0 / code=0 / empty notes means "compatible"; we only require the getters to run.
    let _ = det.reasons_bitmask().expect("reasons bitmask");
    let _ = det.error_code().expect("error code");
    let _ = det.notes().expect("notes");

    // An EP with no registered factory on this build must surface an ORT error, not a panic.
    assert!(
        st_zrt::hardware_device_ep_incompatibility_details(&env, "FakeExecutionProvider", &devs[0])
            .is_err()
    );
}

#[test]
fn diagnostics_providers_and_build_info() {
    let providers = st_zrt::available_providers().expect("providers");
    assert!(
        providers.iter().any(|p| p == "CPUExecutionProvider"),
        "CPU EP must be available: {providers:?}"
    );
    let info = st_zrt::build_info().expect("build info");
    assert!(!info.is_empty(), "build info must be non-empty");
    // GPU device id get/set are best-effort on a CPU host (may error); exercise the FFI only.
    let _ = st_zrt::set_current_gpu_device_id(0);
    let _ = st_zrt::current_gpu_device_id();
}

#[test]
fn environment_telemetry_toggle() {
    let env = Environment::new().expect("env");
    env.enable_telemetry_events().expect("enable telemetry");
    env.disable_telemetry_events().expect("disable telemetry");
}

#[test]
fn environment_language_projection_round_trips() {
    let env = Environment::new().expect("env");
    env.set_language_projection(st_zrt::LanguageProjection::Python)
        .expect("set projection");
    // Setting a different projection must also succeed (the FFI accepts any known value).
    env.set_language_projection(st_zrt::LanguageProjection::CPlusPlus)
        .expect("c++ projection");
}

#[test]
fn environment_per_session_thread_pool_callbacks_reaches_ffi() {
    // The stock ORT release is not built with --enable_session_threadpool_callbacks, so the FFI
    // returns a clean INVALID_ARGUMENT. We assert that build-flag error to prove the call reached
    // ORT with a well-formed versioned config (no UB); a custom ORT build with the flag would
    // accept it.
    let env = Environment::new().expect("env");
    let cfg = st_zrt::ThreadPoolCallbacksConfig::new();
    let err = env
        .set_per_session_thread_pool_callbacks(&cfg)
        .expect_err("stock ORT rejects per-session callbacks");
    assert!(
        err.to_string().contains("session_threadpool_callbacks"),
        "expected the build-flag error, got: {err}"
    );
}

#[test]
fn session_set_ep_dynamic_options_reaches_ffi() {
    // SetEpDynamicOptions on a CPU EP typically returns a clean ORT error (the CPU EP has no
    // dynamic options) — the test asserts only that the call reaches the FFI and returns a
    // well-formed Result (never panics / no UB).
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("overridable_add.onnx");
    if !path.exists() {
        eprintln!(
            "skipping — overridable_add.onnx absent \
             (regenerate: python3 tests/fixtures/gen_overridable.py)"
        );
        return;
    }
    let env = Environment::new().expect("env");
    let sess = Session::new(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
    )
    .expect("session");
    match sess.set_ep_dynamic_options(&[("device_id", "0")]) {
        Ok(()) => eprintln!("set_ep_dynamic_options: accepted"),
        Err(e) => eprintln!("set_ep_dynamic_options: clean ORT error ({e})"),
    }
}

#[test]
fn memory_info_v2_and_equals_round_trip() {
    // CreateMemoryInfo_V2 (richer constructor) + CompareMemoryInfo. Pure — no global state.
    let v2 = MemoryInfo::new_v2(
        "Cpu",
        st_zrt::MemoryInfoDeviceType::Cpu,
        0,
        0,
        st_zrt::DeviceMemoryType::Default,
        0,
        AllocatorType::Device,
    )
    .expect("new_v2");
    assert_eq!(v2.name().expect("name"), "Cpu");
    assert_eq!(v2.class(), MemoryClass::Cpu);
    assert!(v2.is_host_accessible());
    assert_eq!(v2.snapshot().expect("snapshot").class, MemoryClass::Cpu);

    let cpu1 = MemoryInfo::cpu().expect("cpu1");
    let cpu2 = MemoryInfo::cpu().expect("cpu2");
    assert!(
        cpu1.equals(&cpu2).expect("cpu==cpu"),
        "two cpu() memory infos must compare equal"
    );
    let cuda = MemoryInfo::cuda(0).expect("cuda");
    assert_eq!(cuda.class(), MemoryClass::CudaDevice);
    assert!(!cuda.is_host_accessible());
    let cuda_pinned = MemoryInfo::cuda_pinned(0).expect("cuda pinned");
    assert_eq!(cuda_pinned.class(), MemoryClass::CudaPinned);
    assert!(cuda_pinned.is_host_accessible());
    let cuda_shared = MemoryInfo::new_v2(
        "CudaShared",
        st_zrt::MemoryInfoDeviceType::Gpu,
        0,
        0,
        st_zrt::DeviceMemoryType::HostAccessible,
        0,
        AllocatorType::Device,
    )
    .expect("CUDA shared host memory");
    assert_eq!(cuda_shared.class(), MemoryClass::CudaPinned);
    assert!(cuda_shared.is_host_accessible());

    let other = MemoryInfo::new_v2(
        "ExampleNpu",
        st_zrt::MemoryInfoDeviceType::Npu,
        0,
        0,
        st_zrt::DeviceMemoryType::Default,
        0,
        AllocatorType::Device,
    )
    .expect("other device");
    assert_eq!(other.class(), MemoryClass::OtherDevice);
    assert!(!other.is_host_accessible());
    assert!(
        !cpu1.equals(&cuda).expect("cpu!=cuda"),
        "cpu and cuda memory infos must differ"
    );
}

#[test]
fn allocator_memory_info_snapshot_is_host_accessible() {
    // AllocatorGetInfo: the default allocator's memory info is CPU / host-accessible.
    let alloc = Allocator::get_default().expect("default alloc");
    let snap = alloc.memory_info().expect("allocator memory info");
    assert!(
        snap.is_host_accessible(),
        "default allocator is host-accessible"
    );
    assert_eq!(snap.name, "Cpu");
}

#[test]
fn environment_allocator_v2_registration_reaches_ffi() {
    // CreateAndRegisterAllocatorV2 + RegisterAllocator + UnregisterAllocator. These mutate the
    // env's allocator registry; we assert only that the FFI is reached cleanly (Ok or a clean ORT
    // error), not a specific outcome, to avoid coupling to ORT's allocator-registry state.
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("mem");
    let cfg =
        ArenaCfg::new(usize::MAX, ArenaExtendStrategy::NextPowerOfTwo, 1024, 0).expect("arena cfg");
    match env.create_and_register_allocator_v2("CPUExecutionProvider", &mem, &cfg, &[]) {
        Ok(()) => eprintln!("create_and_register_allocator_v2: accepted"),
        Err(e) => eprintln!("create_and_register_allocator_v2: clean ORT error ({e})"),
    }
    let alloc = Allocator::get_default().expect("default alloc");
    let _ = env.register_existing_allocator(&alloc);
    let _ = env.unregister_allocator(&mem);
}

#[test]
fn environment_ep_library_registration_reaches_ffi() {
    // RegisterExecutionProviderLibrary + UnregisterExecutionProviderLibrary. A bogus library path
    // surfaces as a clean error; unregistering an unknown name likewise. Both prove the FFI was
    // reached without UB.
    let env = Environment::new().expect("env");
    let _ = env.register_execution_provider_library("bogus_ep", "/nonexistent/libep.so");
    let _ = unsafe { env.unregister_execution_provider_library("bogus_ep") };
}

#[test]
fn run_options_run_tag_and_unset_terminate() {
    let config = RunOptions::new().with_run_tag("batch-42");
    assert_eq!(config.run_tag(), Some("batch-42"));
    let opts = config.freeze().expect("freeze run opts");
    assert_eq!(opts.run_tag().expect("tag"), "batch-42");
    opts.unset_terminate().expect("unset terminate");
}

#[test]
fn session_overridable_initializers_introspect() {
    // overridable_add.onnx declares C as both a graph input and an initializer, so ORT marks it
    // overridable (count == 1). Exercises the count/name/type-info accessors.
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("overridable_add.onnx");
    if !path.exists() {
        eprintln!(
            "skipping — overridable_add.onnx absent \
             (regenerate: python3 tests/fixtures/gen_overridable.py)"
        );
        return;
    }
    let env = Environment::new().expect("env");
    let sess = Session::new(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
    )
    .expect("session");
    assert_eq!(
        sess.overridable_initializer_count().expect("count"),
        1,
        "C is the single overridable initializer"
    );
    assert_eq!(sess.overridable_initializer_name(0).expect("name 0"), "C");
    let ti = sess
        .overridable_initializer_type_info(0)
        .expect("type info 0");
    assert_eq!(ti.onnx_type().expect("onnx type"), sys::OnnxType::Tensor);
}

#[test]
fn iobinding_output_names_round_trips() {
    // `GetBoundOutputNames` (idx 139): after binding an output, the binding must report it back.
    // Uses a committed fixture so the test always runs (no skip).
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("overridable_add.onnx");
    if !path.exists() {
        eprintln!(
            "skipping — overridable_add.onnx absent \
             (regenerate: python3 tests/fixtures/gen_overridable.py)"
        );
        return;
    }
    let env = Environment::new().expect("env");
    let sess = Session::new(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
    )
    .expect("session");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let mut binding = IoBinding::new(&sess).expect("binding");
    let out0 = sess.output_name(0).expect("output name 0");
    binding
        .bind_output_device(out0, &mem)
        .expect("bind output to device");
    let names = binding.output_names().expect("output names");
    assert!(
        names.iter().any(|n| n.as_str() == out0),
        "bound output {out0:?} must be reported by output_names(): {names:?}"
    );
}

#[test]
fn run_options_config_entry_round_trips() {
    let _env = Environment::new().expect("env");
    let config = RunOptions::new()
        .with_config("test.key", "value-1")
        .expect("add");
    assert_eq!(config.config_entry("unset.key"), None);
    assert_eq!(config.config_entry("test.key"), Some("value-1"));
    let opts = config.freeze().expect("freeze");
    assert_eq!(
        opts.config_entry("test.key").expect("get"),
        Some("value-1".to_string())
    );
}

#[test]
fn run_options_disable_execution_provider_sync_round_trips() {
    let _env = Environment::new().expect("env");
    let config = RunOptions::enqueued(7);
    assert_eq!(
        config.config_entry("disable_synchronize_execution_providers"),
        Some("1")
    );
    assert_eq!(config.config_entry("gpu_graph_id"), Some("7"));
    assert_eq!(config.graph_id(), Some(7));
    let opts = config.freeze().expect("freeze enqueued options");
    assert_eq!(
        opts.config_entry("disable_synchronize_execution_providers")
            .expect("get disable sync"),
        Some("1".to_string())
    );
    assert_eq!(
        opts.config_entry("gpu_graph_id").expect("get graph id"),
        Some("7".to_string())
    );

    let config = RunOptions::new().with_disable_ep_sync(false);
    assert_eq!(
        config.config_entry("disable_synchronize_execution_providers"),
        Some("0")
    );
}

#[test]
fn session_options_has_config_entry() {
    let opts = SessionOptions::new()
        .with_config_entry("my.custom.key", "v")
        .expect("with config");
    assert!(
        opts.has_config_entry("my.custom.key").expect("has"),
        "a set config entry must be present"
    );
}

#[test]
fn key_value_pairs_round_trip() {
    let mut kv = st_zrt::KeyValuePairs::new().expect("new kvps");
    assert_eq!(kv.get("absent").expect("get absent"), None);
    kv.add("k", "v1").expect("add");
    assert_eq!(kv.get("k").expect("get"), Some("v1".to_string()));
    kv.add("k", "v2").expect("replace");
    assert_eq!(kv.get("k").expect("get replaced"), Some("v2".to_string()));
    kv.remove("k").expect("remove");
    assert_eq!(kv.get("k").expect("get removed"), None);
}

#[test]
fn mnist_end_to_end() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping mnist_end_to_end — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem info");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");

    assert_eq!(sess.input_count(), 1, "MNIST has 1 input");
    assert_eq!(sess.output_count(), 1, "MNIST has 1 output");
    assert_eq!(
        sess.input_meta(0).expect("input meta"),
        (
            sys::OnnxType::Tensor,
            st_zrt::ElementType::Float,
            Some(28 * 28)
        )
    );
    assert_eq!(
        sess.output_meta(0).expect("output meta"),
        (sys::OnnxType::Tensor, st_zrt::ElementType::Float, Some(10))
    );
    eprintln!("input[0]  = {}", sess.input_name(0).expect("input name"));
    eprintln!("output[0] = {}", sess.output_name(0).expect("output name"));

    // Zero-copy input: wrap a caller-owned buffer; the engine reads it in place.
    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("zero-copy input");

    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");

    let out = outputs[0].as_ref().expect("output 0 present");
    let logits: &[f32] = out.as_slice().expect("zero-copy output read");
    assert_eq!(logits.len(), 10, "MNIST output should be 10 logits");
    assert!(
        out.get_value(0).is_err(),
        "tensor output has no child values"
    );
    eprintln!("logits: {:?}", logits);
}

#[test]
fn public_misuse_returns_errors_not_panics() {
    let Some((mem, sess)) = mnist_session(&Environment::new().expect("env")) else {
        eprintln!("skipping misuse test — mnist.onnx absent");
        return;
    };
    let buf: Vec<f32> = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("zero-copy input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();

    assert!(sess.run(&[], &mut outputs).is_err());
    assert!(sess.prepare_run(&[]).is_err());
    assert!(sess.run(&[&input], &mut []).is_err());
    assert!(
        sess.prepare_tensor_io_lane::<f32>(&mem, &[], &[&[1, 10]])
            .is_err()
    );
    assert!(sess.input_name(sess.input_count()).is_err());
    assert!(sess.output_name(sess.output_count()).is_err());
    assert!(sess.input_meta(sess.input_count()).is_err());
    assert!(sess.output_meta(sess.output_count()).is_err());
    assert!(sess.input_shape(sess.input_count()).is_err());
    assert!(sess.output_shape(sess.output_count()).is_err());
    assert!(sess.input_symbolic_dims(sess.input_count()).is_err());
    assert!(sess.output_symbolic_dims(sess.output_count()).is_err());
}

#[test]
fn run_is_shared_reentrant() {
    // run(&self) must be safe to call concurrently from multiple threads — ORT's Run
    // is thread-safe on a session, and our safe layer must not introduce shared state.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping reentrancy test — mnist.onnx absent");
        return;
    }

    let env = Arc::new(Environment::new().unwrap());
    let sess = {
        let opts = SessionOptions::new();
        Arc::new(Session::new(&env, path.to_str().unwrap(), opts).unwrap())
    };

    fn run_once(sess: &Session) -> usize {
        let mem = MemoryInfo::cpu().unwrap();
        let buf = vec![0.0_f32; 784];
        let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).unwrap();
        let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
        sess.run(&[&input], &mut out).unwrap();
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len()
    }

    let s2 = sess.clone();
    let h = std::thread::spawn(move || run_once(&s2));
    let main_n = run_once(&sess);
    let thread_n = h.join().unwrap();
    assert_eq!(main_n, 10);
    assert_eq!(thread_n, 10);
    eprintln!("concurrent shared-ref runs OK (both returned 10 logits)");
}

#[test]
fn session_outlives_env_drop() {
    // The Env is dropped right after Session construction — the exact pattern behind the
    // historical ">4MB segfault": ORT sessions reference the Env's thread pools/allocator, so
    // running a Session after its Env was freed corrupted the heap (a use-after-free). Session
    // now holds an `Arc` ref to the Env, keeping it alive for the Session's whole lifetime, so
    // this must run cleanly. Reverting that fix would make this a UAF again. (The large-tensor,
    // sustained-load variant of the same invariant is guarded by bench-c/benches/crash_repro.rs.)
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }
    let sess = {
        let env = Environment::new().expect("env");
        Session::new(&env, path.to_str().unwrap(), SessionOptions::new()).expect("session")
        // `env` drops here; the Env survives via the Session's Arc ref.
    };
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)
        .expect("run after Env dropped");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
    eprintln!("Session outlives Env drop OK — Arc keeps the Env alive (UAF gone)");
}

#[test]
fn iobinding_zero_copy_output() {
    // Bind the output to a CALLER-OWNED buffer (zero-copy: ORT writes logits straight into
    // out_buf). Result must match the regular run() path bit-for-bit.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    // Reference logits from the regular path.
    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let ref_logits: Vec<f32> = out[0].as_ref().unwrap().as_slice::<f32>().unwrap().to_vec();

    // IoBinding: bind input + a preallocated [1,10] output buffer, then run_binding.
    let in2 = vec![0.0_f32; 784];
    let input2 = Tensor::from_buffer(&in2, &[1, 1, 28, 28], &mem).expect("input2");
    let mut out_buf = vec![0.0_f32; 10];
    let out_val = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("output value");
    let mut binding = IoBinding::new(&sess).expect("binding");
    binding
        .bind_input(sess.input_name(0).expect("input name"), &input2)
        .expect("bind input");
    binding
        .bind_output(sess.output_name(0).expect("output name"), &out_val)
        .expect("bind output");
    sess.run_binding(&binding).expect("run_binding");

    let got: &[f32] = out_val.as_slice::<f32>().expect("zero-copy output read");
    assert_eq!(got.len(), 10, "MNIST output is 10 logits");
    assert_eq!(
        got,
        ref_logits.as_slice(),
        "zero-copy output must match the regular path"
    );
    eprintln!(
        "IoBinding zero-copy output OK ({} logits match regular path)",
        got.len()
    );
}

#[test]
fn prepared_run_reuses_hot_path_state() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut run = sess.prepare_run(&[&input]).expect("prepare_run");

    for _ in 0..3 {
        run.run().expect("prepared run");
        let logits = run
            .output(0)
            .expect("output index")
            .unwrap()
            .as_slice::<f32>()
            .unwrap();
        assert_eq!(logits.len(), 10);
    }
    assert!(run.output(sess.output_count()).is_err());
    eprintln!("PreparedRun reused handles and returned 10 logits");
}

#[test]
fn prepared_iobinding_zero_copy_output() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out_buf = vec![0.0_f32; 10];
    let out_val = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("output value");
    let mut prepared = sess
        .prepare_io_binding(&[&input], &[&out_val])
        .expect("prepare_io_binding");

    prepared.run().expect("prepared iobinding run");
    let logits = out_val.as_slice::<f32>().expect("zero-copy output read");
    assert_eq!(logits.len(), 10);
    eprintln!("PreparedIoBinding wrote 10 logits into caller output");
}

#[test]
fn prepared_iobinding_buffer_array_writes_caller_output() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let out = TensorBuffer::<f32>::zeros(&[1, 10], &mem).expect("output buffer");
    let mut prepared = sess
        .prepare_io_binding_buffer_array::<f32, 1, 1>([&input], [&out])
        .expect("prepare fixed binding");

    prepared.run().expect("prepared iobinding run");
    assert_eq!(out.as_slice().len(), 10);
}

#[test]
fn session_with_prepacked_weights_keeps_cache_alive() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let sess = {
        let cache = PrepackedWeightsContainer::new().expect("prepacked cache");
        Session::new_with_prepacked_weights(
            &env,
            path.to_str().unwrap(),
            SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
            &cache,
        )
        .expect("session with prepacked weights")
    };

    let buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out)
        .expect("run after external cache handle dropped");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
}

#[test]
fn session_with_owned_initializer_overrides_model_weight() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let c = TensorBuffer::from_vec(vec![2.0_f32; n], &[1, n as i64], &mem).expect("C buffer");
    let init = OwnedInitializer::tensor("C", c).expect("initializer");
    let sess = Session::new_with_owned_initializers(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::All)
            .with_intra_threads(1),
        vec![init],
    )
    .expect("session with owned initializer");

    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y.len(), n);
    assert_eq!(y[0], 5.0);
    assert_eq!(y[n - 1], 5.0);
}

#[test]
fn session_with_mmap_owned_initializer_overrides_model_weight() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let mmap_path = std::env::temp_dir().join(format!(
        "st-zrt-relay-c-mmap-{}-{}.bin",
        std::process::id(),
        n
    ));
    let c_values = vec![2.0_f32; n];
    std::fs::write(&mmap_path, f32_as_bytes(&c_values)).expect("write mmap initializer");
    let c = TensorBuffer::<f32>::from_mmap_file(&mmap_path, &[1, n as i64], &mem)
        .expect("mmap C buffer");
    let init = OwnedInitializer::tensor("C", c).expect("initializer");
    let sess = Session::new_with_owned_initializers(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::All)
            .with_intra_threads(1),
        vec![init],
    )
    .expect("session with mmap owned initializer");
    let _ = std::fs::remove_file(&mmap_path);

    let x = vec![3.0_f32; n];
    let input = Tensor::from_buffer(&x, &[1, n as i64], &mem).expect("X");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    let y = out[0].as_ref().unwrap().as_slice::<f32>().unwrap();
    assert_eq!(y.len(), n);
    assert_eq!(y[0], 5.0);
    assert_eq!(y[n - 1], 5.0);
}

#[test]
fn allocator_stats_snapshot_is_available_when_ort_supports_it() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let allocator = Allocator::create(&sess, &mem).expect("allocator");
    match allocator.stats() {
        Ok(stats) => {
            let delta = stats.diff(&stats);
            assert!(delta.entries().is_empty());
            assert!(delta.get("missing").is_none());
            eprintln!("allocator stats: {:?}", stats.entries());
        },
        Err(err) => eprintln!("allocator stats unsupported by this allocator: {err}"),
    }
}

#[test]
fn tensor_io_lane_reuses_owned_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("lane");
    assert!(lane.input(1).is_err());
    assert!(lane.input_mut(1).is_err());
    assert!(lane.output(1).is_err());
    assert!(lane.output_mut(1).is_err());
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_buffer(1).is_err());
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    lane.run().expect("second lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    eprintln!("TensorIoLane reused owned input/output buffers");
}

#[test]
fn static_tensor_io_lane_reuses_owned_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_static_tensor_io_lane::<f32, 1, 1>(&mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static lane");
    assert_eq!(lane.inputs().len(), 1);
    assert_eq!(lane.outputs().len(), 1);
    assert!(lane.input(1).is_err());
    assert!(lane.output(1).is_err());
    lane.inputs_mut()[0].as_mut_slice().fill(0.0);
    lane.run().expect("static lane run");
    assert_eq!(lane.outputs()[0].as_slice().len(), 10);

    let allocator = Allocator::create(&sess, &mem).expect("allocator");
    match lane.run_with_allocator_stats(&allocator) {
        Ok(stats) => {
            let _ = stats.delta();
        },
        Err(err) => eprintln!("allocator stats unsupported by this allocator: {err}"),
    }
}

#[test]
fn tensor_io_lane_buffer_policy_controls_alignment() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&[1, 1, 28, 28]],
            &[&[1, 10]],
            BufferSpec::aligned(64).prefault(),
        )
        .expect("aligned lane");
    assert_eq!((lane.input(0).expect("input").as_ptr() as usize) % 64, 0);
    assert_eq!((lane.output(0).expect("output").as_ptr() as usize) % 64, 0);
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.prime(2).expect("aligned lane prime");
    lane.run().expect("aligned lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);

    let locked_lane = sess
        .prepare_tensor_io_lane_with_buffer_policy::<f32>(
            &mem,
            &[&[1, 1, 28, 28]],
            &[&[1, 10]],
            BufferSpec::aligned(4096).mlock().prefault(),
        )
        .expect("mlocked lane");
    assert_eq!(
        (locked_lane.input(0).expect("input").as_ptr() as usize) % 4096,
        0
    );
    assert_eq!(
        (locked_lane.output(0).expect("output").as_ptr() as usize) % 4096,
        0
    );
}

#[test]
fn tensor_io_lane_auto_policy_aligns_large_buffers() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let large_shape = [1, 1_i64 << 19]; // 2 MiB of f32, the Auto hugepage threshold.
    let lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&large_shape], &[&large_shape])
        .expect("auto large lane");
    assert_eq!(
        (lane.input(0).expect("input").as_ptr() as usize) % (2 << 20),
        0
    );
    assert_eq!(
        (lane.output(0).expect("output").as_ptr() as usize) % (2 << 20),
        0
    );
}

#[test]
fn allocated_output_tensor_io_lane_runs() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lane = sess
        .prepare_allocated_output_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("allocated-output lane");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("allocated-output lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_tensor(1).is_err());

    let mut allocated_lane = sess
        .prepare_allocated_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("allocated tensor lane");
    allocated_lane
        .input_mut(0)
        .expect("allocated lane input")
        .fill(0.0);
    allocated_lane.run().expect("allocated tensor lane run");
    assert_eq!(
        allocated_lane
            .output(0)
            .expect("allocated lane output")
            .len(),
        10
    );
    assert!(allocated_lane.input_tensor(1).is_err());
    assert!(allocated_lane.output_tensor(1).is_err());

    let mut device_lane = sess
        .prepare_device_output_tensor_io_lane::<f32>(&mem, &mem, &[&[1, 1, 28, 28]])
        .expect("device-output lane");
    device_lane
        .input_mut(0)
        .expect("device lane input")
        .fill(0.0);
    let outputs = device_lane.run().expect("device-output lane run");
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        device_lane
            .output(0)
            .expect("device lane output")
            .as_slice::<f32>()
            .expect("device lane output slice")
            .len(),
        10
    );
    assert!(device_lane.input_buffer(1).is_err());
    assert!(device_lane.output(1).is_err());
}

#[test]
fn tensor_io_lanes_run_independent_bindings() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lanes = sess
        .prepare_tensor_io_lanes::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
        .expect("lanes");
    for (i, lane) in lanes.iter_mut().enumerate() {
        lane.input_mut(0).expect("lane input").fill(i as f32);
        lane.run().expect("lane run");
        assert_eq!(lane.output(0).expect("lane output").len(), 10);
    }
    eprintln!("TensorIoLane set ran independent bindings");
}

#[test]
fn runtime_shared_session_runs_exclusive_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut runtime =
        Runtime::<f32>::shared_session(sess, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
            .expect("runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let output_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").len())
        })
        .expect("runtime run");
    assert_eq!(output_len, 10);

    let lane = runtime.lane_mut(1).expect("lane");
    assert!(lane.input(1).is_err());
    assert!(lane.input_mut(1).is_err());
    assert!(lane.output(1).is_err());
    assert!(lane.output_mut(1).is_err());
    assert!(lane.input_buffer(1).is_err());
    assert!(lane.output_buffer(1).is_err());
    lane.input_mut(0).expect("lane input").fill(1.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
}

#[test]
fn runtime_runs_without_checkout() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let mut lanes = Runtime::<f32>::shared_session(sess, &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
        .expect("lane set");
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes.session_mode(), RuntimeMode::SharedSession);
    assert!(lanes.lane(2).is_err());
    assert!(lanes.lane_mut(2).is_err());

    let lane = lanes.lane_mut(0).expect("lane 0");
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(lane.output(0).expect("lane output").len(), 10);
}

#[test]
fn runtime_replicated_sessions_run_independent_lanes() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let mut runtime = Runtime::<f32>::replicated_sessions(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::All),
        &mem,
        &[&[1, 1, 28, 28]],
        &[&[1, 10]],
        2,
    )
    .expect("runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::ReplicatedSessions);
    assert!(runtime.lane(2).is_err());
    assert!(runtime.lane_mut(2).is_err());

    let output_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(0.0);
            lane.run()?;
            Ok(lane.output(0).expect("lane output").len())
        })
        .expect("runtime run");
    assert_eq!(output_len, 10);
}

#[test]
fn runtime_session_factory_supports_owned_initializers() {
    let path = relay_path("256k");
    if !path.exists() {
        eprintln!("skipping — relay_256k.onnx absent");
        return;
    }

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let n = 65_536usize;
    let path = path.to_str().unwrap().to_owned();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let mut runtime =
        Runtime::<f32>::from_session_factory(2, &mem, &[&[1, n as i64]], &[&[1, n as i64]], |_| {
            let c = TensorBuffer::from_vec(vec![2.0_f32; n], &[1, n as i64], &mem)?;
            let init = OwnedInitializer::tensor("C", c)?;
            Session::new_with_owned_initializers(&env, &path, opts.clone(), vec![init])
        })
        .expect("runtime with owned initializers");

    let y_len = runtime
        .run_on(0, |lane| {
            lane.input_mut(0).expect("lane input").fill(3.0);
            lane.run()?;
            let y = lane.output(0).expect("lane output");
            assert_eq!(y[0], 5.0);
            assert_eq!(y[n - 1], 5.0);
            Ok(y.len())
        })
        .expect("runtime run");
    assert_eq!(y_len, n);
}

#[test]
fn iobinding_device_output() {
    // Bind the output to device memory (ORT allocates) and read it back via
    // GetBoundOutputValues — the path for dynamic-shape outputs.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut binding = IoBinding::new(&sess).expect("binding");
    binding
        .bind_input(sess.input_name(0).expect("input name"), &input)
        .expect("bind input");
    binding
        .bind_output_device(sess.output_name(0).expect("output name"), &mem)
        .expect("bind output device");
    sess.run_binding(&binding).expect("run_binding");

    let vals = binding.output_values().expect("output_values");
    assert_eq!(vals.len(), 1, "one output value");
    let logits = vals[0].as_slice::<f32>().expect("device output read");
    assert_eq!(logits.len(), 10, "MNIST output is 10 logits");
    eprintln!(
        "IoBinding device output OK ({} logits via GetBoundOutputValues)",
        logits.len()
    );
}

#[test]
fn static_io_lane_runs_static_typed_io() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);
    lane.run().expect("run");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
}

#[test]
fn static_io_lane_runs_unsynchronized_host_binding() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);
    lane.run_unsynchronized().expect("run");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
}

#[test]
fn static_io_lane_enqueued_run_syncs_outputs_explicitly() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);
    lane.run_enqueued().expect("enqueue run");
    lane.synchronize_outputs().expect("sync outputs");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
}

/// Mutable lane accessors must reject while the lane is in flight after the legacy
/// `run_enqueued`: a pending run may still be reading the staging buffers and writing the output
/// buffers, so mutation before `synchronize_outputs` is a data race. After synchronization the
/// mutators succeed again.
#[test]
fn static_io_lane_mutators_reject_in_flight_runs() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("idle input").fill(0.0);
    lane.inputs_mut()
        .expect("idle inputs_mut")
        .iter_mut()
        .for_each(|buffer| buffer.as_mut_slice().fill(0.0));
    lane.outputs_mut().expect("idle outputs_mut");
    lane.run_enqueued().expect("enqueue run");

    for label in [
        "input_mut",
        "input_mut_at",
        "inputs_mut",
        "output_mut",
        "outputs_mut",
    ] {
        let error = match label {
            "input_mut" => lane.input_mut(0).expect_err(label),
            "input_mut_at" => lane.input_mut_at::<0>().expect_err(label),
            "inputs_mut" => lane.inputs_mut().expect_err(label),
            "output_mut" => lane.output_mut(0).expect_err(label),
            _ => lane.outputs_mut().expect_err(label),
        };
        assert!(
            error.to_string().contains("in-flight"),
            "{label} must reject an in-flight lane, got: {error}"
        );
    }
    // Read access that only reports placement/state stays available; mutation does not.
    assert!(lane.input(0).is_ok());

    lane.synchronize_outputs().expect("sync outputs");
    lane.input_mut_at::<0>()
        .expect("input after synchronize")
        .fill(1.0);
    lane.inputs_mut().expect("inputs_mut after synchronize");
    lane.outputs_mut().expect("outputs_mut after synchronize");
    lane.run().expect("lane reusable after sync");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
}

#[test]
fn static_io_lane_owned_in_flight_token_returns_the_lane() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);

    // `enqueue` consumes the lane into an owned token; `synchronize` fences and returns it.
    lane = lane
        .enqueue()
        .expect("enqueue token")
        .synchronize()
        .expect("token sync");
    assert_eq!(lane.output_at::<0>().expect("output").len(), 10);

    // The compatibility split API cannot enqueue twice onto the same staging/output buffers.
    lane.run_enqueued().expect("legacy enqueue");
    assert!(lane.run_enqueued().is_err());
    lane.synchronize_outputs().expect("legacy sync");
    lane.run().expect("lane reusable after sync");
}

#[test]
fn static_io_lane_owned_enqueue_on_busy_lane_returns_the_lane() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut lane = ServingLane::<f32, f32, 1, 1>::new(session, &mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static I/O lane");
    lane.input_mut_at::<0>().expect("input").fill(0.0);
    lane.run_enqueued().expect("legacy enqueue");

    // A busy lane cannot be moved into an owned token; the error returns the lane.
    let failed = lane.enqueue().expect_err("busy lane must not enqueue");
    assert!(
        failed.error.to_string().contains("in-flight"),
        "unexpected error: {}",
        failed.error
    );
    let mut lane = failed.lane;
    lane.synchronize_outputs().expect("fence busy lane");
    lane.run().expect("lane reusable after fence");
}

#[test]
fn static_io_lane_hot_path_audit_reports_zero_copy_plan() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut runtime = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        session,
        &mem,
        [&[1, 1, 28, 28]],
        [&[1, 10]],
        1,
    )
    .expect("static I/O runtime");
    runtime.set_rebind_inputs_each_run(true);
    runtime.assert_zero_copy_plan().expect("zero-copy plan");
    let audits = runtime.audit_hot_path().expect("audit");
    assert_eq!(audits.len(), 1);
    assert!(audits[0].rebind_inputs_each_run);
    assert!(audits[0].input_names_cached);
    assert_eq!(audits[0].inputs.len(), 1);
    assert_eq!(audits[0].outputs.len(), 1);
    assert!(audits[0].inputs[0].pointer_identity);
    assert!(audits[0].outputs[0].pointer_identity);
    assert_eq!(audits[0].inputs[0].memory_info.name, "Cpu");
}

#[test]
fn static_io_runtime_runs_shared_typed_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut runtime = StaticIoRuntime::<f32, f32, 1, 1>::shared_session(
        session,
        &mem,
        [&[1, 1, 28, 28]],
        [&[1, 10]],
        2,
    )
    .expect("static I/O runtime");
    assert_eq!(runtime.len(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let len = runtime
        .run_on(1, |lane| {
            lane.input_mut(0)?.fill(0.0);
            lane.run()?;
            Ok(lane.output(0)?.len())
        })
        .expect("run lane");
    assert_eq!(len, 10);
}

#[test]
fn dynamic_io_runtime_caches_and_runs_shape_bucket() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(session, mem, 2)
        .expect("dynamic I/O runtime");
    assert_eq!(runtime.bucket_count(), 0);
    assert_eq!(runtime.lane_count(), 2);
    assert_eq!(runtime.session_mode(), RuntimeMode::SharedSession);

    let len = runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 1, |lane| {
            lane.input_mut_at::<0>()?.fill(0.0);
            lane.run()?;
            Ok(lane.output_at::<0>()?.len())
        })
        .expect("dynamic run");
    assert_eq!(len, 10);
    assert_eq!(runtime.bucket_count(), 1);

    runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| {
            lane.input_mut(0)?.fill(0.0);
            lane.run()
        })
        .expect("dynamic cached run");
    assert_eq!(runtime.bucket_count(), 1);
    assert_eq!(
        runtime.buckets()[0].key().input_shape(0),
        Some(&[1, 1, 28, 28][..])
    );

    runtime
        .prime_bucket([&[1, 1, 28, 28]], [&[1, 10]], 1)
        .expect("prime cached bucket");
    assert!(runtime.remove_bucket([&[1, 1, 28, 28]], [&[1, 10]]));
    assert_eq!(runtime.bucket_count(), 0);

    runtime
        .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("recreate bucket");
    assert_eq!(runtime.bucket_count(), 1);
    runtime.clear_buckets().expect("clear buckets");
    assert_eq!(runtime.bucket_count(), 0);
}

#[test]
fn dynamic_io_owned_runs_detach_complete_and_restore_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut runtime =
        DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 2).expect("runtime");
    let mut first = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
            lane.input_mut_at::<0>()?.fill(0.0);
            Ok(())
        })
        .expect("first owned run");
    assert!(
        first.lane().is_err(),
        "an owned run must not expose buffers before synchronization"
    );
    let second = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
            lane.input_mut_at::<0>()?.fill(0.0);
            Ok(())
        })
        .expect("second owned run");
    let error = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |_| Ok(()))
        .expect_err("all lanes are detached");
    assert!(error.to_string().contains("all lanes"));
    first.synchronize().expect("explicit first sync");
    assert_eq!(
        first
            .lane()
            .expect("synchronized lane")
            .output_at::<0>()
            .expect("output")
            .len(),
        10
    );
    let first_len = runtime
        .complete_owned(first, |lane| Ok(lane.output_at::<0>()?.len()))
        .expect("complete first");
    assert_eq!(first_len, 10);
    let second_len = runtime
        .complete_owned(second, |lane| Ok(lane.output_at::<0>()?.len()))
        .expect("complete second");
    assert_eq!(second_len, 10);
    assert_eq!(runtime.buckets()[0].lanes().len(), 2);
}

#[test]
fn dynamic_io_runtime_clear_fences_in_flight_lanes() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 1)
        .expect("dynamic I/O runtime");
    {
        let bucket = runtime
            .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
            .expect("bucket");
        let lane = bucket.lane_mut(0).expect("lane");
        lane.input_mut_at::<0>().expect("input").fill(0.0);
        lane.run_enqueued().expect("enqueue");
    }
    runtime.clear_buckets().expect("clear buckets");
    assert_eq!(runtime.bucket_count(), 0);
}

#[test]
fn dynamic_io_runtime_owned_runs_pipeline_out_of_order_completion() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 3)
        .expect("dynamic I/O runtime");

    let mut tokens = Vec::new();
    for i in 0..3 {
        let run = runtime
            .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
                lane.input_mut_at::<0>()
                    .expect("input")
                    .fill(i as f32 * 0.25);
                Ok(())
            })
            .expect("owned enqueue");
        tokens.push(run);
    }
    assert_eq!(
        runtime
            .bucket([&[1, 1, 28, 28]], [&[1, 10]])
            .expect("bucket")
            .lanes()
            .len(),
        0,
        "all lanes must be detached while tokens are alive"
    );
    assert!(
        !runtime.remove_bucket([&[1, 1, 28, 28]], [&[1, 10]]),
        "a bucket with owned runs must not be retired"
    );

    // Complete in reverse enqueue order; every lane must come back and outputs must be readable.
    for (i, run) in tokens.drain(..).enumerate().rev() {
        let output = runtime
            .complete_owned(run, |lane| {
                Ok(lane.output_at::<0>().expect("output").to_vec())
            })
            .expect("complete owned run");
        assert_eq!(output.len(), 10);
        let _ = i;
    }
    let bucket = runtime
        .bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("bucket");
    assert_eq!(bucket.lanes().len(), 3, "all lanes returned to the bucket");
    assert_eq!(bucket.detached_lane_count(), 0);
}

#[test]
fn dynamic_io_runtime_reclaims_dropped_owned_runs_and_guards_clear() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 2)
        .expect("dynamic I/O runtime");

    runtime
        .prebuild_buckets([ShapeSpec::new([&[1, 1, 28, 28]], [&[1, 10]])])
        .expect("prebuild");
    let prepared = runtime
        .prepared_bucket_id([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("prepared id");
    let run = runtime
        .enqueue_prepared(prepared, |lane| {
            lane.input_mut_at::<0>().expect("input").fill(0.5);
            Ok(())
        })
        .expect("owned enqueue");
    let err = runtime
        .clear_buckets()
        .expect_err("live owned run must prevent bucket clearing");
    assert!(err.to_string().contains("owned runs are in flight"));

    drop(run);
    let replacement = runtime
        .enqueue_prepared(prepared, |_| Ok(()))
        .expect("prepared enqueue must reclaim a previously dropped run");
    runtime
        .complete_owned(replacement, |_| Ok(()))
        .expect("complete replacement");
    let bucket = runtime
        .bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("bucket");
    assert_eq!(bucket.detached_lane_count(), 0);
    assert_eq!(bucket.lanes().len(), 2, "dropped run lane was restored");
    runtime.clear_buckets().expect("clear reclaimed bucket");
    runtime
        .prebuild_buckets([ShapeSpec::new([&[1, 1, 28, 28]], [&[1, 10]])])
        .expect("rebuild into a recycled prepared slot");
    let replacement_id = runtime
        .prepared_bucket_id([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("replacement id");
    assert_ne!(
        prepared, replacement_id,
        "recycled slot must change generation"
    );
    let error = runtime
        .enqueue_prepared(prepared, |_| Ok(()))
        .expect_err("retired prepared id must not alias a later bucket");
    assert!(error.to_string().contains("prepared shape bucket"));
}

#[test]
fn dynamic_io_recovery_stack_handles_concurrent_drops_without_loss() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    const LANES: usize = 16;
    const CYCLES: usize = 100;
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, LANES)
        .expect("dynamic I/O runtime");
    runtime
        .prebuild_buckets([ShapeSpec::new([&[1, 1, 28, 28]], [&[1, 10]])])
        .expect("prebuild");
    let prepared = runtime
        .prepared_bucket_id([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("prepared id");

    for cycle in 0..CYCLES {
        let runs = (0..LANES)
            .map(|lane| {
                runtime
                    .enqueue_prepared(prepared, |owned| {
                        owned
                            .input_mut_at::<0>()
                            .expect("input")
                            .fill((cycle * LANES + lane) as f32);
                        Ok(())
                    })
                    .expect("enqueue")
            })
            .collect::<Vec<_>>();
        std::thread::scope(|scope| {
            for run in runs {
                scope.spawn(move || drop(run));
            }
        });
        runtime.reclaim_dropped_runs();
        let bucket = runtime
            .bucket([&[1, 1, 28, 28]], [&[1, 10]])
            .expect("bucket");
        assert_eq!(bucket.detached_lane_count(), 0, "cycle {cycle}");
        assert_eq!(bucket.lanes().len(), LANES, "cycle {cycle}");
    }
}

#[test]
fn dynamic_io_owned_run_can_outlive_its_runtime() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let run = {
        let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 1)
            .expect("dynamic I/O runtime");
        runtime
            .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
                lane.input_mut_at::<0>().expect("input").fill(0.25);
                Ok(())
            })
            .expect("owned enqueue")
    };
    drop(run);
}

#[test]
fn dynamic_io_runtime_owned_enqueue_error_restores_the_lane() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(sess, mem, 2)
        .expect("dynamic I/O runtime");

    let err = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |_lane| {
            Err(st_zrt::Error::local("boom"))
        })
        .expect_err("prepare must fail");
    assert!(err.to_string().contains("boom"), "unexpected error: {err}");
    let bucket = runtime
        .bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("bucket");
    assert_eq!(bucket.lanes().len(), 2, "failed prepare returns the lane");
    assert_eq!(bucket.detached_lane_count(), 0);

    // The lane set still runs normally afterwards.
    runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| {
            lane.input_mut_at::<0>().expect("input").fill(0.0);
            lane.run()
        })
        .expect("lane runs after failed owned enqueue");
}

/// Host-input CUDA-graph configuration fails closed: ORT captures the device buffers it is handed
/// and never repopulates them from host bindings on replay, so a host-input `cuda_graph` runtime
/// would silently serve stale inputs. The guard fires in `DynamicIoOptions::validate` before any
/// session/provider work, so this regression runs on CPU. The supported path is device-resident
/// inputs on the retained user stream (hardware coverage: `cuda_ep::cuda_graph_*`).
#[test]
fn dynamic_io_runtime_rejects_host_input_cuda_graph() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let err = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        mem,
        output_mem,
        2,
        DynamicIoOptions::new(2).with_cuda_graph(true),
    )
    .expect_err("host-input cuda_graph must be rejected at construction");
    let msg = err.to_string();
    assert!(
        msg.contains("device-resident inputs"),
        "guard should name the supported input path, got: {msg}"
    );
}

/// The pipelined owned-run flow that an earlier unpublished cuda-graph test exercised stays valid
/// graph annotation: two lanes of one bucket enqueue before either completes, complete out of
/// order, and the bucket clears cleanly. (Graph-id plan/release coverage now lives on hardware.)
#[test]
fn dynamic_io_runtime_owned_runs_pipeline_and_clear_without_cuda_graph() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        mem,
        output_mem,
        2,
        DynamicIoOptions::new(2),
    )
    .expect("runtime");

    let run_a = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
            lane.input_mut_at::<0>().expect("input").fill(1.0);
            Ok(())
        })
        .expect("owned enqueue a");
    let run_b = runtime
        .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
            lane.input_mut_at::<0>().expect("input").fill(2.0);
            Ok(())
        })
        .expect("owned enqueue b");
    runtime
        .complete_owned(run_b, |lane| {
            assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
            Ok(())
        })
        .expect("complete b");
    runtime
        .complete_owned(run_a, |lane| {
            assert_eq!(lane.output_at::<0>().expect("output").len(), 10);
            Ok(())
        })
        .expect("complete a");

    runtime.clear_buckets().expect("clear buckets");
    assert_eq!(runtime.bucket_count(), 0);
}

#[test]
fn dynamic_io_runtime_prebuilds_and_warms_shape_plan() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let session = sess;
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(session, mem, 2)
        .expect("dynamic I/O runtime");
    let spec = ShapeSpec::new([&[1, 1, 28, 28][..]], [&[1, 10][..]]);

    assert_eq!(runtime.prebuild_buckets([spec]).expect("prebuild"), 1);
    assert_eq!(runtime.bucket_count(), 1);
    runtime
        .prime_cached_buckets(1)
        .expect("warm cached buckets");

    runtime.clear_buckets().expect("clear buckets");
    assert_eq!(runtime.warm_buckets([spec], 1).expect("warm plan"), 1);
    assert_eq!(runtime.bucket_count(), 1);
    assert!(runtime.bucket([&[1, 1, 28, 28]], [&[1, 10]]).is_some());
}

#[test]
fn dynamic_io_runtime_bounds_shape_buckets() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let session = sess;
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(1),
    )
    .expect("dynamic I/O runtime");

    runtime
        .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
        .expect("first bucket");
    assert_eq!(runtime.bucket_count(), 1);
    assert!(runtime.bucket([&[1, 1, 28, 28]], [&[1, 10]]).is_some());

    runtime
        .get_or_create_bucket([&[1, 1, 28, 29]], [&[1, 10]])
        .expect("second bucket");
    assert_eq!(runtime.bucket_count(), 1);
    assert!(runtime.bucket([&[1, 1, 28, 28]], [&[1, 10]]).is_none());
    assert!(runtime.bucket([&[1, 1, 28, 29]], [&[1, 10]]).is_some());
}

#[test]
fn dynamic_io_runtime_strict_shape_cache_rejects_misses() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let session = sess;
    let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        session,
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(2).with_strict_shape_cache(true),
    )
    .expect("dynamic I/O runtime");
    let spec = ShapeSpec::new([&[1, 1, 28, 28][..]], [&[1, 10][..]]);

    assert_eq!(runtime.warm_buckets([spec], 1).expect("warm plan"), 1);
    runtime
        .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| {
            lane.input_mut_at::<0>()?.fill(0.0);
            lane.run()
        })
        .expect("known strict shape should run");

    let err = runtime
        .run_on(
            [&[1, 1, 28, 29]],
            [&[1, 10]],
            0,
            |_| -> st_zrt::Result<()> {
                panic!("strict cache miss should not call the lane closure")
            },
        )
        .expect_err("unknown strict shape should fail before bucket creation");
    let msg = err.to_string();
    assert!(
        msg.contains("strict_shape_cache"),
        "unexpected error: {msg}"
    );
    assert_eq!(runtime.bucket_count(), 1);
}

/// The `cuda_graph` + host-input rejection also applies to the single-lane topology, and even a
/// device-input runtime keeps refusing unplanned shapes. On CPU the host-input guard fires first
/// (the device-input sealed-plan enforcement runs on hardware in `cuda_ep`); this regression pins
/// the fail-closed construction on the exact configuration the stale-input hazard used to allow.
#[test]
fn dynamic_io_runtime_cuda_graph_single_lane_host_input_is_rejected_too() {
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };

    let output_mem = mem.try_clone_descriptor().expect("output mem");
    let err = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session_with_options(
        sess,
        mem,
        output_mem,
        1,
        DynamicIoOptions::new(1).with_cuda_graph(true),
    )
    .expect_err("single-lane host-input cuda_graph must also be rejected");
    assert!(
        err.to_string().contains("device-resident inputs"),
        "unexpected error: {err}"
    );
}

#[test]
fn session_from_bytes_and_metadata() {
    // Load from an in-memory byte buffer (no temp file) and read the model metadata.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping — mnist.onnx absent");
        return;
    }
    let bytes = std::fs::read(&path).expect("read mnist.onnx");

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::from_bytes(&env, &bytes, opts).expect("from_bytes");

    // Inference through the from_bytes session works identically to the file path.
    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );

    // Metadata: producer name + version are readable; producer is Some for a real model.
    let md: ModelMetadata = sess.metadata().expect("metadata");
    let producer = md.producer_name().expect("producer name");
    let version = md.version().expect("version");
    eprintln!("from_bytes OK; producer={:?} version={}", producer, version);
    assert!(producer.is_some(), "a real model has a producer name");
    // version() must not error and must be non-negative for a valid model.
    assert!(version >= 0, "model version is non-negative");
}

#[test]
fn run_with_options_config() {
    // Build a caller RunOptions (log severity + config entry) and pass it to run_with.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let opts = RunOptions::new()
        .with_log_severity(LoggingLevel::Fatal)
        .with_run_tag("zrt-smoke")
        .freeze()
        .expect("run options");

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run_with(&[&input], &mut out, &opts).expect("run_with");
    assert_eq!(
        out[0].as_ref().unwrap().as_slice::<f32>().unwrap().len(),
        10
    );
    eprintln!("run_with (caller RunOptions) OK");
}

#[test]
fn run_options_terminate_cancels() {
    // Pre-terminate the RunOptions; the subsequent run must return an error (ORT checks the
    // terminate flag and aborts). Proves SetTerminate takes effect.
    let env = Environment::new().expect("env");
    let (mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let opts = RunOptions::new().freeze().expect("run options");
    opts.terminate().expect("terminate");

    let in_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&in_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    let res = sess.run_with(&[&input], &mut out, &opts);
    assert!(
        res.is_err(),
        "a run with a pre-terminated RunOptions must error"
    );
    eprintln!(
        "terminate → run_with returned Err (cancelled): {:?}",
        res.err()
    );
}

#[test]
fn memory_info_named() {
    // General named MemoryInfo + introspection getters (round-trip through the engine).
    let mem = MemoryInfo::new_named("Cpu", AllocatorType::Device, 0, MemType::Default)
        .expect("new_named");
    assert_eq!(mem.name().unwrap(), "Cpu");
    assert_eq!(mem.device_id().unwrap(), 0);
    assert_eq!(mem.alloc_type().unwrap(), AllocatorType::Device);
    assert_eq!(mem.mem_type().unwrap(), MemType::Default);
    eprintln!(
        "MemoryInfo::new_named round-trips (name={:?})",
        mem.name().unwrap()
    );
}

#[test]
fn session_io_placement_reports_memory_and_ep_assignment() {
    let env = Environment::new().expect("env");
    let Some((_mem, sess)) = mnist_session(&env) else {
        eprintln!("skipping session_io_placement_reports_memory_and_ep_assignment — mnist absent");
        return;
    };

    let placement = sess.io_placement().expect("io placement");
    assert_eq!(placement.len(), sess.input_count() + sess.output_count());
    assert_eq!(placement[0].direction, IoDirection::Input);
    assert_eq!(placement[0].index, 0);
    assert_eq!(placement[0].element_type, st_zrt::ElementType::Float);
    assert_eq!(placement[0].memory_info.name, "Cpu");
    assert_eq!(placement[0].memory_info.device_id, 0);
    assert_eq!(placement[1].direction, IoDirection::Output);
    assert_eq!(placement[1].memory_info.name, "Cpu");

    let input_mem = sess.input_memory_info(0).expect("input memory info");
    let output_mem = sess.output_memory_info(0).expect("output memory info");
    assert_eq!(input_mem.name, "Cpu");
    assert_eq!(output_mem.name, "Cpu");
    let _ = sess.input_ep_device(0).expect("input ep device query");
    let _ = sess.output_ep_device(0).expect("output ep device query");
}

#[test]
fn copy_tensors_copies_owned_value_to_tensor_buffer() {
    let env = Environment::new().expect("env");
    let Some((mem, sess)) = mnist_session(&env) else {
        eprintln!("skipping copy_tensors_copies_owned_value_to_tensor_buffer — mnist absent");
        return;
    };

    let input_data = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&input_data, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");

    let output = outputs[0].take().expect("output");
    let expected = output.as_slice::<f32>().expect("host output").to_vec();
    let device_value = output.into_device_value().expect("device value");
    let mut dst = TensorBuffer::<f32>::zeros(&[1, 10], &mem).expect("dst");
    device_value
        .copy_to_tensor_buffer(&sess, &mut dst)
        .expect("copy tensors");
    assert_eq!(dst.as_slice(), &expected[..]);
}

#[test]
fn owned_value_copy_convenience_copies_to_tensor_buffer() {
    let env = Environment::new().expect("env");
    let Some((mem, sess)) = mnist_session(&env) else {
        eprintln!("skipping owned_value_copy_convenience_copies_to_tensor_buffer — mnist absent");
        return;
    };

    let input_data = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&input_data, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");
    let output = outputs[0].as_ref().expect("output");
    let expected = output.as_slice::<f32>().expect("host output").to_vec();
    let mut dst = TensorBuffer::<f32>::zeros(&[1, 10], &mem).expect("dst");
    output
        .copy_to_tensor_buffer(&sess, &mut dst)
        .expect("copy tensors");
    assert_eq!(dst.as_slice(), &expected[..]);
}

#[test]
fn release_captured_graph_is_safe_without_captured_graph() {
    // The 1.27 SessionReleaseCapturedGraph wrapper must be safe to call on a session that has no
    // captured CUDA graph (no CUDA EP / enable_cuda_graph). Empirically ORT treats releasing a
    // non-existent graph as a no-op and returns Ok — the guarantee here is that the wrapper does not
    // panic on a CPU session. (The positive path — releasing a real captured graph — is exercised by
    // the CUDA test in tests/cuda_ep.rs.)
    let env = Environment::new().expect("env");
    let Some((_mem, sess)) = mnist_session(&env) else {
        eprintln!("skipping release_captured_graph test — mnist.onnx absent");
        return;
    };
    sess.release_captured_graph(1)
        .expect("releasing a non-existent captured graph is a safe no-op (no panic)");
}

#[test]
fn buffer_spec_presets_and_auto_thresholds_are_stable() {
    assert_eq!(BufferSpec::AUTO, BufferSpec::auto());
    assert_eq!(BufferSpec::LATENCY, BufferSpec::vec().prefault());
    assert_eq!(
        BufferSpec::THROUGHPUT_LARGE,
        BufferSpec::aligned(2 << 20).hugepage().prefault()
    );
    assert_eq!(
        BufferSpec::PINNED_HOST,
        BufferSpec::aligned(2 << 20).hugepage().prefault().mlock()
    );
    assert_eq!(BufferSpec::CUDA_PINNED, BufferSpec::cuda_pinned());

    assert_eq!(BufferSpec::AUTO.resolve((1 << 20) - 1), BufferSpec::vec());
    let aligned = BufferSpec::AUTO.resolve(1 << 20);
    assert_eq!(aligned.storage(), BufferStorage::Aligned);
    assert_eq!(aligned.alignment_bytes(), 4096);
    assert!(aligned.is_prefaulted());
    assert!(!aligned.uses_hugepages());
    assert_eq!(BufferSpec::AUTO.resolve((2 << 20) - 1), aligned);
    assert_eq!(
        BufferSpec::AUTO.resolve(2 << 20),
        BufferSpec::THROUGHPUT_LARGE
    );
    assert_eq!(
        BufferSpec::vec().resolve(usize::MAX),
        BufferSpec::vec(),
        "explicit Vec must not be mistaken for Auto"
    );
    assert_eq!(
        BufferSpec::LATENCY.resolve(usize::MAX),
        BufferSpec::LATENCY,
        "explicit modifiers must remain size-invariant"
    );
}

#[cfg(feature = "model-editor")]
#[test]
fn memory_device_snapshots_are_available_with_ep_sub_api() {
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let device = mem.memory_device().expect("memory device");
    assert_eq!(device.device_id, 0);

    let env = Environment::new().expect("env");
    let Some((mem, sess)) = mnist_session(&env) else {
        eprintln!("skipping memory_device_snapshots_are_available_with_ep_sub_api — mnist absent");
        return;
    };
    let input_data = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&input_data, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");
    let output = outputs[0].as_ref().expect("output");
    let value_device = output.memory_device().expect("value memory device");
    assert_eq!(value_device.device_id, 0);
}

#[test]
fn arena_cfg_construct() {
    // Both ArenaCfg constructors succeed on CPU; register_allocator is best-effort here
    // (its CPU support is ORT-version-specific, so we log rather than hard-assert).
    assert_eq!(ArenaExtendStrategy::NextPowerOfTwo as i32, 0);
    assert_eq!(ArenaExtendStrategy::SameAsRequested as i32, 1);
    let cfg =
        ArenaCfg::new(usize::MAX, ArenaExtendStrategy::NextPowerOfTwo, -1, -1).expect("arena cfg");
    let cfg_v2 = ArenaCfg::with_entries(&[(
        "arena_extend_strategy",
        ArenaExtendStrategy::SameAsRequested as usize,
    )])
    .expect("arena cfg v2");
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let reg = env.register_allocator(&mem, &cfg);
    eprintln!(
        "ArenaCfg OK (v1 + v2); register_allocator(CPU) -> {:?}",
        reg.is_ok()
    );
    drop(cfg_v2);
}

#[test]
fn allocator_create_and_allocate() {
    // Create a session-scoped allocator, allocate through it, let RAII free on drop.
    let env = Environment::new().expect("env");
    let (_mem, sess) = match mnist_session(&env) {
        Some(v) => v,
        None => {
            eprintln!("skipping — mnist.onnx absent");
            return;
        },
    };
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let alloc = Allocator::create(&sess, &mem).expect("create allocator");
    let buf = alloc.allocate(128).expect("allocate 128 bytes");
    assert!(!buf.as_ptr().is_null(), "allocated buffer is non-null");
    // `buf` frees on drop (AllocatorFree); `alloc` releases on drop (ReleaseAllocator).
    eprintln!("Allocator::create + allocate(128) OK (ptr non-null, RAII frees)");
}

// ─── T1.6 Custom logger ───────────────────────────────────────────────────────

#[test]
fn environment_with_custom_logger_constructs() {
    // CreateEnvWithCustomLogger + trampoline registration. A no-op closure; we only assert the
    // Env constructs (and drops) without error — ORT accepted our function pointer + param.
    let env = Environment::new_with_logger(LoggingLevel::Verbose, "zrt-logger-ctor", |_| {})
        .expect("env with custom logger");
    drop(env);
    eprintln!("Environment::new_with_logger constructs + drops cleanly");
}

#[test]
fn environment_with_logger_and_global_thread_pools_constructs() {
    let tp = ThreadingOptions::new().expect("threading");
    let env = Environment::new_with_logger_and_global_thread_pools(
        LoggingLevel::Verbose,
        "zrt-logger-tp",
        |_| {},
        tp,
    )
    .expect("env with logger + global thread pools");
    drop(env);
    eprintln!("Environment::new_with_logger_and_global_thread_pools constructs cleanly");
}

#[test]
fn env_creation_options_builder_constructs_env() {
    // CreateEnvWithOptions via the OrtEnvCreationOptions struct (verified repr(C) layout). Includes
    // a logger + global thread pools + a config entry to exercise every struct field.
    let mut cfg = st_zrt::KeyValuePairs::new().expect("kvps");
    cfg.add("ep_factory.dummy.key", "v").expect("add cfg");
    let opts = EnvCreationOptions::new(LoggingLevel::Warning, "zrt-opts")
        .with_logger(|_| {})
        .with_thread_pools(ThreadingOptions::new().expect("threading"))
        .with_config_entries(cfg);
    let env = Environment::new_with_options(opts).expect("env via CreateEnvWithOptions");
    drop(env);
    eprintln!("EnvCreationOptions builder → CreateEnvWithOptions constructs cleanly");
}

#[test]
fn environment_set_log_level_round_trips() {
    // UpdateEnvWithCustomLogLevel.
    let env = Environment::new().expect("env");
    env.set_log_level(LoggingLevel::Error).expect("set Error");
    env.set_log_level(LoggingLevel::Verbose)
        .expect("set Verbose");
    eprintln!("Environment::set_log_level round-trips cleanly");
}

#[test]
fn environment_custom_logger_runs_session_without_ub() {
    // Drive a real session through an Env whose logs route to a capturing closure. Asserts the env
    // + session + run succeed with a custom logger attached — i.e. the trampoline is registered and
    // ORT can invoke it concurrently without UB. ORT's verbose emission is process-global and not
    // guaranteed across parallel tests, so the deterministic callback-fire proof (an explicit
    // `Logger::log` in a kernel always routing through the env logger) lives in the custom-op tests.
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping environment_custom_logger_runs_session — mnist.onnx absent");
        return;
    }
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let env = Environment::new_with_logger(LoggingLevel::Verbose, "zrt-logger-capture", move |r| {
        sink.lock().unwrap().push(r.message);
    })
    .expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");
    let n = captured.lock().unwrap().len();
    eprintln!("environment_custom_logger_runs_session: run OK; captured {n} message(s)");
}

#[test]
fn session_options_user_logging_function_is_applied() {
    // SetUserLoggingFunction: attach a per-session logger at options-build time and confirm the
    // session constructs + runs without error (the leaked callback stays valid for the run).
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping session_options_user_logging_function — mnist.onnx absent");
        return;
    }
    let captured: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let sink = captured.clone();
    let mut opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    opts.with_user_logging_function(move |_r: LogRecord| {
        *sink.lock().unwrap() += 1;
    })
    .expect("set user logger");
    let env = Environment::new().expect("env");
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let buf: Vec<f32> = vec![0.0; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut outputs).expect("run");
    let n = *captured.lock().unwrap();
    eprintln!(
        "session_options_user_logging_function: callback fired {n} time(s); session ran clean"
    );
}

/// `Session::ep_graph_assignment_info` reaches the FFI and the full borrowed-view chain
/// (`EpAssignedSubgraph` → `ep_name` + `nodes` → `EpAssignedNode` name/domain/operator_type) is
/// sound. Exercises the introspection accessors without asserting a count (a CPU session may or
/// may not report the CPU EP); the populated CUDA path is covered by the bench-c graph bench.
#[cfg(feature = "ep")]
#[test]
fn session_ep_graph_assignment_info_is_sound() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skipping session_ep_graph_assignment_info — mnist.onnx absent");
        return;
    }
    let env = Environment::new().expect("env");
    // `ep_graph_assignment_info` only returns data when the session was created with this entry.
    let opts = SessionOptions::new()
        .with_config_entry("session.record_ep_graph_assignment_info", "1")
        .expect("config entry");
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    let info = sess.ep_graph_assignment_info().expect("assignment info");
    eprintln!(
        "session_ep_graph_assignment_info: {} assigned subgraph(s)",
        info.len()
    );
    for sg in &info {
        let ep = sg.ep_name().unwrap_or_default();
        let nodes = sg.nodes().unwrap_or_default();
        eprintln!("  subgraph ep={ep} nodes={}", nodes.len());
        for n in nodes {
            eprintln!(
                "    node name={} domain={} op={}",
                n.name().unwrap_or_default(),
                n.domain().unwrap_or_default(),
                n.operator_type().unwrap_or_default(),
            );
        }
    }
}
