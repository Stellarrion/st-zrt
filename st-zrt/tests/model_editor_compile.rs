//! AOT model compilation (CompileApi): build a model, compile it to a file via
//! `Model::to_file`, then drive the `ModelCompilationOptions` builder directly (file-input ->
//! file-output, exercising the graph-opt-level / flags / EP-context scalar setters), reloading
//! and running each compiled file to confirm it is a valid, runnable ONNX model.
#![cfg(feature = "model-editor")]

use st_zrt::{
    ElementType, Environment, Graph, GraphOptimizationLevel, MemoryInfo, Model,
    ModelCompilationOptions, Node, OwnedValue, Session, SessionOptions, Tensor,
    TensorTypeAndShapeInfo, TypeInfo, ValueInfo,
};

/// Build `Y = Add(X1, X2)` (float[1]) in memory.
fn build_add_model() -> (Environment, MemoryInfo, Model) {
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let mut tsi = TensorTypeAndShapeInfo::new().expect("tsi");
    tsi.set_element_type(ElementType::Float).expect("elem");
    tsi.set_dimensions(&[1]).expect("dims");
    let ty = TypeInfo::tensor(&tsi).expect("type");
    let g = Graph::new().expect("graph");
    g.set_inputs(vec![
        ValueInfo::new("X1", &ty).expect("X1"),
        ValueInfo::new("X2", &ty).expect("X2"),
    ])
    .expect("inputs");
    g.set_outputs(vec![ValueInfo::new("Y", &ty).expect("Y")])
        .expect("outputs");
    g.add_node(Node::new("Add", "", "add", &["X1", "X2"], &["Y"]).expect("node"))
        .expect("add node");
    let model = Model::new(&[("", 21)]).expect("model");
    model.add_graph(g).expect("add graph");
    (env, mem, model)
}

/// Run an Add session; return the scalar Y.
fn run_add(sess: &Session, mem: &MemoryInfo, x1: f32, x2: f32) -> f32 {
    let b1 = vec![x1];
    let b2 = vec![x2];
    let v1 = Tensor::from_buffer(&b1, &[1], mem).expect("X1");
    let v2 = Tensor::from_buffer(&b2, &[1], mem).expect("X2");
    let inputs: [&dyn st_zrt::RunInput; 2] = [&v1, &v2];
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut out).expect("run");
    let y: &[f32] = out[0].as_ref().expect("out").as_slice().expect("read");
    assert_eq!(y.len(), 1);
    y[0]
}

#[test]
fn compile_to_file_and_reload() {
    let (env, mem, model) = build_add_model();
    let file_a = std::env::temp_dir().join("stzrt_compile_a.onnx");
    let _ = std::fs::remove_file(&file_a);

    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    model
        .to_file(&env, &opts, file_a.to_str().unwrap())
        .expect("to_file");
    assert!(file_a.exists(), "to_file wrote the file");
    let bytes = std::fs::read(&file_a).expect("read compiled file");
    assert!(bytes.len() > 64, "non-trivial compiled file");

    let sess = Session::from_bytes(&env, &bytes, opts).expect("reload from compiled file");
    let y = run_add(&sess, &mem, 2.0, 3.0);
    assert!((y - 5.0).abs() < 1e-5, "Y = 5.0, got {y}");
    eprintln!("Model::to_file -> reload -> run: 2.0 + 3.0 = {y}");
    let _ = std::fs::remove_file(&file_a);
}

#[test]
fn builder_compiles_file_to_file() {
    let (env, mem, model) = build_add_model();
    let file_in = std::env::temp_dir().join("stzrt_compile_in.onnx");
    let file_out = std::env::temp_dir().join("stzrt_compile_out.onnx");
    let _ = std::fs::remove_file(&file_in);
    let _ = std::fs::remove_file(&file_out);

    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    model
        .to_file(&env, &opts, file_in.to_str().unwrap())
        .expect("seed input file");

    // Drive the builder directly: file-input -> file-output, exercising every scalar setter
    // (these reach the FFI; a wrong index/signature crashes rather than erroring cleanly).
    let copts = ModelCompilationOptions::new(&env, &opts).expect("compile opts");
    copts
        .set_input_model_path(file_in.to_str().unwrap())
        .expect("input path");
    copts
        .set_output_model_path(file_out.to_str().unwrap())
        .expect("output path");
    copts
        .set_graph_optimization_level(GraphOptimizationLevel::All)
        .expect("opt level");
    copts.set_flags(0).expect("flags");
    copts
        .set_ep_context_embed_mode(false)
        .expect("ep embed mode");
    copts.compile(&env).expect("compile");
    assert!(file_out.exists(), "builder compiled file written");

    let bytes = std::fs::read(&file_out).expect("read compiled file");
    let sess = Session::from_bytes(&env, &bytes, opts).expect("reload");
    let y = run_add(&sess, &mem, 4.0, 6.0);
    assert!((y - 10.0).abs() < 1e-5, "Y = 10.0, got {y}");
    eprintln!("builder file->file -> reload -> run: 4.0 + 6.0 = {y}");
    let _ = std::fs::remove_file(&file_in);
    let _ = std::fs::remove_file(&file_out);
}

#[test]
fn ep_context_setters_reach_ffi() {
    // The two EP-context path setters are EP-specific (no effect without an EP configured), so
    // they are checked for FFI reach here rather than on the run path: a wrong index/signature
    // crashes, while a correct call errors cleanly with a domain validation message.
    let (env, _mem, _model) = build_add_model();
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let copts = ModelCompilationOptions::new(&env, &opts).expect("compile opts");

    // SetEpContextBinaryInformation validates the directory; a nonexistent dir => clean error.
    let r = copts.set_ep_context_binary_information("/nonexistent/stzrt/dir", "stzrt");
    eprintln!("SetEpContextBinaryInformation -> {r:?}");
    assert!(
        r.is_err(),
        "should error (invalid dir), proving the FFI call works"
    );

    // SetOutputModelExternalInitializersFile accepts any NUL-free path (no validation at set
    // time) — a clean Ok proves the index/signature.
    copts
        .set_output_model_external_initializers_file("/tmp/stzrt_ext_init.bin", 1024)
        .expect("external initializers file path set");
    eprintln!("SetOutputModelExternalInitializersFile reached FFI cleanly");
}
