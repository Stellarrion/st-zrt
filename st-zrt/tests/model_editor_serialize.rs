//! Model serialization round-trip: build an ONNX model in Rust, serialize it to bytes
//! (`Model::to_bytes` via the CompileApi AOT path), reload the bytes, and run — verifying
//! the serialized blob is a valid, runnable ONNX model.
#![cfg(feature = "model-editor")]

use st_zrt::{
    ElementType, Environment, Graph, GraphOptimizationLevel, MemoryInfo, Model, Node, OwnedValue,
    Session, SessionOptions, Tensor, TensorTypeAndShapeInfo, TypeInfo, ValueInfo,
};

#[test]
fn build_serialize_reload_and_run() {
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");

    // Build Add(X1, X2) -> Y (float[1]).
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

    // Serialize to ONNX bytes.
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let bytes = model.to_bytes(&env, &opts).expect("serialize");
    eprintln!("serialized model: {} bytes", bytes.len());
    assert!(bytes.len() > 64, "serialized blob should be non-trivial");

    // Reload the blob as a fresh session and run.
    let sess = Session::from_bytes(&env, &bytes, opts).expect("reload from_bytes");
    let x1 = vec![2.0_f32];
    let x2 = vec![3.0_f32];
    let x1v = Tensor::from_buffer(&x1, &[1], &mem).expect("X1");
    let x2v = Tensor::from_buffer(&x2, &[1], &mem).expect("X2");
    let inputs: [&dyn st_zrt::RunInput; 2] = [&x1v, &x2v];
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut out).expect("run reloaded");
    let y: &[f32] = out[0].as_ref().expect("out").as_slice().expect("read");
    assert_eq!(y.len(), 1, "Y is float[1]");
    assert!((y[0] - 5.0).abs() < 1e-5, "Y = 5.0, got {}", y[0]);
    eprintln!(
        "build -> serialize ({} B) -> reload -> run: 2.0 + 3.0 = {}",
        bytes.len(),
        y[0]
    );
}
