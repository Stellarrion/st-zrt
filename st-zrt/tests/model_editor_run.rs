//! Build an ONNX model from scratch (pure Rust, no proto/protobuf) via the model-editor
//! API and run it. Constructs `Y = Add(X1, X2)` (float[1]), builds a session from the model,
//! runs it, and asserts `2.0 + 3.0 = 5.0`.
#![cfg(feature = "model-editor")]

use st_zrt::{
    ElementType, Environment, Graph, GraphOptimizationLevel, MemoryInfo, Model, Node, OwnedValue,
    Session, SessionOptions, Tensor, TensorTypeAndShapeInfo, TypeInfo, ValueInfo,
};

#[test]
fn build_model_from_scratch_and_run() {
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");

    // float[1] type, reused for both inputs and the output.
    let mut tsi = TensorTypeAndShapeInfo::new().expect("tsi");
    tsi.set_element_type(ElementType::Float)
        .expect("element type");
    tsi.set_dimensions(&[1]).expect("dims");
    let ty = TypeInfo::tensor(&tsi).expect("tensor type-info");
    let x1 = ValueInfo::new("X1", &ty).expect("X1 value-info");
    let x2 = ValueInfo::new("X2", &ty).expect("X2 value-info");
    let y = ValueInfo::new("Y", &ty).expect("Y value-info");

    // Graph: X1 + X2 -> Y.
    let graph = Graph::new().expect("graph");
    graph.set_inputs(vec![x1, x2]).expect("set inputs");
    graph.set_outputs(vec![y]).expect("set outputs");
    graph
        .add_node(Node::new("Add", "", "add1", &["X1", "X2"], &["Y"]).expect("node"))
        .expect("add node");

    // Model with the ONNX default-domain opset, then attach the graph.
    let model = Model::new(&[("", 21)]).expect("model");
    model.add_graph(graph).expect("add graph");

    // Build a session straight from the in-memory model (no serialization).
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::from_model(&env, &model, opts).expect("session from model");

    // Run: X1=[2.0], X2=[3.0] -> Y=[5.0]. (Zero-copy inputs — bind the buffers to locals so
    // they outlive the run.)
    let x1buf = vec![2.0_f32];
    let x2buf = vec![3.0_f32];
    let x1v = Tensor::from_buffer(&x1buf, &[1], &mem).expect("X1 tensor");
    let x2v = Tensor::from_buffer(&x2buf, &[1], &mem).expect("X2 tensor");
    let inputs: [&dyn st_zrt::RunInput; 2] = [&x1v, &x2v];
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut out).expect("run");
    let yv: &[f32] = out[0].as_ref().expect("output").as_slice().expect("read");
    assert_eq!(yv.len(), 1, "Y is a float[1]");
    assert!(
        (yv[0] - 5.0).abs() < 1e-6,
        "Y = Add(X1,X2) = 5.0, got {}",
        yv[0]
    );

    // The session exposes the model's opset for its domain.
    assert_eq!(sess.input_count(), 2, "2 inputs X1,X2");
    assert_eq!(sess.output_count(), 1, "1 output Y");
    let opset = sess.opset_for_domain("").expect("opset for default domain");
    eprintln!(
        "built Add(X1,X2)->Y in pure Rust, ran it: 2.0 + 3.0 = {} (opset {})",
        yv[0], opset
    );
}
