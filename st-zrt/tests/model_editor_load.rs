//! Load an existing model via the model-editor session path (`from_bytes_for_editing`) and
//! run it, asserting the output matches the regular `from_bytes` path.
#![cfg(feature = "model-editor")]

use st_zrt::{
    ElementType, Environment, Graph, GraphOptimizationLevel, MemoryInfo, Model, Node, OwnedValue,
    Session, SessionOptions, Tensor, TensorTypeAndShapeInfo, TypeInfo, ValueInfo,
};

fn mnist_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx")
}

#[test]
fn load_existing_model_via_editor_and_run() {
    let path = mnist_path();
    if !path.exists() {
        eprintln!("skip — mnist.onnx not cached");
        return;
    }
    let bytes = std::fs::read(&path).expect("read mnist");
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let buf = vec![0.0_f32; 28 * 28];
    let input = Tensor::from_buffer(&buf, &[1, 1, 28, 28], &mem).expect("input");

    // Reference run via the regular from_bytes path.
    let ref_sess = Session::from_bytes(&env, &bytes, opts.clone()).expect("from_bytes");
    let mut ref_out: Vec<Option<OwnedValue>> = (0..ref_sess.output_count()).map(|_| None).collect();
    ref_sess.run(&[&input], &mut ref_out).expect("ref run");
    let ref_logits: Vec<f32> = ref_out[0]
        .as_ref()
        .expect("ref output")
        .as_slice()
        .expect("ref read")
        .to_vec();

    // Load the same model via the model-editor session path, finalize, then run.
    let mut ed_sess =
        Session::from_bytes_for_editing(&env, &bytes, opts).expect("from_bytes_for_editing");
    ed_sess
        .finalize(&SessionOptions::new().with_opt_level(GraphOptimizationLevel::All))
        .expect("finalize model-editor session");
    let mut ed_out: Vec<Option<OwnedValue>> = (0..ed_sess.output_count()).map(|_| None).collect();
    ed_sess.run(&[&input], &mut ed_out).expect("editor run");
    let ed_logits: &[f32] = ed_out[0]
        .as_ref()
        .expect("editor output")
        .as_slice()
        .expect("editor read");

    assert_eq!(ed_logits.len(), ref_logits.len(), "same output count");
    for (a, b) in ref_logits.iter().zip(ed_logits.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "editor vs from_bytes logit mismatch: {a} vs {b}"
        );
    }
    eprintln!("loaded mnist via the model-editor session + ran; matches from_bytes ✓");
}

fn scalar_f32_type() -> TypeInfo {
    let mut tsi = TensorTypeAndShapeInfo::new().expect("tsi");
    tsi.set_element_type(ElementType::Float).expect("elem");
    tsi.set_dimensions(&[1]).expect("dims");
    TypeInfo::tensor(&tsi).expect("type")
}

fn build_add_model() -> Model {
    let ty = scalar_f32_type();
    let graph = Graph::new().expect("graph");
    graph
        .set_inputs(vec![
            ValueInfo::new("X1", &ty).expect("X1"),
            ValueInfo::new("X2", &ty).expect("X2"),
        ])
        .expect("inputs");
    graph
        .set_outputs(vec![ValueInfo::new("Y", &ty).expect("Y")])
        .expect("outputs");
    graph
        .add_node(Node::new("Add", "", "add", &["X1", "X2"], &["Y"]).expect("node"))
        .expect("add node");
    let model = Model::new(&[("", 21)]).expect("model");
    model.add_graph(graph).expect("add graph");
    model
}

fn build_identity_fragment() -> Model {
    let ty = scalar_f32_type();
    let graph = Graph::new().expect("graph");
    graph.set_inputs(Vec::new()).expect("fragment input");
    graph
        .set_outputs(vec![ValueInfo::new("Z", &ty).expect("Z")])
        .expect("fragment output");
    graph
        .add_node(Node::new("Identity", "", "promote_y", &["Y"], &["Z"]).expect("node"))
        .expect("fragment node");
    let model = Model::new(&[("", 21)]).expect("fragment model");
    model.add_graph(graph).expect("add fragment graph");
    model
}

#[test]
fn apply_model_fragment_to_existing_output() {
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let source = build_add_model();
    let bytes = source.to_bytes(&env, &opts).expect("serialize source");

    let mut sess =
        Session::from_bytes_for_editing(&env, &bytes, opts.clone()).expect("editor session");
    let fragment = build_identity_fragment();
    sess.apply_model(&fragment).expect("apply fragment");
    sess.finalize(&opts).expect("finalize");

    let x1 = vec![2.0_f32];
    let x2 = vec![3.0_f32];
    let x1v = Tensor::from_buffer(&x1, &[1], &mem).expect("X1");
    let x2v = Tensor::from_buffer(&x2, &[1], &mem).expect("X2");
    let inputs: [&dyn st_zrt::RunInput; 2] = [&x1v, &x2v];
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut out).expect("run augmented");
    assert_eq!(sess.output_name(0).expect("output name"), "Z");
    let z: &[f32] = out[0].as_ref().expect("out").as_slice().expect("read");
    assert_eq!(z.len(), 1);
    assert!(
        (z[0] - 5.0).abs() < 1e-6,
        "Z should forward Y=5, got {}",
        z[0]
    );
}
