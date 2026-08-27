//! Model-editor initializer support: build a graph carrying a named constant tensor (a
//! weight) via `Graph::add_initializer`, then run it — verifying the initializer resolves as
//! a graph value (it is NOT an input) and feeds the Add node. `Y = Add(X, B)` with the
//! constant `B = [10.0]`; running with `X = [5.0]` yields `Y = [15.0]`.
#![cfg(feature = "model-editor")]

use st_zrt::{
    ElementType, Environment, Graph, GraphOptimizationLevel, MemoryInfo, Model, Node, OwnedValue,
    Session, SessionOptions, Tensor, TensorTypeAndShapeInfo, TypeInfo, ValueInfo,
};

#[test]
fn build_model_with_initializer_and_run() {
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");

    // float[1] type for X and Y.
    let mut tsi = TensorTypeAndShapeInfo::new().expect("tsi");
    tsi.set_element_type(ElementType::Float)
        .expect("element type");
    tsi.set_dimensions(&[1]).expect("dims");
    let ty = TypeInfo::tensor(&tsi).expect("tensor type-info");

    // Graph: input X, output Y; constant B = [10.0] registered as an initializer (NOT a graph
    // input); node Add(X, B) -> Y. B is engine-allocated via copy_from_slice, so
    // data_is_external = false (works for any size — a 4-byte scalar is far below the 128-byte
    // floor the external path imposes).
    let g = Graph::new().expect("graph");
    g.set_inputs(vec![ValueInfo::new("X", &ty).expect("X")])
        .expect("set inputs");
    g.set_outputs(vec![ValueInfo::new("Y", &ty).expect("Y")])
        .expect("set outputs");
    let b = Tensor::copy_from_slice(&[10.0_f32], &[1]).expect("B tensor");
    g.add_initializer("B", b, false).expect("add initializer");
    g.add_node(Node::new("Add", "", "add", &["X", "B"], &["Y"]).expect("node"))
        .expect("add node");

    let model = Model::new(&[("", 21)]).expect("model");
    model.add_graph(g).expect("add graph");

    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::from_model(&env, &model, opts).expect("session from model");

    // B is an initializer, not a graph input — the session exposes exactly one input (X).
    assert_eq!(
        sess.input_count(),
        1,
        "only X is an input; B is an initializer"
    );

    // Run: X = [5.0] -> Y = X + B = 15.0. (Zero-copy input bound to a local outliving the run.)
    let xbuf = vec![5.0_f32];
    let xv = Tensor::from_buffer(&xbuf, &[1], &mem).expect("X tensor");
    let inputs: [&dyn st_zrt::RunInput; 1] = [&xv];
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut out).expect("run");
    let yv: &[f32] = out[0].as_ref().expect("output").as_slice().expect("read");
    assert_eq!(yv.len(), 1, "Y is float[1]");
    assert!(
        (yv[0] - 15.0).abs() < 1e-5,
        "Y = Add(X, B) = 5.0 + 10.0 = 15.0, got {}",
        yv[0]
    );
    eprintln!(
        "built Add(X, B)->Y with a constant initializer B=[10.0]; ran: 5.0 + 10.0 = {}",
        yv[0]
    );
}
