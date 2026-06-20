//! Load an existing model via the model-editor session path (`from_bytes_for_editing`) and
//! run it, asserting the output matches the regular `from_bytes` path.
#![cfg(feature = "model-editor")]

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
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
    let ed_sess =
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
