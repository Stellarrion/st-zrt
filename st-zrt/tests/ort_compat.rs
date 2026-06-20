//! Compatibility coverage inspired by the Rust `ort` crate's integration tests.
//!
//! These are not API-compatibility tests. They exercise model surfaces that are easy to miss in
//! a fast-path runtime: dynamic tensor shapes and string-input models.

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions,
    StringTensor, Tensor,
};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("ort_compat")
        .join(name)
}

#[test]
fn ort_compat_dynamic_upsample_shape_and_run() {
    let path = fixture("upsample.onnx");
    assert!(path.exists(), "missing fixture {}", path.display());

    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let sess = Session::new(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::Basic)
            .with_intra_threads(1),
    )
    .expect("session");

    assert_eq!(sess.input_count(), 1);
    assert_eq!(sess.output_count(), 1);
    assert_eq!(sess.input_shape(0).expect("input shape"), &[-1, -1, -1, 3]);
    assert_eq!(
        sess.output_shape(0).expect("output shape"),
        &[-1, -1, -1, 3]
    );
    assert_eq!(sess.input_meta(0).expect("input meta").2, None);
    assert_eq!(sess.output_meta(0).expect("output meta").2, None);

    let input_shape = [1, 2, 3, 3];
    let input: Vec<f32> = (0..18).map(|v| v as f32).collect();
    let input = Tensor::from_buffer(&input, &input_shape, &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");

    let y = out[0].as_ref().expect("output");
    assert_eq!(
        y.tensor_type_and_shape()
            .expect("output type shape")
            .dims()
            .expect("output dims"),
        vec![1, 4, 6, 3]
    );
    assert_eq!(y.as_slice::<f32>().expect("output").len(), 72);
}

#[test]
fn ort_compat_vectorizer_string_input_and_metadata() {
    let path = fixture("vectorizer.onnx");
    assert!(path.exists(), "missing fixture {}", path.display());

    let env = Environment::new().expect("env");
    let sess = Session::new(
        &env,
        path.to_str().unwrap(),
        SessionOptions::new()
            .with_opt_level(GraphOptimizationLevel::Basic)
            .with_intra_threads(1),
    )
    .expect("session");

    let metadata = sess.metadata().expect("metadata");
    assert_eq!(
        metadata.producer_name().expect("producer").as_deref(),
        Some("skl2onnx")
    );
    assert_eq!(
        metadata.description().expect("description").as_deref(),
        Some("test description")
    );
    assert_eq!(
        metadata.lookup("custom_key").expect("custom").as_deref(),
        Some("custom_value")
    );

    let input = StringTensor::new(&["document"], &[1]).expect("string input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&input], &mut out).expect("run");

    let y = out[0].as_ref().expect("output");
    assert_eq!(
        y.tensor_type_and_shape()
            .expect("output type shape")
            .dims()
            .expect("output dims"),
        vec![1, 9]
    );
    assert_eq!(
        y.as_slice::<f32>().expect("output"),
        &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}
