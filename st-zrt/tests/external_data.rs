//! Byte-buffer loading for models whose initializers use ONNX external data.
//!
//! ORT resolves external-data paths relative to the model file's directory, so a plain
//! buffer load cannot resolve them. `from_bytes_with_external_data` spools the buffer
//! next to the external-data directory and loads through the path.

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("external_data")
        .join(name)
}

fn run(session: &Session) -> Vec<f32> {
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let input = Tensor::from_buffer(&[1.0_f32, 2.0, 3.0, 4.0], &[4], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..session.output_count()).map(|_| None).collect();
    session.run(&[&input], &mut out).expect("run");
    out[0]
        .as_ref()
        .unwrap()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec()
}

#[test]
fn external_data_byte_load_resolves_and_runs() {
    let model_path = fixture("ext_add.onnx");
    let dir = model_path.parent().unwrap(); // holds the model and its .data file
    let bytes = std::fs::read(fixture("ext_add.onnx")).expect("model bytes");
    assert!(
        std::fs::read(dir.join("ext_add.onnx.data")).is_ok(),
        "external data file present"
    );

    let env = Environment::new().expect("env");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::Basic);

    // Reference: path-based load (ORT resolves external data relative to the model file).
    let via_path = Session::new(
        &env,
        fixture("ext_add.onnx").to_str().unwrap(),
        SessionOptions::new().with_opt_level(GraphOptimizationLevel::Basic),
    )
    .expect("path load");
    let expected = run(&via_path);
    assert_eq!(expected, vec![3.0, 4.0, 5.0, 6.0]);

    // The generic resolver: same model from bytes, external directory supplied.
    let via_bytes = Session::from_bytes_with_external_data(&env, &bytes, dir, opts)
        .expect("external-data byte load");
    assert_eq!(run(&via_bytes), expected);

    // The spooled temporary file is gone once the session drops.
    drop(via_bytes);
    let leftovers: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(".st-zrt-spooled-")
        })
        .collect();
    assert!(leftovers.is_empty(), "spool cleaned up: {:?}", leftovers);
}

#[test]
fn plain_byte_load_of_external_data_model_is_rejected_or_selfcontained() {
    // Documents the gap being fixed: a plain buffer load has no base directory.
    let dir = fixture("ext_add.onnx").parent().unwrap().to_path_buf();
    let bytes = std::fs::read(dir.join("ext_add.onnx")).expect("model bytes");
    let env = Environment::new().expect("env");
    match Session::from_bytes(&env, &bytes, SessionOptions::new()) {
        Err(_) => {}, // rejected by ORT: expected on this runtime line
        Ok(s) => {
            // If this runtime line tolerates unresolved external data, the values would
            // be wrong; assert they are right to catch silent misbehavior.
            assert_eq!(run(&s), vec![3.0, 4.0, 5.0, 6.0]);
        },
    }
}
