//! Byte-buffer loading for models whose initializers use ONNX external data.
//!
//! ORT resolves external-data paths relative to the model file's directory, so a plain
//! buffer load cannot resolve them. `from_bytes_with_external_data` spools the buffer
//! next to the external-data directory and loads through the path. The external-data
//! blob is materialized by the test into a temp directory — nothing binary is committed.

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

struct TempDir(std::path::PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A temp directory holding a copy of `ext_add.onnx` plus its external-data file
/// (four f32 values of 2.0 — exactly what the model's external reference declares).
fn fixture_dir() -> TempDir {
    let dir = std::env::temp_dir().join(format!(
        "st-zrt-extdata-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/external_data/ext_add.onnx");
    std::fs::copy(&model, dir.join("ext_add.onnx")).expect("copy model");
    let data: Vec<f32> = vec![2.0; 4];
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    std::fs::write(dir.join("ext_add.onnx.data"), bytes).expect("write external data");
    TempDir(dir)
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
    let fx = fixture_dir();
    let dir = &fx.0;
    let bytes = std::fs::read(dir.join("ext_add.onnx")).expect("model bytes");

    let env = Environment::new().expect("env");
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::Basic);

    // Reference: path-based load (ORT resolves external data relative to the model file).
    let via_path = Session::new(
        &env,
        dir.join("ext_add.onnx").to_str().unwrap(),
        opts.clone(),
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
    let fx = fixture_dir();
    let bytes = std::fs::read(fx.0.join("ext_add.onnx")).expect("model bytes");
    let env = Environment::new().expect("env");
    match Session::from_bytes(&env, &bytes, SessionOptions::new()) {
        Err(_) => {}, // rejected by ORT: expected on this runtime line
        Ok(s) => assert_eq!(run(&s), vec![3.0, 4.0, 5.0, 6.0]),
    }
}
