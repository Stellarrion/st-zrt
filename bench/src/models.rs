//! Model download + cache.
//!
//! First slice uses MNIST (ort's own hosted test model — single float input
//! `[1,1,28,28]`, single float output) purely to *prove the harness compiles
//! and runs end-to-end*. The real workload models (MiniLM-L6-v2, MobileNetV3)
//! land in task #3; the harness is model-pluggable.
use std::path::PathBuf;
use std::process::Command;

const MNIST_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx";
const HF_RESNET50_URL: &str =
    "https://huggingface.co/Xenova/resnet-50/resolve/main/onnx/model.onnx";

fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn ensure_cached(name: &str, url: &str) -> std::io::Result<PathBuf> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(name);
    if !path.exists() {
        // Shell out to system curl (verified present); avoids pulling an HTTP
        // client dep into a dev-only harness. -fsSL = fail-on-error, silent, show-error, follow-redirects.
        let status = Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&path)
            .arg(url)
            .status()?;
        if !status.success() {
            return Err(std::io::Error::other(format!("curl failed for {url}")));
        }
    }
    Ok(path)
}

pub fn ensure_mnist() -> std::io::Result<PathBuf> {
    ensure_cached("mnist.onnx", MNIST_URL)
}

pub fn ensure_hf_resnet50() -> std::io::Result<PathBuf> {
    ensure_cached("hf_resnet50.onnx", HF_RESNET50_URL)
}

/// Resolve a synthetic relay model by label (`"4m"`, `"16m"`, etc.) from the
/// shared `bench/models` cache. If it is missing, regenerate the relay set.
pub fn ensure_relay(label: &str) -> std::io::Result<PathBuf> {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("relay_{label}.onnx"));
    if path.exists() {
        return Ok(path);
    }

    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("gen_models.py");
    let status = Command::new("python3").arg(&script).arg(&dir).status()?;
    if !status.success() {
        return Err(std::io::Error::other(format!(
            "gen_models.py failed (is `onnx` installed? path={})",
            script.display()
        )));
    }
    if !path.exists() {
        return Err(std::io::Error::other(format!(
            "relay model {label} not produced by generator"
        )));
    }
    Ok(path)
}
