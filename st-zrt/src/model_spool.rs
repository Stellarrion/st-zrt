//! Spool-through resolution for byte-loaded models that reference ONNX external data.
//!
//! ORT resolves an initializer's external-data path *relative to the model file's
//! directory*. Byte-buffer entry points (`CreateSessionFromArray`,
//! `ModelCompilationOptions_SetInputModelFromBuffer`) have no such base, so any model
//! with external initializers — every model over the 2 GiB protobuf limit, and any
//! model saved with external data — cannot resolve them. The generic resolver: write
//! the (small) serialized graph to a uniquely named temporary file *inside the
//! user-supplied external-data directory* and load through the path-based entry point.
//! The guard keeps the file alive until the owning object (session or compile) is done,
//! because ORT may map or reopen external data lazily.

use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static SPOOL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A uniquely named, self-deleting copy of a serialized model, created inside the
/// directory that holds the model's external-data files.
pub struct SpooledModelFile {
    path: PathBuf,
}

impl SpooledModelFile {
    /// The spooled file's path — valid to load from until `Drop`.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for SpooledModelFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpooledModelFile")
            .field("path", &self.path)
            .finish()
    }
}

impl Drop for SpooledModelFile {
    fn drop(&mut self) {
        // Best-effort: a failure to remove never invalidates the loaded session.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Copy `model_bytes` into a unique file inside `external_data_dir`.
///
/// The directory must be writable; when it is not (e.g. a read-only model bundle),
/// copy the external-data files to a writable directory and pass that instead — ORT
/// resolves the initializer locations relative to the spooled model file either way.
pub(crate) fn spool_model_bytes(
    external_data_dir: &Path, model_bytes: &[u8],
) -> Result<SpooledModelFile> {
    let seq = SPOOL_COUNTER.fetch_add(1, Ordering::Relaxed);
    for _ in 0..32 {
        let name = format!(
            ".st-zrt-spooled-{}-{}-{}.onnx",
            std::process::id(),
            seq,
            // Uniqueness even across processes started in the same millisecond.
            nanos_suffix(),
        );
        let path = external_data_dir.join(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut file) => {
                std::io::Write::write_all(&mut file, model_bytes).map_err(|e| {
                    let _ = std::fs::remove_file(&path);
                    Error::new(-1, format!("zrt: spooling model bytes failed: {e}"))
                })?;
                return Ok(SpooledModelFile { path });
            },
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(Error::new(
                    -1,
                    format!(
                        "zrt: cannot create a temporary model file in {} (external-data \
                         directories must be writable): {e}",
                        external_data_dir.display()
                    ),
                ));
            },
        }
    }
    Err(Error::new(-1, "zrt: could not find a free spool file name"))
}

fn nanos_suffix() -> u128 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}
