//! Variant A — `ort` default inference path (the baseline most users hit).
//!
//! Deliberately exercises the anti-patterns flagged in DESIGN §3:
//!  - O3: `Tensor::from_array` with *owned* data → copies into ORT's allocator.
//!  - O2: `Session::run` takes `&mut self`.
//!  - O5: `SessionOutputs` rebuilt + materialized every call.
use ort::session::Session;
use ort::value::Tensor;

const MNIST_INPUT_LEN: usize = 28 * 28;

pub struct VariantA {
    session: Session,
    input: Vec<f32>,
    shape: Vec<i64>,
}

impl VariantA {
    /// Load the model. (A default ONNX Runtime environment is auto-created.)
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::new_with_intra_threads(model_path, None)
    }

    /// Load with an optional intra-op thread count (control for the thread-pool
    /// overhead bisection — small models are latency-bound by pool scheduling).
    pub fn new_with_intra_threads(model_path: &str, threads: Option<usize>) -> ort::Result<Self> {
        let mut builder = Session::builder()?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n)?;
        }
        let session = builder.commit_from_file(model_path)?;
        Ok(Self {
            session,
            input: vec![0.0; MNIST_INPUT_LEN],
            shape: vec![1, 1, 28, 28],
        })
    }

    /// One inference via the ort default (copying) path.
    #[inline]
    pub fn run_once(&mut self) -> ort::Result<()> {
        // Owned data → Tensor → copy into ORT allocator (O3).
        let tensor = Tensor::<f32>::from_array((self.shape.clone(), self.input.clone()))?;
        // &mut self (O2); SessionOutputs rebuilt per call (O5).
        let outputs = self.session.run(ort::inputs![tensor])?;
        // Force output materialization (O5: per-output Arc::clone + view alloc).
        let _ = outputs[0].try_extract_array::<f32>()?;
        Ok(())
    }
}
