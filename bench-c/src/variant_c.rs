//! Variant C — st-zrt inference path (the runtime this project builds).
//!
//! What the st-zrt layer does differently from ort (the win we measure):
//!  - Input is a **zero-copy view** over a caller-owned buffer
//!    (`CreateTensorWithDataAsOrtValue`) — no copy into ORT's allocator.
//!  - Session pre-marshals input/output names **once** at construction; the hot
//!    path passes stable `*const c_char` ptrs (no `FeedsFetchesManager` rebuild,
//!    no per-call name marshaling).
//!  - `RunOptions` is reused across runs; `run(&self)` is shared-reentrant.
//!  - Output is read **zero-copy** via `GetTensorMutableData`.
//!
//! On MNIST (784-float input ≈ 3 KB) this sits below the ~5 MB crossover, so C is
//! expected to track A/B (compute-bound) — the point is to prove **no binding tax**,
//! i.e. a from-scratch hand-written binding matches the incumbent. Large-tensor /
//! high-QPS workloads are where the architectural wins surface.
use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};

pub struct VariantC {
    // Order matters: env must outlive session; mem must outlive every Tensor.
    _env: Environment,
    mem: MemoryInfo,
    session: Session,
    input_buf: Vec<f32>,
}

impl VariantC {
    pub fn new(model_path: &str) -> st_zrt::Result<Self> {
        let env = Environment::new()?;
        let mem = MemoryInfo::cpu()?;
        let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
        let session = Session::new(&env, model_path, opts)?;
        Ok(Self {
            _env: env,
            mem,
            session,
            input_buf: vec![0.0; 784],
        })
    }

    /// One zero-copy inference through st-zrt.
    #[inline]
    pub fn run_once(&mut self) -> st_zrt::Result<()> {
        // Zero-copy input: wraps self.input_buf in place; no allocator copy.
        let input = Tensor::from_buffer(&self.input_buf, &[1, 1, 28, 28], &self.mem)?;
        let mut outputs: Vec<Option<OwnedValue>> =
            (0..self.session.output_count()).map(|_| None).collect();
        self.session.run(&[&input], &mut outputs)?;
        // Zero-copy output read.
        let _ = outputs[0].as_ref().unwrap().as_slice::<f32>()?;
        Ok(())
    }

    // ── diagnostic accessors (to bisect per-phase cost) ──
    pub fn mem(&self) -> &MemoryInfo {
        &self.mem
    }
    pub fn input_buf(&self) -> &[f32] {
        &self.input_buf
    }
    pub fn session(&self) -> &Session {
        &self.session
    }
}
