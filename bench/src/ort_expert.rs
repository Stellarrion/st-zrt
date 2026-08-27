//! Variant B — `ort` expert path (the ceiling a careful user reaches today).
//!
//! IoBinding + **bind-once-mutate-in-place** + reused binding. Avoids the
//! per-call taxes variant A pays:
//!  - O3: input buffer is ORT-owned and mutated in place → no per-call copy.
//!  - E2: bind once, never rebind.
//!  - inputs! macro / SessionInputs alloc: IoBinding binds by name once.
//!
//! Still inherent to `ort`'s API (only addressable *below* ort — variant C):
//!  - O5: `SessionOutputs` rebuilt every `run_binding`.
//!  - E1: ORT rebuilds `FeedsFetchesManager` internally.
//!  - E4: the simple `run_binding` path uses a default `RunOptions` (no reuse hook).
use ort::memory::MemoryInfo;
use ort::session::{IoBinding, Session};
use ort::value::Tensor;

const MNIST_INPUT_LEN: usize = 28 * 28;

pub struct VariantB {
    session: Session,
    binding: IoBinding,
    input: Tensor<f32>,
}

impl VariantB {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::new_with_intra_threads(model_path, None)
    }

    /// Load with an optional intra-op thread count (1-thread control).
    pub fn new_with_intra_threads(model_path: &str, threads: Option<usize>) -> ort::Result<Self> {
        let mut builder = Session::builder()?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n)?;
        }
        let session = builder.commit_from_file(model_path)?;
        let in_name = session.inputs()[0].name().to_string();
        let out_name = session.outputs()[0].name().to_string();

        // Preallocate the input buffer in ORT's allocator (owned Tensor).
        let input = Tensor::<f32>::from_array((vec![1, 1, 28, 28], vec![0.0; MNIST_INPUT_LEN]))?;

        let mut binding = session.create_binding()?;
        binding.bind_input(in_name, &input)?;
        binding.bind_output_to_device(out_name, &MemoryInfo::default())?;

        Ok(Self {
            session,
            binding,
            input,
        })
    }

    /// One inference via the ort expert (IoBinding, bind-once-mutate) path.
    #[inline]
    pub fn run_once(&mut self) -> ort::Result<()> {
        // Mutate the preallocated input buffer in place (E2 bind-once-mutate; O3 no copy).
        {
            let mut view = self.input.try_extract_array_mut::<f32>()?;
            for v in view.iter_mut() {
                *v = 0.0;
            }
        }
        // &mut session + &binding are disjoint fields; reuse the binding (no rebind).
        let outputs = self.session.run_binding(&self.binding)?;
        let _ = outputs[0].try_extract_array::<f32>()?;
        Ok(())
    }
}
