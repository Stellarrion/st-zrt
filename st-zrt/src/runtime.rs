//! Transport-agnostic serving runtime: fixed, exclusive inference lanes.
//!
//! [`Runtime`] is intentionally not an HTTP/gRPC server. It is the reusable inference
//! service underneath one: a fixed set of zero-copy lanes, each with its own input/output
//! buffers and IoBinding. Server code assigns requests to lanes explicitly, mutates input
//! buffers, runs the lane, then reads outputs. ZRT performs no checkout locking.

use crate::element::TensorElement;
use crate::environment::Environment;
use crate::io_binding::IoBinding;
use crate::memory::MemoryInfo;
use crate::prepacked::PrepackedWeightsContainer;
use crate::session::{lane_tensor_buffer, LaneBufferPolicy, Session};
use crate::session_options::SessionOptions;
use crate::tensor::TensorBuffer;
use crate::{Error, Result};
use std::sync::Arc;

/// How a [`Runtime`] may arrange ONNX Runtime sessions across lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// One shared [`Session`], with one exclusive IoBinding/buffer set per lane.
    SharedSession,
    /// One [`Session`] per lane. This costs more memory but avoids wrapper-level shared
    /// session state and is the preferred latency/concurrency mode for server use.
    ReplicatedSessions,
}

/// One exclusive inference lane.
///
/// A lane owns its input/output buffers and IoBinding. The mutable methods are deliberate:
/// one request owns a lane at a time, so caller code cannot concurrently mutate or run the
/// same bound buffers through the safe runtime API.
pub struct Lane<T: TensorElement> {
    // Drop the binding before the tensor buffers whose ORT value handles it references, and
    // before releasing this lane's session reference.
    binding: IoBinding,
    inputs: Vec<TensorBuffer<T>>,
    outputs: Vec<TensorBuffer<T>>,
    session: Arc<Session>,
}

/// One exclusive static-shape I/O lane with independently typed inputs and outputs.
///
/// This covers models whose input tensors share one scalar type `I` and output tensors share
/// another scalar type `O`, while the input/output arity is fixed in the Rust type.
///
/// The arity is part of the type, so services can keep concrete lane types for hot paths while
/// still using different scalar types for model inputs and outputs. Each lane owns stable input
/// buffers, stable output buffers, and one pre-bound IoBinding.
pub struct StaticIoLane<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    // Drop the binding before buffers and session because it references their ORT handles.
    binding: IoBinding,
    inputs: [TensorBuffer<I>; INPUTS],
    outputs: [TensorBuffer<O>; OUTPUTS],
    session: Arc<Session>,
    rebind_inputs_each_run: bool,
}

impl<T> Lane<T>
where
    T: TensorElement + Clone + Default,
{
    pub(crate) fn new(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if input_shapes.len() != session.input_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: input shape count mismatch: expected {}, got {}",
                    session.input_count(),
                    input_shapes.len()
                ),
            ));
        }
        if output_shapes.len() != session.output_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: output shape count mismatch: expected {}, got {}",
                    session.output_count(),
                    output_shapes.len()
                ),
            ));
        }

        let inputs: Vec<TensorBuffer<T>> = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<_>>()?;
        let outputs: Vec<TensorBuffer<T>> = output_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, mem, policy))
            .collect::<Result<_>>()?;

        let mut binding = IoBinding::new(&session)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(session.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(session.output_name(i)?, output)?;
        }

        Ok(Self {
            binding,
            inputs,
            outputs,
            session,
        })
    }
}

impl<T: TensorElement> Lane<T> {
    /// Execute this lane's prepared binding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
    }

    /// Run this lane `runs` times before serving to prime ORT shape/memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[T]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[T]> {
        self.outputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [T]> {
        self.outputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.inputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane input index {i} out of range")))
    }

    #[inline]
    pub fn output_buffer(&self, i: usize) -> Result<&TensorBuffer<T>> {
        self.outputs
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane output index {i} out of range")))
    }

    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> StaticIoLane<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build one static-shape I/O lane over a shared session.
    pub fn new(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::with_buffer_policy(
            session,
            mem,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one static-shape I/O lane with separate input/output memory descriptors.
    pub fn with_memory(
        session: Arc<Session>, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::with_buffer_policy(
            session,
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build one static-shape I/O lane with explicit input/output buffer policies.
    pub fn with_buffer_policy(
        session: Arc<Session>, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        input_policy: LaneBufferPolicy, output_policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if INPUTS != session.input_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: static I/O lane input count mismatch: expected {}, got {}",
                    session.input_count(),
                    INPUTS
                ),
            ));
        }
        if OUTPUTS != session.output_count() {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: static I/O lane output count mismatch: expected {}, got {}",
                    session.output_count(),
                    OUTPUTS
                ),
            ));
        }

        let inputs: [TensorBuffer<I>; INPUTS] = input_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, input_mem, input_policy))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::new(-1, "zrt: failed to build static I/O input array"))?;
        let outputs: [TensorBuffer<O>; OUTPUTS] = output_shapes
            .iter()
            .map(|shape| lane_tensor_buffer(shape, output_mem, output_policy))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::new(-1, "zrt: failed to build static I/O output array"))?;

        let mut binding = IoBinding::new(&session)?;
        for (i, input) in inputs.iter().enumerate() {
            binding.bind_input(session.input_name(i)?, input)?;
        }
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(session.output_name(i)?, output)?;
        }

        Ok(Self {
            binding,
            inputs,
            outputs,
            session,
            rebind_inputs_each_run: false,
        })
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    StaticIoLane<I, O, INPUTS, OUTPUTS>
{
    /// Execute this lane's pre-bound IoBinding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        if self.rebind_inputs_each_run {
            self.binding.clear_inputs();
            for (i, input) in self.inputs.iter().enumerate() {
                self.binding
                    .bind_input(self.session.input_name(i)?, input)?;
            }
        }
        self.session.run_binding(&self.binding)
    }

    /// Run this lane `runs` times before serving to prime ORT shape/memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    #[inline]
    pub fn inputs(&self) -> &[TensorBuffer<I>; INPUTS] {
        &self.inputs
    }

    #[inline]
    pub fn inputs_mut(&mut self) -> &mut [TensorBuffer<I>; INPUTS] {
        &mut self.inputs
    }

    #[inline]
    pub fn outputs(&self) -> &[TensorBuffer<O>; OUTPUTS] {
        &self.outputs
    }

    #[inline]
    pub fn outputs_mut(&mut self) -> &mut [TensorBuffer<O>; OUTPUTS] {
        &mut self.outputs
    }

    #[inline]
    pub fn input(&self, i: usize) -> Result<&[I]> {
        self.inputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: static I/O lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [I]> {
        self.inputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: static I/O lane input index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn output(&self, i: usize) -> Result<&[O]> {
        self.outputs
            .get(i)
            .map(TensorBuffer::as_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: static I/O lane output index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [O]> {
        self.outputs
            .get_mut(i)
            .map(TensorBuffer::as_mut_slice)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    format!("zrt: static I/O lane output index {i} out of range"),
                )
            })
    }

    #[inline]
    pub fn input_at<const IDX: usize>(&self) -> Result<&[I]> {
        self.input(IDX)
    }

    #[inline]
    pub fn input_mut_at<const IDX: usize>(&mut self) -> Result<&mut [I]> {
        self.input_mut(IDX)
    }

    #[inline]
    pub fn output_at<const IDX: usize>(&self) -> Result<&[O]> {
        self.output(IDX)
    }

    #[inline]
    pub fn output_mut_at<const IDX: usize>(&mut self) -> Result<&mut [O]> {
        self.output_mut(IDX)
    }

    #[inline]
    pub fn input_buffer(&self, i: usize) -> Result<&TensorBuffer<I>> {
        self.inputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: static I/O lane input index {i} out of range"),
            )
        })
    }

    #[inline]
    pub fn output_buffer(&self, i: usize) -> Result<&TensorBuffer<O>> {
        self.outputs.get(i).ok_or_else(|| {
            Error::new(
                -1,
                format!("zrt: static I/O lane output index {i} out of range"),
            )
        })
    }

    #[inline]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Rebind inputs before every run.
    ///
    /// This is not the default because it adds per-run name marshaling and breaks the
    /// bind-once zero-allocation CPU contract. It is useful for CUDA paths where ORT's
    /// reusable CPU input binding can otherwise observe stale mutated input buffers.
    #[inline]
    pub fn set_rebind_inputs_each_run(&mut self, enabled: bool) {
        self.rebind_inputs_each_run = enabled;
    }
}

fn build_shared_lanes<T>(
    session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
    lanes: usize, policy: LaneBufferPolicy, what: &'static str,
) -> Result<Vec<Lane<T>>>
where
    T: TensorElement + Clone + Default,
{
    if lanes == 0 {
        return Err(Error::new(-1, format!("{what} requires at least one lane")));
    }
    (0..lanes)
        .map(|_| Lane::new(session.clone(), mem, input_shapes, output_shapes, policy))
        .collect()
}

/// A fixed, caller-scheduled set of exclusive inference lanes.
///
/// It is for services that already have a deterministic lane assignment strategy, such as
/// sharded workers, per-core loops, or an external scheduler. The hot path is direct
/// `lane_mut(i)`/slice access plus [`Lane::run`]; ZRT does not keep a checkout pool or
/// lock around lane selection.
pub struct Runtime<T: TensorElement> {
    lanes: Vec<Lane<T>>,
    mode: RuntimeMode,
}

/// A fixed set of caller-scheduled I/O lanes with typed inputs and outputs.
pub struct StaticIoRuntime<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    lanes: Vec<StaticIoLane<I, O, INPUTS, OUTPUTS>>,
    mode: RuntimeMode,
}

/// Shape-bucket cache options for [`DynamicIoRuntime`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicIoOptions {
    /// Maximum concrete shape buckets kept by this runtime.
    pub max_buckets: usize,
    /// Buffer policy used for input tensors when a new shape bucket is created.
    pub input_policy: LaneBufferPolicy,
    /// Buffer policy used for output tensors when a new shape bucket is created.
    pub output_policy: LaneBufferPolicy,
    /// Rebind lane input values before each run.
    pub rebind_inputs_each_run: bool,
}

impl DynamicIoOptions {
    /// Build options with a bounded bucket count and default buffer policies.
    #[inline]
    pub fn new(max_buckets: usize) -> Self {
        Self {
            max_buckets,
            ..Self::default()
        }
    }

    /// Set the input buffer policy.
    #[inline]
    pub fn with_input_policy(mut self, policy: LaneBufferPolicy) -> Self {
        self.input_policy = policy;
        self
    }

    /// Set the output buffer policy.
    #[inline]
    pub fn with_output_policy(mut self, policy: LaneBufferPolicy) -> Self {
        self.output_policy = policy;
        self
    }

    /// Enable or disable per-run input rebinding for newly-created static shape buckets.
    #[inline]
    pub fn with_rebind_inputs_each_run(mut self, enabled: bool) -> Self {
        self.rebind_inputs_each_run = enabled;
        self
    }

    fn validate(self) -> Result<Self> {
        if self.max_buckets == 0 {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one shape bucket",
            ));
        }
        Ok(self)
    }
}

impl Default for DynamicIoOptions {
    #[inline]
    fn default() -> Self {
        Self {
            max_buckets: 16,
            input_policy: LaneBufferPolicy::Auto,
            output_policy: LaneBufferPolicy::Auto,
            rebind_inputs_each_run: false,
        }
    }
}

/// Concrete input/output shapes used to select one dynamic runtime bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeKey<const INPUTS: usize, const OUTPUTS: usize> {
    input_shapes: [Vec<i64>; INPUTS],
    output_shapes: [Vec<i64>; OUTPUTS],
}

impl<const INPUTS: usize, const OUTPUTS: usize> ShapeKey<INPUTS, OUTPUTS> {
    /// Copy concrete shape slices into an owned reusable key.
    pub fn new(input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS]) -> Self {
        Self {
            input_shapes: input_shapes.map(<[i64]>::to_vec),
            output_shapes: output_shapes.map(<[i64]>::to_vec),
        }
    }

    /// Borrow one input shape.
    #[inline]
    pub fn input_shape(&self, i: usize) -> Option<&[i64]> {
        self.input_shapes.get(i).map(Vec::as_slice)
    }

    /// Borrow one output shape.
    #[inline]
    pub fn output_shape(&self, i: usize) -> Option<&[i64]> {
        self.output_shapes.get(i).map(Vec::as_slice)
    }

    /// Borrow all input shapes.
    #[inline]
    pub fn input_shapes(&self) -> &[Vec<i64>; INPUTS] {
        &self.input_shapes
    }

    /// Borrow all output shapes.
    #[inline]
    pub fn output_shapes(&self) -> &[Vec<i64>; OUTPUTS] {
        &self.output_shapes
    }

    #[inline]
    fn matches(&self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS]) -> bool {
        self.input_shapes
            .iter()
            .zip(input_shapes)
            .all(|(a, b)| a.as_slice() == b)
            && self
                .output_shapes
                .iter()
                .zip(output_shapes)
                .all(|(a, b)| a.as_slice() == b)
    }
}

/// One concrete shape bucket inside [`DynamicIoRuntime`].
pub struct ShapeBucket<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    key: ShapeKey<INPUTS, OUTPUTS>,
    lanes: StaticIoRuntime<I, O, INPUTS, OUTPUTS>,
    last_used: u64,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    ShapeBucket<I, O, INPUTS, OUTPUTS>
{
    /// Concrete input/output shapes for this bucket.
    #[inline]
    pub fn key(&self) -> &ShapeKey<INPUTS, OUTPUTS> {
        &self.key
    }

    /// Monotonic runtime-local access counter used for eviction.
    #[inline]
    pub fn last_used(&self) -> u64 {
        self.last_used
    }

    /// The static lane set for this concrete shape.
    #[inline]
    pub fn lanes(&self) -> &StaticIoRuntime<I, O, INPUTS, OUTPUTS> {
        &self.lanes
    }

    /// Mutably borrow the static lane set for this concrete shape.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut StaticIoRuntime<I, O, INPUTS, OUTPUTS> {
        &mut self.lanes
    }

    /// Borrow one static lane by index.
    #[inline]
    pub fn lane(&self, i: usize) -> Result<&StaticIoLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes.lane(i)
    }

    /// Mutably borrow one static lane by index.
    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut StaticIoLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes.lane_mut(i)
    }

    /// Run a closure against one lane in this concrete shape bucket.
    #[inline]
    pub fn run_on<R>(
        &mut self, i: usize, f: impl FnOnce(&mut StaticIoLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        self.lanes.run_on(i, f)
    }

    /// Run every lane in this bucket `runs` times to prime ORT shape and memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        self.lanes.prime(runs)
    }
}

enum DynamicSessions {
    Shared(Arc<Session>),
    Replicated(Vec<Arc<Session>>),
}

/// Dynamic-shape runtime backed by fixed, bind-once shape buckets.
///
/// A new concrete shape pays bucket construction cost: tensor allocation plus IoBinding setup.
/// Repeated shapes reuse the cached [`StaticIoRuntime`] and run through the same zero-copy,
/// caller-scheduled lane API as static-shape serving. The runtime itself is intentionally
/// `&mut self` based, so services can shard one instance per worker/core without shared locks.
pub struct DynamicIoRuntime<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    sessions: DynamicSessions,
    input_mem: MemoryInfo,
    output_mem: MemoryInfo,
    options: DynamicIoOptions,
    lane_count: usize,
    buckets: Vec<ShapeBucket<I, O, INPUTS, OUTPUTS>>,
    hot_bucket: Option<usize>,
    tick: u64,
}

impl<T> Runtime<T>
where
    T: TensorElement + Clone + Default,
{
    /// Build a static lane set with one shared session and `lanes` independent bindings.
    pub fn shared_session(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize,
    ) -> Result<Self> {
        Self::from_shared_session(session, mem, input_shapes, output_shapes, lanes)
    }

    /// Build a static lane set with one shared session and an explicit buffer policy.
    pub fn shared_session_with_buffer_policy(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize, policy: LaneBufferPolicy,
    ) -> Result<Self> {
        Self::from_shared_session_with_buffer_policy(
            session,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            policy,
        )
    }

    /// Alias for [`Self::shared_session`].
    pub fn from_shared_session(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize,
    ) -> Result<Self> {
        Self::from_shared_session_with_buffer_policy(
            session,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a fixed shared-session lane set with an explicit buffer policy.
    pub fn from_shared_session_with_buffer_policy(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize, policy: LaneBufferPolicy,
    ) -> Result<Self> {
        let lanes = build_shared_lanes(
            session,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            policy,
            "Runtime",
        )?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::SharedSession,
        })
    }

    /// Build a static lane set from already-created replicated sessions.
    pub fn from_sessions(
        sessions: Vec<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
    ) -> Result<Self> {
        Self::from_sessions_with_buffer_policy(
            sessions,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a static lane set from already-created sessions with an explicit buffer policy.
    pub fn from_sessions_with_buffer_policy(
        sessions: Vec<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(-1, "Runtime requires at least one session"));
        }
        let lanes = sessions
            .into_iter()
            .map(|session| Lane::new(Arc::new(session), mem, input_shapes, output_shapes, policy))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build a fixed replicated-session lane set with a caller-supplied session factory.
    pub fn from_session_factory<F>(
        lanes: usize, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        factory: F,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        Self::from_session_factory_with_buffer_policy(
            lanes,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            factory,
        )
    }

    /// Build a replicated-session lane set with an explicit buffer policy.
    pub fn from_session_factory_with_buffer_policy<F>(
        lanes: usize, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: LaneBufferPolicy, mut factory: F,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        if lanes == 0 {
            return Err(Error::new(-1, "Runtime requires at least one lane"));
        }
        let sessions = (0..lanes).map(&mut factory).collect::<Result<Vec<_>>>()?;
        Self::from_sessions_with_buffer_policy(sessions, mem, input_shapes, output_shapes, policy)
    }

    /// Build a fixed replicated-session lane set from a model path.
    pub fn replicated_sessions(
        env: &Environment, model_path: &str, opts: SessionOptions, mem: &MemoryInfo,
        input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize,
    ) -> Result<Self> {
        Self::from_session_factory(lanes, mem, input_shapes, output_shapes, |_| {
            Session::new(env, model_path, opts.clone())
        })
    }

    /// Build a fixed replicated-session lane set with an explicit buffer policy.
    #[allow(clippy::too_many_arguments)]
    pub fn replicated_sessions_with_buffer_policy(
        env: &Environment, model_path: &str, opts: SessionOptions, mem: &MemoryInfo,
        input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize, policy: LaneBufferPolicy,
    ) -> Result<Self> {
        Self::from_session_factory_with_buffer_policy(
            lanes,
            mem,
            input_shapes,
            output_shapes,
            policy,
            |_| Session::new(env, model_path, opts.clone()),
        )
    }

    /// Build a fixed replicated-session lane set whose lanes share one prepacked cache.
    #[allow(clippy::too_many_arguments)]
    pub fn replicated_sessions_with_prepacked_weights(
        env: &Environment, model_path: &str, opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer, mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], lanes: usize,
    ) -> Result<Self> {
        Self::from_session_factory(lanes, mem, input_shapes, output_shapes, |_| {
            Session::new_with_prepacked_weights(env, model_path, opts.clone(), prepacked)
        })
    }

    /// Build a fixed replicated-session lane set with shared prepacked weights and an
    /// explicit buffer policy.
    #[allow(clippy::too_many_arguments)]
    pub fn replicated_sessions_with_prepacked_weights_and_buffer_policy(
        env: &Environment, model_path: &str, opts: SessionOptions,
        prepacked: &PrepackedWeightsContainer, mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], lanes: usize, policy: LaneBufferPolicy,
    ) -> Result<Self> {
        Self::from_session_factory_with_buffer_policy(
            lanes,
            mem,
            input_shapes,
            output_shapes,
            policy,
            |_| Session::new_with_prepacked_weights(env, model_path, opts.clone(), prepacked),
        )
    }
}

impl<T: TensorElement> Runtime<T> {
    /// Number of lanes in this fixed set.
    #[inline]
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Whether this lane set is empty. Public constructors reject this.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Session arrangement used to create this lane set.
    #[inline]
    pub fn session_mode(&self) -> RuntimeMode {
        self.mode
    }

    /// Borrow all lanes for caller-side scheduling.
    #[inline]
    pub fn lanes(&self) -> &[Lane<T>] {
        &self.lanes
    }

    /// Mutably borrow all lanes for caller-side scheduling.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [Lane<T>] {
        &mut self.lanes
    }

    /// Borrow one lane by index.
    #[inline]
    pub fn lane(&self, i: usize) -> Result<&Lane<T>> {
        self.lanes
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane index {i} out of range")))
    }

    /// Mutably borrow one lane by index.
    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut Lane<T>> {
        self.lanes
            .get_mut(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane index {i} out of range")))
    }

    /// Consume the set and return the raw lanes.
    #[inline]
    pub fn into_lanes(self) -> Vec<Lane<T>> {
        self.lanes
    }

    /// Run a closure against a specific lane.
    #[inline]
    pub fn run_on<R>(&mut self, i: usize, f: impl FnOnce(&mut Lane<T>) -> Result<R>) -> Result<R> {
        f(self.lane_mut(i)?)
    }

    /// Run every lane `runs` times to prime ORT shape and memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for lane in &mut self.lanes {
            lane.prime(runs)?;
        }
        Ok(())
    }

    /// Consume this value. Kept as a no-op migration helper from the previous checkout runtime.
    #[inline]
    pub fn into_lane_set(self) -> Self {
        self
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> StaticIoRuntime<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build a static lane set with one shared session and typed I/O lanes.
    pub fn shared_session(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], lanes: usize,
    ) -> Result<Self> {
        Self::shared_session_with_buffer_policy(
            session,
            mem,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a shared-session set with explicit memory descriptors and buffer policies.
    #[allow(clippy::too_many_arguments)]
    pub fn shared_session_with_buffer_policy(
        session: Arc<Session>, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], lanes: usize,
        input_policy: LaneBufferPolicy, output_policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(-1, "StaticIoRuntime requires at least one lane"));
        }
        let lanes = (0..lanes)
            .map(|_| {
                StaticIoLane::with_buffer_policy(
                    session.clone(),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::SharedSession,
        })
    }

    /// Build a fixed set from already-created replicated sessions.
    pub fn from_sessions(
        sessions: Vec<Session>, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::from_sessions_with_buffer_policy(
            sessions,
            mem,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a replicated-session set with explicit memory descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_sessions_with_buffer_policy(
        sessions: Vec<Session>, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        input_policy: LaneBufferPolicy, output_policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "StaticIoRuntime requires at least one session",
            ));
        }
        let lanes = sessions
            .into_iter()
            .map(|session| {
                StaticIoLane::with_buffer_policy(
                    Arc::new(session),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build a fixed set from already-shared replicated sessions.
    pub fn from_session_arcs(
        sessions: &[Arc<Session>], mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::from_session_arcs_with_buffer_policy(
            sessions,
            mem,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
        )
    }

    /// Build a fixed set from already-shared replicated sessions with explicit memory
    /// descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_arcs_with_buffer_policy(
        sessions: &[Arc<Session>], input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        input_policy: LaneBufferPolicy, output_policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "StaticIoRuntime requires at least one session",
            ));
        }
        let lanes = sessions
            .iter()
            .map(|session| {
                StaticIoLane::with_buffer_policy(
                    session.clone(),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build replicated sessions from a factory.
    pub fn from_session_factory<F>(
        lanes: usize, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], factory: F,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        Self::from_session_factory_with_buffer_policy(
            lanes,
            mem,
            mem,
            input_shapes,
            output_shapes,
            LaneBufferPolicy::Auto,
            LaneBufferPolicy::Auto,
            factory,
        )
    }

    /// Build replicated sessions from a factory with explicit memory descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_factory_with_buffer_policy<F>(
        lanes: usize, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        input_policy: LaneBufferPolicy, output_policy: LaneBufferPolicy, mut factory: F,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        if lanes == 0 {
            return Err(Error::new(-1, "StaticIoRuntime requires at least one lane"));
        }
        let sessions = (0..lanes).map(&mut factory).collect::<Result<Vec<_>>>()?;
        Self::from_sessions_with_buffer_policy(
            sessions,
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            input_policy,
            output_policy,
        )
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    StaticIoRuntime<I, O, INPUTS, OUTPUTS>
{
    #[inline]
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    #[inline]
    pub fn session_mode(&self) -> RuntimeMode {
        self.mode
    }

    #[inline]
    pub fn lanes(&self) -> &[StaticIoLane<I, O, INPUTS, OUTPUTS>] {
        &self.lanes
    }

    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [StaticIoLane<I, O, INPUTS, OUTPUTS>] {
        &mut self.lanes
    }

    #[inline]
    pub fn lane(&self, i: usize) -> Result<&StaticIoLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: static I/O lane index {i} out of range")))
    }

    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut StaticIoLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get_mut(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: static I/O lane index {i} out of range")))
    }

    #[inline]
    pub fn into_lanes(self) -> Vec<StaticIoLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
    }

    #[inline]
    pub fn run_on<R>(
        &mut self, i: usize, f: impl FnOnce(&mut StaticIoLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        f(self.lane_mut(i)?)
    }

    /// Run every lane `runs` times to prime ORT shape and memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for lane in &mut self.lanes {
            lane.prime(runs)?;
        }
        Ok(())
    }

    /// Set whether all lanes rebind inputs before every run.
    #[inline]
    pub fn set_rebind_inputs_each_run(&mut self, enabled: bool) {
        for lane in &mut self.lanes {
            lane.set_rebind_inputs_each_run(enabled);
        }
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build a dynamic-shape runtime with one shared session and `lanes` static lanes per shape.
    pub fn shared_session(session: Arc<Session>, mem: MemoryInfo, lanes: usize) -> Result<Self> {
        let output_mem = mem.try_clone_descriptor()?;
        Self::shared_session_with_options(
            session,
            mem,
            output_mem,
            lanes,
            DynamicIoOptions::default(),
        )
    }

    /// Build a dynamic-shape runtime with one shared session, explicit memory descriptors, and
    /// shape-cache options.
    pub fn shared_session_with_options(
        session: Arc<Session>, input_mem: MemoryInfo, output_mem: MemoryInfo, lanes: usize,
        options: DynamicIoOptions,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one lane",
            ));
        }
        Ok(Self {
            sessions: DynamicSessions::Shared(session),
            input_mem,
            output_mem,
            options: options.validate()?,
            lane_count: lanes,
            buckets: Vec::new(),
            hot_bucket: None,
            tick: 0,
        })
    }

    /// Build a dynamic-shape runtime from replicated sessions.
    pub fn from_sessions(sessions: Vec<Session>, mem: MemoryInfo) -> Result<Self> {
        let output_mem = mem.try_clone_descriptor()?;
        Self::from_sessions_with_options(sessions, mem, output_mem, DynamicIoOptions::default())
    }

    /// Build a dynamic-shape runtime from replicated sessions with explicit memory descriptors
    /// and shape-cache options.
    pub fn from_sessions_with_options(
        sessions: Vec<Session>, input_mem: MemoryInfo, output_mem: MemoryInfo,
        options: DynamicIoOptions,
    ) -> Result<Self> {
        let sessions = sessions.into_iter().map(Arc::new).collect::<Vec<_>>();
        Self::from_session_arcs_with_options(sessions, input_mem, output_mem, options)
    }

    /// Build a dynamic-shape runtime from already-shared replicated sessions.
    pub fn from_session_arcs(sessions: Vec<Arc<Session>>, mem: MemoryInfo) -> Result<Self> {
        let output_mem = mem.try_clone_descriptor()?;
        Self::from_session_arcs_with_options(sessions, mem, output_mem, DynamicIoOptions::default())
    }

    /// Build a dynamic-shape runtime from already-shared replicated sessions with explicit
    /// memory descriptors and shape-cache options.
    pub fn from_session_arcs_with_options(
        sessions: Vec<Arc<Session>>, input_mem: MemoryInfo, output_mem: MemoryInfo,
        options: DynamicIoOptions,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one session",
            ));
        }
        let lane_count = sessions.len();
        Ok(Self {
            sessions: DynamicSessions::Replicated(sessions),
            input_mem,
            output_mem,
            options: options.validate()?,
            lane_count,
            buckets: Vec::new(),
            hot_bucket: None,
            tick: 0,
        })
    }

    /// Build a replicated-session dynamic runtime with a caller-supplied session factory.
    pub fn from_session_factory<F>(lanes: usize, mem: MemoryInfo, factory: F) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        let output_mem = mem.try_clone_descriptor()?;
        Self::from_session_factory_with_options(
            lanes,
            mem,
            output_mem,
            DynamicIoOptions::default(),
            factory,
        )
    }

    /// Build a replicated-session dynamic runtime with explicit memory descriptors and options.
    pub fn from_session_factory_with_options<F>(
        lanes: usize, input_mem: MemoryInfo, output_mem: MemoryInfo, options: DynamicIoOptions,
        mut factory: F,
    ) -> Result<Self>
    where
        F: FnMut(usize) -> Result<Session>,
    {
        if lanes == 0 {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one lane",
            ));
        }
        let sessions = (0..lanes).map(&mut factory).collect::<Result<Vec<_>>>()?;
        Self::from_sessions_with_options(sessions, input_mem, output_mem, options)
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1).max(1);
        self.tick
    }

    fn build_lane_set(
        &self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<StaticIoRuntime<I, O, INPUTS, OUTPUTS>> {
        let mut lanes = match &self.sessions {
            DynamicSessions::Shared(session) => StaticIoRuntime::shared_session_with_buffer_policy(
                session.clone(),
                &self.input_mem,
                &self.output_mem,
                input_shapes,
                output_shapes,
                self.lane_count,
                self.options.input_policy,
                self.options.output_policy,
            ),
            DynamicSessions::Replicated(sessions) => {
                StaticIoRuntime::from_session_arcs_with_buffer_policy(
                    sessions,
                    &self.input_mem,
                    &self.output_mem,
                    input_shapes,
                    output_shapes,
                    self.options.input_policy,
                    self.options.output_policy,
                )
            },
        }?;
        lanes.set_rebind_inputs_each_run(self.options.rebind_inputs_each_run);
        Ok(lanes)
    }

    fn find_bucket_index(
        &self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Option<usize> {
        if self.buckets.len() > 1 {
            if let Some(i) = self.hot_bucket {
                if self
                    .buckets
                    .get(i)
                    .is_some_and(|bucket| bucket.key.matches(input_shapes, output_shapes))
                {
                    return Some(i);
                }
            }
        }
        self.buckets
            .iter()
            .position(|bucket| bucket.key.matches(input_shapes, output_shapes))
    }

    fn evict_one_bucket_if_full(&mut self) {
        if self.buckets.len() < self.options.max_buckets {
            return;
        }
        self.hot_bucket = None;
        if let Some((oldest, _)) = self
            .buckets
            .iter()
            .enumerate()
            .min_by_key(|(_, bucket)| bucket.last_used)
        {
            self.buckets.swap_remove(oldest);
        }
    }

    /// Get the bucket for concrete shapes, creating it on first use.
    ///
    /// Cache hits do not allocate in Rust: shape slices are compared directly against cached
    /// keys. Misses allocate tensor buffers and bind a new static lane set.
    pub fn get_or_create_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<&mut ShapeBucket<I, O, INPUTS, OUTPUTS>> {
        if let Some(i) = self.find_bucket_index(input_shapes, output_shapes) {
            let tick = self.next_tick();
            self.buckets[i].last_used = tick;
            self.hot_bucket = Some(i);
            return Ok(&mut self.buckets[i]);
        }

        self.evict_one_bucket_if_full();
        let key = ShapeKey::new(input_shapes, output_shapes);
        let lanes = self.build_lane_set(input_shapes, output_shapes)?;
        let last_used = self.next_tick();
        self.buckets.push(ShapeBucket {
            key,
            lanes,
            last_used,
        });
        self.hot_bucket = Some(self.buckets.len() - 1);
        self.buckets.last_mut().ok_or_else(|| {
            Error::new(
                -1,
                "zrt: failed to access newly created dynamic shape bucket",
            )
        })
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Number of concrete shape buckets currently cached.
    #[inline]
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Maximum concrete shape buckets retained by this runtime.
    #[inline]
    pub fn max_buckets(&self) -> usize {
        self.options.max_buckets
    }

    /// Number of static lanes in every shape bucket.
    #[inline]
    pub fn lane_count(&self) -> usize {
        self.lane_count
    }

    /// Session arrangement used by this runtime.
    #[inline]
    pub fn session_mode(&self) -> RuntimeMode {
        match &self.sessions {
            DynamicSessions::Shared(_) => RuntimeMode::SharedSession,
            DynamicSessions::Replicated(_) => RuntimeMode::ReplicatedSessions,
        }
    }

    /// Current shape-cache options.
    #[inline]
    pub fn options(&self) -> DynamicIoOptions {
        self.options
    }

    /// Borrow cached shape buckets.
    #[inline]
    pub fn buckets(&self) -> &[ShapeBucket<I, O, INPUTS, OUTPUTS>] {
        &self.buckets
    }

    /// Borrow cached shape buckets mutably.
    #[inline]
    pub fn buckets_mut(&mut self) -> &mut [ShapeBucket<I, O, INPUTS, OUTPUTS>] {
        &mut self.buckets
    }

    /// Drop all cached shape buckets. Existing borrowed lanes must be released first.
    #[inline]
    pub fn clear_buckets(&mut self) {
        self.hot_bucket = None;
        self.buckets.clear();
    }

    /// Borrow an already-cached bucket without creating it.
    #[inline]
    pub fn bucket(
        &self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Option<&ShapeBucket<I, O, INPUTS, OUTPUTS>> {
        self.find_bucket_index(input_shapes, output_shapes)
            .map(|i| &self.buckets[i])
    }

    /// Borrow an already-cached bucket mutably without creating it.
    pub fn bucket_mut(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Option<&mut ShapeBucket<I, O, INPUTS, OUTPUTS>> {
        let i = self.find_bucket_index(input_shapes, output_shapes)?;
        let tick = self.next_tick();
        self.buckets[i].last_used = tick;
        self.hot_bucket = Some(i);
        Some(&mut self.buckets[i])
    }

    /// Remove one cached shape bucket if present.
    pub fn remove_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> bool {
        let Some(i) = self.find_bucket_index(input_shapes, output_shapes) else {
            return false;
        };
        self.hot_bucket = None;
        self.buckets.swap_remove(i);
        true
    }

    /// Create or find a bucket and run every lane `runs` times.
    pub fn prime_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], runs: usize,
    ) -> Result<()> {
        self.get_or_create_bucket(input_shapes, output_shapes)?
            .prime(runs)
    }

    /// Run a closure against one lane in the matching shape bucket, creating the bucket on first
    /// use.
    #[inline]
    pub fn run_on<R>(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], lane: usize,
        f: impl FnOnce(&mut StaticIoLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        self.get_or_create_bucket(input_shapes, output_shapes)?
            .run_on(lane, f)
    }
}
