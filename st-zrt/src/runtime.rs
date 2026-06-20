//! Transport-agnostic serving runtime: fixed, exclusive inference lanes.
//!
//! [`ZrtRuntime`] is intentionally not an HTTP/gRPC server. It is the reusable inference
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

/// How a [`ZrtRuntime`] may arrange ONNX Runtime sessions across lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZrtRuntimeMode {
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
pub struct ZrtLane<T: TensorElement> {
    // Drop the binding before the tensor buffers whose ORT value handles it references, and
    // before releasing this lane's session reference.
    binding: IoBinding,
    inputs: Vec<TensorBuffer<T>>,
    outputs: Vec<TensorBuffer<T>>,
    session: Arc<Session>,
}

impl<T> ZrtLane<T>
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

impl<T: TensorElement> ZrtLane<T> {
    /// Execute this lane's prepared binding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.session.run_binding(&self.binding)
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

fn build_shared_lanes<T>(
    session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
    lanes: usize, policy: LaneBufferPolicy, what: &'static str,
) -> Result<Vec<ZrtLane<T>>>
where
    T: TensorElement + Clone + Default,
{
    if lanes == 0 {
        return Err(Error::new(-1, format!("{what} requires at least one lane")));
    }
    (0..lanes)
        .map(|_| ZrtLane::new(session.clone(), mem, input_shapes, output_shapes, policy))
        .collect()
}

/// A fixed, caller-scheduled set of exclusive inference lanes.
///
/// It is for services that already have a deterministic lane assignment strategy, such as
/// sharded workers, per-core loops, or an external scheduler. The hot path is direct
/// `lane_mut(i)`/slice access plus [`ZrtLane::run`]; ZRT does not keep a checkout pool or
/// lock around lane selection.
pub struct ZrtLaneSet<T: TensorElement> {
    lanes: Vec<ZrtLane<T>>,
    mode: ZrtRuntimeMode,
}

impl<T> ZrtLaneSet<T>
where
    T: TensorElement + Clone + Default,
{
    /// Build a fixed lane set with one shared session and `lanes` independent bindings.
    pub fn shared_session(
        session: Arc<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize,
    ) -> Result<Self> {
        Self::from_shared_session(session, mem, input_shapes, output_shapes, lanes)
    }

    /// Build a fixed lane set with one shared session and an explicit buffer policy.
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
            "ZrtLaneSet",
        )?;
        Ok(Self {
            lanes,
            mode: ZrtRuntimeMode::SharedSession,
        })
    }

    /// Build a fixed lane set from already-created replicated sessions.
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

    /// Build a fixed lane set from already-created sessions with an explicit buffer policy.
    pub fn from_sessions_with_buffer_policy(
        sessions: Vec<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], policy: LaneBufferPolicy,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(-1, "ZrtLaneSet requires at least one session"));
        }
        let lanes = sessions
            .into_iter()
            .map(|session| {
                ZrtLane::new(Arc::new(session), mem, input_shapes, output_shapes, policy)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: ZrtRuntimeMode::ReplicatedSessions,
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
            return Err(Error::new(-1, "ZrtLaneSet requires at least one lane"));
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

impl<T: TensorElement> ZrtLaneSet<T> {
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
    pub fn session_mode(&self) -> ZrtRuntimeMode {
        self.mode
    }

    /// Borrow all lanes for caller-side scheduling.
    #[inline]
    pub fn lanes(&self) -> &[ZrtLane<T>] {
        &self.lanes
    }

    /// Mutably borrow all lanes for caller-side scheduling.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [ZrtLane<T>] {
        &mut self.lanes
    }

    /// Borrow one lane by index.
    #[inline]
    pub fn lane(&self, i: usize) -> Result<&ZrtLane<T>> {
        self.lanes
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane index {i} out of range")))
    }

    /// Mutably borrow one lane by index.
    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut ZrtLane<T>> {
        self.lanes
            .get_mut(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: lane index {i} out of range")))
    }

    /// Consume the set and return the raw lanes.
    #[inline]
    pub fn into_lanes(self) -> Vec<ZrtLane<T>> {
        self.lanes
    }

    /// Run a closure against a specific lane.
    #[inline]
    pub fn run_on<R>(
        &mut self, i: usize, f: impl FnOnce(&mut ZrtLane<T>) -> Result<R>,
    ) -> Result<R> {
        f(self.lane_mut(i)?)
    }

    /// Consume this value. Kept as a no-op migration helper from the previous checkout runtime.
    #[inline]
    pub fn into_lane_set(self) -> Self {
        self
    }
}

/// Lock-free serving runtime: a fixed, caller-scheduled lane set.
///
/// `ZrtRuntime` is currently an alias for [`ZrtLaneSet`]. The runtime surface is direct lane
/// access (`lane_mut`, `lanes_mut`, `run_on`) rather than a cloneable checkout pool, so ZRT does
/// not allocate or lock while selecting a lane.
pub type ZrtRuntime<T> = ZrtLaneSet<T>;
