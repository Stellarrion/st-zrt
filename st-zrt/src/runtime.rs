//! Transport-agnostic serving runtime: fixed, exclusive inference lanes.
//!
//! [`Runtime`] is intentionally not an HTTP/gRPC server. It is the reusable inference
//! service underneath one: a fixed set of zero-copy lanes, each with its own input/output
//! buffers and IoBinding. Server code assigns requests to lanes explicitly, mutates input
//! buffers, runs the lane, then reads outputs. ZRT performs no checkout locking.

use crate::element::TensorElement;
use crate::environment::Environment;
use crate::io_binding::IoBinding;
use crate::memory::{MemoryInfo, MemoryInfoSnapshot};
use crate::prepacked::PrepackedWeightsContainer;
use crate::run_options::{MaterializedRunOptions, RunOptions};
use crate::session::{
    CapturedGraphLease, CapturedGraphRunGuard, IoDirection, RunFuture, Session, lane_tensor_buffer,
};
use crate::session_options::SessionOptions;
use crate::shape_plan::{OutputPolicy, ServingShapePlan};
use crate::tensor::{AllocatedTensor, BufferSpec, TensorBuffer};
use crate::{Error, Result};
use std::cell::UnsafeCell;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorBufferAudit {
    pub direction: IoDirection,
    pub index: usize,
    pub element_type: crate::ElementType,
    pub element_count: usize,
    pub byte_len: usize,
    pub rust_ptr: usize,
    pub ort_ptr: usize,
    pub pointer_identity: bool,
    pub memory_info: MemoryInfoSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneHotPathAudit {
    pub input_count: usize,
    pub output_count: usize,
    pub rebind_inputs_each_run: bool,
    pub input_names_cached: bool,
    pub inputs: Vec<TensorBufferAudit>,
    pub outputs: Vec<TensorBufferAudit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticIoRunTimings {
    pub rebind_inputs: Duration,
    pub device_input_refresh: Duration,
    pub ort_run: Duration,
    pub bound_input_sync: Duration,
    pub run_with_binding: Duration,
    pub bound_output_sync: Duration,
    pub total: Duration,
}

fn audit_tensor_buffer<T: TensorElement>(
    direction: IoDirection, index: usize, buffer: &TensorBuffer<T>,
) -> Result<TensorBufferAudit> {
    let rust_ptr = buffer.as_slice().as_ptr() as usize;
    let ort_ptr = buffer.engine_data_ptr()? as usize;
    Ok(TensorBufferAudit {
        direction,
        index,
        element_type: T::ELEM,
        element_count: buffer.len(),
        byte_len: buffer.byte_len()?,
        rust_ptr,
        ort_ptr,
        pointer_identity: rust_ptr == ort_ptr,
        memory_info: buffer.memory_info()?,
    })
}

fn assert_tensor_buffer_zero_copy<T: TensorElement>(
    what: &str, index: usize, buffer: &TensorBuffer<T>,
) -> Result<()> {
    let audit = audit_tensor_buffer(IoDirection::Input, index, buffer)?;
    if !audit.pointer_identity {
        return Err(Error::new(
            -1,
            format!(
                "zrt: {what} {index} is not zero-copy: rust_ptr=0x{:x}, ort_ptr=0x{:x}",
                audit.rust_ptr, audit.ort_ptr
            ),
        ));
    }
    if !audit.memory_info.is_host_accessible() {
        return Err(Error::new(
            -1,
            format!(
                "zrt: {what} {index} is not host-accessible: {} device {} ({:?}/{:?})",
                audit.memory_info.name,
                audit.memory_info.device_id,
                audit.memory_info.alloc_type,
                audit.memory_info.mem_type
            ),
        ));
    }
    Ok(())
}

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
    session: Session,
}

impl<T: TensorElement> std::fmt::Debug for Lane<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane")
            .field("inputs", &self.inputs.len())
            .field("outputs", &self.outputs.len())
            .field("session", &self.session.as_ptr())
            .finish_non_exhaustive()
    }
}

/// One exclusive static-shape I/O lane with independently typed inputs and outputs.
///
/// This covers models whose input tensors share one scalar type `I` and output tensors share
/// another scalar type `O`, while the input/output arity is fixed in the Rust type.
///
/// The arity is part of the type, so services can keep concrete lane types for hot paths while
/// still using different scalar types for model inputs and outputs. Each lane owns stable input
/// buffers, stable output buffers, and one pre-bound IoBinding.
pub struct ServingLane<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    /// Owned lane state. `Option` so [`Drop`](Drop::drop) can move it out and deliberately leak it
    /// when an unfenced in-flight run may still dereference every provider-visible resource (the
    /// same policy as the owned-run tokens above).
    inner: Option<Box<ServingLaneInner<I, O, INPUTS, OUTPUTS>>>,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::ops::Deref
    for ServingLane<I, O, INPUTS, OUTPUTS>
{
    type Target = ServingLaneInner<I, O, INPUTS, OUTPUTS>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .expect("zrt: serving lane state was already consumed")
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    std::ops::DerefMut for ServingLane<I, O, INPUTS, OUTPUTS>
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
            .as_mut()
            .expect("zrt: serving lane state was already consumed")
    }
}

/// Owned field set of one [`ServingLane`]; split out so lane teardown can leak the whole set when
/// a pending provider run could not be fenced (see [`Drop for ServingLane`](ServingLane)).
///
/// Reached only through [`ServingLane`]'s `Deref`/`DerefMut`. `pub` + hidden solely because a
/// public `Deref` impl must name its `Target`; it is never constructed or named outside this
/// module and is not part of the stable surface.
#[doc(hidden)]
pub struct ServingLaneInner<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    // Field drop order is load-bearing: `binding` and `device_inputs` reference the session's
    // allocator/handles, so they must drop BEFORE `session`. `session` drops before the (raw, no-drop)
    // `stream`. Releasing device tensors into a freed session allocator would be use-after-free.
    binding: IoBinding,
    inputs: [TensorBuffer<I>; INPUTS],
    outputs: [TensorBuffer<O>; OUTPUTS],
    #[cfg(feature = "cuda")]
    device_outputs: Option<Vec<AllocatedTensor<O>>>,
    input_names: [CString; INPUTS],
    /// CUDA-resident input buffers for the device-input mode (cuda-graph-correct path). When `Some`,
    /// these are bound to the IoBinding (the graph bakes their device pointers) and the host `inputs`
    /// act as staging — [`ServingLane::run`] copies each staging buffer → its device buffer before the
    /// run, so a replayed graph reads fresh data. `None` = host-input mode (host `inputs` are bound).
    device_inputs: Option<Vec<AllocatedTensor<I>>>,
    /// Optional per-run options carrying a `gpu_graph_id` CUDA-graph annotation.
    /// When `Some`, [`ServingLane::run`] uses [`Session::run_binding_with`] so ORT
    /// captures/replays one graph per shape instead of one graph for all shapes.
    run_opts: Option<MaterializedRunOptions>,
    /// Frozen no-EP-sync counterpart used only by enqueue paths.
    enqueued_run_opts: MaterializedRunOptions,
    /// Owned ORT sync stream retained by both frozen run-option compositions.
    #[cfg(feature = "ep")]
    sync_stream: Option<Arc<crate::SyncStream>>,
    graph_id: Option<i32>,
    /// Whether a run has already completed with the current `graph_id` — i.e. ORT has captured
    /// this lane's graph and the next run replays instead of capturing. Reset when a new
    /// `graph_id` is assigned. `DynamicIoRuntime` captures eagerly at bucket creation, so every
    /// bucket lane reports `true` before its first served run.
    graph_captured: bool,
    /// Per-graph atomic lease cached at setup; no session HashMap/Mutex lookup on runs.
    graph_lease: Option<CapturedGraphLease>,
    /// Owned graph run guard held from enqueue until output synchronization completes.
    in_flight_graph_guard: Option<CapturedGraphRunGuard>,
    /// Event recorded after an unsynchronized CUDA graph launch on the exact retained user stream.
    #[cfg(feature = "cuda")]
    completion_event: Option<crate::CudaEvent>,
    /// Reusable event recorded after a downstream GPU consumer queued through a chain token.
    #[cfg(feature = "cuda")]
    downstream_completion_event: Option<crate::CudaEvent>,
    #[cfg(feature = "cuda")]
    completion_event_recorded: bool,
    /// Prevents staging-buffer reuse while any execution is pending, including non-graph runs.
    in_flight: bool,
    /// Fixed dynamic-runtime recovery slot; assigned when a bucket is built.
    recovery_slot: Option<usize>,
    session: Session,
    rebind_inputs_each_run: bool,
    /// The CUDA stream retained by the session and used by ORT for graph replay, used to
    /// sequence the host→device refresh before each run. Non-null only in device-input mode. (cuda.)
    #[cfg(feature = "cuda")]
    stream: *mut std::ffi::c_void,
    /// The CUDA device ID for this lane's stream. Used by `ensure_device_active` before manual H2D.
    #[cfg(feature = "cuda")]
    cuda_device_id: Option<i32>,
}

/// Runtime-local, generation-checked handle to one prebuilt dynamic shape bucket.
///
/// Resolve it once with [`DynamicIoRuntime::prepared_bucket_id`] and use it with
/// [`DynamicIoRuntime::enqueue_prepared`]. Lookup is O(1). A retired slot increments its generation,
/// so a stale handle can never alias a later bucket that reuses the same slot. It is intentionally
/// distinct from [`crate::ShapeId`], which identifies a canonical bucket in a
/// [`ServingShapePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PreparedBucketId {
    slot: u32,
    generation: u32,
}

#[derive(Debug, Clone, Copy)]
struct PreparedBucketSlot {
    generation: u32,
    bucket_index: Option<usize>,
}

/// Result of a nonblocking completion query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStatus {
    /// Provider/device work is still using the lane's resources.
    Pending,
    /// The lane is complete and its outputs/resources are safe to access or reuse.
    Ready,
}

/// Error returned by [`ServingLane::enqueue`] when the enqueue fails.
///
/// The lane is returned alongside the underlying error so the caller can retry after fencing or
/// drop it (its [`Drop`] fences pending work best-effort). The lane may be in-flight if the
/// failure happened after ORT enqueued provider work; synchronize it before reuse.
#[derive(Debug)]
pub struct LaneEnqueueError<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    /// The underlying enqueue failure.
    pub error: Error,
    /// The lane that failed to enqueue, still owned by the caller.
    pub lane: ServingLane<I, O, INPUTS, OUTPUTS>,
}

/// An owned in-flight execution of one lane.
///
/// [`ServingLane::enqueue`] moves the lane into this token, so the token owns every staging
/// buffer, the `IoBinding`, and the captured-graph lease involved in the pending run. No
/// reference into the owning runtime or bucket is retained — the token is a plain owned value
/// with no borrows and no interior pointers, so any number of lanes of the same bucket can be
/// enqueued and held concurrently. That is what makes the multi-lane pipeline representable: a
/// token that *borrowed* the lane would keep the whole owning runtime borrowed, and enqueueing
/// the next lane would not compile.
///
/// Call [`Self::synchronize`] to fence device work and regain the lane. Dropping the token
/// performs the same synchronization best-effort, then destroys the lane (while the token lives
/// the lane is not reachable through the runtime). If every fence fails, the lane is deliberately
/// leaked so provider-visible resources are never freed underneath pending device work.
#[must_use = "an in-flight run must be synchronized or dropped before the lane can be reused"]
pub struct InFlightRun<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    lane: Option<ServingLane<I, O, INPUTS, OUTPUTS>>,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    InFlightRun<I, O, INPUTS, OUTPUTS>
{
    /// Query completion without blocking while retaining token ownership.
    #[inline]
    pub fn try_complete(&mut self) -> Result<CompletionStatus> {
        self.lane
            .as_mut()
            .expect("zrt: in-flight lane missing")
            .try_complete()
    }

    /// Wait for bound outputs/device work and return the completed lane.
    ///
    /// On error the lane is poisoned (its staging/outputs and graph lease are not reusable) and
    /// dropped; callers that need the lane back on failure should use
    /// [`DynamicIoRuntime::complete_owned`], which returns the lane to its bucket even when
    /// synchronization or consumption fails.
    pub fn synchronize(mut self) -> Result<ServingLane<I, O, INPUTS, OUTPUTS>> {
        let mut lane = self.lane.take().expect("zrt: in-flight lane missing");
        match lane.synchronize_outputs() {
            Ok(()) => Ok(lane),
            Err(error) => {
                if lane.in_flight {
                    // Both the output fence and device fallback failed. Keep provider-visible
                    // resources alive rather than dropping them underneath pending device work.
                    std::mem::forget(lane);
                }
                Err(error)
            },
        }
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for InFlightRun<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InFlightRun")
            .field("has_lane", &self.lane.is_some())
            .finish()
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> Drop
    for InFlightRun<I, O, INPUTS, OUTPUTS>
{
    fn drop(&mut self) {
        let Some(mut lane) = self.lane.take() else {
            return;
        };
        lane.finish_in_flight_best_effort();
        if lane.in_flight {
            // Never destroy buffers that an unfenced provider run may still dereference.
            std::mem::forget(lane);
        }
    }
}

/// An owned dynamic-runtime execution detached from its shape bucket.
///
/// Like [`InFlightRun`], this token owns its lane outright (no borrows, no interior pointers);
/// it additionally remembers the compact source [`PreparedBucketId`] so [`DynamicIoRuntime::complete_owned`] can
/// return the lane to the exact bucket it came from. It can therefore be stored alongside a
/// serving pipeline while other lanes are enqueued. Dropping it fences pending work best-effort
/// and queues the completed lane for automatic recovery by its runtime. If every fence fails, the
/// lane is deliberately leaked and remains accounted as detached so its graph cannot be retired.
#[must_use = "an owned run must be completed or dropped before its lane can be reused"]
pub struct OwnedDynamicIoRun<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    lane: Option<ServingLane<I, O, INPUTS, OUTPUTS>>,
    bucket_id: PreparedBucketId,
    recovery: Option<Arc<RecoverySlots<I, O, INPUTS, OUTPUTS>>>,
}

struct RecoveredDynamicLane<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    lane: ServingLane<I, O, INPUTS, OUTPUTS>,
    bucket_id: PreparedBucketId,
}

struct RecoveryCell<T> {
    value: UnsafeCell<Option<T>>,
    occupied: AtomicBool,
}

impl<T> RecoveryCell<T> {
    fn empty() -> Self {
        Self {
            value: UnsafeCell::new(None),
            occupied: AtomicBool::new(false),
        }
    }

    fn publish(&self, value: T) -> std::result::Result<(), T> {
        if self
            .occupied
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(value);
        }
        // SAFETY: the successful false->true transition gives this producer exclusive access. The
        // sole consumer reads only after the ready-stack Release/Acquire handoff below.
        unsafe { *self.value.get() = Some(value) };
        Ok(())
    }

    fn take(&self) -> Option<T> {
        // SAFETY: only DynamicIoRuntime consumes, its ready-stack Acquire observed the producer's
        // Release publication, and a node is removed before this cell is read.
        let value = unsafe { (&mut *self.value.get()).take() };
        self.occupied.store(false, Ordering::Release);
        value
    }
}

// SAFETY: producers reserve a cell atomically; its ready-stack Release/Acquire handoff transfers
// access to the sole consumer, which empties the cell before its lane can be reissued.
unsafe impl<T: Send> Sync for RecoveryCell<T> {}

#[cfg(test)]
mod recovery_cell_tests {
    use super::RecoveryCell;

    #[test]
    fn occupied_recovery_cell_returns_second_value_without_replacing_first() {
        let cell = RecoveryCell::empty();
        assert_eq!(cell.publish(7_u32), Ok(()));
        assert_eq!(cell.publish(9_u32), Err(9));
        assert_eq!(cell.take(), Some(7));
        assert_eq!(cell.publish(11_u32), Ok(()));
        assert_eq!(cell.take(), Some(11));
    }
}

struct RecoverySlots<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
{
    slots: Vec<RecoveryCell<RecoveredDynamicLane<I, O, INPUTS, OUTPUTS>>>,
    ready_head: AtomicUsize,
    ready_next: Vec<AtomicUsize>,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    RecoverySlots<I, O, INPUTS, OUTPUTS>
{
    fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity).map(|_| RecoveryCell::empty()).collect(),
            ready_head: AtomicUsize::new(usize::MAX),
            ready_next: (0..capacity)
                .map(|_| AtomicUsize::new(usize::MAX))
                .collect(),
        }
    }

    fn recover(&self, slot: usize, recovered: RecoveredDynamicLane<I, O, INPUTS, OUTPUTS>) {
        let Some(entry) = self.slots.get(slot) else {
            // Drop may run during unwinding. Preserve provider-visible ownership rather than panic
            // again if an internal slot invariant was corrupted.
            eprintln!(
                "st-zrt: dynamic recovery slot index is out of range; leaking recovered lane"
            );
            std::mem::forget(recovered);
            return;
        };
        if let Err(recovered) = entry.publish(recovered) {
            // This function runs from Drop and must not panic, including in debug builds. An occupied
            // unique lane slot indicates corrupted accounting; leak rather than replace or destroy
            // a still-provider-visible lane.
            eprintln!(
                "st-zrt: dynamic recovery slot reused while occupied; leaking recovered lane"
            );
            std::mem::forget(recovered);
        } else {
            // Publish this unique fixed slot through a bounded Treiber stack. A lane cannot be
            // recovered into the same slot again until the sole runtime consumer has popped it,
            // taken the entry, and returned that lane to service, so each node is present at most
            // once. This notification path never allocates and never takes a global lock.
            let mut head = self.ready_head.load(Ordering::Relaxed);
            loop {
                self.ready_next[slot].store(head, Ordering::Relaxed);
                match self.ready_head.compare_exchange_weak(
                    head,
                    slot,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(observed) => head = observed,
                }
            }
        }
    }

    // Only DynamicIoRuntime calls this and all reclaim entry points require `&mut self`, so there is
    // exactly one consumer. Producers only push and a slot cannot be reused until this consumer has
    // removed it; consequently the usual multi-consumer Treiber ABA scenario cannot occur. Both the
    // initial head load and a failed CAS use Acquire: a retry must observe the newly published node's
    // `ready_next` before reading it with Relaxed ordering.
    fn pop_ready(&self) -> Option<usize> {
        let mut head = self.ready_head.load(Ordering::Acquire);
        loop {
            if head == usize::MAX {
                return None;
            }
            let next = self.ready_next[head].load(Ordering::Relaxed);
            match self.ready_head.compare_exchange_weak(
                head,
                next,
                Ordering::Acquire,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(head),
                Err(observed) => head = observed,
            }
        }
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    /// Query completion without blocking the calling thread.
    ///
    /// A `Ready` result also releases the in-flight graph lease, after which [`Self::lane`] and
    /// [`Self::lane_mut`] may expose the completed buffers. `Pending` retains every provider-visible
    /// resource in this owned token.
    #[inline]
    pub fn try_complete(&mut self) -> Result<CompletionStatus> {
        self.lane
            .as_mut()
            .expect("zrt: owned dynamic run missing lane")
            .try_complete()
    }

    /// Read-only exact-stream completion event for batch polling, when available.
    #[cfg(feature = "cuda")]
    pub fn completion_event(&self) -> Option<crate::CompletionEventRef<'_>> {
        self.lane.as_ref().and_then(ServingLane::completion_event)
    }

    /// Queue downstream GPU work that consumes this run's device outputs without blocking the host.
    ///
    /// ZRT first queues `cudaStreamWaitEvent` on `downstream`, then invokes `enqueue` with the stable
    /// device tensors. The closure must synchronously enqueue all work that reads those tensors on
    /// that exact stream and return only after enqueueing it. ZRT then records a downstream event.
    /// The returned token owns the entire resource chain until that event completes.
    #[cfg(feature = "cuda")]
    pub fn chain_on_stream(
        mut self, downstream: &Arc<crate::CudaStream>,
        enqueue: impl FnOnce(&[AllocatedTensor<O>], &crate::CudaStream) -> Result<()>,
    ) -> std::result::Result<
        GpuChainedDynamicIoRun<I, O, INPUTS, OUTPUTS>,
        Box<GpuChainEnqueueError<I, O, INPUTS, OUTPUTS>>,
    > {
        let lane = self
            .lane
            .as_ref()
            .expect("zrt: owned dynamic run missing lane");
        if let Err(error) = lane.device_outputs_for_chain() {
            return Err(Box::new(GpuChainEnqueueError { error, run: self }));
        }
        if let Err(error) = lane.queue_completion_wait(downstream) {
            return Err(Box::new(GpuChainEnqueueError { error, run: self }));
        }
        let enqueue_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enqueue(
                lane.device_outputs_for_chain()
                    .expect("zrt: device outputs validated above"),
                downstream,
            )
        }));
        let record_result = lane.record_downstream_completion(downstream);

        if let Err(payload) = enqueue_result {
            // A panicking safe closure may already have queued consumers. Fence them before Rust
            // unwinding can return the lane; without a completion proof, leak the whole chain.
            let fenced = record_result.is_ok() && lane.synchronize_downstream().is_ok();
            if fenced {
                self.mark_complete_after_gpu_chain();
            } else {
                std::mem::forget(self);
            }
            std::panic::resume_unwind(payload);
        }

        let failure = enqueue_result
            .expect("zrt: panic branch handled above")
            .err();
        if let Err(record_error) = record_result {
            // The enqueue closure may already have queued consumers. There is no completion proof,
            // so the returned token deliberately retains/leaks the full chain on drop.
            return Ok(GpuChainedDynamicIoRun {
                run: Some(self),
                downstream_event_recorded: false,
                downstream_stream: Some(Arc::clone(downstream)),
                failure: Some(record_error),
                complete: false,
            });
        }
        Ok(GpuChainedDynamicIoRun {
            run: Some(self),
            downstream_event_recorded: true,
            downstream_stream: Some(Arc::clone(downstream)),
            failure,
            complete: false,
        })
    }

    #[cfg(feature = "cuda")]
    fn mark_complete_after_gpu_chain(&mut self) {
        self.lane
            .as_mut()
            .expect("zrt: owned dynamic run missing lane")
            .mark_complete_after_gpu_chain();
    }

    /// Wait for output/device completion while retaining ownership of the lane.
    pub fn synchronize(&mut self) -> Result<()> {
        self.lane
            .as_mut()
            .expect("zrt: owned dynamic run missing lane")
            .synchronize_outputs()
    }

    /// Borrow the owned lane after synchronization for typed output consumption.
    pub fn lane(&self) -> Result<&ServingLane<I, O, INPUTS, OUTPUTS>> {
        let lane = self
            .lane
            .as_ref()
            .expect("zrt: owned dynamic run missing lane");
        if lane.in_flight {
            return Err(Error::new(
                -1,
                "zrt: owned run must be synchronized before its lane can be borrowed",
            ));
        }
        Ok(lane)
    }

    /// Mutably borrow the owned lane after synchronization.
    pub fn lane_mut(&mut self) -> Result<&mut ServingLane<I, O, INPUTS, OUTPUTS>> {
        let lane = self
            .lane
            .as_mut()
            .expect("zrt: owned dynamic run missing lane");
        if lane.in_flight {
            return Err(Error::new(
                -1,
                "zrt: owned run must be synchronized before its lane can be mutably borrowed",
            ));
        }
        Ok(lane)
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedDynamicIoRun")
            .field("bucket_id", &self.bucket_id)
            .field("has_lane", &self.lane.is_some())
            .finish()
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> Drop
    for OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    fn drop(&mut self) {
        let Some(mut lane) = self.lane.take() else {
            return;
        };
        lane.finish_in_flight_best_effort();
        if lane.in_flight {
            // Keep the session, binding, device buffers, and graph run guard alive forever rather
            // than risking use-after-free after both the output fence and device fallback failed.
            std::mem::forget(lane);
            return;
        }
        let Some(recovery) = self.recovery.take() else {
            return;
        };
        let Some(slot) = lane.recovery_slot else {
            // Drop must remain nonpanicking. A missing fixed slot indicates corrupted accounting;
            // leak the fenced lane rather than destroy it or unwind from teardown.
            eprintln!("st-zrt: dynamic lane is missing its fixed recovery slot; leaking lane");
            std::mem::forget(lane);
            return;
        };
        // Recovery is a cold cancellation/drop path. The slot is unique to this lane and was
        // preallocated when the runtime was built, so cancellation performs no allocation and never
        // clones or sends through an MPSC channel.
        recovery.recover(
            slot,
            RecoveredDynamicLane {
                lane,
                bucket_id: self.bucket_id,
            },
        );
    }
}

/// Failure to establish a GPU dependency chain before downstream work was exposed.
#[cfg(feature = "cuda")]
pub struct GpuChainEnqueueError<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    pub error: Error,
    pub run: OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>,
}

#[cfg(feature = "cuda")]
impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for GpuChainEnqueueError<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChainEnqueueError")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

/// An owned ORT run whose device outputs are consumed by work queued on another CUDA stream.
///
/// Construction queues a stream wait on the ORT lane's exact completion event, exposes the stable
/// device outputs only to the enqueue closure, then records a second event after the downstream
/// work. The original lane, graph lease, buffers, session, source event, and destination stream all
/// remain owned until that downstream event completes.
#[cfg(feature = "cuda")]
#[must_use = "a GPU-chained run must be completed before its ORT lane can be reused"]
pub struct GpuChainedDynamicIoRun<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    run: Option<OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>>,
    downstream_event_recorded: bool,
    downstream_stream: Option<Arc<crate::CudaStream>>,
    failure: Option<Error>,
    complete: bool,
}

#[cfg(feature = "cuda")]
impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    GpuChainedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    /// Query downstream completion without blocking.
    ///
    /// A `Ready` result proves GPU completion; call [`Self::synchronize`] to recover the ORT run and
    /// surface any error returned by the downstream enqueue closure.
    pub fn try_complete(&mut self) -> Result<CompletionStatus> {
        if self.complete {
            return Ok(CompletionStatus::Ready);
        }
        if !self.downstream_event_recorded {
            return Err(Error::new(
                -1,
                "zrt: GPU chain has no recorded downstream completion event",
            ));
        }
        let complete = self
            .run
            .as_ref()
            .expect("zrt: GPU chain is missing its owned ORT run")
            .lane
            .as_ref()
            .expect("zrt: owned dynamic run missing lane")
            .downstream_is_complete()?;
        if !complete {
            return Ok(CompletionStatus::Pending);
        }
        self.mark_complete();
        Ok(CompletionStatus::Ready)
    }

    /// Block until downstream GPU work completes, then return the original completed ORT run.
    pub fn synchronize(mut self) -> Result<OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>> {
        if !self.downstream_event_recorded {
            let error = self.failure.take().unwrap_or_else(|| {
                Error::new(
                    -1,
                    "zrt: GPU chain has no recorded downstream completion event",
                )
            });
            self.downstream_stream
                .as_ref()
                .expect("zrt: GPU chain is missing its downstream stream")
                .synchronize()?;
            self.mark_complete();
            return Err(error);
        }
        if !self.complete {
            self.run
                .as_ref()
                .expect("zrt: GPU chain is missing its owned ORT run")
                .lane
                .as_ref()
                .expect("zrt: owned dynamic run missing lane")
                .synchronize_downstream()?;
            self.mark_complete();
        }
        self.downstream_stream.take();
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        self.run
            .take()
            .ok_or_else(|| Error::new(-1, "zrt: GPU chain is missing its owned ORT run"))
    }

    fn mark_complete(&mut self) {
        self.run
            .as_mut()
            .expect("zrt: GPU chain is missing its owned ORT run")
            .mark_complete_after_gpu_chain();
        self.complete = true;
    }
}

#[cfg(feature = "cuda")]
impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for GpuChainedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuChainedDynamicIoRun")
            .field("has_run", &self.run.is_some())
            .field("complete", &self.complete)
            .finish()
    }
}

#[cfg(feature = "cuda")]
impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> Drop
    for GpuChainedDynamicIoRun<I, O, INPUTS, OUTPUTS>
{
    fn drop(&mut self) {
        if self.run.is_none() {
            return;
        }
        if !self.complete {
            let event_synchronized = self
                .run
                .as_ref()
                .and_then(|run| run.lane.as_ref())
                .is_some_and(|lane| {
                    self.downstream_event_recorded && lane.synchronize_downstream().is_ok()
                });
            let synchronized = event_synchronized
                || self
                    .downstream_stream
                    .as_ref()
                    .is_some_and(|stream| stream.synchronize().is_ok());
            if !synchronized {
                // Downstream work may still dereference the ORT output. Leak the complete ownership
                // chain rather than destroying the event/stream/lane underneath it.
                if let Some(run) = self.run.take() {
                    std::mem::forget(run);
                }
                if let Some(stream) = self.downstream_stream.take() {
                    std::mem::forget(stream);
                }
                return;
            }
            self.mark_complete();
        }
        // Normal field drop now returns the completed run's lane through its recovery channel.
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for ServingLane<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingLane")
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .field("device_inputs", &self.device_inputs.as_ref().map(Vec::len))
            .field("graph_id", &self.graph_id)
            .field("in_flight", &self.in_flight)
            .field("rebind_inputs_each_run", &self.rebind_inputs_each_run)
            .field("session", &self.session.as_ptr())
            .finish_non_exhaustive()
    }
}

// SAFETY: the `cudaStream_t` handle is safe to share across threads (the CUDA runtime serializes
// per-stream work); every other field is `Send` (`IoBinding`, `TensorBuffer`, `AllocatedTensor`,
// `Session`, `CString`, `MaterializedRunOptions`). Without this the raw stream fields would make the
// lane `!Send` under the `cuda` feature, breaking `ServingLanePool`'s cross-thread checkout.
#[cfg(feature = "cuda")]
unsafe impl<
    I: TensorElement + Send,
    O: TensorElement + Send,
    const INPUTS: usize,
    const OUTPUTS: usize,
> Send for ServingLane<I, O, INPUTS, OUTPUTS>
{
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> Drop
    for ServingLane<I, O, INPUTS, OUTPUTS>
{
    fn drop(&mut self) {
        self.finish_in_flight_best_effort();
        if self.in_flight {
            // Neither the bound-output fence nor the full-device fallback could prove that pending
            // provider work finished. The binding, host staging/output buffers, CUDA-resident
            // tensors, events, graph lease, and session may all still be dereferenced by that work.
            // Leak the complete lane state instead of destructing provider-visible buffers — the
            // same deliberate-leak policy as `InFlightRun::drop`/`OwnedDynamicIoRun::drop`. The
            // leaked memory pins the session and its captured graph; it is never reused.
            if let Some(inner) = self.inner.take() {
                std::mem::forget(inner);
            }
        }
    }
}

impl<T> Lane<T>
where
    T: TensorElement + Clone + Default,
{
    pub(crate) fn new(
        session: Session, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: BufferSpec,
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

    /// Execute this lane without ORT bound-input/output synchronization calls.
    ///
    /// Use this only for fully host-resident bindings or when device stream synchronization is
    /// handled by the caller. See [`Session::run_binding_unsynchronized`].
    ///
    /// # Safety
    /// The caller must uphold the binding lifetime and provider synchronization contract described
    /// by [`Session::run_binding_unsynchronized`].
    #[inline]
    pub unsafe fn run_unsynchronized(&mut self) -> Result<()> {
        unsafe { self.session.run_binding_unsynchronized(&self.binding) }
    }

    /// Start an asynchronous run (`RunAsync`, IDX 260) on an ORT worker thread, returning a
    /// [`RunFuture`] that resolves to the **ORT-owned** outputs (`Vec<OwnedValue>`).
    ///
    /// Unlike [`Self::run`], the async path cannot fill this lane's caller-owned output buffers:
    /// ORT's `RunAsync` takes named input values and allocates fresh output values — there is no
    /// asynchronous IoBinding run in the C API. The lane's input buffers are read directly by
    /// value handle, so ORT observes the current input contents on its worker thread.
    ///
    /// `&mut self` preserves the one-run-per-lane invariant: the input buffers are exclusively
    /// borrowed for the future's lifetime, so no second run can race on them. Keep the lane alive
    /// until the future resolves (same hazard as [`Session::run_async`]). The session's default
    /// run options are used; per-run CUDA-graph annotations stay on the synchronous [`Self::run`]
    /// path.
    #[inline]
    pub fn run_async(&mut self) -> Result<RunFuture<'_>> {
        let handles: Vec<_> = self.inputs.iter().map(|b| b.as_value_ptr()).collect();
        self.session.run_async_owned(handles, None)
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

    /// Snapshot this lane's hot-path pointer and placement plan.
    ///
    /// This is a setup/preflight diagnostic API and may allocate. Do not call it inside the
    /// measured serving loop.
    pub fn audit_hot_path(&self) -> Result<LaneHotPathAudit> {
        let inputs = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, buffer)| audit_tensor_buffer(IoDirection::Input, i, buffer))
            .collect::<Result<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, buffer)| audit_tensor_buffer(IoDirection::Output, i, buffer))
            .collect::<Result<Vec<_>>>()?;
        Ok(LaneHotPathAudit {
            input_count: self.inputs.len(),
            output_count: self.outputs.len(),
            rebind_inputs_each_run: false,
            input_names_cached: true,
            inputs,
            outputs,
        })
    }

    /// Fail if this lane's buffers are not host-accessible pointer-identity zero-copy tensors.
    pub fn assert_zero_copy_plan(&self) -> Result<()> {
        for (i, input) in self.inputs.iter().enumerate() {
            assert_tensor_buffer_zero_copy("lane input", i, input)?;
        }
        for (i, output) in self.outputs.iter().enumerate() {
            assert_tensor_buffer_zero_copy("lane output", i, output)?;
        }
        Ok(())
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> ServingLane<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build one static-shape I/O lane over a shared session.
    pub fn new(
        session: Session, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::with_buffer_policy(
            session,
            mem,
            mem,
            input_shapes,
            output_shapes,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
    }

    /// Build one static-shape I/O lane with separate input/output memory descriptors.
    pub fn with_memory(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::with_buffer_policy(
            session,
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
    }

    /// Build one static-shape I/O lane with explicit input/output buffer policies.
    pub fn with_buffer_policy(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec,
    ) -> Result<Self> {
        Self::build(
            session,
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            input_policy,
            output_policy,
            None,
            std::ptr::null_mut(),
            None,
        )
    }

    /// Build one static-shape I/O lane whose inputs are **device-resident** on `device_id`
    /// (CUDA). The host `input_mem` buffers act as staging the caller fills; [`ServingLane::run`]
    /// copies each staging buffer → its device buffer before the run so a captured CUDA graph reads
    /// fresh data on replay. This is the cuda-graph-correct input path — required when
    /// `enable_cuda_graph` is set, since a captured graph bakes a device input pointer that ORT does
    /// not repopulate from host staging on its own.
    ///
    /// `stream` is the owned CUDA stream ORT replays the captured graph on (the same `Arc`
    /// passed to [`crate::CudaConfig::with_user_stream`] when building the session).
    /// [`ServingLane::run`] enqueues the host→device refresh on this stream — same-stream as the
    /// replay, so the refresh is ordered before it. (feature `cuda`.)
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn with_device_inputs(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec, device_id: i32, stream: &Arc<crate::CudaStream>,
    ) -> Result<Self> {
        if stream.device_id() != device_id || !session.uses_cuda_stream(stream) {
            return Err(Error::new(
                -1,
                "ServingLane requires the exact owned CUDA stream configured on its Session",
            ));
        }
        let input_policy = input_policy.or_if_auto(BufferSpec::CUDA_PINNED);
        Self::build(
            session,
            input_mem,
            output_mem,
            input_shapes,
            output_shapes,
            input_policy,
            output_policy,
            Some(device_id),
            stream.as_ptr(),
            None,
        )
    }

    /// Build a CUDA-graph lane with device-resident inputs and outputs for GPU-to-GPU pipelines.
    ///
    /// The lane retains both allocator-owned device tensors and the exact session stream. Host
    /// output slice access is unavailable; after completion use [`Self::device_output`] to pass the
    /// stable device tensor to another GPU operation.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn with_device_io(
        session: Session, input_mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec, device_id: i32,
        stream: &Arc<crate::CudaStream>,
    ) -> Result<Self> {
        if stream.device_id() != device_id || !session.uses_cuda_stream(stream) {
            return Err(Error::new(
                -1,
                "ServingLane requires the exact owned CUDA stream configured on its Session",
            ));
        }
        let host_output_mem = MemoryInfo::cpu()?;
        Self::build(
            session,
            input_mem,
            &host_output_mem,
            input_shapes,
            output_shapes,
            input_policy.or_if_auto(BufferSpec::CUDA_PINNED),
            BufferSpec::AUTO,
            Some(device_id),
            stream.as_ptr(),
            Some(device_id),
        )
    }

    /// Shared lane builder. `device_id = Some(id)` selects the device-input mode (CUDA-resident
    /// inputs bound to the IoBinding; host `inputs` become staging); `None` is the host-input mode
    /// (host `inputs` bound directly).
    #[allow(clippy::too_many_arguments)]
    fn build(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec, device_id: Option<i32>, stream: *mut std::ffi::c_void,
        device_output_id: Option<i32>,
    ) -> Result<Self> {
        // `stream` is consumed by the `#[cfg(cuda)]` field below; discard it in the non-cuda build.
        let _ = stream;
        #[cfg(not(feature = "cuda"))]
        let _ = device_output_id;
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

        // Host input buffers — bound directly in host mode, or staging (caller-filled, copied to the
        // device buffer each run) in device mode.
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
        let input_names: [CString; INPUTS] = (0..INPUTS)
            .map(|i| {
                CString::new(session.input_name(i)?)
                    .map_err(|_| Error::new(-1, "zrt: input name contains a NUL"))
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::new(-1, "zrt: failed to build static I/O input name array"))?;

        // Device-input mode: allocate CUDA-resident input tensors and bind THOSE (the graph bakes
        // their device pointers). Host mode: bind the host staging buffers directly.
        let device_inputs = match device_id {
            #[cfg(feature = "cuda")]
            Some(id) => {
                let devs: Vec<AllocatedTensor<I>> = input_shapes
                    .iter()
                    .map(|shape| AllocatedTensor::cuda(&session, id, shape))
                    .collect::<Result<Vec<_>>>()?;
                for (i, dev) in devs.iter().enumerate() {
                    binding.bind_input_cstr(&input_names[i], dev)?;
                }
                Some(devs)
            },
            _ => {
                for (i, input) in inputs.iter().enumerate() {
                    binding.bind_input_cstr(&input_names[i], input)?;
                }
                None
            },
        };
        #[cfg(feature = "cuda")]
        let device_outputs = match device_output_id {
            Some(id) => {
                let values = output_shapes
                    .iter()
                    .map(|shape| AllocatedTensor::cuda(&session, id, shape))
                    .collect::<Result<Vec<_>>>()?;
                for (i, output) in values.iter().enumerate() {
                    binding.bind_output_allocated(session.output_name(i)?, output)?;
                }
                Some(values)
            },
            None => {
                for (i, output) in outputs.iter().enumerate() {
                    binding.bind_output_buffer(session.output_name(i)?, output)?;
                }
                None
            },
        };
        #[cfg(not(feature = "cuda"))]
        for (i, output) in outputs.iter().enumerate() {
            binding.bind_output_buffer(session.output_name(i)?, output)?;
        }

        Ok(Self {
            inner: Some(Box::new(ServingLaneInner {
                binding,
                inputs,
                outputs,
                #[cfg(feature = "cuda")]
                device_outputs,
                input_names,
                session,
                rebind_inputs_each_run: false,
                run_opts: None,
                enqueued_run_opts: RunOptions::new().with_disable_ep_sync(true).freeze()?,
                #[cfg(feature = "ep")]
                sync_stream: None,
                graph_id: None,
                graph_captured: false,
                graph_lease: None,
                in_flight_graph_guard: None,
                #[cfg(feature = "cuda")]
                completion_event: device_id.map(crate::CudaEvent::new).transpose()?,
                #[cfg(feature = "cuda")]
                downstream_completion_event: device_output_id
                    .map(crate::CudaEvent::new)
                    .transpose()?,
                #[cfg(feature = "cuda")]
                completion_event_recorded: false,
                in_flight: false,
                recovery_slot: None,
                device_inputs,
                #[cfg(feature = "cuda")]
                stream,
                #[cfg(feature = "cuda")]
                cuda_device_id: device_id,
            })),
        })
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    ServingLane<I, O, INPUTS, OUTPUTS>
{
    #[inline]
    fn ensure_idle(&self) -> Result<()> {
        if self.in_flight {
            Err(Error::new(
                -1,
                "zrt: lane already has an in-flight run; synchronize it before reuse",
            ))
        } else {
            Ok(())
        }
    }

    #[cfg(feature = "cuda")]
    fn try_device_fence_after_sync_error(&self) -> bool {
        let Some(device_id) = self.cuda_device_id else {
            return false;
        };
        crate::cuda_rt::ensure_device_active(device_id).is_ok()
            && crate::cuda_rt::device_synchronize().is_ok()
    }

    fn clear_in_flight(&mut self) {
        self.in_flight = false;
        self.in_flight_graph_guard.take();
        #[cfg(feature = "cuda")]
        {
            self.completion_event_recorded = false;
        }
    }

    #[cfg(feature = "cuda")]
    fn synchronize_completion_event(&self) -> Option<Result<()>> {
        self.completion_event_recorded.then(|| {
            self.completion_event
                .as_ref()
                .expect("zrt: recorded CUDA event missing")
                .synchronize()
        })
    }

    /// Read-only exact-stream completion event for batch polling, when available.
    #[cfg(feature = "cuda")]
    pub fn completion_event(&self) -> Option<crate::CompletionEventRef<'_>> {
        (self.in_flight && self.completion_event_recorded).then(|| {
            crate::CompletionEventRef::new(
                self.completion_event
                    .as_ref()
                    .expect("zrt: recorded CUDA event missing"),
            )
        })
    }

    /// Query provider/device completion without blocking the calling thread.
    ///
    /// CUDA device-input runs use the reusable event recorded after replay on the lane's exact
    /// retained stream. A ready event discharges the graph lease and makes the lane reusable. Other
    /// in-flight configurations have no sound nonblocking completion primitive and return an error
    /// rather than pretending host enqueue completion is device completion.
    #[inline]
    pub fn try_complete(&mut self) -> Result<CompletionStatus> {
        if !self.in_flight {
            return Ok(CompletionStatus::Ready);
        }
        #[cfg(feature = "cuda")]
        if self.completion_event_recorded {
            let complete = self
                .completion_event
                .as_ref()
                .expect("zrt: recorded CUDA event missing")
                .is_complete()?;
            if complete {
                self.clear_in_flight();
                return Ok(CompletionStatus::Ready);
            }
            return Ok(CompletionStatus::Pending);
        }
        Err(Error::new(
            -1,
            "zrt: this in-flight lane has no nonblocking completion event; synchronize it explicitly",
        ))
    }

    #[cfg(feature = "cuda")]
    fn queue_completion_wait(&self, downstream: &crate::CudaStream) -> Result<()> {
        if !self.in_flight || !self.completion_event_recorded {
            return Err(Error::new(
                -1,
                "zrt: GPU chaining requires an in-flight exact-stream CUDA completion event",
            ));
        }
        let source = self
            .completion_event
            .as_ref()
            .expect("zrt: recorded CUDA event missing");
        if source.device_id() != downstream.device_id() {
            return Err(Error::new(
                -1,
                "zrt: GPU chaining currently requires source and downstream streams on the same CUDA device",
            ));
        }
        crate::cuda_rt::stream_wait_event(downstream, source)
    }

    #[cfg(feature = "cuda")]
    fn device_outputs_for_chain(&self) -> Result<&[AllocatedTensor<O>]> {
        self.device_outputs.as_deref().ok_or_else(|| {
            Error::new(
                -1,
                "zrt: GPU chaining requires device-resident lane outputs",
            )
        })
    }

    #[cfg(feature = "cuda")]
    fn record_downstream_completion(&self, downstream: &crate::CudaStream) -> Result<()> {
        self.downstream_completion_event
            .as_ref()
            .ok_or_else(|| Error::new(-1, "zrt: lane has no downstream CUDA completion event"))?
            .record(downstream)
    }

    #[cfg(feature = "cuda")]
    fn downstream_is_complete(&self) -> Result<bool> {
        self.downstream_completion_event
            .as_ref()
            .ok_or_else(|| Error::new(-1, "zrt: lane has no downstream CUDA completion event"))?
            .is_complete()
    }

    #[cfg(feature = "cuda")]
    fn synchronize_downstream(&self) -> Result<()> {
        self.downstream_completion_event
            .as_ref()
            .ok_or_else(|| Error::new(-1, "zrt: lane has no downstream CUDA completion event"))?
            .synchronize()
    }

    #[cfg(feature = "cuda")]
    fn mark_complete_after_gpu_chain(&mut self) {
        debug_assert!(self.in_flight);
        debug_assert!(self.completion_event_recorded);
        self.clear_in_flight();
    }

    fn finish_in_flight_best_effort(&mut self) {
        if !self.in_flight {
            return;
        }
        #[cfg(feature = "cuda")]
        let synchronized = match self.synchronize_completion_event() {
            Some(result) => result.is_ok(),
            None => self.binding.synchronize_outputs().is_ok(),
        };
        #[cfg(not(feature = "cuda"))]
        let synchronized = self.binding.synchronize_outputs().is_ok();
        #[cfg(feature = "cuda")]
        let fenced = synchronized || self.try_device_fence_after_sync_error();
        #[cfg(not(feature = "cuda"))]
        let fenced = synchronized;

        // Teardown cannot return an error. Clear only after a successful fence; an unfenced lane
        // remains poisoned until its fields are dropped, rather than pretending it is reusable.
        if fenced {
            self.clear_in_flight();
        }
    }

    #[inline]
    fn rebind_host_inputs_if_needed(&mut self) -> Result<()> {
        // Host-input rebind only applies in host mode; in device-input mode the device buffers stay
        // bound and are refreshed by `refresh_device_inputs` (rebind would tear down baked pointers).
        // `ServingLane` derefs to its inner state, so take one inner borrow once and split it into
        // the disjoint `binding`/`inputs` field borrows the original direct-field code relied on.
        let inner: &mut ServingLaneInner<I, O, INPUTS, OUTPUTS> = self;
        if inner.rebind_inputs_each_run && inner.device_inputs.is_none() {
            inner.binding.clear_inputs();
            for (i, input) in inner.inputs.iter().enumerate() {
                inner
                    .binding
                    .bind_input_cstr(&inner.input_names[i], input)?;
            }
        }
        Ok(())
    }

    #[inline]
    fn refresh_device_inputs(&mut self) -> Result<()> {
        // `ServingLane` derefs to its inner state, so take one inner borrow once; the loop below
        // splits it into the disjoint `device_inputs`/`inputs`/`stream` field borrows the original
        // direct-field code relied on.
        let inner: &mut ServingLaneInner<I, O, INPUTS, OUTPUTS> = self;
        #[cfg(feature = "cuda")]
        let cuda_device_id = inner.cuda_device_id;
        // Device-input mode: refresh each device buffer from its host staging on the captured-graph
        // stream before the run, so a replayed graph reads fresh data. (CUDA-runtime copy on the
        // user_compute_stream — ORT's CopyTensors has no CPU<->CUDA route for the legacy CUDA EP.)
        if let Some(devs) = inner.device_inputs.as_mut() {
            #[cfg(feature = "cuda")]
            {
                // Ensure the correct CUDA device is active on this thread before manual H2D.
                // The stream may have been created on a different thread (e.g. async model load)
                // and moved into this replica worker thread. cudaSetDevice is thread-local.
                let cuda_device_id = cuda_device_id.ok_or_else(|| {
                    Error::new(-1, "zrt: device-input lane is missing its CUDA device id")
                })?;
                crate::cuda_rt::ensure_device_active(cuda_device_id)?;
                let stream = inner.stream;
                for (i, staging) in inner.inputs.iter().enumerate() {
                    let s = staging.as_slice();
                    // SAFETY: `s` is the lane's host staging slice whose byte length exactly matches
                    // the device-resident input `devs[i]` (both built from the same static shape).
                    // `stream` is the retained CUDA stream the captured graph replays on; the copy
                    // is ordered before the same-stream replay below (ORT syncs the stream at run end).
                    // The staging buffer is lane-owned and not freed/mutated before the copy completes.
                    unsafe {
                        crate::cuda_rt::memcpy_async_h2d(
                            devs[i].raw_mut_ptr()?,
                            s.as_ptr() as *const std::ffi::c_void,
                            std::mem::size_of_val(s),
                            stream,
                        )?;
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = devs;
                return Err(Error::new(
                    -1,
                    "zrt: device-input mode requires the `cuda` feature",
                ));
            }
        }
        Ok(())
    }

    #[inline]
    fn run_bound_binding(&mut self) -> Result<()> {
        let Some(opts) = self.run_opts.as_ref() else {
            return self.session.run_binding(&self.binding);
        };
        let graph_guard = self.graph_lease.as_ref().map(CapturedGraphLease::begin_run);
        let result = self.session.run_binding_with(&self.binding, opts);
        if self.graph_id.is_some() && result.is_ok() {
            // The first successful run with a fresh annotation id is ORT's capture run.
            self.graph_captured = true;
        }
        // `run_binding_with` keeps ORT's end-of-run EP synchronization, so a successful run is
        // fenced and the lease is released on return. A failed run may have queued partial
        // provider work first; settle it with the same fence-or-poison policy as the enqueued path.
        self.settle_graph_run(graph_guard, result, false)
    }

    /// Settle the captured-graph lease after a `RunWithBinding` call returned `result`.
    ///
    /// `retain_on_success` keeps the guard and marks the lane in flight — the enqueued path, whose
    /// disabled EP end-sync means provider work legitimately continues past the return. The
    /// synchronous path passes `false` because its run kept end-of-run EP synchronization.
    ///
    /// On error, ORT may have queued partial provider work before returning. The lease is released
    /// only after a successful bound-output or full-device fence; without one the guard is retained
    /// and the lane is poisoned in flight so a later `release_captured_graph` cannot race that
    /// partial work.
    fn settle_graph_run(
        &mut self, graph_guard: Option<CapturedGraphRunGuard>, result: Result<()>,
        retain_on_success: bool,
    ) -> Result<()> {
        match result {
            Ok(()) => {
                if retain_on_success {
                    self.in_flight_graph_guard = graph_guard;
                    self.in_flight = true;
                }
                Ok(())
            },
            Err(error) => {
                self.fence_or_retain_graph_guard(graph_guard);
                Err(error)
            },
        }
    }

    /// Try to fence provider work a failed run may have left pending, then dispose of `guard`.
    ///
    /// A proven fence (bound-output synchronization or a full device fence) releases the lease and
    /// makes the lane reusable — including an already-retained guard, which is why
    /// [`Self::clear_in_flight`] runs on that path. Without a fence the guard is retained in a
    /// poisoned in-flight state so a later `release_captured_graph` cannot race the partial
    /// provider work ORT may have queued before failing. Passing `None` fences an
    /// already-retained guard (the `run_profiled` output-sync failure path).
    fn fence_or_retain_graph_guard(&mut self, graph_guard: Option<CapturedGraphRunGuard>) -> bool {
        let synchronized = self.binding.synchronize_outputs().is_ok();
        #[cfg(feature = "cuda")]
        let fenced = synchronized || self.try_device_fence_after_sync_error();
        #[cfg(not(feature = "cuda"))]
        let fenced = synchronized;
        match (fenced, graph_guard) {
            (true, guard) => {
                self.clear_in_flight();
                drop(guard);
            },
            (false, Some(guard)) => {
                self.in_flight_graph_guard = Some(guard);
                self.in_flight = true;
            },
            (false, None) => {
                // The guard was already retained (and the lane poisoned) by the earlier
                // retain-on-success settle; it stays pinned until an explicit fence succeeds.
            },
        }
        fenced
    }

    fn run_bound_binding_enqueued(&mut self) -> Result<()> {
        self.ensure_idle()?;

        // The owned guard intentionally survives RunWithBinding return: disabling EP sync means
        // device work may still be using the captured graph. It is released only after output sync.
        let graph_guard = self.graph_lease.as_ref().map(CapturedGraphLease::begin_run);
        let result = (|| {
            unsafe {
                self.session
                    .run_binding_unsynchronized_with(&self.binding, &self.enqueued_run_opts)?;
            }
            #[cfg(feature = "cuda")]
            if self.device_inputs.is_some() && self.graph_id.is_some() {
                let event = self
                    .completion_event
                    .as_ref()
                    .expect("zrt: device-input lane is missing its CUDA completion event");
                // ORT 1.27 enqueues captured-graph replay on the configured user stream before
                // returning when EP synchronization is disabled. Recording on that exact retained
                // stream fences this lane's H2D refresh and replay without provider-wide Sync().
                event.record_raw_stream(self.stream)?;
                self.completion_event_recorded = true;
            }
            Ok(())
        })();
        if self.graph_id.is_some() && result.is_ok() {
            // The first successful enqueued run with a fresh annotation id is ORT's capture run.
            self.graph_captured = true;
        }
        self.settle_graph_run(graph_guard, result, true)
    }

    /// Execute this lane's pre-bound IoBinding.
    #[inline]
    pub fn run(&mut self) -> Result<()> {
        self.ensure_idle()?;
        self.rebind_host_inputs_if_needed()?;
        self.refresh_device_inputs()?;
        #[cfg(feature = "cuda")]
        if self.device_inputs.is_some() && self.graph_id.is_some() {
            // The device buffers and graph replay share the exact retained user_compute_stream.
            // Avoid IoBinding's provider-wide input/output Sync() calls: enqueue replay without
            // end-of-run EP synchronization, record a CUDA event after the launch, and fence only
            // that stream before exposing outputs or making the lane reusable.
            self.run_bound_binding_enqueued()?;
            return self.synchronize_outputs();
        }
        self.run_bound_binding()
    }

    /// Enqueue this lane's pre-bound IoBinding without synchronizing bound inputs or outputs.
    ///
    /// This is the explicit pipelining variant of [`Self::run`]. It still performs the lane-local
    /// preparation steps (`rebind_host_inputs_if_needed` and device-input H2D refresh), then calls
    /// `RunWithBinding` with ORT's `disable_synchronize_execution_providers=1` run option. Call
    /// [`Self::synchronize_outputs`] before reading host-visible outputs, reusing staging buffers for
    /// another in-flight run, or dropping resources whose device work may still be pending.
    #[inline]
    pub fn run_enqueued(&mut self) -> Result<()> {
        self.ensure_idle()?;
        self.rebind_host_inputs_if_needed()?;
        self.refresh_device_inputs()?;
        self.run_bound_binding_enqueued()
    }

    /// Synchronize bound outputs and release any owned in-flight graph lease.
    #[inline]
    pub fn synchronize_outputs(&mut self) -> Result<()> {
        if !self.in_flight {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        let result = match self.synchronize_completion_event() {
            Some(result) => result,
            None => self.binding.synchronize_outputs(),
        };
        #[cfg(not(feature = "cuda"))]
        let result = self.binding.synchronize_outputs();
        let synchronized = result.is_ok();
        #[cfg(feature = "cuda")]
        let fenced = synchronized || self.try_device_fence_after_sync_error();
        #[cfg(not(feature = "cuda"))]
        let fenced = synchronized;
        if fenced {
            self.clear_in_flight();
        }
        result
    }

    /// Enqueue an exclusive run and return an owned token that owns this lane until completion.
    ///
    /// Prefer this over the legacy split [`Self::run_enqueued`] / [`Self::synchronize_outputs`]
    /// API: the returned [`InFlightRun`] *owns* the lane, so its staging buffers, outputs, and
    /// captured-graph lease cannot be reused until [`InFlightRun::synchronize`] returns the lane.
    /// No borrow is retained: once `enqueue` returns, the lane's former slot in its bucket or
    /// runtime is free for other lanes, which is what allows a whole pipeline to be enqueued
    /// before any synchronization.
    ///
    /// On failure the lane is returned inside [`LaneEnqueueError`].
    pub fn enqueue(
        mut self,
    ) -> std::result::Result<
        InFlightRun<I, O, INPUTS, OUTPUTS>,
        Box<LaneEnqueueError<I, O, INPUTS, OUTPUTS>>,
    > {
        match self.run_enqueued() {
            Ok(()) => Ok(InFlightRun { lane: Some(self) }),
            Err(error) => Err(Box::new(LaneEnqueueError { error, lane: self })),
        }
    }

    /// Execute the event-fenced device-input CUDA-graph path and return host-side phase timings.
    ///
    /// Unlike [`Self::run_profiled`], this measures the optimized path: no IoBinding provider-wide
    /// input/output synchronization is issued. `bound_input_sync` is zero and
    /// `bound_output_sync` measures waiting on this lane's exact-stream CUDA completion event.
    #[cfg(feature = "cuda")]
    pub fn run_event_profiled(&mut self) -> Result<StaticIoRunTimings> {
        self.ensure_idle()?;
        if self.device_inputs.is_none() || self.graph_id.is_none() {
            return Err(Error::new(
                -1,
                "zrt: event-profiled run requires device inputs and a CUDA graph id",
            ));
        }
        let total_start = Instant::now();
        let refresh_start = Instant::now();
        self.refresh_device_inputs()?;
        let device_input_refresh = refresh_start.elapsed();

        let run_start = Instant::now();
        self.run_bound_binding_enqueued()?;
        let run_with_binding = run_start.elapsed();

        let sync_start = Instant::now();
        self.synchronize_outputs()?;
        let completion_event_sync = sync_start.elapsed();
        Ok(StaticIoRunTimings {
            rebind_inputs: Duration::ZERO,
            device_input_refresh,
            ort_run: run_with_binding + completion_event_sync,
            bound_input_sync: Duration::ZERO,
            run_with_binding,
            bound_output_sync: completion_event_sync,
            total: total_start.elapsed(),
        })
    }

    /// Execute this lane and return coarse host-side timings for diagnostics.
    ///
    /// This preserves [`Self::run`] semantics and is intended for benchmarks/profiling, not for
    /// production hot paths.
    pub fn run_profiled(&mut self) -> Result<StaticIoRunTimings> {
        self.ensure_idle()?;
        let total_start = Instant::now();

        let rebind_start = Instant::now();
        self.rebind_host_inputs_if_needed()?;
        let rebind_inputs = rebind_start.elapsed();

        let refresh_start = Instant::now();
        self.refresh_device_inputs()?;
        let device_input_refresh = refresh_start.elapsed();

        let ort_start = Instant::now();
        let sync_inputs_start = Instant::now();
        self.binding.synchronize_inputs()?;
        let bound_input_sync = sync_inputs_start.elapsed();

        let run_with_binding_start = Instant::now();
        // Keep the graph lease through output synchronization, not merely through enqueue return.
        // Early errors below settle with the same fence-or-poison policy as the enqueued path: a
        // partial native enqueue must not release the lease unfenced.
        let graph_guard = self.graph_lease.as_ref().map(CapturedGraphLease::begin_run);
        let run_result = match self.run_opts.as_ref() {
            Some(opts) => unsafe {
                self.session
                    .run_binding_unsynchronized_with(&self.binding, opts)
            },
            None => unsafe { self.session.run_binding_unsynchronized(&self.binding) },
        };
        let run_with_binding = run_with_binding_start.elapsed();
        if self.graph_id.is_some() && run_result.is_ok() {
            // The first successful run with a fresh annotation id is ORT's capture run.
            self.graph_captured = true;
        }
        // The unsynchronized run keeps provider work past its return, so retain the lease and mark
        // the lane in flight — exactly the enqueued-path policy — until output synchronization
        // completes below. A run failure fence-or-poisons through the same helper and propagates.
        self.settle_graph_run(graph_guard, run_result, true)?;

        let sync_outputs_start = Instant::now();
        let sync_result = self.binding.synchronize_outputs();
        let bound_output_sync = sync_outputs_start.elapsed();
        let ort_run = ort_start.elapsed();
        match sync_result {
            Ok(()) => {
                // Synchronization is the fence: discharge the retained lease and reuse state.
                self.clear_in_flight();
            },
            Err(error) => {
                // The unsynchronized run already enqueued provider work, so a failed output fence
                // must not free the lease or the buffers underneath it. Retry-fence or keep the
                // lane poisoned in flight, then surface the original synchronization failure.
                let _ = self.fence_or_retain_graph_guard(None);
                return Err(error);
            },
        }

        Ok(StaticIoRunTimings {
            rebind_inputs,
            device_input_refresh,
            ort_run,
            bound_input_sync,
            run_with_binding,
            bound_output_sync,
            total: total_start.elapsed(),
        })
    }

    /// Execute this lane without ORT bound-input/output synchronization calls.
    ///
    /// If `rebind_inputs_each_run` is enabled, inputs are still rebound before the run. Use this
    /// only for fully host-resident bindings or when device stream synchronization is handled by
    /// the caller. **Errors in device-input mode** — that mode requires the host→device refresh and
    /// run-end stream synchronization performed by [`Self::run`]; the unsynchronized path would
    /// replay the graph over stale device buffers, so use [`Self::run`] there. See
    /// [`Session::run_binding_unsynchronized`].
    #[inline]
    pub fn run_unsynchronized(&mut self) -> Result<()> {
        self.ensure_idle()?;
        // Device-input mode needs the H2D refresh + run-end stream sync that `run` performs; the
        // unsynchronized path skips both and would replay the graph over stale device buffers.
        // Refuse rather than silently serve incorrect results.
        if self.device_inputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: run_unsynchronized is unsafe in device-input mode — the H2D refresh and \
                 stream synchronization performed by `run` are required; use `run` instead",
            ));
        }
        if self.graph_id.is_some() {
            return Err(Error::new(
                -1,
                "zrt: run_unsynchronized cannot retain a captured-graph lease through caller-managed \
                 synchronization; use enqueue() or run() instead",
            ));
        }
        self.rebind_host_inputs_if_needed()?;
        match &self.run_opts {
            Some(opts) => unsafe {
                self.session
                    .run_binding_unsynchronized_with(&self.binding, opts)
            },
            None => unsafe { self.session.run_binding_unsynchronized(&self.binding) },
        }
    }

    /// Start an asynchronous run (`RunAsync`, IDX 260) on an ORT worker thread, returning a
    /// [`RunFuture`] that resolves to the **ORT-owned** outputs (`Vec<OwnedValue>`).
    ///
    /// Unlike [`Self::run`], the async path cannot fill this lane's caller-owned output buffers:
    /// ORT's `RunAsync` takes named input values and allocates fresh output values — there is no
    /// asynchronous IoBinding run in the C API. The input buffers are read directly by value
    /// handle, so the async path bypasses [`Self::run`]'s input-rebind logic
    /// (`rebind_inputs_each_run` has no effect here) and ORT reads the current input contents on
    /// its worker thread.
    ///
    /// `&mut self` preserves the one-run-per-lane invariant: the input buffers are exclusively
    /// borrowed for the future's lifetime, so no second run can race on them. Keep the lane alive
    /// until the future resolves. The session's default run options are used; per-run CUDA-graph
    /// annotations (`gpu_graph_id`) stay on the synchronous [`Self::run`] path.
    #[inline]
    pub fn run_async(&mut self) -> Result<RunFuture<'_>> {
        self.ensure_idle()?;
        let handles: Vec<_> = self.inputs.iter().map(|b| b.as_value_ptr()).collect();
        self.session.run_async_owned(handles, None)
    }

    /// Run this lane `runs` times before serving to prime ORT shape/memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run()?;
        }
        Ok(())
    }

    /// Run this lane `runs` times through [`Self::run_enqueued`] before serving.
    ///
    /// CUDA-graph services that use the enqueued path in production should prime with this method so
    /// ORT captures the same run-option mode used later on live traffic.
    pub fn prime_enqueued(&mut self, runs: usize) -> Result<()> {
        for _ in 0..runs {
            self.run_enqueued()?;
            self.synchronize_outputs()?;
        }
        Ok(())
    }

    #[inline]
    pub fn inputs(&self) -> &[TensorBuffer<I>; INPUTS] {
        &self.inputs
    }

    /// Mutable host-resident input staging buffers.
    ///
    /// Errors while the lane is in flight (after [`Self::run_enqueued`]): a pending host→device
    /// refresh or host-bound run may still be reading these buffers, so mutation before
    /// [`Self::synchronize_outputs`] is a data race. This mirrors [`Self::fill_inputs`].
    #[inline]
    pub fn inputs_mut(&mut self) -> Result<&mut [TensorBuffer<I>; INPUTS]> {
        self.ensure_idle()?;
        Ok(&mut self.inputs)
    }

    /// Host-resident output buffers.
    ///
    /// Device-output lanes return an error because their host placeholders are not bound to ORT.
    #[inline]
    pub fn outputs(&self) -> Result<&[TensorBuffer<O>; OUTPUTS]> {
        #[cfg(feature = "cuda")]
        if self.device_outputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: lane outputs are CUDA-resident; use device_output",
            ));
        }
        Ok(&self.outputs)
    }

    /// Mutable host-resident output buffers.
    ///
    /// Errors while the lane is in flight (after [`Self::run_enqueued`]): pending device→host
    /// output copies may still be writing these buffers. Errors for CUDA-resident outputs.
    #[inline]
    pub fn outputs_mut(&mut self) -> Result<&mut [TensorBuffer<O>; OUTPUTS]> {
        self.ensure_idle()?;
        #[cfg(feature = "cuda")]
        if self.device_outputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: lane outputs are CUDA-resident; use device_output",
            ));
        }
        Ok(&mut self.outputs)
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

    /// Mutably borrow one host-resident input staging buffer.
    ///
    /// Errors while the lane is in flight (after [`Self::run_enqueued`]) — see [`Self::inputs_mut`].
    #[inline]
    pub fn input_mut(&mut self, i: usize) -> Result<&mut [I]> {
        self.ensure_idle()?;
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
        #[cfg(feature = "cuda")]
        if self.device_outputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: lane output is CUDA-resident; use device_output",
            ));
        }
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

    /// Mutably borrow one host-resident output buffer.
    ///
    /// Errors while the lane is in flight (after [`Self::run_enqueued`]) — see
    /// [`Self::outputs_mut`]. Errors for CUDA-resident outputs.
    #[inline]
    pub fn output_mut(&mut self, i: usize) -> Result<&mut [O]> {
        self.ensure_idle()?;
        #[cfg(feature = "cuda")]
        if self.device_outputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: lane output is CUDA-resident; use device_output",
            ));
        }
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

    /// Fill all lane-owned input staging buffers directly.
    ///
    /// For CUDA device-input lanes these buffers are page-locked and the subsequent run enqueues
    /// H2D from them. Upstream producers should write here instead of materializing an intermediate
    /// `Vec` and copying it into the lane. The closure runs only while the lane is idle.
    pub fn fill_inputs(
        &mut self, fill: impl FnOnce(&mut [&mut [I]; INPUTS]) -> Result<()>,
    ) -> Result<()> {
        self.ensure_idle()?;
        let mut inputs = self.inputs.iter_mut();
        let mut slices: [&mut [I]; INPUTS] = std::array::from_fn(|_| {
            inputs
                .next()
                .expect("zrt: lane input arity diverged from const generic")
                .as_mut_slice()
        });
        fill(&mut slices)
    }

    #[inline]
    pub fn input_mut_at<const IDX: usize>(&mut self) -> Result<&mut [I]> {
        self.input_mut(IDX)
    }

    /// Completed CUDA-resident output tensor for GPU-to-GPU consumption.
    #[cfg(feature = "cuda")]
    pub fn device_output(&self, i: usize) -> Result<&AllocatedTensor<O>> {
        if self.in_flight {
            return Err(Error::new(
                -1,
                "zrt: device output is unavailable before lane completion",
            ));
        }
        self.device_outputs
            .as_ref()
            .and_then(|outputs| outputs.get(i))
            .ok_or_else(|| Error::new(-1, "zrt: lane has no CUDA-resident output at this index"))
    }

    #[inline]
    pub fn output_at<const IDX: usize>(&self) -> Result<&[O]> {
        self.output(IDX)
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
        #[cfg(feature = "cuda")]
        if self.device_outputs.is_some() {
            return Err(Error::new(
                -1,
                "zrt: lane output is CUDA-resident; use device_output",
            ));
        }
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

    /// Snapshot this static lane's hot-path pointer and placement plan.
    ///
    /// This is a setup/preflight diagnostic API and may allocate. Do not call it inside the
    /// measured serving loop.
    pub fn audit_hot_path(&self) -> Result<LaneHotPathAudit> {
        let inputs = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, buffer)| audit_tensor_buffer(IoDirection::Input, i, buffer))
            .collect::<Result<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .enumerate()
            .map(|(i, buffer)| audit_tensor_buffer(IoDirection::Output, i, buffer))
            .collect::<Result<Vec<_>>>()?;
        Ok(LaneHotPathAudit {
            input_count: INPUTS,
            output_count: OUTPUTS,
            rebind_inputs_each_run: self.rebind_inputs_each_run,
            input_names_cached: self.input_names.len() == INPUTS,
            inputs,
            outputs,
        })
    }

    /// Fail if this lane's buffers are not host-accessible pointer-identity zero-copy tensors.
    pub fn assert_zero_copy_plan(&self) -> Result<()> {
        for (i, input) in self.inputs.iter().enumerate() {
            assert_tensor_buffer_zero_copy("static I/O lane input", i, input)?;
        }
        for (i, output) in self.outputs.iter().enumerate() {
            assert_tensor_buffer_zero_copy("static I/O lane output", i, output)?;
        }
        Ok(())
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

    /// Assign this lane a CUDA-graph annotation id by setting the typed `gpu_graph_id`
    /// configuration. Subsequent [`run`](Self::run) calls replay the graph ORT captured for this id
    /// (one graph per shape) when the session was built with `enable_cuda_graph=true`.
    ///
    /// Requires a **device-input** lane (built with `with_device_inputs` or
    /// `with_device_io`; `cuda` feature). A host-input captured graph bakes device buffers ORT never
    /// repopulates on replay, so replays silently read stale inputs; host-input lanes are rejected.
    ///
    /// The FIRST run after assigning a fresh id captures ORT's graph; capture is device-wide
    /// serialized and must not overlap any live replay (prime the lane before serving).
    ///
    /// Leak warning: if this lane is ever deliberately leaked unfenced (see
    /// [`Drop for ServingLane`](Drop)), its retained `CapturedGraphRunGuard` pins the annotation
    /// id's lease forever, and a later `Session::release_captured_graph` with the SAME id blocks
    /// indefinitely waiting for that lease to drain. Owners that leak lanes must not release those
    /// ids; `DynamicIoRuntime`'s bucket paths deliberately skip release for buckets they leak.
    pub fn set_gpu_graph_id(&mut self, id: i32) -> Result<()> {
        self.ensure_idle()?;
        self.ensure_graph_id_assignable()?;
        let (synchronized, enqueued) = self.materialize_run_options(
            Some(id),
            #[cfg(feature = "ep")]
            self.sync_stream.as_ref(),
        )?;
        self.run_opts = Some(synchronized);
        self.enqueued_run_opts = enqueued;
        self.graph_lease = Some(self.session.captured_graph_lease(id));
        self.graph_id = Some(id);
        // A fresh annotation id means the next run captures ORT's graph again.
        self.graph_captured = false;
        Ok(())
    }

    /// Whether the run that triggers ORT graph capture for this lane's current `gpu_graph_id`
    /// has already completed — i.e. the next [`run`](Self::run) /
    /// [`run_enqueued`](Self::run_enqueued) replays a captured graph instead of capturing.
    ///
    /// [`DynamicIoRuntime`] captures eagerly while creating a CUDA-graph bucket, so every lane of
    /// a prebuilt bucket reports `true` before its first served run. Direct
    /// [`ServingLane`]/[`StaticIoRuntime`] owners assign ids without capturing; prime such lanes
    /// (see [`Self::prime`]/[`Self::prime_enqueued`]) before serving, because capture is
    /// device-wide serialized and must not overlap a live replay.
    #[inline]
    pub fn graph_captured(&self) -> bool {
        self.graph_captured
    }

    /// Pure validation half of [`Self::set_gpu_graph_id`]: the same fail-closed host-input
    /// rejection without mutating any state, so multi-lane setters can validate every lane before
    /// assigning to any of them.
    fn ensure_graph_id_assignable(&self) -> Result<()> {
        if self.device_inputs.is_none() {
            return Err(Error::new(
                -1,
                "zrt: gpu_graph_id requires a device-input lane on the retained user stream; a \
                 host-input captured graph replays stale inputs — build the lane with \
                 `with_device_inputs`/`with_device_io`",
            ));
        }
        Ok(())
    }

    /// Build the synchronized and enqueued compositions together. Callers install both only after
    /// every fallible FFI operation succeeds, so setup errors cannot leave mismatched handles.
    fn materialize_run_options(
        &self, graph_id: Option<i32>,
        #[cfg(feature = "ep")] sync_stream: Option<&Arc<crate::SyncStream>>,
    ) -> Result<(MaterializedRunOptions, MaterializedRunOptions)> {
        let synchronized = match graph_id {
            Some(id) => RunOptions::graph_replay(id),
            None => RunOptions::new(),
        };
        let enqueued = match graph_id {
            Some(id) => RunOptions::enqueued(id),
            None => RunOptions::new().with_disable_ep_sync(true),
        };
        #[cfg(feature = "ep")]
        let synchronized = match sync_stream {
            Some(stream) => synchronized.with_sync_stream(stream),
            None => synchronized,
        };
        #[cfg(feature = "ep")]
        let enqueued = match sync_stream {
            Some(stream) => enqueued.with_sync_stream(stream),
            None => enqueued,
        };
        Ok((synchronized.freeze()?, enqueued.freeze()?))
    }

    /// Attach an owned sync stream so subsequent [`run`](Self::run) calls launch captured replay
    /// on it. Both synchronized and enqueued frozen options retain an `Arc`, and the stream itself
    /// retains its originating environment.
    #[cfg(feature = "ep")]
    pub fn set_sync_stream(&mut self, stream: &Arc<crate::SyncStream>) -> Result<()> {
        self.ensure_idle()?;
        self.session.check_sync_stream(stream)?;
        let (synchronized, enqueued) = self.materialize_run_options(self.graph_id, Some(stream))?;
        self.run_opts = Some(synchronized);
        self.enqueued_run_opts = enqueued;
        self.sync_stream = Some(Arc::clone(stream));
        Ok(())
    }
}

fn build_shared_lanes<T>(
    session: Session, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
    lanes: usize, policy: BufferSpec, what: &'static str,
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

/// An interior-synchronized pool of [`ServingLane`]s: N prebuilt lanes handed out one at a time
/// via check-out / check-in so they can run in parallel from multiple threads.
///
/// A [`ServingLanePoolGuard`] returned by [`Self::checkout`] / [`Self::try_checkout`] carries an
/// exclusive lane and **returns it to the pool on drop**. The guard derefs to [`ServingLane`], so
/// the full lane API (`input_mut_at`, [`ServingLane::run`], [`ServingLane::run_async`], …) is
/// available on a checked-out lane.
///
/// Pair with the async path: check out several lanes and call [`ServingLane::run_async`] on each
/// — the runs overlap on ORT's worker threads while the lanes stay mutually exclusive, so a single
/// driver thread can keep a replicated set busy. With synchronous [`ServingLane::run`], drive one
/// thread per checked-out lane.
///
/// Lanes may share one `Session` (ORT sessions are `Sync`) or — for least contention — be
/// built over replicated sessions before being handed to [`Self::from_lanes`].
pub struct ServingLanePool<I, O, const INPUTS: usize, const OUTPUTS: usize>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    idle: Mutex<Vec<ServingLane<I, O, INPUTS, OUTPUTS>>>,
    cv: Condvar,
    total: usize,
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for ServingLanePool<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingLanePool")
            .field("total", &self.total)
            .field("idle", &self.idle_count())
            .finish()
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> ServingLanePool<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    /// Wrap a set of prebuilt lanes as a pool. The lanes may share a session or be replicated.
    pub fn from_lanes(lanes: Vec<ServingLane<I, O, INPUTS, OUTPUTS>>) -> Result<Self> {
        if lanes.is_empty() {
            return Err(Error::new(
                -1,
                "zrt: ServingLanePool requires at least one lane",
            ));
        }
        let total = lanes.len();
        Ok(Self {
            idle: Mutex::new(lanes),
            cv: Condvar::new(),
            total,
        })
    }

    /// Total number of lanes the pool was built with.
    pub fn len(&self) -> usize {
        self.total
    }

    /// Whether the pool holds zero lanes (always false for a pool built via [`Self::from_lanes`]).
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Number of lanes currently idle (not checked out). Diagnostic only — do not synchronize on it.
    pub fn idle_count(&self) -> usize {
        self.idle.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check out a lane without blocking. Returns `None` if every lane is currently held.
    pub fn try_checkout(&self) -> Option<ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS>> {
        self.idle
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
            .map(|lane| ServingLanePoolGuard {
                lane: Some(lane),
                pool: self,
            })
    }

    /// Check out a lane, blocking until one is returned by another guard's drop.
    pub fn checkout(&self) -> ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS> {
        let mut idle = self.idle.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(lane) = idle.pop() {
                return ServingLanePoolGuard {
                    lane: Some(lane),
                    pool: self,
                };
            }
            idle = self.cv.wait(idle).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// An exclusive lease on one [`ServingLane`] from a [`ServingLanePool`]. Derefs to the lane and
/// returns it to the pool on drop.
pub struct ServingLanePoolGuard<'p, I, O, const INPUTS: usize, const OUTPUTS: usize>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    lane: Option<ServingLane<I, O, INPUTS, OUTPUTS>>,
    pool: &'p ServingLanePool<I, O, INPUTS, OUTPUTS>,
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingLanePoolGuard")
            .field("has_lane", &self.lane.is_some())
            .field("pool_total", &self.pool.total)
            .finish()
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> std::ops::Deref
    for ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    type Target = ServingLane<I, O, INPUTS, OUTPUTS>;
    fn deref(&self) -> &Self::Target {
        self.lane
            .as_ref()
            .expect("lane present while guard is held")
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> std::ops::DerefMut
    for ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.lane
            .as_mut()
            .expect("lane present while guard is held")
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> Drop
    for ServingLanePoolGuard<'_, I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default + Send,
    O: TensorElement + Clone + Default + Send,
{
    fn drop(&mut self) {
        if let Some(mut lane) = self.lane.take() {
            lane.finish_in_flight_best_effort();
            let mut idle = self.pool.idle.lock().unwrap_or_else(|e| e.into_inner());
            idle.push(lane);
            self.pool.cv.notify_one();
        }
    }
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

impl<T: TensorElement> std::fmt::Debug for Runtime<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime")
            .field("lanes", &self.lanes.len())
            .field("mode", &self.mode)
            .finish()
    }
}

/// A fixed set of caller-scheduled I/O lanes with typed inputs and outputs.
pub struct StaticIoRuntime<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    lanes: Vec<ServingLane<I, O, INPUTS, OUTPUTS>>,
    mode: RuntimeMode,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for StaticIoRuntime<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticIoRuntime")
            .field("lanes", &self.lanes.len())
            .field("mode", &self.mode)
            .field("inputs", &INPUTS)
            .field("outputs", &OUTPUTS)
            .finish()
    }
}

/// Shape-bucket cache options for [`DynamicIoRuntime`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicIoOptions {
    /// Maximum concrete shape buckets kept by this runtime.
    pub max_buckets: usize,
    /// Buffer policy used for input tensors when a new shape bucket is created.
    pub input_policy: BufferSpec,
    /// Buffer policy used for output tensors when a new shape bucket is created.
    pub output_policy: BufferSpec,
    /// Rebind lane input values before each run.
    pub rebind_inputs_each_run: bool,
    /// Assign each shape bucket a distinct ORT CUDA-graph annotation (`gpu_graph_id`)
    /// so the CUDA EP captures and replays one graph per shape. Requires the session
    /// to be built with `enable_cuda_graph=true`, **device-resident inputs** on the retained user
    /// stream (`with_device_inputs`, or per-lane streams on a replicated runtime; `cuda`
    /// feature) — host-input capture replays stale inputs and is rejected — and, with a shared
    /// session, exactly one lane per bucket (multiple lanes need replicated sessions).
    ///
    /// ORT's legacy CUDA EP keeps captured graphs for the session lifetime. Size `max_buckets` for
    /// the expected hot shape set and prewarm buckets when low tail latency matters; evicting a ZRT
    /// bucket does not reclaim ORT's captured CUDA graph or make its `gpu_graph_id` reusable.
    /// Lazy bucket creation is refused while any lane of the runtime is still in flight, so the
    /// whole planned shape set must be prebuilt/warmed before serving traffic.
    pub cuda_graph: bool,
    /// Keep bound outputs in reusable CUDA allocations instead of copying them to host as part of
    /// `RunWithBinding`. This is the nonblocking CUDA submission path; completed outputs are exposed
    /// through `ServingLane::device_output`. Requires device-input mode and the `cuda` feature.
    pub device_outputs: bool,
    /// CUDA device id for **device-resident** lane inputs (the cuda-graph-correct input path).
    /// `None` (default) = host-input mode; `Some(id)` binds CUDA-resident input tensors and refreshes
    /// them from host staging on the `DynamicIoOptions::with_device_inputs` stream before each run.
    /// Requires the `cuda` feature; mutually exclusive with `rebind_inputs_each_run`.
    pub device_id: Option<i32>,
    /// Reject shape-cache misses during serving instead of creating buckets on demand.
    ///
    /// Use [`DynamicIoRuntime::prebuild_buckets`] or [`DynamicIoRuntime::warm_buckets`] to populate
    /// the allowed shape set before calling [`DynamicIoRuntime::run_on`]. This is useful for CUDA
    /// graph serving, where first-use capture and unexpected shapes can create tail spikes.
    pub strict_shape_cache: bool,
    /// Owned CUDA stream for the host→device input refresh (device-input mode).
    /// Set via [`Self::with_device_inputs`]. Requires the `cuda` feature + `device_id` set.
    #[cfg(feature = "cuda")]
    stream: Option<Arc<crate::CudaStream>>,
    /// Owned caller CUDA streams, one per replicated lane/session.
    ///
    /// Set via [`Self::with_device_input_streams`]. This is only valid with replicated sessions;
    /// each ORT session must be constructed with the matching stream as its CUDA EP
    /// `user_compute_stream`.
    #[cfg(feature = "cuda")]
    lane_streams: Vec<Arc<crate::CudaStream>>,
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
    pub fn with_input_policy(mut self, policy: BufferSpec) -> Self {
        self.input_policy = policy;
        self
    }

    /// Set the output buffer policy.
    #[inline]
    pub fn with_output_policy(mut self, policy: BufferSpec) -> Self {
        self.output_policy = policy;
        self
    }

    /// Enable or disable per-run input rebinding for newly-created static shape buckets.
    #[inline]
    pub fn with_rebind_inputs_each_run(mut self, enabled: bool) -> Self {
        self.rebind_inputs_each_run = enabled;
        self
    }

    /// Enable per-shape CUDA-graph capture: each new shape bucket gets a distinct
    /// `gpu_graph_id` annotation so ORT captures/replays one graph per shape. The
    /// session must be built with `enable_cuda_graph=true`.
    ///
    /// `gpu_graph_id` values are never reused. The legacy CUDA EP's graph release hook is a no-op,
    /// so captured graphs live until session drop.
    #[inline]
    pub fn with_cuda_graph(mut self, enabled: bool) -> Self {
        self.cuda_graph = enabled;
        self
    }

    /// Enable **device-resident** CUDA inputs for newly-created shape buckets — the cuda-graph-correct
    /// input path. Each bucket's lanes bind CUDA-resident input tensors (the host staging buffers are
    /// refreshed → device on `stream` before every run, so a replayed graph reads fresh data instead of
    /// its capture-time snapshot). Pair with [`Self::with_cuda_graph`] and a session built with
    /// `CudaConfig::graph_replay` with the same owned stream. Requires the `cuda` feature; mutually
    /// exclusive with [`Self::with_rebind_inputs_each_run`].
    ///
    /// `stream` is the owned CUDA stream ORT replays the captured graph on (the same `Arc` passed
    /// to `CudaConfig::with_user_stream`). The returned options retain the stream with `Arc`, and
    /// the constructed runtime and session retain their own guards through native teardown.
    #[inline]
    #[cfg(feature = "cuda")]
    pub fn with_device_inputs(
        mut self, device_id: i32, stream: &Arc<crate::CudaStream>,
    ) -> Result<Self> {
        if stream.device_id() != device_id {
            return Err(Error::new(-1, "CUDA stream belongs to a different device"));
        }
        self.device_id = Some(device_id);
        self.stream = Some(Arc::clone(stream));
        self.lane_streams.clear();
        Ok(self)
    }

    /// Keep outputs device-resident for nonblocking CUDA submission and GPU-to-GPU pipelines.
    ///
    /// This requires device-input mode because completion is fenced by the exact retained CUDA
    /// stream event. Host output slice access is unavailable; use [`ServingLane::device_output`]
    /// only after the owned run reports [`CompletionStatus::Ready`].
    #[inline]
    #[cfg(feature = "cuda")]
    pub fn with_device_outputs(mut self, enabled: bool) -> Self {
        self.device_outputs = enabled;
        self
    }

    /// Enable device-resident CUDA inputs with one owned stream per replicated lane.
    ///
    /// This is the production shape for overlapping CUDA graph replays: lane `i` refreshes its device
    /// inputs and ORT replays its captured graph on `streams[i]`. The runtime must be built from
    /// replicated sessions, and each session must have been constructed with the corresponding stream
    /// as the CUDA EP `user_compute_stream`.
    #[inline]
    #[cfg(feature = "cuda")]
    pub fn with_device_input_streams(
        mut self, device_id: i32, streams: Vec<Arc<crate::CudaStream>>,
    ) -> Result<Self> {
        if streams.iter().any(|stream| stream.device_id() != device_id) {
            return Err(Error::new(-1, "CUDA stream belongs to a different device"));
        }
        self.device_id = Some(device_id);
        self.stream = None;
        self.lane_streams = streams;
        Ok(self)
    }

    /// Reject shape-cache misses during serving.
    ///
    /// Prebuild or warm the accepted shape set first. Once enabled, [`DynamicIoRuntime::run_on`] and
    /// [`DynamicIoRuntime::get_or_create_bucket`] fail fast if a request uses an unknown shape.
    #[inline]
    pub fn with_strict_shape_cache(mut self, enabled: bool) -> Self {
        self.strict_shape_cache = enabled;
        self
    }

    fn validate(self) -> Result<Self> {
        if self.max_buckets == 0 {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one shape bucket",
            ));
        }
        // CUDA-graph capture bakes the lane's input/output buffer addresses into the graph. Per-run
        // input rebinding (`rebind_inputs_each_run`) tears those baked bindings down, so the two are
        // mutually exclusive — combining them crashes at replay (rebind mid-capture invalidates the
        // captured pointer set). Refresh inputs between replays with an on-stream host→device copy onto a
        // device-resident lane buffer instead, not with rebinding. The same hazard applies to the
        // device-input mode: its device buffers stay bound and are refreshed by copy, not rebind.
        if self.cuda_graph && self.rebind_inputs_each_run {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime: `cuda_graph` and `rebind_inputs_each_run` are mutually exclusive \
                 (rebind tears down the pointers a captured CUDA graph bakes); refresh device-resident \
                 inputs with an on-stream host→device copy instead",
            ));
        }
        // The supported graph path is device-resident inputs on the retained user stream. ORT
        // captures the device buffers it is handed and never repopulates them from host bindings on
        // replay, so a host-input graph silently serves stale/never-initialized inputs. There is no
        // opt-in acknowledgment for that mode, so it fails closed here.
        if self.cuda_graph && self.device_id.is_none() {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime: `cuda_graph` requires device-resident inputs refreshed on the \
                 retained user stream (`with_device_inputs`/`with_device_input_streams`); a \
                 host-input captured graph replays stale inputs",
            ));
        }
        if self.device_outputs && self.device_id.is_none() {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime: device outputs require device-input mode and an owned CUDA stream",
            ));
        }
        #[cfg(not(feature = "cuda"))]
        if self.device_outputs {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime: device outputs require the `cuda` feature",
            ));
        }
        if self.device_id.is_some() {
            // Same baked-pointer hazard as `cuda_graph` + rebind: device-input lanes bind CUDA-resident
            // buffers the captured graph bakes; rebinding them tears that down.
            if self.rebind_inputs_each_run {
                return Err(Error::new(
                    -1,
                    "DynamicIoRuntime: `device_inputs` and `rebind_inputs_each_run` are mutually \
                     exclusive (rebind tears down the device pointers a captured CUDA graph bakes); \
                     refresh device-resident inputs with the on-stream copy instead",
                ));
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::new(
                    -1,
                    "DynamicIoRuntime: device-input mode requires the `cuda` feature",
                ));
            }
            // The refresh uses the CUDA runtime directly; a null stream is rejected at lane build.
            #[cfg(feature = "cuda")]
            if self.lane_streams.is_empty() && self.stream.is_none() {
                return Err(Error::new(
                    -1,
                    "DynamicIoRuntime: device-input mode requires a non-null CUDA stream handle",
                ));
            }
        }
        Ok(self)
    }

    #[allow(clippy::needless_return)]
    fn validate_runtime_mode(&self, mode: RuntimeMode, lane_count: usize) -> Result<()> {
        // One shape bucket assigns ONE `gpu_graph_id` to every lane it builds. With a shared
        // session every lane of that bucket replays the same ORT-captured graph, and that graph
        // baked whichever lane's input/output pointers were live at capture — every other lane
        // would silently read and write that lane's buffers. Only a single lane per bucket, or
        // replicated sessions (one session + exact stream per lane, each capturing its own graph),
        // is a sound topology. This is an ORT graph-semantics constraint, not a build-feature one,
        // so it is enforced on every feature combination.
        if self.cuda_graph && lane_count > 1 && mode != RuntimeMode::ReplicatedSessions {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime: `cuda_graph` with a shared session supports exactly one lane \
                 per shape bucket — one bucket mints one gpu_graph_id, and ORT captures whichever \
                 lane ran first into that graph, so other lanes would silently use that lane's \
                 buffers; use replicated sessions (one session + exact stream per lane) or a \
                 single lane",
            ));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (mode, lane_count);
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if self.lane_streams.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        {
            if mode != RuntimeMode::ReplicatedSessions {
                return Err(Error::new(
                    -1,
                    "DynamicIoRuntime: per-lane device-input streams require replicated sessions",
                ));
            }
            if self.lane_streams.len() != lane_count {
                return Err(Error::new(
                    -1,
                    format!(
                        "DynamicIoRuntime: expected {lane_count} per-lane CUDA streams, got {}",
                        self.lane_streams.len()
                    ),
                ));
            }
            Ok(())
        }
    }
}

impl Default for DynamicIoOptions {
    #[inline]
    fn default() -> Self {
        Self {
            max_buckets: 16,
            input_policy: BufferSpec::AUTO,
            output_policy: BufferSpec::AUTO,
            rebind_inputs_each_run: false,
            cuda_graph: false,
            device_outputs: false,
            device_id: None,
            strict_shape_cache: false,
            #[cfg(feature = "cuda")]
            stream: None,
            #[cfg(feature = "cuda")]
            lane_streams: Vec::new(),
        }
    }
}

/// Borrowed concrete input/output shapes for prebuilding or warming dynamic runtime buckets.
#[derive(Debug, Clone, Copy)]
pub struct ShapeSpec<'a, const INPUTS: usize, const OUTPUTS: usize> {
    pub input_shapes: [&'a [i64]; INPUTS],
    pub output_shapes: [&'a [i64]; OUTPUTS],
}

impl<'a, const INPUTS: usize, const OUTPUTS: usize> ShapeSpec<'a, INPUTS, OUTPUTS> {
    #[inline]
    pub fn new(input_shapes: [&'a [i64]; INPUTS], output_shapes: [&'a [i64]; OUTPUTS]) -> Self {
        Self {
            input_shapes,
            output_shapes,
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
    id: PreparedBucketId,
    lanes: Vec<ServingLane<I, O, INPUTS, OUTPUTS>>,
    last_used: u64,
    /// Lanes temporarily owned by [`OwnedDynamicIoRun`] tokens.
    detached_lanes: usize,
    /// The `gpu_graph_id` annotation assigned to this bucket's lanes when `cuda_graph` is enabled,
    /// so the provider can be notified when the bucket is evicted. ORT's legacy CUDA EP does not
    /// release captured graphs for evicted ids; ids remain one-shot and session-scoped.
    graph_id: Option<i32>,
}

struct RetiredShapeBucket<
    I: TensorElement,
    O: TensorElement,
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    _bucket: ShapeBucket<I, O, INPUTS, OUTPUTS>,
    release_done: Receiver<()>,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for ShapeBucket<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShapeBucket")
            .field("key", &self.key)
            .field("id", &self.id)
            .field("lanes", &self.lanes.len())
            .field("last_used", &self.last_used)
            .field("detached_lanes", &self.detached_lanes)
            .field("graph_id", &self.graph_id)
            .finish()
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    ShapeBucket<I, O, INPUTS, OUTPUTS>
{
    /// Concrete input/output shapes for this bucket.
    #[inline]
    pub fn key(&self) -> &ShapeKey<INPUTS, OUTPUTS> {
        &self.key
    }

    /// Number of lanes temporarily detached into owned in-flight tokens.
    #[inline]
    pub fn detached_lane_count(&self) -> usize {
        self.detached_lanes
    }

    /// Monotonic runtime-local access counter used for eviction.
    #[inline]
    pub fn last_used(&self) -> u64 {
        self.last_used
    }

    /// Prepared lanes for this concrete shape.
    #[inline]
    pub fn lanes(&self) -> &[ServingLane<I, O, INPUTS, OUTPUTS>] {
        &self.lanes
    }

    /// Mutably borrow the prepared lanes for this concrete shape.
    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [ServingLane<I, O, INPUTS, OUTPUTS>] {
        &mut self.lanes
    }

    /// Borrow one prepared lane by index.
    #[inline]
    pub fn lane(&self, i: usize) -> Result<&ServingLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: serving lane index {i} out of range")))
    }

    /// Mutably borrow one prepared lane by index.
    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut ServingLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get_mut(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: serving lane index {i} out of range")))
    }

    /// Run a closure against one lane in this concrete shape bucket.
    #[inline]
    pub fn run_on<R>(
        &mut self, i: usize, f: impl FnOnce(&mut ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        f(self.lane_mut(i)?)
    }

    /// Run every lane in this bucket `runs` times to prime ORT shape and memory caches.
    pub fn prime(&mut self, runs: usize) -> Result<()> {
        for lane in &mut self.lanes {
            lane.prime(runs)?;
        }
        Ok(())
    }

    /// Run every lane in this bucket `runs` times through the enqueued path.
    pub fn prime_enqueued(&mut self, runs: usize) -> Result<()> {
        for lane in &mut self.lanes {
            lane.prime_enqueued(runs)?;
        }
        Ok(())
    }

    /// Snapshot every lane's hot-path pointer and placement plan for this cached shape.
    ///
    /// Diagnostic/setup API; may allocate.
    pub fn audit_hot_path(&self) -> Result<Vec<LaneHotPathAudit>> {
        self.lanes.iter().map(ServingLane::audit_hot_path).collect()
    }

    /// Fail if any lane in this cached shape is not pointer-identity zero-copy.
    pub fn assert_zero_copy_plan(&self) -> Result<()> {
        for lane in &self.lanes {
            lane.assert_zero_copy_plan()?;
        }
        Ok(())
    }
}

#[derive(Clone)]
enum DynamicSessions {
    Shared(Session),
    Replicated(Vec<Session>),
}

impl DynamicSessions {
    fn allocate_captured_graph_id(&self) -> Result<i32> {
        match self {
            DynamicSessions::Shared(session) => session.allocate_captured_graph_id(),
            DynamicSessions::Replicated(sessions) => sessions
                .first()
                .ok_or_else(|| Error::new(-1, "zrt: replicated runtime has no sessions"))?
                .allocate_captured_graph_id(),
        }
    }

    fn release_captured_graph(&self, id: i32) -> Result<()> {
        match self {
            DynamicSessions::Shared(session) => session.release_captured_graph(id),
            DynamicSessions::Replicated(sessions) => {
                for session in sessions {
                    session.release_captured_graph(id)?;
                }
                Ok(())
            },
        }
    }
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
    // Field drop order is load-bearing: `buckets` (lanes whose device tensors reference the session
    // allocator) and `retired_buckets` (evicted buckets waiting for graph release) must drop BEFORE
    // `sessions`, else those device tensors release into a freed allocator → use-after-free.
    buckets: Vec<ShapeBucket<I, O, INPUTS, OUTPUTS>>,
    retired_buckets: Vec<RetiredShapeBucket<I, O, INPUTS, OUTPUTS>>,
    recovery: Arc<RecoverySlots<I, O, INPUTS, OUTPUTS>>,
    input_mem: MemoryInfo,
    output_mem: MemoryInfo,
    options: DynamicIoOptions,
    shape_plan: Option<Arc<ServingShapePlan>>,
    sessions: DynamicSessions,
    lane_count: usize,
    hot_bucket: Option<usize>,
    prepared_slots: Vec<PreparedBucketSlot>,
    free_prepared_slots: Vec<u32>,
    tick: u64,
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> std::fmt::Debug
    for DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicIoRuntime")
            .field("buckets", &self.buckets.len())
            .field("max_buckets", &self.options.max_buckets)
            .field("lane_count", &self.lane_count)
            .field(
                "shape_plan",
                &self.shape_plan.as_ref().map(|plan| plan.len()),
            )
            .field(
                "session_mode",
                &match &self.sessions {
                    DynamicSessions::Shared(_) => RuntimeMode::SharedSession,
                    DynamicSessions::Replicated(_) => RuntimeMode::ReplicatedSessions,
                },
            )
            .field("hot_bucket", &self.hot_bucket)
            .finish_non_exhaustive()
    }
}

impl<T> Runtime<T>
where
    T: TensorElement + Clone + Default,
{
    /// Build a static lane set with one shared session and `lanes` independent bindings.
    ///
    /// Formerly also available as the `from_shared_session` alias.
    pub fn shared_session(
        session: Session, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize,
    ) -> Result<Self> {
        Self::shared_session_with_buffer_policy(
            session,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            BufferSpec::AUTO,
        )
    }

    /// Build a fixed shared-session lane set with an explicit buffer policy.
    ///
    /// Formerly also available as the `from_shared_session_with_buffer_policy` alias.
    pub fn shared_session_with_buffer_policy(
        session: Session, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        lanes: usize, policy: BufferSpec,
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
            BufferSpec::AUTO,
        )
    }

    /// Build a static lane set from already-created sessions with an explicit buffer policy.
    pub fn from_sessions_with_buffer_policy(
        sessions: Vec<Session>, mem: &MemoryInfo, input_shapes: &[&[i64]],
        output_shapes: &[&[i64]], policy: BufferSpec,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(-1, "Runtime requires at least one session"));
        }
        let lanes = sessions
            .into_iter()
            .map(|session| Lane::new(session, mem, input_shapes, output_shapes, policy))
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
            BufferSpec::AUTO,
            factory,
        )
    }

    /// Build a replicated-session lane set with an explicit buffer policy.
    pub fn from_session_factory_with_buffer_policy<F>(
        lanes: usize, mem: &MemoryInfo, input_shapes: &[&[i64]], output_shapes: &[&[i64]],
        policy: BufferSpec, mut factory: F,
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
        input_shapes: &[&[i64]], output_shapes: &[&[i64]], lanes: usize, policy: BufferSpec,
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
        output_shapes: &[&[i64]], lanes: usize, policy: BufferSpec,
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

    /// Snapshot every lane's hot-path pointer and placement plan.
    ///
    /// Diagnostic/setup API; may allocate.
    pub fn audit_hot_path(&self) -> Result<Vec<LaneHotPathAudit>> {
        self.lanes.iter().map(Lane::audit_hot_path).collect()
    }

    /// Fail if any lane is not pointer-identity zero-copy over host-accessible buffers.
    pub fn assert_zero_copy_plan(&self) -> Result<()> {
        for lane in &self.lanes {
            lane.assert_zero_copy_plan()?;
        }
        Ok(())
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> StaticIoRuntime<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build a static lane set with one shared session and typed I/O lanes.
    pub fn shared_session(
        session: Session, mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], lanes: usize,
    ) -> Result<Self> {
        Self::shared_session_with_buffer_policy(
            session,
            mem,
            mem,
            input_shapes,
            output_shapes,
            lanes,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
    }

    /// Build a shared-session set with explicit memory descriptors and buffer policies.
    #[allow(clippy::too_many_arguments)]
    pub fn shared_session_with_buffer_policy(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], lanes: usize,
        input_policy: BufferSpec, output_policy: BufferSpec,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(-1, "StaticIoRuntime requires at least one lane"));
        }
        let lanes = (0..lanes)
            .map(|_| {
                ServingLane::with_buffer_policy(
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
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
    }

    /// Build a replicated-session set with explicit memory descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_sessions_with_buffer_policy(
        sessions: Vec<Session>, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec,
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
                ServingLane::with_buffer_policy(
                    session,
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

    /// Build a fixed set from borrowed replicated session handles.
    pub fn from_session_refs(
        sessions: &[Session], mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<Self> {
        Self::from_session_refs_with_buffer_policy(
            sessions,
            mem,
            mem,
            input_shapes,
            output_shapes,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
    }

    /// Build a fixed set from borrowed replicated session handles with explicit memory
    /// descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_refs_with_buffer_policy(
        sessions: &[Session], input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec,
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
                ServingLane::with_buffer_policy(
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

    /// Build a shared-session set whose lanes bind **device-resident** CUDA inputs (the
    /// cuda-graph-correct input path). Mirrors [`Self::shared_session_with_buffer_policy`] but
    /// builds lanes via [`ServingLane::with_device_inputs`]. The runtime retains the stream through
    /// each lane session; it must be the exact stream configured through `CudaConfig`. (feature `cuda`.)
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn shared_session_with_device_inputs(
        session: Session, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], lanes: usize,
        input_policy: BufferSpec, output_policy: BufferSpec, device_id: i32,
        stream: &Arc<crate::CudaStream>,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(-1, "StaticIoRuntime requires at least one lane"));
        }
        let lanes = (0..lanes)
            .map(|_| {
                ServingLane::with_device_inputs(
                    session.clone(),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                    device_id,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::SharedSession,
        })
    }

    /// Build a shared-session set with device-resident CUDA inputs and outputs.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn shared_session_with_device_io(
        session: Session, input_mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], lanes: usize, input_policy: BufferSpec, device_id: i32,
        stream: &Arc<crate::CudaStream>,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(-1, "StaticIoRuntime requires at least one lane"));
        }
        let lanes = (0..lanes)
            .map(|_| {
                ServingLane::with_device_io(
                    session.clone(),
                    input_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    device_id,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::SharedSession,
        })
    }

    /// Build a replicated-session set whose lanes bind **device-resident** CUDA inputs (the
    /// cuda-graph-correct input path). Mirrors [`Self::from_session_refs_with_buffer_policy`] but
    /// builds lanes via [`ServingLane::with_device_inputs`]. (feature `cuda`.)
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_refs_with_device_inputs(
        sessions: &[Session], input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec, device_id: i32, stream: &Arc<crate::CudaStream>,
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
                ServingLane::with_device_inputs(
                    session.clone(),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                    device_id,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build a replicated-session set whose lanes bind device-resident CUDA inputs with one
    /// owned stream per lane/session. Each session must have been configured with its
    /// corresponding stream as the CUDA EP `user_compute_stream`. (feature `cuda`.)
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_refs_with_device_input_streams(
        sessions: &[Session], input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec, device_id: i32, streams: &[Arc<crate::CudaStream>],
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "StaticIoRuntime requires at least one session",
            ));
        }
        if sessions.len() != streams.len() {
            return Err(Error::new(
                -1,
                format!(
                    "StaticIoRuntime requires one CUDA stream per session, got {} sessions and {} streams",
                    sessions.len(),
                    streams.len()
                ),
            ));
        }
        let lanes = sessions
            .iter()
            .zip(streams.iter())
            .map(|(session, stream)| {
                ServingLane::with_device_inputs(
                    session.clone(),
                    input_mem,
                    output_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    output_policy,
                    device_id,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build replicated-session lanes with device-resident inputs and outputs on one retained stream.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_refs_with_device_io(
        sessions: &[Session], input_mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec, device_id: i32,
        stream: &Arc<crate::CudaStream>,
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
                ServingLane::with_device_io(
                    session.clone(),
                    input_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    device_id,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lanes,
            mode: RuntimeMode::ReplicatedSessions,
        })
    }

    /// Build replicated-session device-I/O lanes with one exact stream per session/lane.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_refs_with_device_io_streams(
        sessions: &[Session], input_mem: &MemoryInfo, input_shapes: [&[i64]; INPUTS],
        output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec, device_id: i32,
        streams: &[Arc<crate::CudaStream>],
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "StaticIoRuntime requires at least one session",
            ));
        }
        if sessions.len() != streams.len() {
            return Err(Error::new(
                -1,
                format!(
                    "StaticIoRuntime requires one CUDA stream per session, got {} sessions and {} streams",
                    sessions.len(),
                    streams.len()
                ),
            ));
        }
        let lanes = sessions
            .iter()
            .zip(streams)
            .map(|(session, stream)| {
                ServingLane::with_device_io(
                    session.clone(),
                    input_mem,
                    input_shapes,
                    output_shapes,
                    input_policy,
                    device_id,
                    stream,
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
            BufferSpec::AUTO,
            BufferSpec::AUTO,
            factory,
        )
    }

    /// Build replicated sessions from a factory with explicit memory descriptors and policies.
    #[allow(clippy::too_many_arguments)]
    pub fn from_session_factory_with_buffer_policy<F>(
        lanes: usize, input_mem: &MemoryInfo, output_mem: &MemoryInfo,
        input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], input_policy: BufferSpec,
        output_policy: BufferSpec, mut factory: F,
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
    pub fn lanes(&self) -> &[ServingLane<I, O, INPUTS, OUTPUTS>] {
        &self.lanes
    }

    #[inline]
    pub fn lanes_mut(&mut self) -> &mut [ServingLane<I, O, INPUTS, OUTPUTS>] {
        &mut self.lanes
    }

    #[inline]
    pub fn lane(&self, i: usize) -> Result<&ServingLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: static I/O lane index {i} out of range")))
    }

    #[inline]
    pub fn lane_mut(&mut self, i: usize) -> Result<&mut ServingLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
            .get_mut(i)
            .ok_or_else(|| Error::new(-1, format!("zrt: static I/O lane index {i} out of range")))
    }

    #[inline]
    pub fn into_lanes(self) -> Vec<ServingLane<I, O, INPUTS, OUTPUTS>> {
        self.lanes
    }

    #[inline]
    pub fn run_on<R>(
        &mut self, i: usize, f: impl FnOnce(&mut ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
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

    /// Run every lane `runs` times through the enqueued path to prime ORT caches.
    pub fn prime_enqueued(&mut self, runs: usize) -> Result<()> {
        for lane in &mut self.lanes {
            lane.prime_enqueued(runs)?;
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

    /// Assign every lane in this set the same CUDA-graph annotation id. Each shape bucket should
    /// get a distinct id so ORT captures one graph per shape.
    ///
    /// **Topology constraint:** every lane sharing this id also shares ORT's single captured graph
    /// for it, and that graph bakes the input/output pointers of whichever lane captured it. This
    /// is only sound when this set has exactly one lane, or when every lane has its own replicated
    /// session (and its own retained stream) so each captures an independent graph.
    /// [`DynamicIoRuntime`] enforces that topology at construction; direct
    /// `StaticIoRuntime`/[`ServingLane`] callers must enforce it themselves. Lanes must be
    /// device-input lanes — host-input capture replays stale inputs (see
    /// [`ServingLane::set_gpu_graph_id`]).
    #[inline]
    pub fn set_gpu_graph_id(&mut self, id: i32) -> Result<()> {
        // Two-pass: validate every lane first (idle + device-input fail-closed checks), then
        // assign, so a mid-set rejection cannot leave earlier lanes annotated and later lanes
        // not. Only the run-option freeze inside `set_gpu_graph_id` remains fallible in the
        // second pass; the configuration rejections above are atomic.
        for lane in &mut self.lanes {
            lane.ensure_idle()?;
            lane.ensure_graph_id_assignable()?;
        }
        for lane in &mut self.lanes {
            lane.set_gpu_graph_id(id)?;
        }
        Ok(())
    }

    /// Snapshot every lane's hot-path pointer and placement plan.
    ///
    /// Diagnostic/setup API; may allocate.
    pub fn audit_hot_path(&self) -> Result<Vec<LaneHotPathAudit>> {
        self.lanes.iter().map(ServingLane::audit_hot_path).collect()
    }

    /// Fail if any lane is not pointer-identity zero-copy over host-accessible buffers.
    pub fn assert_zero_copy_plan(&self) -> Result<()> {
        for lane in &self.lanes {
            lane.assert_zero_copy_plan()?;
        }
        Ok(())
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize>
    DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
{
    fn allocate_prepared_bucket_id(&mut self, bucket_index: usize) -> Result<PreparedBucketId> {
        if let Some(slot) = self.free_prepared_slots.pop() {
            let Some(entry) = self.prepared_slots.get_mut(slot as usize) else {
                return Err(Error::new(
                    -1,
                    "zrt: prepared bucket free-slot index is invalid",
                ));
            };
            if entry.bucket_index.is_some() {
                return Err(Error::new(
                    -1,
                    "zrt: prepared bucket free-slot accounting is inconsistent",
                ));
            }
            entry.bucket_index = Some(bucket_index);
            return Ok(PreparedBucketId {
                slot,
                generation: entry.generation,
            });
        }
        let slot = u32::try_from(self.prepared_slots.len())
            .map_err(|_| Error::new(-1, "zrt: prepared bucket slot space exhausted"))?;
        self.prepared_slots.push(PreparedBucketSlot {
            generation: 1,
            bucket_index: Some(bucket_index),
        });
        Ok(PreparedBucketId {
            slot,
            generation: 1,
        })
    }

    #[inline]
    fn prepared_bucket_index(&self, id: PreparedBucketId) -> Option<usize> {
        self.prepared_slots
            .get(id.slot as usize)
            .filter(|entry| entry.generation == id.generation)
            .and_then(|entry| entry.bucket_index)
            .filter(|&index| {
                self.buckets
                    .get(index)
                    .is_some_and(|bucket| bucket.id == id)
            })
    }

    fn retire_prepared_bucket_id(&mut self, id: PreparedBucketId) {
        let Some(entry) = self.prepared_slots.get_mut(id.slot as usize) else {
            return;
        };
        if entry.generation != id.generation || entry.bucket_index.is_none() {
            return;
        }
        entry.bucket_index = None;
        if let Some(next) = entry.generation.checked_add(1) {
            entry.generation = next;
            self.free_prepared_slots.push(id.slot);
        }
    }

    fn repair_prepared_bucket_indices_from(&mut self, start: usize) {
        for (index, bucket) in self.buckets.iter().enumerate().skip(start) {
            let Some(entry) = self.prepared_slots.get_mut(bucket.id.slot as usize) else {
                eprintln!(
                    "st-zrt: prepared bucket references an invalid slot while repairing indices"
                );
                continue;
            };
            if entry.generation != bucket.id.generation {
                eprintln!("st-zrt: prepared bucket generation mismatch while repairing indices");
                continue;
            }
            entry.bucket_index = Some(index);
        }
    }

    fn reclaim_dropped_runs_inner(&mut self) -> usize {
        let mut recovered_count = 0;
        while let Some(slot_index) = self.recovery.pop_ready() {
            let recovered = self.recovery.slots[slot_index].take();
            let Some(recovered) = recovered else {
                // Reclaim also runs from DynamicIoRuntime::drop; diagnose a corrupted notification
                // without panicking during teardown.
                eprintln!("st-zrt: ready recovery index referenced an empty slot");
                continue;
            };
            let Some(bucket_index) = self.prepared_bucket_index(recovered.bucket_id) else {
                eprintln!(
                    "st-zrt: recovering a dropped owned run failed: source bucket is gone; leaking recovered lane"
                );
                std::mem::forget(recovered);
                continue;
            };
            let Some(bucket) = self.buckets.get_mut(bucket_index) else {
                eprintln!(
                    "st-zrt: recovering a dropped owned run failed: source bucket is gone; leaking recovered lane"
                );
                std::mem::forget(recovered);
                continue;
            };
            if bucket.detached_lanes == 0 {
                eprintln!(
                    "st-zrt: recovering a dropped owned run found zero detached-lane accounting; leaking recovered lane"
                );
                std::mem::forget(recovered);
                continue;
            }
            bucket.detached_lanes -= 1;
            bucket.lanes.push(recovered.lane);
            recovered_count += 1;
        }
        recovered_count
    }

    /// Fence-and-return lanes whose owned run tokens were dropped instead of explicitly completed.
    ///
    /// This recovery path is also invoked automatically before mutable cache operations. It is
    /// exposed so diagnostics can force reclamation before inspecting bucket lane counts.
    pub fn reclaim_dropped_runs(&mut self) -> usize {
        self.reclaim_dropped_runs_inner()
    }

    fn notify_bucket_graph_release_best_effort(&self, graph_id: Option<i32>, action: &str) {
        let Some(id) = graph_id else {
            return;
        };
        if let Err(e) = self.sessions.release_captured_graph(id) {
            eprintln!("st-zrt: release_captured_graph({id}) on {action} failed: {e}");
        }
    }

    fn drain_active_buckets_and_release_graphs(&mut self, action: &str) -> bool {
        let mut all_fenced = true;
        let mut graph_ids = Vec::new();
        let active_ids = self
            .buckets
            .iter()
            .map(|bucket| bucket.id)
            .collect::<Vec<_>>();
        for id in active_ids {
            self.retire_prepared_bucket_id(id);
        }
        for mut bucket in std::mem::take(&mut self.buckets) {
            for lane in &mut bucket.lanes {
                lane.finish_in_flight_best_effort();
            }
            if bucket.lanes.iter().any(|lane| lane.in_flight) {
                // An unfenced provider run may still dereference every resource in the bucket. A
                // deliberate leak is the only safe teardown fallback: dropping tensors/bindings or
                // releasing the captured graph could otherwise cause a device-side use-after-free.
                eprintln!(
                    "st-zrt: {action} could not fence an in-flight lane; leaking its shape bucket for safety"
                );
                std::mem::forget(bucket);
                all_fenced = false;
                continue;
            }
            if bucket.detached_lanes == 0 {
                if let Some(graph_id) = bucket.graph_id {
                    graph_ids.push(graph_id);
                }
            }
            // Drop lanes (and therefore any retained leases) before asking ORT to release graphs.
            drop(bucket);
        }
        for graph_id in graph_ids {
            self.notify_bucket_graph_release_best_effort(Some(graph_id), action);
        }
        all_fenced
    }

    fn spawn_bucket_graph_release_notification(
        &self, graph_id: i32, action: &'static str,
    ) -> Receiver<()> {
        let (done_tx, done_rx) = mpsc::channel();
        let sessions = self.sessions.clone();
        let spawn_result = std::thread::Builder::new()
            .name("st-zrt-graph-release".to_string())
            .spawn(move || {
                if let Err(e) = sessions.release_captured_graph(graph_id) {
                    eprintln!("st-zrt: release_captured_graph({graph_id}) on {action} failed: {e}");
                }
                let _ = done_tx.send(());
            });
        if let Err(err) = spawn_result {
            eprintln!(
                "st-zrt: spawning graph release thread for {action} failed: {err}; releasing synchronously"
            );
            self.notify_bucket_graph_release_best_effort(Some(graph_id), action);
            let (done_tx, done_rx) = mpsc::channel();
            let _ = done_tx.send(());
            return done_rx;
        }
        done_rx
    }

    fn retire_bucket_at(&mut self, index: usize, action: &'static str) -> bool {
        if self.buckets[index].detached_lanes != 0 {
            eprintln!("st-zrt: refusing to retire a bucket with owned runs during {action}");
            return false;
        }
        // A compatibility split-enqueue caller may still have work pending in this bucket. Fence
        // every lane before removing access to it or asking the provider to release its graph.
        for lane in &mut self.buckets[index].lanes {
            lane.finish_in_flight_best_effort();
        }
        if self.buckets[index].lanes.iter().any(|lane| lane.in_flight) {
            eprintln!(
                "st-zrt: {action} could not fence an in-flight lane; leaking its shape bucket for safety"
            );
            let retired_id = self.buckets[index].id;
            self.retire_prepared_bucket_id(retired_id);
            let bucket = self.buckets.swap_remove(index);
            if index < self.buckets.len() {
                self.repair_prepared_bucket_indices_from(index);
            }
            std::mem::forget(bucket);
            return true;
        }
        let release_done = self.buckets[index]
            .graph_id
            .map(|graph_id| self.spawn_bucket_graph_release_notification(graph_id, action));
        let retired_id = self.buckets[index].id;
        self.retire_prepared_bucket_id(retired_id);
        let bucket = self.buckets.swap_remove(index);
        if index < self.buckets.len() {
            self.repair_prepared_bucket_indices_from(index);
        }
        if let Some(release_done) = release_done {
            self.retired_buckets.push(RetiredShapeBucket {
                _bucket: bucket,
                release_done,
            });
        }
        true
    }

    fn reap_retired_buckets(&mut self) {
        let mut i = 0;
        while i < self.retired_buckets.len() {
            match self.retired_buckets[i].release_done.try_recv() {
                Ok(()) | Err(TryRecvError::Disconnected) => {
                    self.retired_buckets.swap_remove(i);
                },
                Err(TryRecvError::Empty) => i += 1,
            }
        }
    }

    fn wait_for_retired_buckets(&mut self) {
        for retired in &self.retired_buckets {
            let _ = retired.release_done.recv();
        }
        self.retired_buckets.clear();
    }
}

impl<I: TensorElement, O: TensorElement, const INPUTS: usize, const OUTPUTS: usize> Drop
    for DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
{
    fn drop(&mut self) {
        self.reclaim_dropped_runs_inner();
        self.wait_for_retired_buckets();
        let _ = self.drain_active_buckets_and_release_graphs("runtime drop");
    }
}

impl<I, O, const INPUTS: usize, const OUTPUTS: usize> DynamicIoRuntime<I, O, INPUTS, OUTPUTS>
where
    I: TensorElement + Clone + Default,
    O: TensorElement + Clone + Default,
{
    /// Build a dynamic-shape runtime with one shared session and `lanes` static lanes per shape.
    pub fn shared_session(session: Session, mem: MemoryInfo, lanes: usize) -> Result<Self> {
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
        session: Session, input_mem: MemoryInfo, output_mem: MemoryInfo, lanes: usize,
        options: DynamicIoOptions,
    ) -> Result<Self> {
        if lanes == 0 {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one lane",
            ));
        }
        let options = options.validate()?;
        options.validate_runtime_mode(RuntimeMode::SharedSession, lanes)?;
        let recovery_capacity = options
            .max_buckets
            .checked_mul(lanes)
            .ok_or_else(|| Error::new(-1, "zrt: dynamic recovery slot capacity overflow"))?;
        let recovery = Arc::new(RecoverySlots::new(recovery_capacity));
        Ok(Self {
            sessions: DynamicSessions::Shared(session),
            input_mem,
            output_mem,
            options,
            shape_plan: None,
            lane_count: lanes,
            buckets: Vec::new(),
            retired_buckets: Vec::new(),
            recovery,
            hot_bucket: None,
            prepared_slots: Vec::new(),
            free_prepared_slots: Vec::new(),
            tick: 0,
        })
    }

    /// Build a dynamic-shape runtime from replicated sessions.
    pub fn from_sessions(sessions: Vec<Session>, mem: MemoryInfo) -> Result<Self> {
        let output_mem = mem.try_clone_descriptor()?;
        Self::from_sessions_with_options(sessions, mem, output_mem, DynamicIoOptions::default())
    }

    /// Build a dynamic-shape runtime from replicated sessions with explicit memory descriptors
    /// and shape-cache options. `Session` is already a cheap-clone shared handle, so no outer
    /// `Arc<Session>` layer is needed.
    pub fn from_sessions_with_options(
        sessions: Vec<Session>, input_mem: MemoryInfo, output_mem: MemoryInfo,
        options: DynamicIoOptions,
    ) -> Result<Self> {
        if sessions.is_empty() {
            return Err(Error::new(
                -1,
                "DynamicIoRuntime requires at least one session",
            ));
        }
        let lane_count = sessions.len();
        let options = options.validate()?;
        options.validate_runtime_mode(RuntimeMode::ReplicatedSessions, lane_count)?;
        let recovery_capacity = options
            .max_buckets
            .checked_mul(lane_count)
            .ok_or_else(|| Error::new(-1, "zrt: dynamic recovery slot capacity overflow"))?;
        let recovery = Arc::new(RecoverySlots::new(recovery_capacity));
        Ok(Self {
            sessions: DynamicSessions::Replicated(sessions),
            input_mem,
            output_mem,
            options,
            shape_plan: None,
            lane_count,
            buckets: Vec::new(),
            retired_buckets: Vec::new(),
            recovery,
            hot_bucket: None,
            prepared_slots: Vec::new(),
            free_prepared_slots: Vec::new(),
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
        // Device-input mode (cuda-graph-correct path): lanes bind CUDA-resident inputs refreshed on
        // the retained stream. Otherwise the host-input path binds host staging buffers directly.
        let mut lanes = if let Some(device_id) = self.options.device_id {
            #[cfg(feature = "cuda")]
            {
                match &self.sessions {
                    DynamicSessions::Shared(session) => {
                        let stream = self.options.stream.as_ref().ok_or_else(|| {
                            Error::new(-1, "device-input mode is missing its owned CUDA stream")
                        })?;
                        if self.options.device_outputs {
                            StaticIoRuntime::shared_session_with_device_io(
                                session.clone(),
                                &self.input_mem,
                                input_shapes,
                                output_shapes,
                                self.lane_count,
                                self.options.input_policy,
                                device_id,
                                stream,
                            )
                        } else {
                            StaticIoRuntime::shared_session_with_device_inputs(
                                session.clone(),
                                &self.input_mem,
                                &self.output_mem,
                                input_shapes,
                                output_shapes,
                                self.lane_count,
                                self.options.input_policy,
                                self.options.output_policy,
                                device_id,
                                stream,
                            )
                        }
                    },
                    DynamicSessions::Replicated(sessions) => {
                        if self.options.lane_streams.is_empty() {
                            let stream = self.options.stream.as_ref().ok_or_else(|| {
                                Error::new(-1, "device-input mode is missing its owned CUDA stream")
                            })?;
                            if self.options.device_outputs {
                                StaticIoRuntime::from_session_refs_with_device_io(
                                    sessions,
                                    &self.input_mem,
                                    input_shapes,
                                    output_shapes,
                                    self.options.input_policy,
                                    device_id,
                                    stream,
                                )
                            } else {
                                StaticIoRuntime::from_session_refs_with_device_inputs(
                                    sessions,
                                    &self.input_mem,
                                    &self.output_mem,
                                    input_shapes,
                                    output_shapes,
                                    self.options.input_policy,
                                    self.options.output_policy,
                                    device_id,
                                    stream,
                                )
                            }
                        } else if self.options.device_outputs {
                            StaticIoRuntime::from_session_refs_with_device_io_streams(
                                sessions,
                                &self.input_mem,
                                input_shapes,
                                output_shapes,
                                self.options.input_policy,
                                device_id,
                                &self.options.lane_streams,
                            )
                        } else {
                            StaticIoRuntime::from_session_refs_with_device_input_streams(
                                sessions,
                                &self.input_mem,
                                &self.output_mem,
                                input_shapes,
                                output_shapes,
                                self.options.input_policy,
                                self.options.output_policy,
                                device_id,
                                &self.options.lane_streams,
                            )
                        }
                    },
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                // `validate()` already rejected device-input without `cuda`, so this is unreachable.
                let _ = device_id;
                return Err(Error::new(
                    -1,
                    "zrt: device-input mode requires the `cuda` feature",
                ));
            }
        } else {
            match &self.sessions {
                DynamicSessions::Shared(session) => {
                    StaticIoRuntime::shared_session_with_buffer_policy(
                        session.clone(),
                        &self.input_mem,
                        &self.output_mem,
                        input_shapes,
                        output_shapes,
                        self.lane_count,
                        self.options.input_policy,
                        self.options.output_policy,
                    )
                },
                DynamicSessions::Replicated(sessions) => {
                    StaticIoRuntime::from_session_refs_with_buffer_policy(
                        sessions,
                        &self.input_mem,
                        &self.output_mem,
                        input_shapes,
                        output_shapes,
                        self.options.input_policy,
                        self.options.output_policy,
                    )
                },
            }
        }?;
        lanes.set_rebind_inputs_each_run(self.options.rebind_inputs_each_run);
        Ok(lanes)
    }

    /// Install the finite canonical shape plan before creating any buckets.
    ///
    /// CUDA-graph runtimes require this call: live shapes outside the plan are rejected, so graph
    /// capture remains finite and startup-controlled. Replacing a plan after allocation is refused.
    pub fn install_shape_plan(&mut self, plan: Arc<ServingShapePlan>) -> Result<()> {
        if !self.buckets.is_empty() || !self.retired_buckets.is_empty() {
            return Err(Error::new(
                -1,
                "zrt: install the serving shape plan before building or warming buckets",
            ));
        }
        if plan.len() > self.options.max_buckets {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: serving plan has {} buckets but runtime capacity is {}",
                    plan.len(),
                    self.options.max_buckets
                ),
            ));
        }
        for bucket in plan.buckets() {
            if bucket.input_shapes().len() != INPUTS || bucket.output_shapes().len() != OUTPUTS {
                return Err(Error::new(
                    -1,
                    format!(
                        "zrt: serving plan arity mismatch: runtime expects {INPUTS} inputs/{OUTPUTS} outputs, plan bucket has {}/{}",
                        bucket.input_shapes().len(),
                        bucket.output_shapes().len()
                    ),
                ));
            }
            let policy_matches = match bucket.output_policy() {
                OutputPolicy::HostBuffer => !self.options.device_outputs,
                OutputPolicy::CudaPinned => false,
                OutputPolicy::DeviceResident => self.options.device_outputs,
            };
            if !policy_matches {
                return Err(Error::new(
                    -1,
                    "zrt: serving plan output placement does not match DynamicIoOptions",
                ));
            }
        }
        self.shape_plan = Some(plan);
        Ok(())
    }

    /// The installed finite shape plan, if any.
    #[inline]
    pub fn shape_plan(&self) -> Option<&ServingShapePlan> {
        self.shape_plan.as_deref()
    }

    fn validate_planned_shape(
        &self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<()> {
        let Some(plan) = &self.shape_plan else {
            if self.options.cuda_graph {
                return Err(Error::new(
                    -1,
                    "zrt: CUDA-graph runtime requires a sealed ServingShapePlan before bucket creation",
                ));
            }
            return Ok(());
        };
        let id = plan.classify(&input_shapes).map_err(|error| {
            Error::new(
                -1,
                format!("zrt: serving shape rejected by sealed plan: {error}"),
            )
        })?;
        let bucket = plan
            .bucket(id)
            .ok_or_else(|| Error::new(-1, "zrt: serving plan returned an invalid shape id"))?;
        let exact_inputs = bucket
            .input_shapes()
            .iter()
            .zip(input_shapes)
            .all(|(planned, actual)| planned.as_slice() == actual);
        let exact_outputs = bucket
            .output_shapes()
            .iter()
            .zip(output_shapes)
            .all(|(planned, actual)| planned.as_slice() == actual);
        if !exact_inputs || !exact_outputs {
            return Err(Error::new(
                -1,
                "zrt: request does not exactly match the canonical input/output shapes; pad to the classified bucket before running",
            ));
        }
        Ok(())
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

    /// Refuse lazy CUDA-graph bucket creation while any lane of this runtime may still have
    /// provider work pending: detached lanes owned by in-flight tokens, and lanes left in flight by
    /// the legacy `run_enqueued` API. Both are visible without hardware because the check only
    /// reads lane/bucket accounting.
    /// CUDA-graph capture is device-wide serialized, but this runtime's idleness guard is
    /// per-runtime and per-session-clone only. Concurrent capture across *separate*
    /// `DynamicIoRuntime`s (including runtimes built from cloned sessions), or between a runtime
    /// and a directly owned `ServingLane` capturing its first run, is NOT serialized by st-zrt
    /// and is unsupported unless callers serialize it externally (the same way
    /// `tests/common::cuda_graph_capture_lock` serializes capture across test binaries).
    fn refuse_graph_bucket_creation_while_in_flight(&self) -> Result<()> {
        let Some(index) = self.buckets.iter().position(|bucket| {
            bucket.detached_lanes > 0 || bucket.lanes.iter().any(|lane| lane.in_flight)
        }) else {
            return Ok(());
        };
        let bucket = &self.buckets[index];
        let detached = bucket.detached_lanes;
        let legacy_in_flight = bucket.lanes.iter().filter(|lane| lane.in_flight).count();
        Err(Error::new(
            -1,
            format!(
                "zrt: refusing to create a new CUDA-graph shape bucket while bucket {index} still \
                 has {detached} detached lane(s) and {legacy_in_flight} in-flight lane(s) — graph \
                 capture must not overlap a live replay; prebuild or warm every bucket before \
                 serving traffic"
            ),
        ))
    }

    fn evict_one_bucket_if_full(&mut self) -> Result<()> {
        if self.buckets.len() < self.options.max_buckets {
            return Ok(());
        }
        if self.options.cuda_graph {
            return Err(Error::new(
                -1,
                format!(
                    "zrt: cuda_graph dynamic shape cache is full (max_buckets={}); \
                     refusing to evict because ORT's legacy CUDA EP keeps captured graphs for \
                     the session lifetime. Increase max_buckets and prewarm the hot shape set, \
                     or disable cuda_graph for unbounded shapes.",
                    self.options.max_buckets
                ),
            ));
        }
        self.hot_bucket = None;
        let (oldest, _) = self
            .buckets
            .iter()
            .enumerate()
            .filter(|(_, bucket)| bucket.detached_lanes == 0)
            .min_by_key(|(_, bucket)| bucket.last_used)
            .ok_or_else(|| {
                Error::new(
                    -1,
                    "zrt: dynamic shape cache is full and every bucket has owned runs in flight",
                )
            })?;
        if !self.retire_bucket_at(oldest, "eviction") {
            return Err(Error::new(
                -1,
                "zrt: dynamic shape cache eviction refused inconsistent detached-lane accounting",
            ));
        }
        Ok(())
    }

    fn get_or_create_bucket_inner(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        allow_create: bool,
    ) -> Result<&mut ShapeBucket<I, O, INPUTS, OUTPUTS>> {
        self.reclaim_dropped_runs_inner();
        self.validate_planned_shape(input_shapes, output_shapes)?;
        self.reap_retired_buckets();
        if let Some(i) = self.find_bucket_index(input_shapes, output_shapes) {
            let tick = self.next_tick();
            self.buckets[i].last_used = tick;
            self.hot_bucket = Some(i);
            return Ok(&mut self.buckets[i]);
        }

        if !allow_create {
            return Err(Error::new(
                -1,
                "zrt: dynamic shape cache miss rejected by strict_shape_cache; prebuild or warm \
                 this shape before serving",
            ));
        }

        // Lazy creation is the capture path: the first run of a new CUDA-graph bucket captures on
        // whatever thread/shape arrives, and capture is device-wide serialized — it must not
        // overlap a live replay (or any other pending provider work) on another lane of this
        // runtime. Refuse while any lane is detached into an owned run or still in flight after a
        // legacy `run_enqueued`; the whole planned shape set must be built before traffic.
        if self.options.cuda_graph {
            self.refuse_graph_bucket_creation_while_in_flight()?;
        }

        self.evict_one_bucket_if_full()?;
        let key = ShapeKey::new(input_shapes, output_shapes);
        let mut lane_set = self.build_lane_set(input_shapes, output_shapes)?;
        let graph_id = if self.options.cuda_graph {
            let id = self.sessions.allocate_captured_graph_id()?;
            lane_set.set_gpu_graph_id(id)?;
            Some(id)
        } else {
            None
        };
        let mut lanes = lane_set.into_lanes();
        // Eager capture (cuda_graph): a CUDA-graph lane captures ORT's graph on its FIRST run,
        // and capture is device-wide serialized — it must never overlap a live replay. Bucket
        // creation is the one place this runtime's idleness was just proven
        // (`refuse_graph_bucket_creation_while_in_flight` above), so capture every fresh lane
        // NOW, before the bucket becomes reachable through later cache hits. This closes the
        // prebuild-then-serve window: without it, `prebuild A+B` created both buckets without
        // capturing, and the first later run of B (a plain cache HIT, which nothing guarded)
        // would capture while prebuilt A is already replaying. For device-input graph lanes —
        // the only lanes `set_gpu_graph_id` accepts — `ServingLane::run` *is* the enqueued
        // capture path plus an exact-stream fence (`run_bound_binding_enqueued` +
        // `synchronize_outputs`), so the eager capture uses the same run options production
        // replays use; `prime`/`prime_enqueued` afterwards only warm caches on the
        // already-captured graph. On capture failure the bucket is not created; its
        // `gpu_graph_id` stays allocated and, per the never-reuse invariant, is never recycled.
        if self.options.cuda_graph {
            for (lane_index, lane) in lanes.iter_mut().enumerate() {
                lane.run().map_err(|error| {
                    Error::new(
                        -1,
                        format!(
                            "zrt: eager CUDA-graph capture for lane {lane_index} of a new shape \
                             bucket failed: {error}"
                        ),
                    )
                })?;
            }
        }
        let last_used = self.next_tick();
        let bucket_id = self.allocate_prepared_bucket_id(self.buckets.len())?;
        let slot_base = (bucket_id.slot as usize)
            .checked_mul(self.lane_count)
            .ok_or_else(|| Error::new(-1, "zrt: dynamic recovery slot index overflow"))?;
        for (lane_index, lane) in lanes.iter_mut().enumerate() {
            lane.recovery_slot = Some(slot_base + lane_index);
        }
        self.buckets.push(ShapeBucket {
            key,
            id: bucket_id,
            lanes,
            last_used,
            detached_lanes: 0,
            graph_id,
        });
        self.hot_bucket = Some(self.buckets.len() - 1);
        self.buckets.last_mut().ok_or_else(|| {
            Error::new(
                -1,
                "zrt: failed to access newly created dynamic shape bucket",
            )
        })
    }

    /// Get the bucket for concrete shapes, creating it on first use unless strict shape-cache mode is
    /// enabled.
    ///
    /// Cache hits do not allocate in Rust: shape slices are compared directly against cached
    /// keys. Misses allocate tensor buffers and bind a new static lane set.
    pub fn get_or_create_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<&mut ShapeBucket<I, O, INPUTS, OUTPUTS>> {
        self.get_or_create_bucket_inner(
            input_shapes,
            output_shapes,
            !self.options.strict_shape_cache,
        )
    }

    /// Resolve the runtime-local id of an already-built bucket during setup.
    ///
    /// Cache this id and use [`Self::enqueue_prepared`] on the serving hot path.
    pub fn prepared_bucket_id(
        &self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> Result<PreparedBucketId> {
        self.find_bucket_index(input_shapes, output_shapes)
            .map(|index| self.buckets[index].id)
            .ok_or_else(|| Error::new(-1, "zrt: shape bucket has not been prepared"))
    }

    /// Prepare and enqueue one lane, transferring exclusive lane ownership to the returned token.
    ///
    /// The runtime remains usable for other lanes while the token is alive. Completion returns the
    /// lane to its original shape bucket; dropping the token instead fences it and queues it for
    /// automatic recovery before the runtime's next mutable cache operation.
    pub fn enqueue_owned(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
        prepare: impl FnOnce(&mut ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<()>,
    ) -> Result<OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>> {
        let bucket_id = self.get_or_create_bucket(input_shapes, output_shapes)?.id;
        self.enqueue_prepared(bucket_id, prepare)
    }

    /// Enqueue directly into a prebuilt bucket identified during setup.
    ///
    /// Unlike [`Self::enqueue_owned`], this performs no shape classification, cache lookup, LRU
    /// update, or shape allocation. The id is runtime-local and remains valid until that bucket is
    /// explicitly removed or evicted. CUDA-graph runtimes never evict captured buckets.
    pub fn enqueue_prepared(
        &mut self, bucket_id: PreparedBucketId,
        prepare: impl FnOnce(&mut ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<()>,
    ) -> Result<OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>> {
        self.reclaim_dropped_runs_inner();
        let lane_count = self.lane_count;
        let (mut lane, bucket_id) = {
            let bucket_index = self
                .prepared_bucket_index(bucket_id)
                .ok_or_else(|| Error::new(-1, "zrt: prepared shape bucket is unavailable"))?;
            let bucket = &mut self.buckets[bucket_index];
            if bucket.detached_lanes.checked_add(bucket.lanes.len()) != Some(lane_count) {
                return Err(Error::new(
                    -1,
                    "zrt: detached-lane accounting diverged from the lane slot count",
                ));
            }
            let lane = bucket.lanes.pop().ok_or_else(|| {
                Error::new(-1, "zrt: all lanes for this serving shape are in flight")
            })?;
            bucket.detached_lanes += 1;
            (lane, bucket.id)
        };

        if let Err(error) = prepare(&mut lane) {
            let _ = self.return_detached_lane(bucket_id, lane);
            return Err(error);
        }
        if let Err(error) = lane.run_enqueued() {
            lane.finish_in_flight_best_effort();
            if lane.in_flight {
                // Preserve detached accounting and leak the unfenced lane. That intentionally pins
                // the bucket/graph instead of returning unsafe-to-reuse resources to the pool.
                std::mem::forget(lane);
            } else {
                let _ = self.return_detached_lane(bucket_id, lane);
            }
            return Err(error);
        }
        Ok(OwnedDynamicIoRun {
            lane: Some(lane),
            bucket_id,
            recovery: Some(Arc::clone(&self.recovery)),
        })
    }

    /// Query multiple owned CUDA runs after one calling-thread device validation.
    ///
    /// Ready runs are marked complete in-place, releasing their graph leases. A subsequent
    /// [`Self::complete_owned`] therefore returns them without another event query or synchronization.
    /// This performs one nonblocking pass only; callers own timeout and backoff policy. Device,
    /// event, and result-length validation happens before any run is changed. If a raw CUDA query
    /// fails mid-pass, earlier ready runs may already be completed safely in place and callers must
    /// inspect the owned tokens rather than assume all-or-nothing progress.
    #[cfg(feature = "cuda")]
    pub fn poll_owned_runs(
        &self, poller: crate::CudaCompletionPoller,
        runs: &mut [OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>], statuses: &mut [CompletionStatus],
    ) -> Result<()> {
        if statuses.len() < runs.len() {
            return Err(Error::new(
                -1,
                "zrt: owned-run completion result buffer is smaller than the run batch",
            ));
        }
        let mut has_in_flight = false;
        for run in runs.iter() {
            let lane = run
                .lane
                .as_ref()
                .expect("zrt: owned dynamic run missing lane");
            if !lane.in_flight {
                continue;
            }
            has_in_flight = true;
            let event = run.completion_event().ok_or_else(|| {
                Error::new(
                    -1,
                    "zrt: batch polling requires every in-flight run to have an exact CUDA completion event",
                )
            })?;
            if event.device_id() != poller.device_id() {
                return Err(Error::new(
                    -1,
                    "zrt: owned-run completion batch contains another CUDA device",
                ));
            }
        }
        if has_in_flight {
            poller.validate_current_device()?;
        }
        for (run, status) in runs.iter_mut().zip(statuses) {
            let lane = run
                .lane
                .as_ref()
                .expect("zrt: owned dynamic run missing lane");
            if !lane.in_flight {
                *status = CompletionStatus::Ready;
                continue;
            }
            let event = run
                .completion_event()
                .expect("zrt: owned run event preflighted above");
            if poller.query_validated(event)? {
                run.lane
                    .as_mut()
                    .expect("zrt: owned dynamic run missing lane")
                    .clear_in_flight();
                *status = CompletionStatus::Ready;
            } else {
                *status = CompletionStatus::Pending;
            }
        }
        Ok(())
    }

    /// Synchronize an owned run, expose its completed lane to `consume`, then return the lane to
    /// the runtime even when consumption reports an error. A synchronization error also returns a
    /// successfully fallback-fenced lane; an unfenced lane is deliberately leaked and pins its
    /// bucket rather than being made reusable.
    pub fn complete_owned<R>(
        &mut self, mut run: OwnedDynamicIoRun<I, O, INPUTS, OUTPUTS>,
        consume: impl FnOnce(&ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        let sync_result = run.synchronize();
        let consume_result = match sync_result {
            Ok(()) => consume(
                run.lane
                    .as_ref()
                    .expect("zrt: owned dynamic run missing lane after synchronization"),
            ),
            Err(error) => Err(error),
        };
        let reusable = !run
            .lane
            .as_ref()
            .expect("zrt: owned dynamic run missing lane after synchronization")
            .in_flight;
        let lane = run
            .lane
            .take()
            .expect("zrt: owned dynamic run missing lane during completion");
        let bucket_id = run.bucket_id;
        run.recovery.take();
        if !reusable {
            // The detached count deliberately remains nonzero so this bucket and its graph cannot
            // be retired. See the owned-token Drop path for the safety rationale.
            std::mem::forget(lane);
            return consume_result;
        }
        let return_result = self.return_detached_lane(bucket_id, lane);
        match (consume_result, return_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }

    fn return_detached_lane(
        &mut self, bucket_id: PreparedBucketId, lane: ServingLane<I, O, INPUTS, OUTPUTS>,
    ) -> Result<()> {
        let Some(bucket_index) = self.prepared_bucket_index(bucket_id) else {
            return Err(Error::new(
                -1,
                "zrt: shape bucket retired while an owned run was in flight",
            ));
        };
        let Some(bucket) = self.buckets.get_mut(bucket_index) else {
            return Err(Error::new(
                -1,
                "zrt: shape bucket retired while an owned run was in flight",
            ));
        };
        if bucket.detached_lanes == 0 || bucket.lanes.len() >= self.lane_count {
            eprintln!(
                "st-zrt: detached-lane accounting is inconsistent while returning a lane; leaking lane"
            );
            std::mem::forget(lane);
            return Err(Error::new(
                -1,
                "zrt: detached-lane accounting is inconsistent",
            ));
        }
        bucket.detached_lanes -= 1;
        bucket.lanes.push(lane);
        Ok(())
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
        self.options.clone()
    }

    /// Borrow cached shape buckets.
    #[inline]
    pub fn buckets(&self) -> &[ShapeBucket<I, O, INPUTS, OUTPUTS>] {
        &self.buckets
    }

    /// Borrow cached shape buckets mutably after reclaiming any dropped owned runs.
    #[inline]
    pub fn buckets_mut(&mut self) -> &mut [ShapeBucket<I, O, INPUTS, OUTPUTS>] {
        self.reclaim_dropped_runs_inner();
        &mut self.buckets
    }

    /// Drop all cached shape buckets.
    ///
    /// Clearing is rejected while an owned run is still alive. Dropped tokens are reclaimed first,
    /// so callers only need to explicitly complete tokens that remain in use.
    pub fn clear_buckets(&mut self) -> Result<()> {
        self.reclaim_dropped_runs_inner();
        if self.buckets.iter().any(|bucket| bucket.detached_lanes != 0) {
            return Err(Error::new(
                -1,
                "zrt: cannot clear shape buckets while owned runs are in flight",
            ));
        }
        self.hot_bucket = None;
        self.wait_for_retired_buckets();
        if !self.drain_active_buckets_and_release_graphs("clear_buckets") {
            return Err(Error::new(
                -1,
                "zrt: could not fence every in-flight lane while clearing buckets; affected resources were leaked for safety",
            ));
        }
        Ok(())
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
        self.reclaim_dropped_runs_inner();
        let i = self.find_bucket_index(input_shapes, output_shapes)?;
        let tick = self.next_tick();
        self.buckets[i].last_used = tick;
        self.hot_bucket = Some(i);
        Some(&mut self.buckets[i])
    }

    /// Remove one cached shape bucket if present and no owned runs are detached from it.
    pub fn remove_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS],
    ) -> bool {
        self.reclaim_dropped_runs_inner();
        let Some(i) = self.find_bucket_index(input_shapes, output_shapes) else {
            return false;
        };
        if self.buckets[i].detached_lanes != 0 {
            return false;
        }
        self.hot_bucket = None;
        self.retire_bucket_at(i, "remove_bucket")
    }

    /// Create or find a set of concrete shape buckets.
    ///
    /// Returns the number of specs processed. If more unique shapes are passed than
    /// [`Self::max_buckets`], the runtime's normal bounded-cache eviction policy applies.
    pub fn prebuild_buckets<'a>(
        &mut self, specs: impl IntoIterator<Item = ShapeSpec<'a, INPUTS, OUTPUTS>>,
    ) -> Result<usize> {
        let mut count = 0usize;
        for spec in specs {
            self.get_or_create_bucket_inner(spec.input_shapes, spec.output_shapes, true)?;
            count += 1;
        }
        Ok(count)
    }

    /// Create or find a bucket and run every lane `runs` times.
    pub fn prime_bucket(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], runs: usize,
    ) -> Result<()> {
        self.get_or_create_bucket_inner(input_shapes, output_shapes, true)?
            .prime(runs)
    }

    /// Create or find a bucket and run every lane through the enqueued path `runs` times.
    pub fn prime_bucket_enqueued(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], runs: usize,
    ) -> Result<()> {
        self.get_or_create_bucket_inner(input_shapes, output_shapes, true)?
            .prime_enqueued(runs)
    }

    /// Run every currently cached bucket's every lane `runs` times.
    ///
    /// This does not create new buckets. Pair it with [`Self::prebuild_buckets`] to warm a fixed
    /// shape plan before serving.
    pub fn prime_cached_buckets(&mut self, runs: usize) -> Result<()> {
        for bucket in &mut self.buckets {
            bucket.prime(runs)?;
        }
        Ok(())
    }

    /// Create/find each bucket in `specs`, then warm every lane in that bucket.
    ///
    /// Returns the number of specs processed.
    pub fn warm_buckets<'a>(
        &mut self, specs: impl IntoIterator<Item = ShapeSpec<'a, INPUTS, OUTPUTS>>, runs: usize,
    ) -> Result<usize> {
        let mut count = 0usize;
        for spec in specs {
            self.get_or_create_bucket_inner(spec.input_shapes, spec.output_shapes, true)?
                .prime(runs)?;
            count += 1;
        }
        Ok(count)
    }

    /// Create/find each bucket in `specs`, then warm every lane through the enqueued path.
    pub fn warm_buckets_enqueued<'a>(
        &mut self, specs: impl IntoIterator<Item = ShapeSpec<'a, INPUTS, OUTPUTS>>, runs: usize,
    ) -> Result<usize> {
        let mut count = 0usize;
        for spec in specs {
            self.get_or_create_bucket_inner(spec.input_shapes, spec.output_shapes, true)?
                .prime_enqueued(runs)?;
            count += 1;
        }
        Ok(count)
    }

    /// Run a closure against one lane in the matching shape bucket, creating the bucket on first
    /// use.
    #[inline]
    pub fn run_on<R>(
        &mut self, input_shapes: [&[i64]; INPUTS], output_shapes: [&[i64]; OUTPUTS], lane: usize,
        f: impl FnOnce(&mut ServingLane<I, O, INPUTS, OUTPUTS>) -> Result<R>,
    ) -> Result<R> {
        self.get_or_create_bucket_inner(
            input_shapes,
            output_shapes,
            !self.options.strict_shape_cache,
        )?
        .run_on(lane, f)
    }

    /// Snapshot every cached bucket's hot-path pointer and placement plan.
    ///
    /// This only audits buckets that already exist. It does not create new buckets.
    /// Diagnostic/setup API; may allocate.
    pub fn audit_cached_hot_paths(&self) -> Result<Vec<Vec<LaneHotPathAudit>>> {
        self.buckets
            .iter()
            .map(ShapeBucket::audit_hot_path)
            .collect()
    }

    /// Fail if any cached bucket is not pointer-identity zero-copy.
    pub fn assert_cached_zero_copy_plan(&self) -> Result<()> {
        for bucket in &self.buckets {
            bucket.assert_zero_copy_plan()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape_plan::OutputPolicy;

    fn mnist_path() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("bench")
            .join("models")
            .join("mnist.onnx");
        path.exists().then_some(path)
    }

    /// Fail-closed topology matrix for the shared-session CUDA-graph hazard: one bucket mints one
    /// `gpu_graph_id`, and ORT captures whichever lane ran first into that graph, so a shared
    /// session with more than one lane per bucket would silently cross lanes. Pure configuration
    /// validation — no session, no provider, no hardware.
    #[test]
    fn cuda_graph_shared_session_multi_lane_topology_is_rejected() {
        let graph = DynamicIoOptions::new(4).with_cuda_graph(true);
        // The sound topologies pass runtime-mode validation.
        assert!(
            graph
                .clone()
                .validate_runtime_mode(RuntimeMode::SharedSession, 1)
                .is_ok()
        );
        assert!(
            graph
                .clone()
                .validate_runtime_mode(RuntimeMode::ReplicatedSessions, 4)
                .is_ok()
        );
        // The unsound topology fails with the remedy in the message.
        let error = graph
            .validate_runtime_mode(RuntimeMode::SharedSession, 2)
            .expect_err("shared session + cuda_graph + 2 lanes must fail");
        assert!(
            error.message.contains("replicated sessions"),
            "guard should direct callers to replicated sessions or one lane, got: {error}"
        );
        // Non-graph runtimes keep the historical shared-session multi-lane topology.
        assert!(
            DynamicIoOptions::new(4)
                .validate_runtime_mode(RuntimeMode::SharedSession, 8)
                .is_ok()
        );
    }

    /// Host-input `cuda_graph` fails closed in `DynamicIoOptions::validate` before any session or
    /// provider work, on every feature combination.
    #[test]
    fn cuda_graph_host_input_options_are_rejected() {
        let error = DynamicIoOptions::new(4)
            .with_cuda_graph(true)
            .validate()
            .expect_err("host-input cuda_graph options must fail validation");
        assert!(
            error.message.contains("device-resident inputs"),
            "guard should name the supported input path, got: {error}"
        );
    }

    /// Lazy CUDA-graph bucket creation is refused while any lane of the runtime may still have
    /// provider work pending — detached into an owned token, or left in flight by the legacy
    /// `run_enqueued` — because the new bucket's first run captures, and capture must not overlap a
    /// live replay. CPU-runnable by flipping the internal `cuda_graph` flag the way a device-input
    /// runtime would set it; the check reads only lane/bucket accounting.
    #[test]
    fn graph_bucket_creation_refuses_inflight_lanes() {
        let _envs = crate::lock_default_env_creation();
        let Ok(env) = Environment::new() else {
            return;
        };
        let Some(path) = mnist_path() else {
            eprintln!("skip — mnist.onnx absent");
            return;
        };
        let Ok(session) = Session::new(&env, path.to_str().expect("path"), SessionOptions::new())
        else {
            return;
        };
        let mut runtime = DynamicIoRuntime::<f32, f32, 1, 1>::shared_session(
            session,
            MemoryInfo::cpu().expect("cpu mem"),
            2,
        )
        .expect("runtime");
        let mut builder = crate::shape_plan::ServingShapePlan::builder();
        builder.add_shape(
            [vec![1, 1, 28, 28]],
            [vec![1, 10]],
            OutputPolicy::HostBuffer,
        );
        builder.add_shape(
            [vec![2, 1, 28, 28]],
            [vec![2, 10]],
            OutputPolicy::HostBuffer,
        );
        runtime
            .install_shape_plan(Arc::new(builder.build().expect("shape plan")))
            .expect("install plan");
        // Build the first bucket as a plain (non-graph) bucket, then flip the internal
        // `cuda_graph` flag the way a device-input runtime would carry it. The refusal guard reads
        // only lane/bucket accounting, so it is fully observable on CPU; the flip also proves the
        // guard fires before any graph-id assignment could touch a host-input lane.
        runtime
            .get_or_create_bucket([&[1, 1, 28, 28]], [&[1, 10]])
            .expect("first bucket while idle");
        runtime.options.cuda_graph = true;

        // Detached lane (owned token in flight): lazy creation of the second bucket is refused.
        let run = runtime
            .enqueue_owned([&[1, 1, 28, 28]], [&[1, 10]], |lane| {
                lane.input_mut_at::<0>()?.fill(0.0);
                Ok(())
            })
            .expect("owned enqueue");
        let error = runtime
            .get_or_create_bucket([&[2, 1, 28, 28]], [&[2, 10]])
            .expect_err("creation under a detached lane must be refused");
        assert!(
            error.message.contains("prebuild or warm"),
            "guard should direct callers to prebuild/warm before traffic, got: {error}"
        );
        assert_eq!(runtime.bucket_count(), 1, "no bucket was created");

        runtime
            .complete_owned(run, |_| Ok(()))
            .expect("complete owned run");

        // Legacy in-flight lane (`run_enqueued` without a token): still refused.
        runtime
            .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| lane.run_enqueued())
            .expect("legacy enqueue");
        let error = runtime
            .get_or_create_bucket([&[2, 1, 28, 28]], [&[2, 10]])
            .expect_err("creation under a legacy in-flight lane must be refused");
        assert!(
            error.message.contains("1 in-flight lane"),
            "guard should report the legacy in-flight lane, got: {error}"
        );

        // Once every lane is fenced the guard no longer blocks: creation proceeds past it and then
        // fails closed for a different reason — CPU host-input lanes cannot take a graph id.
        runtime
            .run_on([&[1, 1, 28, 28]], [&[1, 10]], 0, |lane| {
                lane.synchronize_outputs()
            })
            .expect("fence legacy lane");
        let error = runtime
            .get_or_create_bucket([&[2, 1, 28, 28]], [&[2, 10]])
            .expect_err("creation must proceed past the in-flight guard once fenced");
        assert!(
            error.message.contains("device-input lane"),
            "the post-guard failure should be the host-input graph-id rejection, got: {error}"
        );
    }

    /// A failed synchronized graph run must not release its captured-graph lease unfenced: the
    /// lease stays held while the run guard is alive, and a proven fence releases it. Uses a real
    /// CPU lane with an installed lease (the public `set_gpu_graph_id` path now requires
    /// device-input lanes, so the lease is installed directly, as `DynamicIoRuntime` does
    /// internally after its own validation).
    #[test]
    fn synchronized_graph_run_error_fences_before_releasing_lease() {
        let _envs = crate::lock_default_env_creation();
        let Ok(env) = Environment::new() else {
            return;
        };
        let Some(path) = mnist_path() else {
            eprintln!("skip — mnist.onnx absent");
            return;
        };
        let Ok(session) = Session::new(&env, path.to_str().expect("path"), SessionOptions::new())
        else {
            return;
        };
        let mem = MemoryInfo::cpu().expect("cpu mem");
        let mut lanes = StaticIoRuntime::<f32, f32, 1, 1>::shared_session_with_buffer_policy(
            session.clone(),
            &mem,
            &mem,
            [&[1, 1, 28, 28]],
            [&[1, 10]],
            1,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
        .expect("static lane set");
        let lane = lanes.lane_mut(0).expect("lane");
        lane.graph_lease = Some(session.captured_graph_lease(911));

        // Hold an active run guard, then prove a concurrent `release_captured_graph` waits for it:
        // the lease is genuinely retained while the lane believes a run is executing.
        let guard = session.captured_graph_lease(911).begin_run();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let releaser = {
            let session = session.clone();
            std::thread::spawn(move || {
                let released = session.release_captured_graph(911);
                let _ = done_tx.send(());
                released
            })
        };
        assert!(
            rx_timeout(&done_rx, std::time::Duration::from_millis(200)).is_none(),
            "release must wait while the run guard is held"
        );

        // Settle a failed run: on CPU the bound-output fence succeeds, so the lane is not poisoned
        // and the dropped guard lets the waiting release complete.
        let error = lane
            .settle_graph_run(
                Some(guard),
                Err(Error::local("injected run failure")),
                false,
            )
            .expect_err("settling must return the failure it was handed");
        assert_eq!(error.message, "injected run failure");
        assert!(!lane.in_flight, "a fenced failure must not poison the lane");
        assert!(
            rx_timeout(&done_rx, std::time::Duration::from_secs(5)).is_some(),
            "release must complete after the fenced failure settles"
        );
        releaser
            .join()
            .expect("releaser thread")
            .expect("release result");
    }

    /// `ServingLane::drop` fences best-effort and then drops its state normally once a fence is
    /// proven, releasing the lane's `Session` handle. (The unfenced branch deliberately leaks the
    /// whole inner state box; forcing it needs a failing `SynchronizeBoundOutputs` and a failing
    /// device fence, which cannot be injected without fault injection — the shared policy is also
    /// exercised by the owned-token drop paths.)
    #[test]
    fn serving_lane_drop_fences_and_releases_state_when_fence_succeeds() {
        let _envs = crate::lock_default_env_creation();
        let Ok(env) = Environment::new() else {
            return;
        };
        let Some(path) = mnist_path() else {
            eprintln!("skip — mnist.onnx absent");
            return;
        };
        let Ok(session) = Session::new(&env, path.to_str().expect("path"), SessionOptions::new())
        else {
            return;
        };
        fn strong_count(session: &Session) -> usize {
            // `Session`'s own count is the LAST `strong_count:` field in its debug output — the
            // materialized run options embedded in it print their own guard count first.
            let debug = format!("{session:?}");
            let marker = "strong_count: ";
            let start = debug
                .rfind(marker)
                .expect("session debug output carries the Arc strong count");
            debug[start + marker.len()..]
                .split(|c: char| !c.is_ascii_digit())
                .next()
                .and_then(|digits| digits.parse::<usize>().ok())
                .expect("strong count digits")
        }
        let before = strong_count(&session);
        let mem = MemoryInfo::cpu().expect("cpu mem");
        let lanes = StaticIoRuntime::<f32, f32, 1, 1>::shared_session_with_buffer_policy(
            session.clone(),
            &mem,
            &mem,
            [&[1, 1, 28, 28]],
            [&[1, 10]],
            1,
            BufferSpec::AUTO,
            BufferSpec::AUTO,
        )
        .expect("static lane set");
        let mut lane = lanes.into_lanes().pop().expect("lane");
        // The lane state plus its `IoBinding` guard each retain one session handle.
        assert_eq!(strong_count(&session), before + 2);
        // Simulate a pending run; drop must fence it (CPU bound-output sync succeeds) and then
        // release both retained session handles rather than leak the inner state.
        lane.in_flight = true;
        drop(lane);
        assert_eq!(
            strong_count(&session),
            before,
            "a fenced lane drop must release its session handles, not leak the inner state"
        );
    }

    /// `recv_timeout` that treats a disconnect as completion (the releaser thread may finish and
    /// drop its sender while this thread is still waiting).
    fn rx_timeout(rx: &std::sync::mpsc::Receiver<()>, timeout: std::time::Duration) -> Option<()> {
        match rx.recv_timeout(timeout) {
            Ok(value) => Some(value),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Some(()),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => None,
        }
    }
}
