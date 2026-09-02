//! Bounded single-producer/single-consumer queue for worker-owned serving lanes.
//!
//! This is intentionally small and std-only. It is not a general MPMC channel replacement: each
//! queue has exactly one sender and one receiver. The blocking operations spin briefly, then park
//! with a short timeout; successful sends/receives unpark the peer if it has registered by blocking.

use std::cell::UnsafeCell;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, Thread};
use std::time::Duration;

const DEFAULT_SPINS: usize = 256;
const PARK_TIMEOUT: Duration = Duration::from_nanos(50);

#[repr(align(64))]
struct PaddedAtomicUsize(AtomicUsize);

impl PaddedAtomicUsize {
    fn new(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }

    fn load(&self, order: Ordering) -> usize {
        self.0.load(order)
    }

    fn store(&self, value: usize, order: Ordering) {
        self.0.store(value, order);
    }
}

struct Slot<T>(UnsafeCell<MaybeUninit<T>>);

// SAFETY: the SPSC protocol guarantees only the producer writes a slot before publishing `tail`,
// and only the consumer reads/drops that slot after observing `tail`.
unsafe impl<T: Send> Sync for Slot<T> {}

struct Inner<T> {
    slots: Vec<Slot<T>>,
    mask: usize,
    head: PaddedAtomicUsize,
    tail: PaddedAtomicUsize,
    sender_closed: AtomicBool,
    receiver_closed: AtomicBool,
    sender_thread: OnceLock<Thread>,
    receiver_thread: OnceLock<Thread>,
    spins: usize,
}

// SAFETY: all cross-thread access is mediated by atomics and SPSC ownership of slots.
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    fn new(capacity: usize, spins: usize) -> Self {
        assert!(capacity > 0, "SPSC capacity must be non-zero");
        let capacity = capacity
            .checked_next_power_of_two()
            .expect("SPSC capacity is too large");
        Self {
            slots: (0..capacity)
                .map(|_| Slot(UnsafeCell::new(MaybeUninit::uninit())))
                .collect(),
            mask: capacity - 1,
            head: PaddedAtomicUsize::new(0),
            tail: PaddedAtomicUsize::new(0),
            sender_closed: AtomicBool::new(false),
            receiver_closed: AtomicBool::new(false),
            sender_thread: OnceLock::new(),
            receiver_thread: OnceLock::new(),
            spins,
        }
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn len(&self) -> usize {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
    }

    fn unpark_sender(&self) {
        if let Some(thread) = self.sender_thread.get() {
            thread.unpark();
        }
    }

    fn unpark_receiver(&self) {
        if let Some(thread) = self.receiver_thread.get() {
            thread.unpark();
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        for idx in head..tail {
            let slot = &self.slots[idx & self.mask];
            // SAFETY: `head..tail` are the still-published, not-yet-consumed elements. `Inner` is
            // dropping after both halves are gone, so no producer/consumer can race this cleanup.
            unsafe { (*slot.0.get()).assume_init_drop() };
        }
    }
}

/// Sending half of a bounded single-producer/single-consumer queue.
pub struct SpscSender<T> {
    inner: Arc<Inner<T>>,
}

/// Receiving half of a bounded single-producer/single-consumer queue.
pub struct SpscReceiver<T> {
    inner: Arc<Inner<T>>,
}

/// Error returned by [`SpscSender::try_send`].
#[derive(Debug, PartialEq, Eq)]
pub enum TrySendError<T> {
    Full(T),
    Closed(T),
}

/// Error returned by [`SpscSender::send`].
#[derive(Debug, PartialEq, Eq)]
pub struct SendError<T>(pub T);

/// Error returned by [`SpscReceiver::try_recv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    Empty,
    Closed,
}

impl<T> fmt::Debug for SpscSender<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpscSender")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("closed", &self.is_closed())
            .finish()
    }
}

impl<T> fmt::Debug for SpscReceiver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpscReceiver")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("closed", &self.is_closed())
            .finish()
    }
}

/// Create a bounded SPSC queue.
///
/// `capacity` must be non-zero; internally it is rounded up to the next power of two.
pub fn bounded_spsc<T>(capacity: usize) -> (SpscSender<T>, SpscReceiver<T>) {
    bounded_spsc_with_spins(capacity, DEFAULT_SPINS)
}

/// Create a bounded SPSC queue with an explicit spin budget before each park/yield.
///
/// `capacity` must be non-zero; internally it is rounded up to the next power of two.
pub fn bounded_spsc_with_spins<T>(
    capacity: usize, spins: usize,
) -> (SpscSender<T>, SpscReceiver<T>) {
    let inner = Arc::new(Inner::new(capacity, spins));
    (
        SpscSender {
            inner: Arc::clone(&inner),
        },
        SpscReceiver { inner },
    )
}

impl<T> SpscSender<T> {
    /// Queue capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Approximate queued element count.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the queue is observed empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the receiving half has closed.
    pub fn is_closed(&self) -> bool {
        self.inner.sender_closed.load(Ordering::Acquire)
            || self.inner.receiver_closed.load(Ordering::Acquire)
    }

    /// Close this sender and wake the receiver.
    pub fn close(&self) {
        self.inner.sender_closed.store(true, Ordering::Release);
        self.inner.unpark_receiver();
    }

    /// Try to enqueue without blocking.
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        if self.inner.sender_closed.load(Ordering::Acquire)
            || self.inner.receiver_closed.load(Ordering::Acquire)
        {
            return Err(TrySendError::Closed(value));
        }
        let tail = self.inner.tail.load(Ordering::Relaxed);
        let head = self.inner.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) == self.inner.capacity() {
            return Err(TrySendError::Full(value));
        }
        let slot = &self.inner.slots[tail & self.inner.mask];
        // SAFETY: only this producer writes the next tail slot, and the full check above proves the
        // consumer has already released it if the indices wrapped.
        unsafe { (*slot.0.get()).write(value) };
        self.inner
            .tail
            .store(tail.wrapping_add(1), Ordering::Release);
        self.inner.unpark_receiver();
        Ok(())
    }

    /// Enqueue, spinning briefly before parking while the queue is full.
    pub fn send(&self, mut value: T) -> Result<(), SendError<T>> {
        let _ = self.inner.sender_thread.set(thread::current());
        loop {
            match self.try_send(value) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Closed(v)) => return Err(SendError(v)),
                Err(TrySendError::Full(v)) => value = v,
            }
            for _ in 0..self.inner.spins {
                match self.try_send(value) {
                    Ok(()) => return Ok(()),
                    Err(TrySendError::Closed(v)) => return Err(SendError(v)),
                    Err(TrySendError::Full(v)) => value = v,
                }
                std::hint::spin_loop();
            }
            self.inner.unpark_receiver();
            thread::park_timeout(PARK_TIMEOUT);
        }
    }
}

impl<T> Drop for SpscSender<T> {
    fn drop(&mut self) {
        self.close();
    }
}

impl<T> SpscReceiver<T> {
    /// Queue capacity.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Approximate queued element count.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the queue is observed empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the sending half is closed and no more values can arrive.
    pub fn is_closed(&self) -> bool {
        self.inner.receiver_closed.load(Ordering::Acquire)
            || (self.inner.sender_closed.load(Ordering::Acquire) && self.is_empty())
    }

    /// Close this receiver and wake the sender.
    pub fn close(&self) {
        self.inner.receiver_closed.store(true, Ordering::Release);
        self.inner.unpark_sender();
    }

    /// Try to dequeue without blocking.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let head = self.inner.head.load(Ordering::Relaxed);
        let tail = self.inner.tail.load(Ordering::Acquire);
        if head == tail {
            if self.inner.sender_closed.load(Ordering::Acquire) {
                return Err(TryRecvError::Closed);
            }
            return Err(TryRecvError::Empty);
        }
        let slot = &self.inner.slots[head & self.inner.mask];
        // SAFETY: only this consumer reads the next head slot, and tail acquire proves the producer
        // initialized it before publishing.
        let value = unsafe { (*slot.0.get()).assume_init_read() };
        self.inner
            .head
            .store(head.wrapping_add(1), Ordering::Release);
        self.inner.unpark_sender();
        Ok(value)
    }

    /// Dequeue, spinning briefly before parking while the queue is empty. Returns `None` after the
    /// sender closes and all queued elements have been drained.
    pub fn recv(&self) -> Option<T> {
        let _ = self.inner.receiver_thread.set(thread::current());
        loop {
            match self.try_recv() {
                Ok(value) => return Some(value),
                Err(TryRecvError::Closed) => return None,
                Err(TryRecvError::Empty) => {},
            }
            for _ in 0..self.inner.spins {
                match self.try_recv() {
                    Ok(value) => return Some(value),
                    Err(TryRecvError::Closed) => return None,
                    Err(TryRecvError::Empty) => {},
                }
                std::hint::spin_loop();
            }
            thread::park_timeout(PARK_TIMEOUT);
        }
    }
}

impl<T> Drop for SpscReceiver<T> {
    fn drop(&mut self) {
        self.close();
    }
}
