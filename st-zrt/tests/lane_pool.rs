//! Concurrency-safe multi-lane pool over cheap-cloned `Session`-owned `ServingLane`s.
//!
//! `ServingLanePool` hands out exclusive lanes via check-out / check-in (the guard returns its
//! lane on drop), so N prebuilt lanes can run in parallel from multiple threads — or be overlapped
//! from a single driver thread by calling `run_async` (see `tests/async_run.rs`) on several
//! checked-out lanes. This closes the gap that today every run is `&mut self` and rs-celer drives
//! one lane at a time, leaving the replicated set idle.

use std::sync::Arc;
use std::thread;

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, ServingLane, ServingLanePool, Session,
    SessionOptions,
};

fn mnist_session() -> Option<(MemoryInfo, Session)> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx");
    if !path.exists() {
        eprintln!("skip — mnist.onnx absent");
        return None;
    }
    let env = Environment::new().ok()?;
    let mem = MemoryInfo::cpu().ok()?;
    let opts = SessionOptions::new().with_opt_level(GraphOptimizationLevel::All);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).ok()?;
    Some((mem, sess))
}

fn build_pool<const N: usize>(
    mem: &MemoryInfo, session: Session,
) -> ServingLanePool<f32, f32, 1, 1> {
    let lanes: Vec<ServingLane<f32, f32, 1, 1>> = (0..N)
        .map(|_| ServingLane::new(session.clone(), mem, [&[1, 1, 28, 28]], [&[1, 10]]))
        .collect::<Result<_, _>>()
        .expect("lanes");
    ServingLanePool::from_lanes(lanes).expect("pool")
}

#[test]
fn pool_checkout_runs_and_auto_returns_lane() {
    let (mem, sess) = match mnist_session() {
        Some(v) => v,
        None => return,
    };
    let session = sess;
    let pool = build_pool::<2>(&mem, session);
    assert_eq!(pool.len(), 2);
    assert_eq!(pool.idle_count(), 2);

    {
        let mut guard = pool.try_checkout().expect("checkout");
        guard.input_mut_at::<0>().expect("input").fill(0.0);
        guard.run().expect("run");
        assert_eq!(guard.output_at::<0>().expect("output").len(), 10);
        assert_eq!(pool.idle_count(), 1, "one lane checked out, one still idle");
    } // guard drops → lane auto-returned
    assert_eq!(pool.idle_count(), 2, "lane returned on drop");

    // A second checkout observes the just-returned lane.
    let _again = pool.try_checkout().expect("checkout after return");
    assert_eq!(pool.idle_count(), 1);
}

#[test]
fn pool_try_checkout_exhausts_at_capacity() {
    let (mem, sess) = match mnist_session() {
        Some(v) => v,
        None => return,
    };
    let session = sess;
    let pool = build_pool::<2>(&mem, session);

    let g0 = pool.try_checkout().expect("lane 0");
    let g1 = pool.try_checkout().expect("lane 1");
    assert!(pool.try_checkout().is_none(), "pool must be exhausted");
    drop(g1);
    assert!(pool.try_checkout().is_some(), "freed slot available");
    drop(g0);
    assert_eq!(pool.idle_count(), 2);
}

#[test]
fn pool_concurrent_checkouts_run_in_parallel() {
    let (mem, sess) = match mnist_session() {
        Some(v) => v,
        None => return,
    };
    let session = sess;
    const N: usize = 4;
    let pool = Arc::new(build_pool::<N>(&mem, session));

    // N threads, N lanes — every checkout succeeds immediately; validates Send/Sync + no data
    // races on concurrently run exclusive lanes.
    let lens: Vec<usize> = thread::scope(|s| {
        (0..N)
            .map(|_| {
                let pool = pool.clone();
                s.spawn(move || {
                    let mut guard = pool.checkout();
                    guard.input_mut_at::<0>().expect("input").fill(0.0);
                    guard.run().expect("run");
                    guard.output_at::<0>().expect("output").len()
                })
            })
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    assert_eq!(lens.len(), N);
    assert!(
        lens.iter().all(|&l| l == 10),
        "every lane produced 10 logits"
    );
    assert_eq!(pool.idle_count(), N, "all lanes returned");
}

#[test]
fn pool_blocking_checkout_waits_for_a_returned_lane() {
    let (mem, sess) = match mnist_session() {
        Some(v) => v,
        None => return,
    };
    let session = sess;
    // 2 lanes, 6 workers — demand exceeds supply, so `checkout()` must block on the condvar and
    // be woken as guards drop. Every worker still completes exactly one run.
    let pool = Arc::new(build_pool::<2>(&mem, session));
    const WORKERS: usize = 6;

    let done: Vec<()> = thread::scope(|s| {
        (0..WORKERS)
            .map(|_| {
                let pool = pool.clone();
                s.spawn(move || {
                    let mut guard = pool.checkout();
                    guard.input_mut_at::<0>().expect("input").fill(1.0);
                    guard.run().expect("run");
                    assert_eq!(guard.output_at::<0>().expect("output").len(), 10);
                })
            })
            .map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    assert_eq!(done.len(), WORKERS);
    assert_eq!(pool.idle_count(), 2, "all lanes returned after the storm");
}

#[test]
fn pool_guard_fences_enqueued_lane_before_checkin() {
    let (mem, sess) = match mnist_session() {
        Some(v) => v,
        None => return,
    };
    let pool = build_pool::<1>(&mem, sess);
    {
        let mut guard = pool.checkout();
        guard.input_mut_at::<0>().expect("input").fill(0.0);
        guard.run_enqueued().expect("enqueue");
        // Deliberately omit synchronize_outputs: guard drop must fence before checkin.
    }
    let mut guard = pool.checkout();
    guard.run().expect("returned lane is reusable");
}
