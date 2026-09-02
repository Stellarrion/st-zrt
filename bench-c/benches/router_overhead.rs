//! Request distribution overhead microbenchmarks.
//!
//! This isolates the host-side cost behind the observed tiny-model behavior where N=4 lockless
//! serving can underperform N=1. For sub-microsecond GPU work, channel hops and acknowledgements can
//! dominate throughput.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use st_zrt::{SpscReceiver, SpscSender, bounded_spsc};
use std::sync::Barrier;
use std::sync::mpsc;

const SPSC_CAPACITY: usize = 1024;
const STOP: usize = usize::MAX;

enum Msg {
    Run(usize),
    Stop,
}

fn worker(rx: mpsc::Receiver<Msg>, ack: mpsc::Sender<usize>) {
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Run(value) => {
                black_box(value.wrapping_mul(31).wrapping_add(7));
                ack.send(value).expect("ack");
            },
            Msg::Stop => break,
        }
    }
}

fn spsc_worker(req: SpscReceiver<usize>, ack: SpscSender<usize>, ready: std::sync::Arc<Barrier>) {
    ready.wait();
    while let Some(value) = req.recv() {
        if value == STOP {
            break;
        }
        black_box(value.wrapping_mul(31).wrapping_add(7));
        ack.send(value).expect("ack");
    }
}

fn bench_direct_dispatch(c: &mut Criterion) {
    let mut value = 0usize;
    c.bench_function("router/direct_function_call", |b| {
        b.iter(|| {
            value = value.wrapping_add(1);
            black_box(value.wrapping_mul(31).wrapping_add(7));
        });
    });
}

fn bench_single_lane_mpsc_roundtrip(c: &mut Criterion) {
    let (tx, rx) = mpsc::channel();
    let (ack_tx, ack_rx) = mpsc::channel();
    let handle = std::thread::spawn(move || worker(rx, ack_tx));
    let mut value = 0usize;

    c.bench_function("router/mpsc_roundtrip_1_lane", |b| {
        b.iter(|| {
            value = value.wrapping_add(1);
            tx.send(Msg::Run(value)).expect("send");
            black_box(ack_rx.recv().expect("ack"));
        });
    });

    tx.send(Msg::Stop).expect("stop");
    handle.join().expect("worker");
}

fn bench_single_lane_spsc_roundtrip(c: &mut Criterion) {
    let (req_tx, req_rx) = bounded_spsc(SPSC_CAPACITY);
    let (ack_tx, ack_rx) = bounded_spsc(SPSC_CAPACITY);
    let ready = std::sync::Arc::new(Barrier::new(2));
    let handle = {
        let ready = std::sync::Arc::clone(&ready);
        std::thread::spawn(move || spsc_worker(req_rx, ack_tx, ready))
    };
    ready.wait();
    let mut value = 0usize;

    c.bench_function("router/spsc_spin_park_roundtrip_1_lane", |b| {
        b.iter(|| {
            value = value.wrapping_add(1);
            req_tx.send(value).expect("send");
            black_box(ack_rx.recv().expect("ack"));
        });
    });

    req_tx.send(STOP).expect("stop");
    handle.join().expect("worker");
}

fn bench_round_robin_mpsc_4_lanes(c: &mut Criterion) {
    let (ack_tx, ack_rx) = mpsc::channel();
    let mut senders = Vec::new();
    let mut handles = Vec::new();
    for _ in 0..4 {
        let (tx, rx) = mpsc::channel();
        senders.push(tx);
        let ack = ack_tx.clone();
        handles.push(std::thread::spawn(move || worker(rx, ack)));
    }
    drop(ack_tx);
    let mut value = 0usize;
    let mut lane = 0usize;

    c.bench_function("router/mpsc_roundtrip_4_lane_round_robin", |b| {
        b.iter(|| {
            value = value.wrapping_add(1);
            senders[lane].send(Msg::Run(value)).expect("send");
            lane = (lane + 1) % senders.len();
            black_box(ack_rx.recv().expect("ack"));
        });
    });

    for tx in senders {
        tx.send(Msg::Stop).expect("stop");
    }
    for handle in handles {
        handle.join().expect("worker");
    }
}

fn bench_round_robin_spsc_4_lanes(c: &mut Criterion) {
    let mut reqs = Vec::new();
    let mut acks = Vec::new();
    let mut handles = Vec::new();
    let ready = std::sync::Arc::new(Barrier::new(5));
    for _ in 0..4 {
        let (req_tx, req_rx) = bounded_spsc(SPSC_CAPACITY);
        let (ack_tx, ack_rx) = bounded_spsc(SPSC_CAPACITY);
        let worker_ready = std::sync::Arc::clone(&ready);
        handles.push(std::thread::spawn(move || {
            spsc_worker(req_rx, ack_tx, worker_ready)
        }));
        reqs.push(req_tx);
        acks.push(ack_rx);
    }
    ready.wait();
    let mut value = 0usize;
    let mut lane = 0usize;

    c.bench_function("router/spsc_spin_park_roundtrip_4_lane_round_robin", |b| {
        b.iter(|| {
            value = value.wrapping_add(1);
            reqs[lane].send(value).expect("send");
            black_box(acks[lane].recv().expect("ack"));
            lane = (lane + 1) % reqs.len();
        });
    });

    for req in &reqs {
        req.send(STOP).expect("stop");
    }
    for handle in handles {
        handle.join().expect("worker");
    }
}

criterion_group!(
    benches,
    bench_direct_dispatch,
    bench_single_lane_mpsc_roundtrip,
    bench_single_lane_spsc_roundtrip,
    bench_round_robin_mpsc_4_lanes,
    bench_round_robin_spsc_4_lanes
);
criterion_main!(benches);
