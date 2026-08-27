//! Compare per-event CUDA completion queries with one-device-validation batch polling.
//! Run: cargo run --release --features cuda --example completion_poller

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("completion_poller requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;
    use st_zrt::{CompletionEventRef, CudaCompletionPoller, CudaEvent, CudaStream};

    const ITERATIONS: usize = 100_000;
    for count in [1usize, 2, 4, 8, 16] {
        let streams = (0..count)
            .map(|_| CudaStream::new(0).expect("stream"))
            .collect::<Vec<_>>();
        let events = streams
            .iter()
            .map(|stream| {
                let event = CudaEvent::new(0).expect("event");
                event.record(stream).expect("record");
                event.synchronize().expect("warm event");
                event
            })
            .collect::<Vec<_>>();
        let refs = events
            .iter()
            .map(CompletionEventRef::from)
            .collect::<Vec<_>>();
        let poller = CudaCompletionPoller::new(0).expect("poller");
        let mut ready = vec![false; count];

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            for event in &events {
                std::hint::black_box(event.is_complete().expect("single query"));
            }
        }
        let individual_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            poller.query(&refs, &mut ready).expect("batch query");
            std::hint::black_box(&ready);
        }
        let batch_ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        println!(
            "events={count:2}: individual={individual_ns:.1}ns batch={batch_ns:.1}ns speedup={:.2}x",
            individual_ns / batch_ns,
        );
    }
}
