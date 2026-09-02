//! Rust heap allocation probe for the `st-zrt` lane hot path.
//!
//! This counts Rust global allocator calls after warmup. Native libonnxruntime allocations are not
//! counted here; use `native_alloc_lane` with LD_PRELOAD for that lower-level view.
use st_zrt::{Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions};
use st_zrt_bench_c::models;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, old_layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn n_for(label: &str) -> usize {
    match label {
        "mnist" => 784,
        "hf_resnet50" => 3 * 224 * 224,
        "4m" => 1 << 20,
        "16m" => 1 << 22,
        _ => panic!("unknown ZRT_LABEL={label}; use mnist|hf_resnet50|4m|16m"),
    }
}

fn main() {
    let label = std::env::var("ZRT_LABEL").unwrap_or_else(|_| "mnist".into());
    let iters: usize = std::env::var("ZRT_ITERS")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .unwrap();
    let n = n_for(&label);
    let path = match label.as_str() {
        "mnist" => models::ensure_mnist().expect("mnist"),
        "hf_resnet50" => models::ensure_hf_resnet50().expect("hf resnet50"),
        _ => models::ensure_relay(&label).expect("relay"),
    };
    let input_shape: &[i64] = match label.as_str() {
        "mnist" => &[1, 1, 28, 28],
        "hf_resnet50" => &[1, 3, 224, 224],
        _ => &[1, n as i64],
    };
    let output_shape: &[i64] = match label.as_str() {
        "mnist" => &[1, 10],
        "hf_resnet50" => &[1, 1000],
        _ => &[1, n as i64],
    };

    let env = Environment::new().unwrap();
    let mem = MemoryInfo::cpu().unwrap();
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = st_zrt::Session::new(&env, path.to_str().unwrap(), opts).unwrap();
    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[input_shape], &[output_shape])
        .unwrap();
    lane.input_mut(0).expect("lane input").fill(3.0);

    for _ in 0..64 {
        lane.run().unwrap();
    }

    ALLOCS.store(0, Ordering::Relaxed);
    let mut checksum = 0u64;
    let start = Instant::now();
    for i in 0..iters {
        lane.run().unwrap();
        let out = lane.output(0).expect("lane output");
        checksum = checksum.wrapping_add(out[i % out.len()].to_bits() as u64);
    }
    let elapsed = start.elapsed();
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    println!(
        "rust_alloc_zrt label={label} mode=lane iters={iters} avg_us={avg_us:.3} rust_allocs={allocs} rust_allocs_per_run={:.3} checksum={checksum:#x}",
        allocs as f64 / iters as f64
    );
}
