//! Rust heap allocation probe for Rust `ort` hot paths.
//!
//! This counts Rust global allocator calls after warmup. It intentionally does not count
//! allocations made inside libonnxruntime's native allocator.
use ort::memory::MemoryInfo;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{IoBinding, Session};
use ort::value::Tensor;
use st_zrt_bench::models;
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
        _ => panic!("unknown ORT_LABEL={label}; use mnist|hf_resnet50|4m|16m"),
    }
}

fn shape_for(label: &str, n: usize) -> Vec<i64> {
    match label {
        "mnist" => vec![1, 1, 28, 28],
        "hf_resnet50" => vec![1, 3, 224, 224],
        _ => vec![1, n as i64],
    }
}

fn session(path: &str) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(1)?
        .commit_from_file(path)
}

fn bind(session: &Session, input: &Tensor<f32>) -> ort::Result<IoBinding> {
    let input_name = session.inputs()[0].name().to_string();
    let output_name = session.outputs()[0].name().to_string();
    let mut binding = session.create_binding()?;
    binding.bind_input(input_name, input)?;
    binding.bind_output_to_device(output_name, &MemoryInfo::default())?;
    Ok(binding)
}

fn checksum(values: &[f32], i: usize) -> u64 {
    if values.is_empty() {
        0
    } else {
        values[i % values.len()].to_bits() as u64
    }
}

fn main() {
    let label = std::env::var("ORT_LABEL").unwrap_or_else(|_| "mnist".into());
    let mode = std::env::var("ORT_MODE").unwrap_or_else(|_| "iobinding".into());
    let iters: usize = std::env::var("ORT_ITERS")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .unwrap();
    let n = n_for(&label);
    let shape = shape_for(&label, n);
    let path = match label.as_str() {
        "mnist" => models::ensure_mnist().expect("mnist"),
        "hf_resnet50" => models::ensure_hf_resnet50().expect("hf resnet50"),
        _ => models::ensure_relay(&label).expect("relay"),
    };
    let mut session = session(path.to_str().unwrap()).unwrap();
    let input_data = vec![3.0_f32; n];

    let mut checksum_acc = 0u64;
    let elapsed = match mode.as_str() {
        "default" => {
            for _ in 0..64 {
                let tensor = Tensor::<f32>::from_array((shape.clone(), input_data.clone())).unwrap();
                let outputs = session.run(ort::inputs![tensor]).unwrap();
                let view = outputs[0].try_extract_array::<f32>().unwrap();
                std::hint::black_box(&view);
            }
            ALLOCS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            for i in 0..iters {
                let tensor = Tensor::<f32>::from_array((shape.clone(), input_data.clone())).unwrap();
                let outputs = session.run(ort::inputs![tensor]).unwrap();
                let view = outputs[0].try_extract_array::<f32>().unwrap();
                checksum_acc = checksum_acc.wrapping_add(checksum(view.as_slice().unwrap_or(&[]), i));
            }
            start.elapsed()
        }
        "iobinding" => {
            let input = Tensor::<f32>::from_array((shape.clone(), input_data)).unwrap();
            let binding = bind(&session, &input).unwrap();
            for _ in 0..64 {
                let outputs = session.run_binding(&binding).unwrap();
                let view = outputs[0].try_extract_array::<f32>().unwrap();
                std::hint::black_box(&view);
            }
            ALLOCS.store(0, Ordering::Relaxed);
            let start = Instant::now();
            for i in 0..iters {
                let outputs = session.run_binding(&binding).unwrap();
                let view = outputs[0].try_extract_array::<f32>().unwrap();
                checksum_acc = checksum_acc.wrapping_add(checksum(view.as_slice().unwrap_or(&[]), i));
            }
            start.elapsed()
        }
        other => panic!("unknown ORT_MODE={other}; use default|iobinding"),
    };

    let allocs = ALLOCS.load(Ordering::Relaxed);
    let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    println!(
        "rust_alloc_ort label={label} mode={mode} iters={iters} avg_us={avg_us:.3} rust_allocs={allocs} rust_allocs_per_run={:.3} checksum={checksum_acc:#x}",
        allocs as f64 / iters as f64
    );
}
