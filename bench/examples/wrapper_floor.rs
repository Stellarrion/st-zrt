//! Wrapper-overhead floor: per-run time and allocation counts on an Identity model
//! (kernel ~= no-op) plus the standard fixtures, for the ort naive and expert paths.
//! Intra-op threads pinned to 1 so scheduling noise stays out of the measurement.
use std::alloc::{GlobalAlloc, Layout, System};
use std::time::Instant;

use ort::memory::MemoryInfo;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use st_zrt_bench::models;

static ALLOCS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

const WARM: usize = 100;
const ITERS: usize = 2000;

fn session(path: &str) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(1)?
        .commit_from_file(path)
}

fn measure<F: FnMut() -> ort::Result<()>>(label: &str, mut f: F) {
    for _ in 0..WARM {
        f().expect("warmup run");
    }
    let before = ALLOCS.load(std::sync::atomic::Ordering::Relaxed);
    let t = Instant::now();
    for _ in 0..ITERS {
        f().expect("measured run");
    }
    let dt = t.elapsed();
    let allocs = ALLOCS.load(std::sync::atomic::Ordering::Relaxed) - before;
    println!(
        "{label},{:.1},{:.2}",
        dt.as_nanos() as f64 / ITERS as f64 / 1000.0,
        allocs as f64 / ITERS as f64
    );
}

fn naive(session: &mut Session, shape: &[i64], n: usize, label: &str) -> ort::Result<()> {
    let data = vec![0.0_f32; n];
    measure(label, || {
        let input = Tensor::<f32>::from_array((shape.to_vec(), data.clone()))?;
        let outputs = session.run(ort::inputs![input])?;
        let _ = outputs[0].try_extract_array::<f32>()?;
        Ok(())
    });
    Ok(())
}

fn expert(session: &mut Session, shape: &[i64], n: usize, label: &str) -> ort::Result<()> {
    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();
    let input = Tensor::<f32>::from_array((shape.to_vec(), vec![0.0_f32; n]))?;
    let mut binding = session.create_binding()?;
    binding.bind_input(in_name, &input)?;
    binding.bind_output_to_device(out_name, &MemoryInfo::default())?;
    measure(label, || {
        let outputs = session.run_binding(&binding)?;
        let _ = outputs[0].try_extract_array::<f32>()?;
        Ok(())
    });
    Ok(())
}

fn main() -> ort::Result<()> {
    println!("variant,us_per_run,allocs_per_run");
    let id = models::identity_path().unwrap();
    let mut s = session(id.to_str().unwrap())?;
    naive(&mut s, &[1, 65536], 65536, "ort_naive_identity")?;
    expert(&mut s, &[1, 65536], 65536, "ort_expert_identity")?;

    let m = models::ensure_mnist().unwrap();
    let mut s = session(m.to_str().unwrap())?;
    naive(&mut s, &[1, 1, 28, 28], 784, "ort_naive_mnist")?;
    expert(&mut s, &[1, 1, 28, 28], 784, "ort_expert_mnist")?;

    let r = models::ensure_relay("4m").unwrap();
    let mut s = session(r.to_str().unwrap())?;
    naive(&mut s, &[1, 1 << 20], 1 << 20, "ort_naive_relay4m")?;
    expert(&mut s, &[1, 1 << 20], 1 << 20, "ort_expert_relay4m")?;
    Ok(())
}
