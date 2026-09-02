//! Wrapper-overhead floor: per-run time and allocation counts on an Identity model
//! (kernel ~= no-op) plus the standard fixtures, for the st-zrt naive and prepared-lane
//! paths. Intra-op threads pinned to 1 so scheduling noise stays out of the measurement.
use std::alloc::{GlobalAlloc, Layout, System};
use std::time::Instant;

use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue, Session, SessionOptions, Tensor,
};
use st_zrt_bench_c::models;

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

fn session(path: &str) -> st_zrt::Result<Session> {
    let env = Environment::new()?;
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    Session::new(&env, path, opts)
}

fn measure<F: FnMut() -> st_zrt::Result<()>>(label: &str, mut f: F) {
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

fn naive(session: &Session, shape: &[i64], n: usize, label: &str) -> st_zrt::Result<()> {
    let env_mem = MemoryInfo::cpu()?;
    let data = vec![0.0_f32; n];
    measure(label, || {
        let input = Tensor::from_buffer(&data, shape, &env_mem)?;
        let mut out: Vec<Option<OwnedValue>> =
            (0..session.output_count()).map(|_| None).collect();
        session.run(&[&input], &mut out)?;
        let _ = out[0].as_ref().unwrap().as_slice::<f32>()?;
        Ok(())
    });
    Ok(())
}

fn lane(session: &Session, in_shape: &[i64], out_shape: &[i64], label: &str) -> st_zrt::Result<()> {
    let mem = MemoryInfo::cpu()?;
    let mut lane = session.prepare_tensor_io_lane::<f32>(&mem, &[in_shape], &[out_shape])?;
    lane.input_mut(0)?.fill(0.0);
    lane.prime(WARM)?;
    measure(label, || {
        lane.run()?;
        let _ = lane.output(0)?;
        Ok(())
    });
    Ok(())
}

fn main() -> st_zrt::Result<()> {
    println!("variant,us_per_run,allocs_per_run");
    let id = models::identity_path().unwrap();
    let s = session(id.to_str().unwrap())?;
    naive(&s, &[1, 65536], 65536, "zrt_naive_identity")?;
    lane(&s, &[1, 65536], &[1, 65536], "zrt_lane_identity")?;

    let m = models::ensure_mnist().unwrap();
    let s = session(m.to_str().unwrap())?;
    naive(&s, &[1, 1, 28, 28], 784, "zrt_naive_mnist")?;
    lane(&s, &[1, 1, 28, 28], &[1, 10], "zrt_lane_mnist")?;

    let r = models::ensure_relay("4m").unwrap();
    let s = session(r.to_str().unwrap())?;
    naive(&s, &[1, 1 << 20], 1 << 20, "zrt_naive_relay4m")?;
    lane(&s, &[1, 1 << 20], &[1, 1 << 20], "zrt_lane_relay4m")?;
    Ok(())
}
