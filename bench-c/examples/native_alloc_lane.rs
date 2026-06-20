//! Native allocation probe for the lane-local hot path.
//!
//! Run via `bench/native_alloc/run_lane_alloc_trace.sh`. The script LD_PRELOADs a small
//! malloc interposer, this example resolves its reset/count functions with dlsym, warms up
//! the model, resets the counters, then runs only `TensorIoLane::run` in the measured loop.
use st_zrt::{Allocator, Environment, GraphOptimizationLevel, MemoryInfo, SessionOptions};
use st_zrt_bench_c::models;
use std::ffi::{c_char, c_void, CStr, CString};

#[link(name = "dl")]
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type ResetFn = unsafe extern "C" fn();
type CountFn = unsafe extern "C" fn() -> u64;

struct Counter {
    reset: ResetFn,
    allocs: CountFn,
    frees: CountFn,
    bytes: CountFn,
}

fn load_symbol<T: Copy>(name: &CStr) -> T {
    let ptr = unsafe { dlsym(std::ptr::null_mut(), name.as_ptr()) };
    if ptr.is_null() {
        panic!(
            "missing symbol {}; run through bench/native_alloc/run_lane_alloc_trace.sh",
            name.to_string_lossy()
        );
    }
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<*mut c_void>(),
        "function pointer size mismatch"
    );
    unsafe { std::mem::transmute_copy(&ptr) }
}

fn counter() -> Counter {
    Counter {
        reset: load_symbol(&CString::new("zrt_malloc_counter_reset").unwrap()),
        allocs: load_symbol(&CString::new("zrt_malloc_counter_allocs").unwrap()),
        frees: load_symbol(&CString::new("zrt_malloc_counter_frees").unwrap()),
        bytes: load_symbol(&CString::new("zrt_malloc_counter_bytes").unwrap()),
    }
}

fn n_for(label: &str) -> usize {
    match label {
        "mnist" => 784,
        "4m" => 1 << 20,
        "16m" => 1 << 22,
        _ => panic!("unknown ZRT_LABEL={label}; use mnist|4m|16m"),
    }
}

fn main() {
    let label = std::env::var("ZRT_LABEL").unwrap_or_else(|_| "mnist".into());
    let iters: usize = std::env::var("ZRT_ITERS")
        .unwrap_or_else(|_| "10000".into())
        .parse()
        .unwrap();
    let n = n_for(&label);
    let path = if label == "mnist" {
        models::ensure_mnist().expect("mnist")
    } else {
        models::ensure_relay(&label).expect("relay")
    };
    let input_shape: &[i64] = if label == "mnist" {
        &[1, 1, 28, 28]
    } else {
        &[1, n as i64]
    };
    let output_shape: &[i64] = if label == "mnist" {
        &[1, 10]
    } else {
        &[1, n as i64]
    };

    let ctr = counter();
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

    let allocator = Allocator::create(&sess, &mem).unwrap();
    match lane.run_with_allocator_stats(&allocator) {
        Ok(stats) => eprintln!(
            "ort_allocator_stats before={:?} after={:?}",
            stats.before.entries(),
            stats.after.entries()
        ),
        Err(err) => eprintln!("ort_allocator_stats unavailable: {err}"),
    }

    unsafe { (ctr.reset)() };
    let mut checksum = 0u64;
    for i in 0..iters {
        lane.run().unwrap();
        let out = lane.output(0).expect("lane output");
        checksum = checksum.wrapping_add(out[i % out.len()].to_bits() as u64);
    }

    let allocs = unsafe { (ctr.allocs)() };
    let frees = unsafe { (ctr.frees)() };
    let bytes = unsafe { (ctr.bytes)() };
    println!(
        "native_alloc_lane label={label} iters={iters} allocs={allocs} frees={frees} bytes={bytes} checksum={checksum:#x}"
    );
}
