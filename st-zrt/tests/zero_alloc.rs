use st_zrt::{
    Environment, GraphOptimizationLevel, MemoryInfo, OutputValue, Session, SessionOptions, Tensor,
    TensorBuffer, ZrtLaneSet,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard};

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

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

fn mnist_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("bench")
        .join("models")
        .join("mnist.onnx")
}

fn mnist_session() -> Option<(Environment, MemoryInfo, Session)> {
    let path = mnist_path();
    if !path.exists() {
        return None;
    }
    let env = Environment::new().expect("env");
    let mem = MemoryInfo::cpu().expect("cpu mem");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = Session::new(&env, path.to_str().unwrap(), opts).expect("session");
    Some((env, mem, sess))
}

fn measured_allocs(f: impl FnOnce()) -> usize {
    ALLOCS.store(0, Ordering::SeqCst);
    f();
    ALLOCS.load(Ordering::SeqCst)
}

fn test_guard() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap()
}

#[test]
fn tensor_io_lane_run_is_rust_zero_alloc() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("lane");
    lane.input_mut(0).expect("lane input").fill(0.0);
    for _ in 0..8 {
        lane.run().expect("warmup");
    }

    let allocs = measured_allocs(|| {
        lane.run().expect("lane run");
        assert_eq!(lane.output(0).expect("lane output").len(), 10);
    });
    assert_eq!(allocs, 0, "TensorIoLane::run allocated {allocs} times");
}

#[test]
fn static_tensor_io_lane_run_is_rust_zero_alloc() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut lane = sess
        .prepare_static_tensor_io_lane::<f32, 1, 1>(&mem, [&[1, 1, 28, 28]], [&[1, 10]])
        .expect("static lane");
    lane.inputs_mut()[0].as_mut_slice().fill(0.0);
    for _ in 0..8 {
        lane.run().expect("warmup");
    }

    let allocs = measured_allocs(|| {
        lane.run().expect("static lane run");
        assert_eq!(lane.outputs()[0].as_slice().len(), 10);
    });
    assert_eq!(
        allocs, 0,
        "StaticTensorIoLane::run allocated {allocs} times"
    );
}

#[test]
fn prepared_iobinding_run_is_rust_zero_alloc() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &mem).expect("input");
    let mut output = st_zrt::TensorBuffer::<f32>::zeros(&[1, 10], &mem).expect("output");
    output.as_mut_slice().fill(0.0);
    let mut prepared = sess
        .prepare_io_binding_buffers(&[&input], &[&output])
        .expect("prepared binding");
    prepared.run().expect("warmup");

    let allocs = measured_allocs(|| {
        prepared.run().expect("prepared run");
        assert_eq!(output.as_slice().len(), 10);
    });
    assert_eq!(allocs, 0, "PreparedIoBinding::run allocated {allocs} times");
}

#[test]
fn tensor_from_buffer_is_pointer_identity_zero_copy() {
    let _guard = test_guard();
    let Some((_env, mem, _sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &[1, 1, 28, 28], &mem).expect("input");
    let engine_slice = input.as_slice::<f32>().expect("engine slice");
    assert_eq!(
        engine_slice.as_ptr(),
        input_buf.as_ptr(),
        "Tensor::from_buffer copied instead of pointing at caller input"
    );
}

#[test]
fn output_value_is_pointer_identity_zero_copy() {
    let _guard = test_guard();
    let Some((_env, mem, _sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut out_buf = vec![0.0_f32; 10];
    let expected_ptr = out_buf.as_ptr();
    let output = OutputValue::from_buffer(&mut out_buf, &[1, 10], &mem).expect("output");
    let engine_slice = output.as_slice::<f32>().expect("engine output slice");
    assert_eq!(
        engine_slice.as_ptr(),
        expected_ptr,
        "OutputValue copied instead of pointing at caller output"
    );
}

#[test]
fn tensor_buffer_is_pointer_identity_zero_copy() {
    let _guard = test_guard();
    let Some((_env, mem, _sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let buffer = TensorBuffer::<f32>::zeros(&[1, 10], &mem).expect("buffer");
    assert_eq!(
        buffer.engine_data_ptr().expect("engine ptr"),
        buffer.as_slice().as_ptr(),
        "TensorBuffer ORT pointer differs from backing Vec pointer"
    );
}

#[test]
fn tensor_buffer_prefaulted_and_aligned_are_pointer_identity_zero_copy() {
    let _guard = test_guard();
    let Some((_env, mem, _sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let prefaulted = TensorBuffer::<f32>::zeros_prefaulted(&[1, 1024], &mem).expect("prefaulted");
    assert_eq!(
        prefaulted.engine_data_ptr().expect("engine ptr"),
        prefaulted.as_slice().as_ptr(),
        "prefaulted TensorBuffer ORT pointer differs from backing storage"
    );

    let aligned =
        TensorBuffer::<f32>::zeros_aligned_prefaulted(&[1, 1024], 64, &mem).expect("aligned");
    assert_eq!(
        aligned.engine_data_ptr().expect("engine ptr"),
        aligned.as_slice().as_ptr(),
        "aligned TensorBuffer ORT pointer differs from backing storage"
    );
    assert_eq!(
        (aligned.as_slice().as_ptr() as usize) % 64,
        0,
        "aligned TensorBuffer pointer is not 64-byte aligned"
    );

    let hugepage =
        TensorBuffer::<f32>::zeros_aligned_hugepage_prefaulted(&[1, 1 << 19], 2 << 20, &mem)
            .expect("hugepage");
    assert_eq!(
        hugepage.engine_data_ptr().expect("engine ptr"),
        hugepage.as_slice().as_ptr(),
        "hugepage TensorBuffer ORT pointer differs from backing storage"
    );
    assert_eq!(
        (hugepage.as_slice().as_ptr() as usize) % (2 << 20),
        0,
        "hugepage TensorBuffer pointer is not 2 MiB aligned"
    );

    let locked = TensorBuffer::<f32>::zeros_aligned_mlocked_prefaulted(&[1, 1024], 4096, &mem)
        .expect("mlocked");
    assert_eq!(
        locked.engine_data_ptr().expect("engine ptr"),
        locked.as_slice().as_ptr(),
        "mlocked TensorBuffer ORT pointer differs from backing storage"
    );
    assert_eq!(
        (locked.as_slice().as_ptr() as usize) % 4096,
        0,
        "mlocked TensorBuffer pointer is not 4096-byte aligned"
    );

    let path = std::env::temp_dir().join(format!(
        "st-zrt-mmap-zero-copy-{}-{}.bin",
        std::process::id(),
        ALLOCS.load(Ordering::Relaxed)
    ));
    let mmap_values = [1.0f32, 2.0, 3.0, 4.0];
    std::fs::write(&path, f32_as_bytes(&mmap_values)).expect("write mmap tensor");
    let mmap = TensorBuffer::<f32>::from_mmap_file(&path, &[2, 2], &mem).expect("mmap tensor");
    assert_eq!(mmap.as_slice(), &mmap_values);
    assert_eq!(
        mmap.engine_data_ptr().expect("engine ptr"),
        mmap.as_slice().as_ptr(),
        "mmap TensorBuffer ORT pointer differs from mapped storage"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn tensor_io_lanes_are_rust_zero_alloc() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut lanes = sess
        .prepare_tensor_io_lanes::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
        .expect("lanes");
    for lane in lanes.iter_mut() {
        lane.run().expect("warmup");
    }

    let allocs = measured_allocs(|| {
        for lane in lanes.iter_mut() {
            lane.run().expect("lane run");
            assert_eq!(lane.output(0).expect("lane output").len(), 10);
        }
    });
    assert_eq!(allocs, 0, "TensorIoLane set allocated {allocs} times");
}

#[test]
fn zrt_lane_set_runs_are_rust_zero_alloc() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut lanes =
        ZrtLaneSet::<f32>::shared_session(Arc::new(sess), &mem, &[&[1, 1, 28, 28]], &[&[1, 10]], 2)
            .expect("lane set");
    for lane in lanes.lanes_mut() {
        lane.run().expect("warmup");
    }

    let allocs = measured_allocs(|| {
        for lane in lanes.lanes_mut() {
            lane.run().expect("lane run");
            assert_eq!(lane.output(0).expect("lane output").len(), 10);
        }
    });
    assert_eq!(allocs, 0, "ZrtLaneSet runs allocated {allocs} times");
}

#[test]
fn tensor_io_lane_output_is_pointer_identity_zero_copy() {
    let _guard = test_guard();
    let Some((_env, mem, sess)) = mnist_session() else {
        eprintln!("skipping — mnist.onnx absent");
        return;
    };

    let mut lane = sess
        .prepare_tensor_io_lane::<f32>(&mem, &[&[1, 1, 28, 28]], &[&[1, 10]])
        .expect("lane");
    let output_ptr = lane.output(0).expect("lane output").as_ptr();
    assert_eq!(
        lane.output_buffer(0)
            .expect("lane output buffer")
            .engine_data_ptr()
            .expect("engine ptr"),
        output_ptr,
        "lane output binding does not point at the owned output buffer"
    );
    lane.input_mut(0).expect("lane input").fill(0.0);
    lane.run().expect("lane run");
    assert_eq!(
        lane.output(0).expect("lane output").as_ptr(),
        output_ptr,
        "lane output buffer moved or was replaced across run"
    );
    assert_eq!(
        lane.output_buffer(0)
            .expect("lane output buffer")
            .engine_data_ptr()
            .expect("engine ptr"),
        output_ptr,
        "ORT output pointer changed after run"
    );
}
