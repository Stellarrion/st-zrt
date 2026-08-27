//! Tensor accessor microbenchmarks.
//!
//! These isolate the wrapper-side costs optimized in the v0.3 performance pass: cached
//! `TensorView::shape`, cached `AllocatedTensor::raw_mut_ptr`, and lazy `OwnedValue`
//! host-accessibility probing.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use st_zrt::{
    AllocatedTensor, Allocator, Environment, GraphOptimizationLevel, MemoryInfo, OwnedValue,
    RunInput, Session, SessionOptions, Tensor,
};
use st_zrt_bench_c::models;

const INPUT: [i64; 4] = [1, 1, 28, 28];

fn session(env: &Environment) -> (Session, MemoryInfo) {
    let model = models::ensure_mnist().expect("mnist");
    let mem = MemoryInfo::cpu().expect("cpu memory");
    let opts = SessionOptions::new()
        .with_opt_level(GraphOptimizationLevel::All)
        .with_intra_threads(1);
    let sess = Session::new(env, model.to_str().unwrap(), opts).expect("session");
    (sess, mem)
}

fn owned_output(sess: &Session, mem: &MemoryInfo) -> OwnedValue {
    let input_buf = vec![0.0_f32; 784];
    let input = Tensor::from_buffer(&input_buf, &INPUT, mem).expect("input tensor");
    let inputs: [&dyn RunInput; 1] = [&input];
    let mut outputs: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&inputs, &mut outputs).expect("run");
    outputs[0].take().expect("output")
}

fn bench_tensor_view_shape_cached(c: &mut Criterion) {
    let mem = MemoryInfo::cpu().expect("cpu memory");
    let data = vec![1.0_f32; 1024];
    let tensor = Tensor::from_buffer(&data, &[1, 1024], &mem).expect("tensor");
    let view = tensor.as_view();
    black_box(view.shape().expect("prime shape cache"));

    c.bench_function("tensor/view_shape_cached", |b| {
        b.iter(|| {
            black_box(view.shape().expect("shape"));
        });
    });
}

fn bench_tensor_view_dims_cached_clone(c: &mut Criterion) {
    let mem = MemoryInfo::cpu().expect("cpu memory");
    let data = vec![1.0_f32; 1024];
    let tensor = Tensor::from_buffer(&data, &[1, 1024], &mem).expect("tensor");
    let view = tensor.as_view();
    black_box(view.shape().expect("prime shape cache"));

    c.bench_function("tensor/view_dims_cached_clone", |b| {
        b.iter(|| {
            black_box(view.dims().expect("dims"));
        });
    });
}

fn bench_allocated_tensor_raw_mut_ptr_cached(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let alloc = Allocator::create(&sess, &mem).expect("allocator");
    let data = vec![1.0_f32; 1024];
    let tensor =
        AllocatedTensor::copy_from_slice(alloc, &[1, 1024], &data).expect("allocated tensor");
    black_box(tensor.raw_mut_ptr().expect("prime pointer"));

    c.bench_function("tensor/allocated_raw_mut_ptr_cached", |b| {
        b.iter(|| {
            black_box(tensor.raw_mut_ptr().expect("raw ptr"));
        });
    });
}

fn bench_owned_value_as_slice_first_access(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);

    c.bench_function("tensor/owned_value_as_slice_first_access", |b| {
        b.iter_batched(
            || owned_output(&sess, &mem),
            |output| {
                black_box(output.as_slice::<f32>().expect("output slice"));
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_owned_value_as_slice_repeated_access(c: &mut Criterion) {
    let env = Environment::new().expect("env");
    let (sess, mem) = session(&env);
    let output = owned_output(&sess, &mem);
    black_box(output.as_slice::<f32>().expect("prime host-access cache"));

    c.bench_function("tensor/owned_value_as_slice_repeated_access", |b| {
        b.iter(|| {
            black_box(output.as_slice::<f32>().expect("output slice"));
        });
    });
}

criterion_group!(
    benches,
    bench_tensor_view_shape_cached,
    bench_tensor_view_dims_cached_clone,
    bench_allocated_tensor_raw_mut_ptr_cached,
    bench_owned_value_as_slice_first_access,
    bench_owned_value_as_slice_repeated_access
);
criterion_main!(benches);
