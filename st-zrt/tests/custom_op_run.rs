//! End-to-end custom-op run (feature `custom-ops`).
//!
//! Loads a bundled ONNX model whose single node is `y = MyRelu(x)` in the `com.example`
//! domain, registers that domain — carrying a `MyRelu` `custom_op!` vtable — on the
//! session, and runs one inference. This is the proof the custom-op surface runs
//! end-to-end: ORT resolves the unknown op to the Rust kernel and invokes
//! `create` / `compute` / `destroy` (previously only compile-verified), and the output
//! matches the ReLU we authored in safe Rust.
//!
//! Fixture: `fixtures/custom_relu.onnx` (regenerate with `fixtures/gen_custom_relu.py`).
use st_zrt::{
    AllocatedTensor, Allocator, CustomOp, CustomOpDomain, Environment, KernelContext, KernelInfo,
    MemoryInfo, Op, OpIoSpec, OwnedValue, Session, SessionOptions, ShapeInferContext, Tensor,
    custom_op,
};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Callback count from `MyReluPar::compute`'s `parallel_for` — bumped once per iteration ORT
/// dispatches. Read by the test to prove `KernelContext_ParallelFor` fires the closure.
static PAR_FOR_CALLS: AtomicUsize = AtomicUsize::new(0);

/// `com.example::MyRelu`: float in, relu'd float out.
struct MyRelu;

impl CustomOp for MyRelu {
    const NAME: &'static str = "MyRelu";
    const DOMAIN: &'static str = "com.example";

    fn create(info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        // Exercise KernelInfo config-entries + string-array-attribute reads
        // (KernelInfo_GetConfigEntries + KernelInfoGetAttributeArray_string). For a model without
        // these, config_entries returns an empty container and attr_strings errors cleanly (missing
        // attribute) — both prove the FFI was reached without UB.
        let _ = info.config_entries();
        let _ = info.attr_strings("missing_attr");
        Ok(Self)
    }

    fn compute(&mut self, ctx: &KernelContext<'_>) -> st_zrt::Result<()> {
        // Exercise the kernel-time Logger: read its severity threshold (Logger_GetLoggingSeverityLevel)
        // and emit an Info line (Logger_LogMessage). The message routes through the session→env
        // logger: under a Verbose env it reaches the custom callback (deterministic capture); under
        // the default Warning env it is dropped below threshold.
        if let Some(logger) = ctx.logger()? {
            let lvl = logger
                .severity_level()
                .unwrap_or(st_zrt::sys::LoggingLevel::Warning);
            logger.log(
                st_zrt::sys::LoggingLevel::Info,
                &format!("MyRelu::compute level={lvl:?}"),
                file!(),
                line!() as i32,
                module_path!(),
            )?;
        }
        let input = ctx.input(0)?.expect("MyRelu: input[0] required");
        let dims = input.dims()?;
        let inp = input.as_slice::<f32>()?;
        ctx.output_mut::<f32>(0, &dims, |out| {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.max(0.0);
            }
            Ok(())
        })
    }

    fn inputs() -> &'static [OpIoSpec] {
        static IN: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &IN
    }
    fn outputs() -> &'static [OpIoSpec] {
        static OUT: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &OUT
    }
}

custom_op!(MyRelu, "MyRelu", as MY_RELU_VTABLE);

/// The whole custom-op path, live: domain registration → session load → `create` →
/// `compute` → result read → `destroy` (on session drop). `compute` also exercises the
/// kernel-time Logger (`KernelContext::logger` → `Logger::severity_level` + `Logger::log`); a
/// correct ReLU output proves those FFI calls returned `Ok`.
#[test]
fn custom_op_runs_end_to_end() {
    // Register the custom domain on the session options BEFORE building the session:
    // ORT resolves `com.example::MyRelu` from it during graph instantiation.
    let domain = CustomOpDomain::new(MyRelu::DOMAIN).expect("new domain");
    domain.add_op(&MY_RELU_VTABLE).expect("add_op");

    let env = Environment::new().expect("env");
    let opts = SessionOptions::default().with_custom_op_domain(&domain);
    let model = include_bytes!("fixtures/custom_relu.onnx");
    let sess = Session::from_bytes(&env, model, opts).expect("session from bytes");

    let mem = MemoryInfo::cpu().expect("cpu mem");
    let input = [-2.0f32, 3.0, -1.0, 5.0];
    let view = Tensor::from_buffer(&input, &[input.len() as i64], &mem).expect("input");

    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&view], &mut out).expect("run");

    let y = out[0]
        .as_ref()
        .expect("output[0]")
        .as_slice::<f32>()
        .expect("as_slice");
    assert_eq!(y, &[0.0f32, 3.0, 0.0, 5.0], "relu output");
    eprintln!("custom_op_runs_end_to_end: MyRelu fired end-to-end, y = {y:?}");

    // Drop order matters: the session before the domain (a domain must outlive its
    // sessions — an ORT invariant; ORT retains the domain, it does not copy it).
    drop(sess);
    drop(domain);
}

/// `com.example::MyRelu` WITH shape inference: the output mirrors the input's type+shape.
struct MyReluInfer;

impl CustomOp for MyReluInfer {
    const NAME: &'static str = "MyRelu";
    const DOMAIN: &'static str = "com.example";

    fn create(_info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        Ok(Self)
    }

    fn compute(&mut self, ctx: &KernelContext<'_>) -> st_zrt::Result<()> {
        let input = ctx.input(0)?.expect("MyRelu: input[0] required");
        let dims = input.dims()?;
        let inp = input.as_slice::<f32>()?;
        ctx.output_mut::<f32>(0, &dims, |out| {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.max(0.0);
            }
            Ok(())
        })
    }

    /// Output type+shape == input type+shape (elementwise relu). Reads the input's type+shape
    /// (releasing that owning info), then builds a fresh output info for `set_output_type_shape`
    /// (which hands ownership to ORT).
    fn infer_shapes(ctx: &ShapeInferContext<'_>) -> st_zrt::Result<()> {
        let in_info = ctx.input_type_shape(0)?;
        let elem = in_info.element_type()?;
        let dims = in_info.dims()?;
        drop(in_info); // release the input's owning info
        let mut out = st_zrt::TensorTypeAndShapeInfo::new()?;
        out.set_element_type(elem)?;
        out.set_dimensions(&dims)?;
        ctx.set_output_type_shape(0, out) // consumes `out`; ORT takes ownership
    }

    fn inputs() -> &'static [OpIoSpec] {
        static IN: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &IN
    }
    fn outputs() -> &'static [OpIoSpec] {
        static OUT: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &OUT
    }
}

custom_op!(MyReluInfer, "MyRelu", as MY_RELU_INFER_VTABLE);

/// The unshaped fixture's output `y` has no static shape, so ORT MUST call `infer_shapes`
/// to learn it — without a firing hook the session would fail to build (unknown output
/// shape). A successful load + the inferred [4] shape + the correct ReLU output proves the
/// `InferOutputShapeFn` trampoline runs and `ShapeInferContext` works end-to-end.
#[test]
fn custom_op_shape_inference_runs() {
    let domain = CustomOpDomain::new(MyReluInfer::DOMAIN).expect("new domain");
    domain.add_op(&MY_RELU_INFER_VTABLE).expect("add_op");

    let env = Environment::new().expect("env");
    let opts = SessionOptions::default().with_custom_op_domain(&domain);
    let model = include_bytes!("fixtures/custom_relu_unshaped.onnx");
    // If infer_shapes didn't fire, this errors (unknown output shape):
    let sess = Session::from_bytes(&env, model, opts).expect("session (infer_shapes fired)");

    // The output shape was INFERRED from the input; the cached meta reflects it.
    assert_eq!(
        sess.output_shape(0).expect("output shape"),
        &[4],
        "inferred output shape"
    );

    let mem = MemoryInfo::cpu().expect("cpu mem");
    let input = [-2.0f32, 3.0, -1.0, 5.0];
    let view = Tensor::from_buffer(&input, &[input.len() as i64], &mem).expect("input");
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&view], &mut out).expect("run");
    let y = out[0]
        .as_ref()
        .expect("output[0]")
        .as_slice::<f32>()
        .expect("as_slice");
    assert_eq!(y, &[0.0f32, 3.0, 0.0, 5.0], "relu output");
    eprintln!(
        "custom_op_shape_inference_runs: infer_shapes fired, output shape inferred + relu OK"
    );

    drop(sess);
    drop(domain);
}

/// `com.example::MyRelu` whose `compute` dispatches work through
/// [`KernelContext::parallel_for`] — proving the `KernelContext_ParallelFor` C bridge fires the
/// Rust closure exactly once per iteration, then still produces a correct ReLU.
struct MyReluPar;

impl CustomOp for MyReluPar {
    const NAME: &'static str = "MyRelu";
    const DOMAIN: &'static str = "com.example";

    fn create(_info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        Ok(Self)
    }

    fn compute(&mut self, ctx: &KernelContext<'_>) -> st_zrt::Result<()> {
        let input = ctx.input(0)?.expect("MyReluPar: input[0] required");
        let dims = input.dims()?;
        let n = input.as_slice::<f32>()?.len();

        // Fire one callback per element via the engine's parallel-for. The closure is Send + Sync
        // and touches no per-thread state, so concurrent dispatch is safe; it just bumps the shared
        // counter to record that each iteration ran. (Asserted by the test, not here.)
        ctx.parallel_for(n, 2, |_| {
            PAR_FOR_CALLS.fetch_add(1, Ordering::SeqCst);
        })?;

        // The actual ReLU (sequential) into the output buffer.
        let inp = input.as_slice::<f32>()?;
        ctx.output_mut::<f32>(0, &dims, |out| {
            for (o, &v) in out.iter_mut().zip(inp) {
                *o = v.max(0.0);
            }
            Ok(())
        })
    }

    fn inputs() -> &'static [OpIoSpec] {
        static IN: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &IN
    }
    fn outputs() -> &'static [OpIoSpec] {
        static OUT: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &OUT
    }
}

custom_op!(MyReluPar, "MyRelu", as MY_RELU_PAR_VTABLE);

/// `parallel_for` dispatches the closure once per iteration (`KernelContext_ParallelFor`), and
/// the op still computes a correct ReLU afterward. Reuses the shaped `custom_relu.onnx` fixture
/// (node `com.example::MyRelu`), resolved to `MyReluPar`'s vtable.
#[test]
fn custom_op_parallel_for_runs() {
    let domain = CustomOpDomain::new(MyReluPar::DOMAIN).expect("new domain");
    domain.add_op(&MY_RELU_PAR_VTABLE).expect("add_op");

    let env = Environment::new().expect("env");
    let opts = SessionOptions::default().with_custom_op_domain(&domain);
    let model = include_bytes!("fixtures/custom_relu.onnx");
    let sess = Session::from_bytes(&env, model, opts).expect("session from bytes");

    let mem = MemoryInfo::cpu().expect("cpu mem");
    let input = [-2.0f32, 3.0, -1.0, 5.0];
    let view = Tensor::from_buffer(&input, &[input.len() as i64], &mem).expect("input");

    let before = PAR_FOR_CALLS.load(Ordering::SeqCst);
    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&view], &mut out).expect("run");
    let fired = PAR_FOR_CALLS.load(Ordering::SeqCst) - before;

    // One callback per element (4) — the engine dispatched every iteration of `parallel_for`.
    assert_eq!(fired, input.len(), "parallel_for fired once per element");

    let y = out[0]
        .as_ref()
        .expect("output[0]")
        .as_slice::<f32>()
        .expect("as_slice");
    assert_eq!(
        y,
        &[0.0f32, 3.0, 0.0, 5.0],
        "relu output after parallel_for"
    );
    eprintln!("custom_op_parallel_for_runs: parallel_for fired {fired} callbacks, relu OK");

    drop(sess);
    drop(domain);
}

/// `com.example::InvokeAdd` does NOT add itself: it creates the **builtin** ONNX `Add` op via
/// [`Op::create`] (`CreateOp`) and runs it via [`Op::invoke`] (`InvokeOp`) into a separately
/// allocated tensor, then copies the result to the kernel output. Proves the op-invoke-op FFI
/// bridge end-to-end. Invoking builtin `Add` (a different op than this one) cannot recurse into
/// `InvokeAdd::compute`; and because `KernelContext` exposes outputs only through `output_mut`'s
/// slice (not a `TensorView`), the invoke target is a fresh [`AllocatedTensor`] instead.
struct InvokeAdd {
    add: Op,
}

impl CustomOp for InvokeAdd {
    const NAME: &'static str = "InvokeAdd";
    const DOMAIN: &'static str = "com.example";

    fn create(info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        // Builtin `Add` (domain "", since-version 14 — its latest schema): two float inputs, one
        // float output.
        let add = Op::create(
            info,
            "Add",
            "",
            14,
            &[("T", st_zrt::sys::ElementType::Float)],
            &[],
            2,
            1,
        )?;
        Ok(Self { add })
    }

    fn compute(&mut self, ctx: &KernelContext<'_>) -> st_zrt::Result<()> {
        let x = ctx.input(0)?.expect("InvokeAdd: input[0] required");
        let dims = x.dims()?;
        let n = x.as_slice::<f32>()?.len();

        // Second addend: all ones, so Add(x, ones) == x + 1.
        let mem = MemoryInfo::cpu()?;
        let ones = vec![1.0_f32; n];
        let ones_t = Tensor::from_buffer(&ones, &dims, &mem)?;

        // Invoke builtin Add into a separate allocated tensor (the context's own output is only
        // reachable as a slice via output_mut, not as a TensorView invoke can write into).
        let tmp = AllocatedTensor::<f32>::new(Allocator::get_default()?, &dims)?;
        let mut y = tmp.as_view();
        self.add
            .invoke(ctx, &[&x, ones_t.as_view()], &mut [&mut y])?;

        // Hand the invoked result to the kernel output.
        let got = tmp.as_slice()?;
        ctx.output_mut::<f32>(0, &dims, |out| {
            out.copy_from_slice(got);
            Ok(())
        })
    }

    fn inputs() -> &'static [OpIoSpec] {
        static IN: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &IN
    }
    fn outputs() -> &'static [OpIoSpec] {
        static OUT: [OpIoSpec; 1] = [OpIoSpec::required(st_zrt::sys::ElementType::Float)];
        &OUT
    }
}

custom_op!(InvokeAdd, "InvokeAdd", as INVOKE_ADD_VTABLE);

/// `Op::create` + `Op::invoke` round-trip: the `InvokeAdd` kernel builds the builtin `Add` op and
/// invokes it, so `[1,2,3,4]` → `[2,3,4,5]` (`Add(x, ones)`). A correct result proves both the
/// `CreateOp` and `InvokeOp` FFI bridges fire.
#[test]
fn custom_op_invoke_builtin_add() {
    let domain = CustomOpDomain::new(InvokeAdd::DOMAIN).expect("new domain");
    domain.add_op(&INVOKE_ADD_VTABLE).expect("add_op");

    let env = Environment::new().expect("env");
    let opts = SessionOptions::default().with_custom_op_domain(&domain);
    let model = include_bytes!("fixtures/invoke_add.onnx");
    let sess = Session::from_bytes(&env, model, opts).expect("session from bytes");

    let mem = MemoryInfo::cpu().expect("cpu mem");
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let view = Tensor::from_buffer(&input, &[input.len() as i64], &mem).expect("input");

    let mut out: Vec<Option<OwnedValue>> = (0..sess.output_count()).map(|_| None).collect();
    sess.run(&[&view], &mut out).expect("run");

    let y = out[0]
        .as_ref()
        .expect("output[0]")
        .as_slice::<f32>()
        .expect("as_slice");
    assert_eq!(
        y,
        &[2.0_f32, 3.0, 4.0, 5.0],
        "InvokeAdd = Add(x, ones) = x + 1"
    );
    eprintln!("custom_op_invoke_builtin_add: CreateOp + InvokeOp fired, y = {y:?}");

    drop(sess);
    drop(domain);
}
