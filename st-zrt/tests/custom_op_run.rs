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
    custom_op, CustomOp, CustomOpDomain, Environment, KernelContext, KernelInfo, MemoryInfo,
    OpIoSpec, OwnedValue, Session, SessionOptions, ShapeInferContext, Tensor,
};

/// `com.example::MyRelu`: float in, relu'd float out.
struct MyRelu;

impl CustomOp for MyRelu {
    const NAME: &'static str = "MyRelu";
    const DOMAIN: &'static str = "com.example";

    fn create(_info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        // ReLU needs no attributes, so the kernel state is a unit.
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
/// `compute` → result read → `destroy` (on session drop).
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
