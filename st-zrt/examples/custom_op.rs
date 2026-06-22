//! Author a custom ONNX operator in safe Rust and register it.
//!
//! Defines a `MyRelu` op via the [`CustomOp`] trait, emits its `OrtCustomOp` vtable with the
//! [`custom_op!`](st_zrt::custom_op!) macro, registers it on a [`CustomOpDomain`], and
//! attaches the domain to [`SessionOptions`] — the full authoring + registration path.
//!
//! No inference is run: there is no bundled model referencing the `com.example` domain, so
//! ORT never instantiates the kernel. The `compute` body is therefore **compile-verified
//! only** (it exercises the kernel-time API in `st_zrt::custom_ops`). Running a custom op
//! end-to-end needs a model with a node in the `com.example` domain — a separate
//! model-builder effort.
//!
//! ```text
//! cargo run --example custom_op --features custom-ops
//! ```
use st_zrt::{
    CustomOp, CustomOpDomain, KernelContext, KernelInfo, OpIoSpec, SessionOptions, custom_op,
};

/// A `com.example::MyRelu` custom op: float in, float out (compile-verified kernel body).
struct MyRelu;

impl CustomOp for MyRelu {
    const NAME: &'static str = "MyRelu";
    const DOMAIN: &'static str = "com.example";

    fn create(_info: &KernelInfo<'_>) -> st_zrt::Result<Self> {
        // A real op reads attributes / precomputes state from `info` here (e.g.
        // `info.attr_int64("alpha")?`). ReLU needs none, so the kernel state is a unit.
        Ok(Self)
    }

    fn compute(&mut self, ctx: &KernelContext<'_>) -> st_zrt::Result<()> {
        // Compile-verified only — ORT never calls this without a model referencing this op.
        // Real ReLU: read input[0], clamp at zero, write output[0].
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

fn main() -> st_zrt::Result<()> {
    let domain = CustomOpDomain::new(MyRelu::DOMAIN)?;
    domain.add_op(&MY_RELU_VTABLE)?;
    let _opts = SessionOptions::default().with_custom_op_domain(&domain);
    println!(
        "custom op '{}' registered on domain '{}'",
        MyRelu::NAME,
        MyRelu::DOMAIN
    );
    Ok(())
}
