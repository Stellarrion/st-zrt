# st-zrt-sys

Generated, zrt-namespaced raw FFI for ONNX Runtime 1.27.

This crate exposes the `OrtApi` function-pointer table, stable ORT enums, opaque handle types, and
build logic used by `st-zrt`. The version mirrors `libonnxruntime`: `st-zrt-sys 1.27.0` targets
ONNX Runtime 1.27.0.

What is different:

- no `bindgen`;
- no public legacy `Ort*` type names;
- checked-in generated table from the workspace codegen tool;
- newer ONNX element metadata variants including complex, float8 (incl. e8m0), int4/uint4, float4, and int2/uint2;
- pure-Rust download, SHA-256 verification, and archive extraction in `build.rs`;
- optional feature gates for EP, CUDA, custom-op, model-editor, and training symbols.

The `cuda` feature downloads the ONNX Runtime 1.27 GPU CUDA 13 archive. CUDA runtime libraries are
resolved from a system CUDA 13 toolkit (`ST_ZRT_CUDA13_PATH`, then `CUDA_PATH`, then `/opt/cuda`);
they are not vendored by this crate.

Most users should depend on `st-zrt`, not this crate directly.

Override automatic ONNX Runtime discovery with:

```bash
ST_ZRT_ORT_PATH=/path/to/onnxruntime cargo build
```

License: `Apache-2.0`.
