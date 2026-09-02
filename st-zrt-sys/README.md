# st-zrt-sys

Generated, zrt-namespaced raw FFI for ONNX Runtime 1.27.

This crate exposes the `OrtApi` function-pointer table, stable ORT enums, opaque handle types, and
build logic used by `st-zrt`. The major/minor version mirrors `libonnxruntime`: `st-zrt-sys 1.27.1` is the first Rust-binding
revision targeting ONNX Runtime 1.27.0 after the already-published `st-zrt-sys 1.27.0`. The patch
component revisions the binding surface; the native ABI/API target remains ONNX Runtime 1.27.0/API 27.
A 1.28.x runtime is also supported: the C API is append-only (API 27 → 28 adds
`KernelContext_GetSyncStream`, removes nothing), so binding `ST_ZRT_ORT_PATH` to an
extracted 1.28 runtime is valid; `st_zrt_sys::SUPPORTED_RUNTIME_LINES` is the machine-readable policy.

What is different:

- no `bindgen`;
- the generated accessor table and sub-APIs are zrt-namespaced, with the upstream `Ort*` C names
  preserved in source comments for header cross-reference; the only deliberately exported upstream
  `Ort*` Rust names are the ABI-faithful `OrtErrorCode` enum, the `OrtCustomOp` opaque handle, and
  the `OrtGetApiBase` link symbol;
- checked-in generated table from the workspace codegen tool;
- newer ONNX element metadata variants including complex, float8 (incl. e8m0), int4/uint4, float4, and int2/uint2;
- pure-Rust download (bounded timeouts/retries, atomic `.part` rename), SHA-256 verification, and
  `.tgz` archive extraction in `build.rs`;
- optional feature gates for EP, CUDA, custom-op, model-editor, and training symbols.

Most users should depend on `st-zrt`, not this crate directly.

## Acquiring ONNX Runtime

On a supported target (linux-x64, linux-aarch64, osx-arm64) the build script downloads the
SHA-256-pinned official release archive and extracts it into `OUT_DIR`. The `cuda` feature does the
same for the GPU CUDA 13 archive (linux-x64 only). CUDA runtime libraries are resolved from a system
CUDA 13 toolkit (`ST_ZRT_CUDA13_PATH`, then `CUDA_PATH`, then `/opt/cuda`); they are not vendored by
this crate.

Windows is not auto-acquired: upstream publishes the win-x64 CPU package as a `.zip` and this build
script extracts `.tgz` archives only. Point `ST_ZRT_ORT_PATH` at an extracted directory (from the
release ZIP or the NuGet `Microsoft.ML.OnnxRuntime` package).

Override automatic discovery with an already-extracted directory (relative paths resolve against the
workspace root, not the working directory):

```bash
ST_ZRT_ORT_PATH=/path/to/onnxruntime cargo build
```

The directory must contain `include/` and `lib/`.

## Runtime loading

Linking `libonnxruntime` is not the same as loading it. The rpath this crate emits covers only its
own linkable units (for example `cargo test -p st-zrt-sys`); it does **not** propagate through an
rlib to downstream final binaries. A downstream binary must arrange for the dynamic loader to find
the library:

- Linux: `LD_LIBRARY_PATH=<ort>/lib`, or an rpath emitted by the final binary's own build script;
- macOS: `DYLD_LIBRARY_PATH=<ort>/lib`, or a consumer-side rpath;
- Windows: place `onnxruntime.dll` next to the executable or add its directory to `PATH`.

To support that, the build script exports the acquired directories to direct dependents' build
scripts as Cargo metadata: `DEP_ONNXRUNTIME_ROOT`, `DEP_ONNXRUNTIME_LIBDIR`, and
`DEP_ONNXRUNTIME_INCLUDE` (available because this crate declares `links = "onnxruntime"`). A
consumer's build script can emit `cargo:rustc-link-arg=-Wl,-rpath,$DEP_ONNXRUNTIME_LIBDIR` for its
own binaries. The metadata exports data only; nothing is loaded automatically.

With the `cuda` feature, the same loader requirements apply to the CUDA 13 runtime libraries
(`libcudart.so.13`, `libcublas.so.13`, `libcufft.so.12`, `libcurand.so.10`, `libnvrtc.so.13`) and to
cuDNN 9 (`libcudnn.so.9`), which must be resolvable on the host.

License: `Apache-2.0`.
