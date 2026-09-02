# Support

## Where to ask

- **Bugs and feature requests for `st-zrt`/`st-zrt-sys`:** open a GitHub issue with the
  crate, feature set, platform, and ONNX Runtime acquisition mode (`ST_ZRT_ORT_PATH` vs
  build-script download).
- **Security vulnerabilities:** follow [SECURITY.md](SECURITY.md) — private GitHub
  vulnerability reporting only, no public issues.
- **ONNX Runtime questions** (model behavior, kernels, execution providers): ONNX
  Runtime is a separate project; use its upstream documentation and issue tracker at
  <https://github.com/microsoft/onnxruntime>.
- **CUDA/cuDNN problems** in the native stack: NVIDIA documentation and forums; `st-zrt`
  links against a system CUDA 13.x toolkit and cuDNN 9 but does not ship them.

## Supported platforms

| Target | Support level |
| --- | --- |
| Linux x86_64 | reference platform; the only one with automated native link and test coverage |
| Linux aarch64 | ORT archive download supported; CI checks are compile-only |
| Windows x64 | compile-only in CI; no automatic ORT acquisition — set `ST_ZRT_ORT_PATH` from the release ZIP or the NuGet package |
| macOS arm64 | ORT archive download supported; no automated coverage |
| macOS x86_64 | not supported by the ONNX Runtime 1.29.0 archive set |

MSRV: Rust 1.85 (edition 2024). Published line: `st-zrt` 0.3.0 /
`st-zrt-sys` 1.27.1 (ORT 1.27 bundled, 1.28 bring-your-own). Release candidate:
`st-zrt` 0.4.0 / `st-zrt-sys` 1.29.0 (ORT 1.29 only). See the wrapper-to-ORT mapping
table in the README.

On compile-only platforms, issues are limited to what can be reproduced without native
linking; runtime behavior there is best-effort and unverified by this project's CI.

## Before filing

Reproduce with a minimal model and check the README sections on ONNX Runtime
acquisition, the dynamic-library loader path for downstream binaries, and the CUDA
feature requirements — most setup failures come from those three areas.
