# Third-Party Notices

This file is an informational summary, not a complete legal inventory. Distribution and
deployment decisions remain your responsibility; review the license texts of everything
you actually ship.

## This project

`st-zrt` and `st-zrt-sys` are licensed under the Apache License 2.0 (see [LICENSE](LICENSE)).

## ONNX Runtime

- `st-zrt-sys` does not bundle ONNX Runtime. At build time it downloads a SHA-256-pinned
  ONNX Runtime 1.27.0 archive from Microsoft's GitHub releases (Linux x86_64, Linux
  aarch64, macOS arm64; a GPU package when the `cuda` feature is enabled), or uses a
  user-provided distribution via `ST_ZRT_ORT_PATH` (required on Windows).
- ONNX Runtime is made available by Microsoft and its contributors under the MIT
  License (see the upstream repository for the authoritative text and notice).
- Binaries you produce link `libonnxruntime` dynamically; complying with ONNX Runtime's
  license and any applicable notices is the responsibility of the party distributing the
  resulting binary.

## CUDA toolkit and cuDNN

The `cuda` feature links a system-provided CUDA 13.x toolkit and cuDNN 9, resolved from
`ST_ZRT_CUDA13_PATH` → `CUDA_PATH` → `/opt/cuda`. These components are licensed by
NVIDIA under their own terms; they are neither bundled nor redistributed by this project.

## Rust dependencies

- Runtime and build dependencies of the workspace crates are resolved from crates.io and
  pinned in `Cargo.lock`.
- The dependency-policy allowlist (see `deny.toml`) covers MIT, Apache-2.0,
  BSD-3-Clause, ISC, CDLA-Permissive-2.0, and Unicode-3.0 licensed crates.
- The standalone benchmark crate `bench/` uses the incumbent `ort`/`ort-sys` crates
  (MIT OR Apache-2.0), which download their own ONNX Runtime binaries; `bench-c/`
  depends on `st-zrt` itself.
