# Contributing to st-zrt

Thanks for improving `st-zrt`. This project wraps ONNX Runtime through generated unsafe
FFI, so correctness of the native boundary outranks convenience or micro-optimization.

## Prerequisites

- Rust **1.85 or newer** (MSRV, edition 2024). `rust-toolchain.toml` selects a stable
  channel with rustfmt/clippy automatically.
- The build downloads a SHA-pinned ONNX Runtime 1.27.0 archive on first use, or set
  `ST_ZRT_ORT_PATH` to a pre-extracted distribution containing `include/` and `lib/`
  (required on Windows; the build script extracts `.tgz` archives only).
- Native tests need the downloaded `lib/` directory on the loader path
  (`LD_LIBRARY_PATH` on Linux) because the sys crate rpaths only its own units.
- CUDA work additionally needs the `cuda` feature, a CUDA 13.x toolkit
  (`ST_ZRT_CUDA13_PATH` → `CUDA_PATH` → `/opt/cuda`), and cuDNN 9, on Linux x86_64.
- `cargo-deny` 0.19.x or 0.20.x for the local dependency-policy gates, matching the pinned
  `cargo-deny-action` container used in CI.

## Development loop

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p st-zrt            # set RUST_TEST_THREADS=1: ORT allows one default LoggingManager
scripts/release-check.sh local  # full non-mutating local gate
```

Notes:

- `RUST_TEST_THREADS=1` serializes tests that create ONNX Runtime environments.
- The feature matrix (no-default-features combinations for `st-zrt-sys` and `st-zrt`)
  is enforced by CI and `scripts/release-check.sh`; keep changes inside MSRV 1.85.
- Changes to the API 27 bindings must regenerate byte-for-byte
  (`scripts/check-generated-bindings.sh`).
- CUDA lanes/graphs: test with fresh changing inputs, capture/replay ordering, and
  real hardware where possible; follow `docs/v0.3-release-checklist.md`.
- The benchmark crates `bench/` (incumbent `ort`) and `bench-c/` (`st-zrt`) are
  standalone workspaces with their own lockfiles; keep their dependency policy green.

## Release boundaries

Commits, tags, and publication are explicit approval boundaries. Do not push, tag, or
publish; the tag-bound gates in `scripts/release-check.sh`
(`pre-sys-publish`, `post-sys-publish`) document the reviewed sequence.
See `docs/v0.3-release-checklist.md`.

Security issues go through [SECURITY.md](SECURITY.md), never public issues.
