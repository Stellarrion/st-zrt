# Security Policy

`st-zrt` is a Rust wrapper whose guarantees rest on unsafe FFI code in `st-zrt-sys`
(generated bindings plus acquisition/link logic) against the ONNX Runtime 1.27.0 C API.
Soundness bugs in that boundary — use-after-free or double-release of native handles,
thread-safety violations, unsound `unsafe` blocks reachable from safe APIs — are treated
as security issues.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting for this repository:
**Security → "Report a vulnerability"**, or
<https://github.com/Stellarrion/st-zrt/security/advisories/new>.

- Do not open a public issue, PR, or discussion for a suspected vulnerability.
- There is no dedicated security email address; private GitHub reporting is the only
  supported channel.
- Please include: the crate and feature set involved, how ONNX Runtime was acquired
  (build-script download vs `ST_ZRT_ORT_PATH`), the platform and toolchain version, and
  a minimal reproducer if possible.

## Scope

**In scope:** the `st-zrt` and `st-zrt-sys` source in this repository, including the
unsafe FFI boundary, build-script download/extraction logic, and CI definitions.

**Out of scope:**

- ONNX Runtime itself — report problems in the native library upstream at
  <https://github.com/microsoft/onnxruntime>.
- CUDA, cuDNN, or other system libraries supplied by the user.
- Model files, serving frameworks built on top of `st-zrt`, or misconfiguration of
  caller-owned buffers.

## Supported versions and platforms

- The current published line is `st-zrt` 0.4.0 / `st-zrt-sys` 1.29.0 (publication is
  automated from the default branch). The previous line, `st-zrt` 0.3.0 /
  `st-zrt-sys` 1.27.1, remains on crates.io.
- Linux x86_64 is the only platform with native link and test coverage (CI). Linux
  aarch64 and Windows x64 are compile-only in CI; macOS arm64 builds download the ORT
  archive but have no automated coverage; macOS x86_64 is not supported by the ONNX
  Runtime 1.29.0 archive set. See the platform support table in SUPPORT.md.
- Reports against other platforms are accepted as compile-only findings.

Fixes are released through merge-reviewed changes to the default branch: a version
bump that lands there triggers the publish workflow once CI passes. The tag-bound
release gates remain in `scripts/release-check.sh`.
