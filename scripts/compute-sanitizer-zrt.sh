#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Build first so this script also works from a fresh target directory.
cargo test -p st-zrt --locked --features cuda --test cuda_ep --no-run
ORT="${ST_ZRT_ORT_PATH:-$(find "$TARGET_DIR/debug/build" -path '*/out/onnxruntime' -type d -exec test -f '{}/lib/libonnxruntime_providers_cuda.so' ';' -print -quit)}"
[[ -n "$ORT" ]] || { echo "CUDA ONNX Runtime extraction not found after building the CUDA test" >&2; exit 1; }
ORT="$(realpath "$ORT")"
export ST_ZRT_ORT_PATH="$ORT"
export LD_LIBRARY_PATH="$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
BIN="$(find "$TARGET_DIR/debug/deps" -maxdepth 1 -type f -name 'cuda_ep-*' ! -name '*.d' -printf '%T@ %p\n' | sort -nr | sed -n '1s/^[^ ]* //p')"
[[ -x "$BIN" ]] || { echo "cuda_ep test binary not found" >&2; exit 1; }
CS="${COMPUTE_SANITIZER:-/opt/cuda/bin/compute-sanitizer}"
for tool in memcheck racecheck initcheck synccheck; do
  "$CS" --tool "$tool" --error-exitcode 1 --log-file "${TMPDIR:-/tmp}/st-zrt-${tool}.log" \
    "$BIN" gpu_chain --test-threads=1
  echo "$tool: $(tail -1 "${TMPDIR:-/tmp}/st-zrt-${tool}.log")"
done
