#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/../.." && pwd)"
out="$here/target"
mkdir -p "$out"

cc -O2 -fPIC -shared "$here/malloc_counter.c" -ldl -o "$out/libzrt_malloc_counter.so"

cargo build --manifest-path "$repo/bench-c/Cargo.toml" --example native_alloc_lane --release

ort_lib="$(find "$repo/bench-c/target/release/build" -path '*/out/onnxruntime/lib/libonnxruntime.so' -print -quit 2>/dev/null || true)"
if [[ -z "$ort_lib" ]]; then
  echo "Could not find st-zrt-sys ONNX Runtime library under bench-c/target." >&2
  exit 1
fi
ort_dir="$(dirname "$ort_lib")"

label="${ZRT_LABEL:-mnist}"
iters="${ZRT_ITERS:-10000}"

LD_PRELOAD="$out/libzrt_malloc_counter.so" \
LD_LIBRARY_PATH="$ort_dir:${LD_LIBRARY_PATH:-}" \
ZRT_LABEL="$label" \
ZRT_ITERS="$iters" \
"$repo/bench-c/target/release/examples/native_alloc_lane"
