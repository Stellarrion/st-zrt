#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$here/.." && pwd)"

if [[ -n "${ST_ZRT_ORT_PATH:-}" ]]; then
  ort_root="$ST_ZRT_ORT_PATH"
else
  ort_lib="$(find "$repo/bench-c/target" -path '*/out/onnxruntime/lib/libonnxruntime.so' -print -quit 2>/dev/null || true)"
  if [[ -z "$ort_lib" ]]; then
    echo "Could not find st-zrt-sys ONNX Runtime. Build bench-c first or set ST_ZRT_ORT_PATH." >&2
    exit 1
  fi
  ort_root="$(cd -- "$(dirname -- "$ort_lib")/.." && pwd)"
fi

case_name="${1:-mnist}"
iters="${2:-}"
intra="${3:-0}"

case "$case_name" in
  mnist)
    model="$repo/bench/models/mnist.onnx"
    default_iters=200000
    ;;
  relay4m)
    model="$repo/bench/models/relay_4m.onnx"
    default_iters=30000
    ;;
  relay16m)
    model="$repo/bench/models/relay_16m.onnx"
    default_iters=10000
    ;;
  *)
    echo "unknown case '$case_name'; use mnist, relay4m, or relay16m" >&2
    exit 2
    ;;
esac

if [[ ! -f "$model" ]]; then
  echo "model missing: $model" >&2
  exit 1
fi

if [[ -z "$iters" ]]; then
  iters="$default_iters"
fi

mkdir -p "$here/target"
c++ -O3 -DNDEBUG -std=c++17 \
  -I"$ort_root/include" "$here/ort_cpp_expert.cpp" \
  -L"$ort_root/lib" -Wl,-rpath,"$ort_root/lib" -lonnxruntime \
  -o "$here/target/ort_cpp_expert"

"$here/target/ort_cpp_expert" "$model" "$case_name" "$iters" "$intra"
