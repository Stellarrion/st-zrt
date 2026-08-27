#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
# Build the lifecycle test first so this script also works from a fresh target directory.
cargo test -p st-zrt --locked --features cuda --test cuda_ep --no-run
ORT="${ST_ZRT_ORT_PATH:-$(find "$TARGET_DIR/debug/build" -path '*/out/onnxruntime' -type d -exec test -f '{}/lib/libonnxruntime_providers_cuda.so' ';' -print -quit)}"
[[ -n "$ORT" ]] || { echo "CUDA ONNX Runtime extraction not found after building the CUDA test" >&2; exit 1; }
ORT="$(realpath "$ORT")"
export ST_ZRT_ORT_PATH="$ORT"
export LD_LIBRARY_PATH="$ORT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ST_ZRT_DEVICE_OUTPUT_ITERATIONS="${ST_ZRT_DEVICE_OUTPUT_ITERATIONS:-10000}"
cargo run --locked --release --manifest-path bench-c/Cargo.toml --features cuda --example device_output_no_sync
cycles="${ST_ZRT_LIFECYCLE_CYCLES:-100}"
for ((cycle=1; cycle<=cycles; cycle++)); do
  cargo test -q -p st-zrt --locked --features cuda --test cuda_ep \
    cuda_graph_device_output_gpu_chain_owns_lane_until_downstream_completion -- --test-threads=1
  if (( cycle % 10 == 0 )); then echo "completed lifecycle cycle $cycle/$cycles"; fi
done
