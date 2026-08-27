#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${ZRT_BENCH_OUT:-"$ROOT/target/benchmark-runs/$(date -u +%Y%m%dT%H%M%SZ)"}
BASELINE=${ZRT_BENCH_BASELINE:-}
CPU_SET=${ZRT_BENCH_CPU_SET:-}
GPU=${ZRT_BENCH_GPU:-0}
REQUIRE_EXCLUSIVE_GPU=${ZRT_REQUIRE_EXCLUSIVE_GPU:-0}
mkdir -p "$OUT"

run() {
  printf '+ ' | tee -a "$OUT/commands.log"
  printf '%q ' "$@" | tee -a "$OUT/commands.log"
  printf '\n' | tee -a "$OUT/commands.log"
  "$@"
}
pinned() {
  if [[ -n "$CPU_SET" ]]; then run taskset --cpu-list "$CPU_SET" "$@"; else run "$@"; fi
}
criterion_args=()
if [[ -n "$BASELINE" ]]; then criterion_args+=(--save-baseline "$BASELINE"); fi

{
  date --iso-8601=seconds
  uname -a
  git -C "$ROOT" status --short
  git -C "$ROOT" rev-parse HEAD
  lscpu
  if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    printf 'governor='; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  fi
  command -v rustc >/dev/null && rustc -Vv
  command -v cargo >/dev/null && cargo -V
  command -v nvidia-smi >/dev/null && nvidia-smi -q
} >"$OUT/machine.txt" 2>&1

if [[ "$REQUIRE_EXCLUSIVE_GPU" == 1 ]]; then
  mapfile -t gpu_pids < <(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
  if ((${#gpu_pids[@]} != 0)); then
    printf 'refusing GPU benchmark: device %s has compute PIDs: %s\n' "$GPU" "${gpu_pids[*]}" >&2
    exit 2
  fi
fi

pinned cargo bench --manifest-path "$ROOT/bench-c/Cargo.toml" --bench inference -- "${criterion_args[@]}" 2>&1 | tee "$OUT/inference.txt"
pinned cargo bench --manifest-path "$ROOT/bench-c/Cargo.toml" --bench runtime_shapes -- "${criterion_args[@]}" 2>&1 | tee "$OUT/runtime-shapes.txt"
pinned cargo bench --manifest-path "$ROOT/bench-c/Cargo.toml" --bench large -- "${criterion_args[@]}" 2>&1 | tee "$OUT/large.txt"
pinned cargo bench --manifest-path "$ROOT/bench-c/Cargo.toml" --bench router_overhead -- "${criterion_args[@]}" 2>&1 | tee "$OUT/router.txt"

if [[ ${ZRT_BENCH_CUDA:-0} == 1 ]]; then
  : "${ZRT_GTE_MODEL:?set ZRT_GTE_MODEL for CUDA benchmarking}"
  run env ZRT_GTE_MODEL="$ZRT_GTE_MODEL" cargo bench --manifest-path "$ROOT/bench-c/Cargo.toml" --features cuda --bench cuda_graph_gte 2>&1 | tee "$OUT/cuda-gte.txt"
fi

printf 'benchmark artifacts: %s\n' "$OUT"
