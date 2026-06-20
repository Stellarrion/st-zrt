#!/usr/bin/env bash
set -euo pipefail

cpu_list="${ZRT_BENCH_CPU_LIST:-0}"
iters="${ZRT_BENCH_ITERS:-200}"
warmups="${ZRT_BENCH_WARMUPS:-20}"
models="${ZRT_BENCH_MODELS:-relay4m relay16m}"

if command -v taskset >/dev/null 2>&1; then
  runner=(taskset -c "$cpu_list")
else
  echo "warning: taskset not found; running without CPU affinity" >&2
  runner=()
fi

echo "runtime,model,mode,warmups,iters,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
for model in $models; do
  ZRT_BENCH_WARMUPS="$warmups" "${runner[@]}" \
    cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" lane
  ZRT_BENCH_WARMUPS="$warmups" "${runner[@]}" \
    cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" allocated-output
  ZRT_BENCH_WARMUPS="$warmups" "${runner[@]}" \
    cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" device-output
  ZRT_BENCH_WARMUPS="$warmups" "${runner[@]}" \
    cargo run --quiet --release --manifest-path bench/Cargo.toml --example mem_probe -- "$model" "$iters"
done
