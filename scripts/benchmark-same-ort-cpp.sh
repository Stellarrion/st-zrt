#!/usr/bin/env bash
set -euo pipefail

iters="${ZRT_BENCH_ITERS:-200}"
warmups="${ZRT_BENCH_WARMUPS:-20}"
models="${ZRT_BENCH_MODELS:-relay4m relay16m}"

echo "runtime,model,mode,warmups,iters,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
for model in $models; do
  ZRT_BENCH_WARMUPS="$warmups" \
    cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" lane
  ZRT_BENCH_WARMUPS="$warmups" \
    cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" allocated-output
done

echo
for model in $models; do
  case "$model" in
    relay4m)
      bench-cpp/run_ort_cpp_expert.sh relay4m "$iters" 1
      ;;
    relay16m)
      bench-cpp/run_ort_cpp_expert.sh relay16m "$iters" 1
      ;;
    mnist)
      bench-cpp/run_ort_cpp_expert.sh mnist "$iters" 1
      ;;
  esac
done
