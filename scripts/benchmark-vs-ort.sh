#!/usr/bin/env bash
set -euo pipefail

iters="${ZRT_BENCH_ITERS:-30}"
models="${ZRT_BENCH_MODELS:-mnist relay4m relay16m hf_resnet50}"

echo "runtime,model,mode,warmups,iters,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
for model in $models; do
  cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" lane
  case "$model" in
    relay4m|relay16m)
      cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" allocated-output
      cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" allocated-io
      cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" device-output
      ;;
  esac
  cargo run --quiet --release --manifest-path bench/Cargo.toml --example mem_probe -- "$model" "$iters"
done

if [[ "${ZRT_BENCH_CRITERION:-0}" == "1" ]]; then
  cargo bench --manifest-path bench-c/Cargo.toml --bench inference -- --warm-up-time 1 --measurement-time 2 --sample-size 10
  cargo bench --manifest-path bench/Cargo.toml --bench inference -- --warm-up-time 1 --measurement-time 2 --sample-size 10
  cargo bench --manifest-path bench-c/Cargo.toml --bench large -- --warm-up-time 1 --measurement-time 2 --sample-size 10
  cargo bench --manifest-path bench/Cargo.toml --bench relay_ort -- --warm-up-time 1 --measurement-time 2 --sample-size 10
  cargo bench --manifest-path bench-c/Cargo.toml --bench hf_resnet -- --warm-up-time 1 --measurement-time 2 --sample-size 10
  cargo bench --manifest-path bench/Cargo.toml --bench hf_resnet -- --warm-up-time 1 --measurement-time 2 --sample-size 10
fi
