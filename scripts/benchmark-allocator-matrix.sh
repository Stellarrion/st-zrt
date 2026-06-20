#!/usr/bin/env bash
set -euo pipefail

models="${ZRT_BENCH_MODELS:-relay4m relay16m}"
modes="${ZRT_ZRT_MODES:-lane allocated-output device-output}"
iters="${ZRT_BENCH_ITERS:-200}"
warmups="${ZRT_BENCH_WARMUPS:-20}"

echo "arena,mem_pattern,runtime,model,mode,warmups,iters,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
for arena in default disabled; do
  for mem_pattern in default disabled; do
    disable_arena=0
    disable_mem_pattern=0
    [[ "$arena" == "disabled" ]] && disable_arena=1
    [[ "$mem_pattern" == "disabled" ]] && disable_mem_pattern=1
    for model in $models; do
      for mode in $modes; do
        row=$(ZRT_BENCH_WARMUPS="$warmups" \
          ZRT_DISABLE_ARENA="$disable_arena" \
          ZRT_DISABLE_MEM_PATTERN="$disable_mem_pattern" \
          cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" "$mode")
        echo "$arena,$mem_pattern,$row"
      done
    done
  done
done
