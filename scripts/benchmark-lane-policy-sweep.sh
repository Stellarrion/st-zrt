#!/usr/bin/env bash
set -euo pipefail

model="${ZRT_BENCH_MODEL:-relay16m}"
iters="${ZRT_BENCH_ITERS:-200}"
warmups="${ZRT_BENCH_WARMUPS:-20}"
policies="${ZRT_LANE_POLICIES:-vec prefaulted aligned aligned-prefaulted hugepage hugepage-prefaulted aligned-hugepage-prefaulted auto}"
alignments="${ZRT_LANE_ALIGNMENTS:-64 128 4096 2097152}"

echo "policy,alignment,runtime,model,mode,warmups,iters,load_ms,warmup_ms,avg_us,rss_start_kb,rss_loaded_kb,rss_done_kb,hwm_kb,checksum"
for policy in $policies; do
  if [[ "$policy" == "aligned" || "$policy" == "aligned-prefaulted" || "$policy" == "aligned-hugepage-prefaulted" || "$policy" == "aligned-mlocked" || "$policy" == "aligned-mlocked-prefaulted" || "$policy" == "aligned-hugepage-mlocked-prefaulted" ]]; then
    for alignment in $alignments; do
      row=$(ZRT_BENCH_WARMUPS="$warmups" ZRT_LANE_POLICY="$policy" ZRT_LANE_ALIGNMENT="$alignment" \
        cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" lane)
      echo "$policy,$alignment,$row"
    done
  else
    row=$(ZRT_BENCH_WARMUPS="$warmups" ZRT_LANE_POLICY="$policy" \
      cargo run --quiet --release --manifest-path bench-c/Cargo.toml --example mem_probe -- "$model" "$iters" lane)
    echo "$policy,,$row"
  fi
done
