#!/usr/bin/env bash
set -euo pipefail

models="${ZRT_MMAP_INIT_MODELS:-256k 4m 16m}"
iters="${ZRT_BENCH_ITERS:-100}"
warmups="${ZRT_BENCH_WARMUPS:-20}"
mode="${ZRT_MMAP_INIT_MODE:-all}"

first=1
for model in $models; do
  if [[ "$mode" == "all" ]]; then
    modes="embedded external vec mmap"
  else
    modes="$mode"
  fi
  for one_mode in $modes; do
    out=$(ZRT_BENCH_WARMUPS="$warmups" cargo run --quiet --release \
      --manifest-path bench-c/Cargo.toml \
      --example mmap_initializer_probe -- "$model" "$one_mode" "$iters")
    if [[ "$first" == "1" ]]; then
      echo "$out"
      first=0
    else
      echo "$out" | tail -n +2
    fi
  done
done
