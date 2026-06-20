#!/usr/bin/env bash
set -euo pipefail

models="${ZRT_BATCH_MODELS:-mnist hf_resnet50}"
iters="${ZRT_BENCH_ITERS:-20}"
sizes="${ZRT_BATCH_SIZES:-1 2 4 8}"
eps="${ZRT_BATCH_EPS:-cpu}"
intra_threads="${ZRT_INTRA_THREADS_LIST:-1}"
inter_threads="${ZRT_INTER_THREADS_LIST:-}"
execution_modes="${ZRT_EXECUTION_MODES:-default sequential parallel}"
feature_args=()

if [[ "${ZRT_BENCH_EP_FEATURES:-0}" == "1" ]]; then
  feature_args+=(--features ep)
fi
if [[ "${ZRT_BENCH_CUDA_FEATURES:-0}" == "1" ]]; then
  feature_args+=(--features cuda)
fi

for model in $models; do
  for ep in $eps; do
    for intra in $intra_threads; do
      if [[ -n "$inter_threads" ]]; then
        inter_values="$inter_threads"
      else
        inter_values="__default__"
      fi
      for inter in $inter_values; do
        for mode in $execution_modes; do
          export ZRT_BATCH_SIZES="$sizes"
          export ZRT_EP="$ep"
          export ZRT_INTRA_THREADS="$intra"
          export ZRT_EXECUTION_MODE="$mode"
          if [[ "$inter" == "__default__" ]]; then
            unset ZRT_INTER_THREADS
          else
            export ZRT_INTER_THREADS="$inter"
          fi
          cargo run --quiet --release --manifest-path bench-c/Cargo.toml \
            "${feature_args[@]}" \
            --example batch_probe -- "$model" "$iters"
        done
      done
    done
  done
done
