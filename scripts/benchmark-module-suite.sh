#!/usr/bin/env bash
set -euo pipefail

# Focused module/functionality benchmark suite for st-zrt's wrapper/runtime hot paths.
#
# CPU/default coverage:
# - tensor metadata/data access
# - router/channel overhead
# - static/dynamic runtime dispatch
# - tail latency
# - dynamic bucket churn/eviction
# - batch and allocator sweeps
#
# Optional CUDA coverage:
#   ST_ZRT_BENCH_CUDA=1 scripts/benchmark-module-suite.sh

cargo bench --manifest-path bench-c/Cargo.toml --bench tensor_access
cargo bench --manifest-path bench-c/Cargo.toml --bench router_overhead
cargo bench --manifest-path bench-c/Cargo.toml --bench runtime_shapes
cargo bench --manifest-path bench-c/Cargo.toml --bench tail_latency
cargo bench --manifest-path bench-c/Cargo.toml --bench dynamic_bucket_churn

scripts/benchmark-batching.sh
scripts/benchmark-allocator-matrix.sh

if [[ "${ST_ZRT_BENCH_CUDA:-0}" == "1" ]]; then
  cargo bench --manifest-path bench-c/Cargo.toml --features cuda --bench cuda_graph_gte
  ST_ZRT_CHURN_CUDA=1 cargo bench --manifest-path bench-c/Cargo.toml --features cuda --bench dynamic_bucket_churn
fi
