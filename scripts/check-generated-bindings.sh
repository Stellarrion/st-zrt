#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TARGET_DIR="${CARGO_TARGET_DIR:-target}"
header="${ST_ZRT_ORT_HEADER:-}"
if [[ -z "$header" && -n "${ST_ZRT_ORT_PATH:-}" ]]; then
  header="$ST_ZRT_ORT_PATH/include/onnxruntime_c_api.h"
fi
if [[ -z "$header" && -d "$TARGET_DIR/debug/build" ]]; then
  # Stale build dirs from older ORT versions can shadow the current one; take the most
  # recently extracted header. The API-version assertion below still rejects mismatches.
  header="$(find "$TARGET_DIR/debug/build" -path '*/out/onnxruntime/include/onnxruntime_c_api.h' -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
fi
if [[ ! -f "$header" ]]; then
  # Populate a fresh target directory through the crate's checksum-verified normal build path.
  cargo check -p st-zrt-sys --locked
  header="$(find "$TARGET_DIR/debug/build" -path '*/out/onnxruntime/include/onnxruntime_c_api.h' -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
fi
[[ -f "$header" ]] || {
  echo "generated binding check requires an extracted ONNX Runtime 1.29 onnxruntime_c_api.h" >&2
  exit 1
}
grep -Eq '^#define ORT_API_VERSION +29$' "$header" || {
  echo "generated binding check requires an ONNX Runtime API 29 header: $header" >&2
  exit 1
}
tmp="$(mktemp "${TMPDIR:-/tmp}/st-zrt-generated.XXXXXX.rs")"
trap 'rm -f "$tmp"' EXIT
cargo run -p st-zrt-sys-codegen --locked -- "$header" "$tmp"
rustfmt --edition 2021 "$tmp"
if ! cmp -s st-zrt-sys/src/generated.rs "$tmp"; then
  diff -u st-zrt-sys/src/generated.rs "$tmp" || true
  echo "st-zrt-sys/src/generated.rs is stale; regenerate it from the API 29 header and run rustfmt" >&2
  exit 1
fi
echo "generated bindings match ONNX Runtime API 29"
