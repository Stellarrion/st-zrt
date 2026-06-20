#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pre-sys-publish}"

workspace_version() {
  sed -n '/^\[workspace.package\]/,/^\[/{s/^version = "\(.*\)"/\1/p}' Cargo.toml
}

release_tag() {
  printf '%s\n' "${ST_ZRT_RELEASE_TAG:-st-zrt-v$(workspace_version)}"
}

require_clean_release_tag() {
  local tag
  tag="$(release_tag)"

  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "release check requires a clean worktree" >&2
    exit 1
  fi

  if ! git tag --points-at HEAD | grep -Fxq "$tag"; then
    echo "release check must run from tag '$tag' at HEAD" >&2
    echo "override with ST_ZRT_RELEASE_TAG=<tag> when cutting a non-default release tag" >&2
    exit 1
  fi
}

case "$mode" in
  pre-sys-publish)
    require_clean_release_tag
    cargo test -p st-zrt-sys
    cargo test -p st-zrt --tests
    cargo check -p st-zrt --all-features
    cargo clippy -p st-zrt --all-targets --all-features -- -D warnings
    RUSTDOCFLAGS=-Dwarnings cargo doc -p st-zrt --no-deps
    cargo test -p st-zrt --doc
    cargo test -p st-zrt --examples
    cargo test -p st-zrt --features ep --example ep_config
    cargo test -p st-zrt --features cuda --example cuda_inference --no-run
    cargo test -p st-zrt --features custom-ops --example custom_op
    cargo package -p st-zrt-sys --offline
    cargo package -p st-zrt --list --offline
    ;;
  post-sys-publish)
    require_clean_release_tag
    cargo package -p st-zrt
    ;;
  *)
    echo "usage: $0 [pre-sys-publish|post-sys-publish]" >&2
    exit 2
    ;;
esac
