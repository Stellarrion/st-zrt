#!/usr/bin/env bash
set -euo pipefail

# ORT permits only one default LoggingManager instance at a time; serialize native environment tests.
export RUST_TEST_THREADS="${RUST_TEST_THREADS:-1}"
MSRV_TOOLCHAIN="${ST_ZRT_MSRV_TOOLCHAIN:-1.85.1}"

require_local_tooling() {
  # Match the pinned cargo-deny-action container: accept cargo-deny 0.19.x or 0.20.x locally.
  command -v cargo-deny >/dev/null || {
    echo "release check requires cargo-deny 0.19.x or 0.20.x" >&2
    exit 1
  }
  cargo deny --version | grep -Eq '^cargo-deny 0\.(19|20)\.' || {
    echo "release check requires cargo-deny 0.19.x or 0.20.x" >&2
    exit 1
  }
  rustup run "$MSRV_TOOLCHAIN" rustc --version >/dev/null 2>&1 || {
    echo "release check requires Rust toolchain $MSRV_TOOLCHAIN" >&2
    exit 1
  }
}

check_msrv() {
  DOCS_RS=1 cargo "+$MSRV_TOOLCHAIN" check --workspace --all-targets --all-features --locked
}

check_dependency_policy() {
  # The workspace graph plus both auxiliary benchmark graphs (cargo-deny resolves each
  # manifest's own lockfile). --all-features matches the pinned cargo-deny-action, which
  # checks all features in CI.
  cargo deny --all-features check
  cargo deny --manifest-path bench/Cargo.toml --all-features check
  cargo deny --manifest-path bench-c/Cargo.toml --all-features check
}

check_lockfile_sources() {
  local lock source
  for lock in Cargo.lock bench/Cargo.lock bench-c/Cargo.lock; do
    if grep -q '^\[\[patch\.unused\]\]' "$lock"; then
      echo "release check rejects unused patch entries in $lock" >&2
      exit 1
    fi
    while IFS= read -r source; do
      if [[ "$source" != 'source = "registry+https://github.com/rust-lang/crates.io-index"' ]]; then
        echo "release check rejects non-crates.io source in $lock: $source" >&2
        exit 1
      fi
    done < <(grep '^source = ' "$lock" || true)
  done
}

check_feature_matrix() {
  cargo check -p st-zrt-sys --no-default-features --locked
  RUSTDOCFLAGS=-Dwarnings cargo doc -p st-zrt-sys --no-deps --no-default-features --locked
  for features in ep custom-ops model-editor cuda training ep,model-editor custom-ops,model-editor; do
    cargo check -p st-zrt-sys --no-default-features --features "$features" --locked
    RUSTDOCFLAGS=-Dwarnings cargo doc -p st-zrt-sys --no-deps --no-default-features --features "$features" --locked
  done
  cargo check -p st-zrt --no-default-features --locked
  cargo test -p st-zrt --tests --no-default-features --locked
  RUSTDOCFLAGS=-Dwarnings cargo doc -p st-zrt --no-deps --no-default-features --locked
  cargo test -p st-zrt --doc --no-default-features --locked
  for features in half serde ep custom-ops model-editor ep,serde custom-ops,model-editor half,serde,ep,custom-ops,model-editor; do
    cargo check -p st-zrt --no-default-features --features "$features" --locked
    RUSTDOCFLAGS=-Dwarnings cargo doc -p st-zrt --no-deps --no-default-features --features "$features" --locked
    cargo test -p st-zrt --doc --no-default-features --features "$features" --locked
  done
}

check_package_licenses() {
  local files
  files="$(cargo package -p st-zrt-sys --list --allow-dirty --locked)"
  grep -Fxq LICENSE <<<"$files"
  files="$(cargo package -p st-zrt --list --allow-dirty --locked)"
  grep -Fxq LICENSE <<<"$files"
}

check_bench_workspace() {
  # Compile-only gate for the standalone A/B harness graph (incumbent `ort` crate):
  # DOCS_RS makes ort-sys's build script skip the prebuilt ORT download/link, so no
  # network fetch or native loader is involved. bench-c keeps its full native
  # download/link check below.
  DOCS_RS=1 cargo check --manifest-path bench/Cargo.toml --all-targets --locked
  cargo check --manifest-path bench-c/Cargo.toml --all-targets --features cuda --locked
}

check_generated_bindings() {
  scripts/check-generated-bindings.sh
}

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

  if [[ -n "$(git status --porcelain)" ]]; then
    echo "release check requires a clean worktree, including no untracked release files" >&2
    git status --short >&2
    exit 1
  fi

  local tags
  tags="$(git tag --points-at HEAD)"
  if ! grep -Fxq "$tag" <<<"$tags"; then
    echo "release check must run from tag '$tag' at HEAD" >&2
    echo "override with ST_ZRT_RELEASE_TAG=<tag> when cutting a non-default release tag" >&2
    exit 1
  fi
}

case "$mode" in
  local)
    require_local_tooling
    check_msrv
    check_dependency_policy
    check_lockfile_sources
    check_feature_matrix
    check_generated_bindings
    check_package_licenses
    check_bench_workspace
    cargo test -p st-zrt-sys --locked
    cargo test -p st-zrt-sys --all-features --locked
    cargo test -p st-zrt --tests --locked
    cargo check -p st-zrt --all-features --locked
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
    RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --all-features --no-deps --locked
    cargo test -p st-zrt --doc --locked
    cargo test -p st-zrt --examples --locked
    cargo test -p st-zrt --features ep --example ep_config --locked
    cargo test -p st-zrt --features cuda --example cuda_inference --no-run --locked
    cargo test -p st-zrt --features custom-ops --example custom_op --locked
    cargo package -p st-zrt-sys --allow-dirty --locked --offline
    ;;
  pre-sys-publish)
    require_clean_release_tag
    require_local_tooling
    check_msrv
    check_dependency_policy
    check_lockfile_sources
    check_feature_matrix
    check_generated_bindings
    check_package_licenses
    check_bench_workspace
    cargo test -p st-zrt-sys --locked
    cargo test -p st-zrt-sys --all-features --locked
    cargo test -p st-zrt --tests --locked
    cargo check -p st-zrt --all-features --locked
    cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
    RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --all-features --no-deps --locked
    cargo test -p st-zrt --doc --locked
    cargo test -p st-zrt --examples --locked
    cargo test -p st-zrt --features ep --example ep_config --locked
    cargo test -p st-zrt --features cuda --example cuda_inference --no-run --locked
    cargo test -p st-zrt --features custom-ops --example custom_op --locked
    cargo package -p st-zrt-sys --locked --offline
    cargo publish -p st-zrt-sys --dry-run --locked
    cargo package -p st-zrt --list --locked --offline
    ;;
  post-sys-publish)
    require_clean_release_tag
    cargo publish -p st-zrt --dry-run --locked
    ;;
  *)
    echo "usage: $0 [local|pre-sys-publish|post-sys-publish]" >&2
    exit 2
    ;;
esac
