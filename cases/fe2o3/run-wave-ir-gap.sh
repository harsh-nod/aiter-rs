#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=${FE2O3_REPO:-"$here/../../../fe2o3"}
revision=182e989a279454346953b990ce14aceeb1ebbb70
worktree=$(mktemp -d "${TMPDIR:-/tmp}/aiter-rs-fe2o3-wave.XXXXXX")
rmdir "$worktree"

cleanup() {
    git -C "$repo" worktree remove --force "$worktree" 2>/dev/null || true
}
trap cleanup EXIT

git -C "$repo" cat-file -e "$revision^{commit}"
git -C "$repo" worktree add --quiet --detach "$worktree" "$revision"
cp "$here/wave_participation_ir.rs" \
    "$worktree/crates/fe2o3-kernel-ir/tests/aiter_rs_wave_participation.rs"
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"${TMPDIR:-/tmp}/aiter-rs-fe2o3-target"} \
    cargo test --locked \
    --manifest-path "$worktree/crates/fe2o3-kernel-ir/Cargo.toml" \
    --test aiter_rs_wave_participation
