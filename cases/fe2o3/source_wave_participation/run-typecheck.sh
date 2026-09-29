#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fe2o3_repo=${FE2O3_REPO:-"$(dirname "$root")/fe2o3"}
fe2o3_sha=4fb8ae500e38ad22475861aae7c3b3ef238068f6
scratch=$(mktemp -d)
worktree="$scratch/fe2o3"

cleanup() {
    if [[ -d "$worktree/.git" || -f "$worktree/.git" ]]; then
        git -C "$fe2o3_repo" worktree remove --force "$worktree"
    fi
    rm -rf "$scratch"
}
trap cleanup EXIT

git -C "$fe2o3_repo" cat-file -e "$fe2o3_sha^{commit}"
git -C "$fe2o3_repo" worktree add --quiet --detach "$worktree" "$fe2o3_sha"
printf 'fe2o3 source: %s\n' "$fe2o3_sha"

fixture="$worktree/crates/rustc-codegen-fe2o3/tests/fixtures/memory-v1-compiler/src/bin/gfx942_wave_lds_v1.rs"
manifest="$worktree/crates/rustc-codegen-fe2o3/tests/fixtures/memory-v1-compiler/Cargo.toml"
for variant in bad_divergent_collective repaired_uniform_collective; do
    source="$root/cases/fe2o3/source_wave_participation/$variant.rs"
    printf '%s source sha256: %s\n' "$variant" "$(sha256sum "$source" | cut -d ' ' -f 1)"
    cp "$source" "$fixture"
    (
        cd "$worktree"
        printf 'rustc: %s\n' "$(rustc --version)"
        CARGO_TARGET_DIR="${FE2O3_TARGET_DIR:-$scratch/target}" \
            cargo check --quiet --locked --manifest-path "$manifest" --bin gfx942_wave_lds_v1
    )
    printf '%s: ordinary Rust typecheck accepted\n' "$variant"
done

printf 'No authenticated source-to-KIR or gfx950 claim is made by this check.\n'
