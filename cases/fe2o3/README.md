# Wave-shuffle participation gap

This case translates the causal control from quant trial r002 into fe2o3's
semantic Kernel IR. The condition is computed from `LaneId & 1`, so only even
lanes execute the branch-local shuffle while requesting their odd neighbor's
value (`lane ^ 1` keeps the source index valid for every lane). The IR node
still declares `active_lanes = 64` and uniform subgroup
convergence. The separate repaired module executes the shuffle before the
lane-varying branch; only publication would be predicated in the original HIP
algorithm.

Run `./cases/fe2o3/run-wave-ir-gap.sh` from this repository. The script tests
an isolated, clean worktree at fe2o3 commit
`182e989a279454346953b990ce14aceeb1ebbb70` and does not modify the live
fe2o3 checkout. It requires that commit to exist in `FE2O3_REPO` (default:
the adjacent `fe2o3` checkout).

Expected observations:

- Explicit `active_lanes = 32` is rejected by `verify_module`.
- The uniform-call repair is accepted.
- The branch-local shuffle with *false* full-wave metadata is also accepted.

The last result is a **coverage gap at the IR verification boundary**, not a
claim that an authenticated Rust frontend accepts this program. The IR's
`Convergence` documentation assigns truth of the claim to uniformity analysis
or a proof artifact. This case does not test source-to-IR authentication,
gfx950 lowering, machine instruction semantics, or hardware execution.
Those require separately pinned evidence before fe2o3 can be credited with
catching the observed agent error end to end.
