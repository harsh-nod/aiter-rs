# First observed agent failure: wave-shuffle participation

Evidence: [isolated agent trajectory](agent-previews/quant-mxfp4-even-preview03/README.md)
and [reproduced visible-case result](2026-09-25-quant-mxfp4-preview03-visible.md).
The agent put `__shfl_down(code, 1, 64)` inside an even-lane-only branch.
The odd source lanes did not execute the shuffle; every high FP4 nibble was
zero. A separate analyst edit moving only the shuffle before the branch
made all three public cases byte-exact. This is an observed agent-authored
wave-participation error, but the run was **unscored** and supplies no
frequency estimate. It is not an AITER bug.

## What a verifier would need to check

For each dynamic wave-shuffle call, establish that the source lane for every
reading lane participates in the same wave operation at that program point.
In the minimal example, a 64-lane wave calls shuffle-down only on even
lanes, yet each even lane reads the next odd lane. Reject that version;
accept the version where all lanes shuffle and only even lanes store.
The check must follow control flow to the actual execution mask; a
caller-supplied or stale mask is not sufficient evidence. This property is
about lane participation, not multi-wave workgroup scheduling or forward
progress.

At local fe2o3 revision `182e989a279454346953b990ce14aceeb1ebbb70`,
`crates/fe2o3-kernel-ir/src/verify.rs` rejects a wave operation whose
declared `active_lanes` differs from its wave width and requires a uniform
subgroup convergence claim (`verify_wave`, around line 1435). The relevant
`crates/fe2o3-kernel-ir/tests/wave_operations.rs` mutation already rejects
`active_lanes=32` for a wave64 operation. This is a useful existing IR gate,
**provided** the source-to-IR path authenticates the actual conditional
execution mask. An IR node merely declaring 64 active lanes would not by
itself explain the observed HIP control flow.

The documented authenticated Rust-facing Wave64 collective path is scoped
to exact `gfx942:xnack-` and remains partial through finalization
(`docs/gfx942-wave-lds-v2.md` in that fe2o3 revision). The current evidence
does not show an admitted `gfx950` Rust source proof, a proof for arbitrary
HIP shuffle code, or machine-instruction refinement. The next fe2o3 test
should therefore be a pair of Rust-source mutations that reaches an
authenticated wave operation: a divergent shuffle equivalent must fail
before IR admission, and a uniform shuffle with predicated stores must pass.
Record the source, IR, compiler, and hardware evidence separately.

**Priority signal:** strong mechanism relevance to fe2o3's wave model;
unknown incidence and unknown current end-to-end detection. Do not rank its
frequency above other bug classes from this single exploratory attempt.
