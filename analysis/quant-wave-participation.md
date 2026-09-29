# Quantization wave-participation case study

**Population:** six independent, frozen no-feedback HIP agent runs on one
MXFP4 Even task. This is one challenge type, not six types. Runs r001-r003
used Codex CLI `0.157.0` with `gpt-5.5` medium reasoning; r004-r006 used CLI
`0.159.0` with high reasoning. The CLI change confounds any reasoning-effort
comparison. Every run had a fresh bubblewrap workspace, the same
prompt/starter, and a 900-second wall limit. A controllable sampling seed was
not available. Trusted correctness replay used pinned AITER on one
MI350X/gfx950, seven committed cases including four withheld, without
performance timing.

| Run | Final correctness | Agent trajectory | Mechanism evidence |
| --- | --- | --- | --- |
| [r001](../results/agent-trials/quant-mxfp4-even-r001/README.md) | 7/7 | Submitted a correct-on-matrix HIP implementation. | No correctness failure observed; a later public-only graph control found it slower than AITER. |
| [r002](../results/agent-trials/quant-mxfp4-even-r002/README.md) | 1/7; six packed-output mismatches, scales correct | Final source calls `__shfl_down(code,1)` only in an even-lane branch, asking for odd-lane values that do not participate. | A separate one-line analyst repair moves the shuffle before the branch and passes 7/7, including withheld cases. This supports a source-lane participation root cause. |
| [r003](../results/agent-trials/quant-mxfp4-even-r003/README.md) | 1/7; six packed-output mismatches, scales correct | Its first compiled shared-memory source (snapshot 2) passes 7/7. The agent then replaces reduction/packing with shuffles for speed. Final source calls `__shfl` under `lane < 16` but requests values from lanes 16-31. | Trusted replay confirms the regression occurred between source snapshots 2 and 3. The precise contribution of each changed instruction has not been isolated by a one-line repair. |
| [r004](../results/agent-trials/quant-mxfp4-even-r004/README.md) | 7/7 | Submitted a correct-on-matrix HIP implementation. | No final correctness failure observed. |
| [r005](../results/agent-trials/quant-mxfp4-even-r005/README.md) | 7/7 | Submitted a correct-on-matrix HIP implementation. | No final correctness failure observed. |
| [r006](../results/agent-trials/quant-mxfp4-even-r006/README.md) | 1/7; six packed-output mismatches, scales correct | Final source calls `__shfl_down` only inside an even-lane branch while requesting an odd-lane source. | A separate [one-line analyst repair](../results/analyst-controls/quant-mxfp4-even-r006-shuffle/README.md) moves the shuffle before the branch and passes 7/7, including withheld cases. |

**Classification:** kernel-source mechanism `subgroup source-lane
participation`; symptom `silent wrong packed output`; affected layer `HIP
kernel`; likely transferable GPU concept, subject to a faithful Rust lowering.
The r002/r006 causal controls and r003 within-run comparison are analyst
replays, not extra agent trials. All three final failures escaped compile-only
checks. These runs
provided no visible correctness feedback, so they do not measure escape from
a realistic visible-test suite. The separate public-only r001
[graph control](../references/quant_mxfp4_graph_control.md) found its correct
HIP binary slower than AITER in both buckets; there is no scored performance
comparison for the six-run cohort. The agents' stated speed motivations are
not evidence that their rewrites were faster.

The fe2o3 [wave-operation target](../results/2026-09-25-fe2o3-wave-shuffle-target.md)
describes a source-to-IR proof obligation: every reading lane's source lane
must participate in the same collective. Its current declared-active-lane
IR gate is promising, but neither an authenticated gfx950 Rust source check
nor arbitrary HIP detection has been demonstrated. This case says nothing
about one wave per SIMD, cross-workgroup synchronization, or GPU forward
progress. Such properties require different tasks and hardware/semantics
assumptions.

A later [one-wave HIP control](../results/2026-09-29-hip-wave-shuffle-control.md)
on the same gfx950 class produced 32/32 wrong values with the divergent call
and 0/32 with the uniform-call repair. The corresponding
[fe2o3 IR experiment](../results/2026-09-29-fe2o3-wave-ir-gap.md) shows that
the core verifier rejects an explicit partial-lane claim but accepts a false
full-wave claim in a lane-varying branch. Neither control is another agent
trial or an end-to-end fe2o3 source proof.

**Counting limit:** three failing final submissions among six fresh runs of
one task is an early mechanism signal, not a frequency estimate for AITER,
gfx950 kernels, megakernels, or agent-written HIP generally. Performance and
joint parity remain pending; the type has not been admitted to a complete
parity cohort. Do not pool the unscored preview03 or analyst controls
into the denominator.
