# Quantization wave-participation case study

**Population:** three independent, frozen no-feedback HIP agent runs on one
MXFP4 Even task. This is one challenge type, not three types. All three used
Codex CLI `0.157.0`, `gpt-5.5` medium, a fresh bubblewrap workspace, the same
prompt/starter, and a 900-second wall limit. A controllable sampling seed was
not available. The trusted correctness replay used pinned AITER on one
MI350X/gfx950, seven committed cases including four withheld, and no timing.

| Run | Final correctness | Agent trajectory | Mechanism evidence |
| --- | --- | --- | --- |
| [r001](../results/agent-trials/quant-mxfp4-even-r001/README.md) | 7/7 | Submitted a correct-on-matrix HIP implementation. | No failure observed; performance unmeasured. |
| [r002](../results/agent-trials/quant-mxfp4-even-r002/README.md) | 1/7; six packed-output mismatches, scales correct | Final source calls `__shfl_down(code,1)` only in an even-lane branch, asking for odd-lane values that do not participate. | A separate one-line analyst repair moves the shuffle before the branch and passes 7/7, including withheld cases. This supports a source-lane participation root cause. |
| [r003](../results/agent-trials/quant-mxfp4-even-r003/README.md) | 1/7; six packed-output mismatches, scales correct | Its first compiled shared-memory source (snapshot 2) passes 7/7. The agent then replaces reduction/packing with shuffles for speed. Final source calls `__shfl` under `lane < 16` but requests values from lanes 16-31. | Trusted replay confirms the regression occurred between source snapshots 2 and 3. The precise contribution of each changed instruction has not been isolated by a one-line repair. |

**Classification:** kernel-source mechanism `subgroup source-lane
participation`; symptom `silent wrong packed output`; affected layer `HIP
kernel`; likely transferable GPU concept, subject to a faithful Rust lowering.
The r002 causal control and r003 within-run comparison are analyst replays,
not extra agent trials. Both failures escaped compile-only checks. These runs
provided no visible correctness feedback, so they do not measure escape from
a realistic visible-test suite. There was no performance comparison, so the
agent's stated speed motivation is not evidence that its rewrite was faster.

The fe2o3 [wave-operation target](../results/2026-09-25-fe2o3-wave-shuffle-target.md)
describes a source-to-IR proof obligation: every reading lane's source lane
must participate in the same collective. Its current declared-active-lane
IR gate is promising, but neither an authenticated gfx950 Rust source check
nor arbitrary HIP detection has been demonstrated. This case says nothing
about one wave per SIMD, cross-workgroup synchronization, or GPU forward
progress. Such properties require different tasks and hardware/semantics
assumptions.

**Counting limit:** two failing final submissions among three fresh runs of
one task is an early mechanism signal, not a frequency estimate for AITER,
gfx950 kernels, megakernels, or agent-written HIP generally. Performance and
joint parity remain pending; the type has not been admitted to a complete
parity cohort. Do not pool the unscored preview03 or the two analyst controls
into the denominator.
