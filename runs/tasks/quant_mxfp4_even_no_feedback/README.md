# Frozen MXFP4 Even no-feedback task

This task uses the [public operator contract](../../../tasks/quant_mxfp4_even/spec.md)
and [scorer spec](../../../references/quant_mxfp4_gfx950.json) at pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. The private seven-case
matrix (three public, four withheld) was committed by SHA256 before agent
launch. Pinned AITER and the non-agent HIP reference passed it on the
host-attested MI350X; see the [baseline admission result](../../../results/2026-09-25-quant-mxfp4-baseline-admission.md).

The agent receives only `prompt.md` and the nonfunctional HIP starter in a
fresh bubblewrap workspace, with no GPU or SSH access and no supplied tests.
The model, CLI version, and wall limit are recorded per run. The fixed
performance rule is at most 1.05 times pinned AITER median latency in **each**
of the medium and large workload buckets. Correctness and performance will
be scored separately from verified source snapshots on the same MI350X.

`scored_eligible=true` authorizes an independent trial capture; it is not a
claim that any submission has passed. GPU contention blocked timing on
September 25; [uncontended r001 replays](../../../results/2026-09-28-quant-r001-performance-replays.md)
later produced two joint passes and one noise-gate failure for the same
submission. Do not count those measurements as independent agent trials or
as stable type-level parity admission.
