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
claim that any submission has passed. The current GPU contention prevents a
valid timing comparison. Until an uncontended replay completes, captured
trials have no joint-parity outcome and must not be reported as completed
performance experiments.
