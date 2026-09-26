# gfx950 MXFP4 Even baseline correctness admission

Status: **baseline correctness pass only**. This is not an agent run or a
performance-parity result.

- Hardware: one host-attested AMD Instinct MI350X (`gfx950`, PCI model
  `0x75a0`), cross-checked against the container's architecture and PCI model.
- AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public spec raw-file SHA256:
  `127e3bc54b2fbd312b5b34d3d95526bbc7603e148a854aa3ed5de9ec986a6d61`.
  The scorer's filename-inclusive spec provenance hash is
  `94a090d850e4017dcf2bf82f19a41a4149e43c7b355a07a9752e95be117063a2`.
- HIP candidate: the non-agent correctness-first reference in
  `references/quant_mxfp4_baseline.hip`.
- Correctness: pinned AITER and the HIP reference each passed all **7**
  oracle comparisons, including **4** withheld cases. Packed FP4 and E8M0
  scales matched the independent CPU oracle byte-for-byte; input remained
  unchanged, and output guards were intact.
- Scorer status: `candidate_kind=reference`, `correctness=pass`,
  `withheld_cases_evaluated=true`, `performance=not_run` (`correctness_only`),
  `joint_pass=false`.

No latency was measured because the GPU was contended. The private inputs,
case identifiers, and raw result remain in runner-owned storage. Passing this
matrix admits the baseline for further task feasibility work; it does not
establish full operator coverage or AITER-level performance and contributes
nothing to the agent-error incidence denominator.
