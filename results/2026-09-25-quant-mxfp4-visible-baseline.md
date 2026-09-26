# gfx950 MXFP4 Even visible baseline

Status: **visible correctness pass only**. No hidden cases, agent candidate,
or performance bucket was scored.

- Hardware: one host-attested AMD Instinct MI350X (`gfx950`, PCI model
  `0x75a0`) on `mi350-2`; container device architecture and PCI model
  matched the host report.
- AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public scorer spec SHA256:
  `94a090d850e4017dcf2bf82f19a41a4149e43c7b355a07a9752e95be117063a2`.
- HIP candidate: non-agent correctness-first reference from
  `references/quant_mxfp4_baseline.hip`, compiled binary SHA256
  `97a5fb2f0e578f0a1be2243d06866fef067c9a67a108ab596ee35b1e5eb258e0`.
- Software: container PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`.
- Cases: `f16_edges`, `f16_medium`, and `bf16_large` all passed. For each,
  AITER and the HIP reference matched the separate CPU oracle byte-for-byte
  for packed FP4 and E8M0 scales; neither changed input, and output guards
  remained intact.
- Scorer status: `correctness=visible_pass`, `performance=not_run`
  (`correctness_only`), `joint_pass=false`, `withheld_cases_evaluated=false`.

The visible GPU was occupied by an unrelated GEMM workload, so latency
would not be interpretable. This result establishes neither AITER-level
performance nor eligibility for a scored agent trial. The private withheld
manifest was not supplied to this run and is not published while trials are
being prepared.
