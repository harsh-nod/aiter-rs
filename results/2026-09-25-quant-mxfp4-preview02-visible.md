# gfx950 MXFP4 Even agent preview 02

Status: **visible correctness pass only**. This was an unscored agent preview,
not a parity result or an agent-error incidence observation.

- Hardware: one host-attested AMD Instinct MI350X (`gfx950`, PCI model
  `0x75a0`). The container-reported architecture and PCI model matched the
  host report.
- AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public scorer spec SHA256:
  `94a090d850e4017dcf2bf82f19a41a4149e43c7b355a07a9752e95be117063a2`.
- Agent final HIP source SHA256:
  `a7b1dc3084ce5cd1edb0e0ea01c1730606ab59acc1439ac9e18023bd174840b5`.
  It compiled for `gfx950` with `hipcc -O3 -shared -fPIC --offload-arch=gfx950`.
- Compiled library raw SHA256:
  `d7d687ae6cd95f431e1cf50344a711f3b128f75cfed64ee1b7cd357fea6ed131`.
  The scorer's filename-inclusive binary provenance hash is
  `3c527c0b2512961b2ccefd5820d11c3ccbb08e590246a67be8373291f397a646`.
- Software: container PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`.
- Visible cases: `f16_edges`, `f16_medium`, and `bf16_large` each passed.
  For each case, pinned AITER and the agent HIP candidate matched the
  independent CPU oracle byte-for-byte for packed FP4 and E8M0 scales.
  Neither modified input, and both output guard regions remained intact.
- Scorer status: `candidate_kind=agent_preview`,
  `correctness=visible_pass`, `performance=not_run` (`correctness_only`),
  `joint_pass=false`, `withheld_cases_evaluated=false`.

The private withheld matrix was not supplied. GPU contention prevented a
meaningful latency comparison, so this preview establishes neither full
functionality nor AITER-level performance. It is not included in the scored
agent-run denominator.
