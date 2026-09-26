# gfx950 packed BF16 GDR decode baseline correctness admission

Status: **AITER baseline correctness pass only**. This is not an agent run,
a HIP candidate comparison, or a performance-parity result.

- Hardware: one host-attested AMD Instinct MI350X (`gfx950`, PCI model
  `0x75a0`), cross-checked against the container's architecture and PCI model.
- AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public spec raw-file SHA256:
  `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
  The scorer's filename-inclusive spec provenance hash is
  `104b36e546879566b75a4ea1eeb7ca56512da78562805854990827c5dd2875e0`.
- Private case-manifest SHA256 commitment:
  `4bb61ab49dab61bff62e124cc5554ec92af3a846f41341f77229f58e38e5c9d0`.
- Correctness: pinned AITER passed the independent CPU oracle on **8/8**
  cases, including **4 withheld** cases, across **28/28** sequential invocation
  steps. Output and updated state passed `rtol=1e-2, atol=1e-3` comparisons;
  untouched state slots and invalid-row positive zeros matched bitwise. Input
  tensors, padded lanes, output guards, state boundary guards, and state-slot
  gap guards remained intact where exercised.
- Scorer status: `candidate_kind=aiter_baseline_admission`,
  `correctness=pass`, `withheld_cases_evaluated=true`,
  `performance=not_run` (`correctness_only`), `joint_pass=false` because there
  was no candidate-performance stage.

The run was correctness-only; no latency was measured. Private inputs, case
identifiers, and the raw result remain outside the repository. Passing this
matrix admits AITER as a correctness baseline for the pilot, but a vetted HIP
baseline and uncontended performance comparison are still required before
agent scoring. It does not establish full operator coverage or contribute to
the agent-error incidence denominator.
