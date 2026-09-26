# gfx950 MXFP4 Even agent preview 03

Status: **visible correctness failure, reproduced twice**. This is an
isolated, unscored agent-authored preview, not a performance-parity result or
an observation in the scored agent-error incidence denominator.

- Hardware: one host-attested AMD Instinct MI350X (`gfx950`, PCI model
  `0x75a0`), cross-checked against the container-reported architecture and
  PCI model. Container PyTorch was `2.11.0+gitd0c8b1f`; HIP was `7.2.53211`.
- AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public spec raw-file SHA256:
  `127e3bc54b2fbd312b5b34d3d95526bbc7603e148a854aa3ed5de9ec986a6d61`.
  The scorer's filename-inclusive spec provenance hash was
  `94a090d850e4017dcf2bf82f19a41a4149e43c7b355a07a9752e95be117063a2`.
- Agent final `starter.hip` source SHA256:
  `1a6db9ab07e94a2973e61a9e1275bd274c699a3cd1420d7ed34f8b065b6194d2`.
  It compiled with `hipcc -O3 -shared -fPIC --offload-arch=gfx950`.
- Compiled library raw SHA256:
  `0a208ebe5596783f3963f77ff2170bd6e8730f617eebe8fe3ae4cbf52b35d694`.
  The scorer's filename-inclusive binary provenance hash was
  `83d639f6204bbfb0e91cfe9aff21aa9f41908f7525d23b13ca28a0e7b0b7a00a`.

| Public case | AITER vs. CPU oracle | Candidate packed FP4 mismatch | Candidate E8M0 scale |
| --- | --- | ---: | --- |
| `f16_edges` | exact | 133 / 144 bytes | exact |
| `f16_medium` | exact | 62,305 / 65,536 bytes | exact |
| `bf16_large` | exact | 249,277 / 262,144 bytes | exact |

Each case left input unchanged, and the output guard checks passed for both
AITER and the candidate. The second full visible-only run reproduced all three
mismatch counts with the same spec and binary hashes. No withheld cases were
supplied. Scorer status was `candidate_kind=agent_preview`,
`correctness=fail`, `performance=not_run` (`correctness_not_passed`),
`joint_pass=false`, and `withheld_cases_evaluated=false`.

## Minimized failure and mechanism

The public `f16_edges` input contains the FP16 pair `1.25, -1.25` in its
first 32-element group. That group's E8M0 scale byte is `127` (scale `1.0`).
The independent oracle and pinned AITER pack the pair as `0xA2`; the candidate
writes `0x02`. Across all 144 bytes in that public case, the candidate's low
nibble matches the oracle and its high nibble is zero. This distinguishes the
failure from inverse-scale or FP4 threshold arithmetic.

In the frozen candidate source, `starter.hip:122-125` calls
`__shfl_down(code, 1, 64)` **inside** an even-lane-only branch, then writes the
returned code into the high nibble. The adjacent odd source lane has not
executed the shuffle. The observed all-zero high nibbles are consistent with
reading an inactive lane; this is a **wave-lane participation / divergent
shuffle** bug, not a packing-order error. A repair would execute the shuffle
with all source lanes participating and branch only for the final byte store.

This is a useful proposed fe2o3 verifier challenge: model the active-lane
precondition of wave shuffles and reject reads from lanes that do not
participate at that program point. The preview does **not** establish that
fe2o3 currently proves this property. Private inputs and raw results remain
outside the repository; no latency or AITER-level performance claim was made.
