# GDR live-feedback preview 001

This is one **unscored**, incidence-ineligible agent session on the frozen
GDR native-seed v2 task. It is not evidence of a scored kernel error or
performance parity. Local capture ran from main
`45a534b19a95f82589910be36d37298766e2d7c4`; the clean remote public
scorer checkout was `7d397f2b4e6698adfd62d14a7bea0c9fc7464df2`.
The trusted GDR runner/scorer pin was
`95a11671df486eebfecd050ab2b5c50dc37e57a9`; those trusted files were
identical across the two checkouts.

Codex CLI `0.159.1`, model `gpt-5.5`, high reasoning, 900-second wall limit,
fresh replicate `live-preview-001`, no sampling seed. It completed in
502.619 seconds with exit 0, nine immutable source snapshots, no snapshot or
broker errors, and exactly three agent-visible public benchmark requests.
The full private capture transport SHA256 was
`322e2b8eb80d57ee7e4620503230e94da60797c857982c9a700cefaf7fd21729`;
remote `validate_capture` confirmed the final tree. The broker event-log
SHA256 was
`8463e486a23f12a372bafac53eb1c3e077adfed9cbe1109b0fe874254ca0d2b9`.

| Request | Source SHA256 | Visible | Public graph ratios (valid / strided / large) |
| --- | --- | --- | --- |
| 1, unchanged seed | `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519` | 6/6 | 0.9984 / 1.0027 / 1.0108, all pass |
| 2, first edit | `f9c26e433a62681101472140e831c490fa03bac5af961e62dcef7f43a5cff3ee` | 6/6 | 1.0103 / 1.0011 / 1.0295, all pass |
| 3, second edit | `8316577ecaf83a2e24f9517f0b717efcd8f06f3e03c4926dff73223af0873bd8` | 6/6 | 1.0508 / 1.0621 / 1.0280, first two fail the 1.05 bucket gate |

All request-2 and request-3 buckets were noise-qualified. For request 3,
the AITER/candidate relative MAD pairs were 0.003022/0.001353 (valid),
0.002374/0.001547 (strided), and 0.014216/0.006118 (large). The valid ratio
missed the 1.05 limit by only 0.000834; report that exact threshold miss,
not a broader claim about stable slowdown. Raw latency samples remain private.

The first edit doubled workgroup warps from four to eight, halved V blocks
from four to two, and moved Q/K BF16 conversion into the load loop. The
second edit restored the original workgroup layout and added aligned 32-bit
gate loads; the third public response shows a performance regression on two
buckets but no visible correctness failure. These source changes are
descriptive, not a proven causal explanation of the timings.

After spending all three requests, the agent produced an **untested final
source** with SHA256
`c5f5ae5569ced69d4991a843b3b57e0bc2916665a7f136f259df40eaf3930145`
and final tree SHA256
`1c15d1300e9670b4c47b8c9d91f25220aa75a5b5d69a9d93648c97c2027f17d5`.
It removed the gate-load change, restored seed Q/K conversion and workgroup
layout, and changed the invalid-index path so tile zero writes all V blocks.
The final source is distinct from every live-scored snapshot. At capture close,
its correctness and performance were unknown; only the separate trusted replay
below evaluated this exact final version.

## Post-agent replay

After credential/source review and a SHA-verified private transfer, the final
snapshot was validated again on the clean remote checkout. The trusted
public stage compiled it to binary SHA256
`5d55923c427d9a4352e4b0b5d664f643662e82d9f057bb6bb5a2803037794399`
and passed **6/6 visible cases**. A separate network-disabled withheld stage
reused that exact binary and passed **4/4 withheld cases**. This is a
nonadversarial correctness observation, not a scored incidence result.

All three final graph buckets were noise-qualified and passed the 1.05
non-inferiority gate: valid 0.996620, strided 1.000364, large 1.025296.
The geomean ratio was 1.007347, so the proposed 0.95 improvement gate did
**not** pass. AITER/candidate relative MAD pairs were 0.001427/0.001785
(valid), 0.001638/0.001273 (strided), and 0.003325/0.005569 (large).
Public raw result SHA256 was
`54c1d7acd726e8441d278ca6287db2836132ecdf9dfd0cdafc91799f3b6681da`;
withheld raw result SHA256 was
`f437ea94a13522ef0fd49e58d6cd4676272376124090ad909657ee681cb5bf5f`.
Raw cases and latency samples remain private. The scorer reported
`exploratory_replay_complete`, `incidence_eligible=false`, and
`agent_parity_claim=false`.

The private raw trace and snapshots were scanned for credential/private-host
patterns and stayed outside the repository. Both the capture and feedback
roots were mode 0700; the copied workspace itself was mode 0755 inside a
non-traversable 0700 run directory. No SSH credential or withheld manifest was
mounted for the agent. Post-agent hidden scoring remains a nonadversarial
same-process confidentiality boundary.
