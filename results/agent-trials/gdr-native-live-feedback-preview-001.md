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
The final source is distinct from every live-scored snapshot; its correctness
and performance remain **unknown** pending separate trusted replay.

The private raw trace and snapshots were scanned for credential/private-host
patterns and stayed outside the repository. Both the capture and feedback
roots were mode 0700; the copied workspace itself was mode 0755 inside a
non-traversable 0700 run directory. No SSH credential or withheld manifest was
mounted for the agent. Post-agent hidden scoring remains a nonadversarial
same-process confidentiality boundary.
