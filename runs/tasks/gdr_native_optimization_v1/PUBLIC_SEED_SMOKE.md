# Public-only native seed smoke

One bounded 2026-09-29 MI350X/gfx950 run used the frozen native HIP seed and
the public-only Docker scorer from repository commit
`61179e23544b0faa1fdbc62664f40f95167f50a1`. AITER was pinned at
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. The source SHA256 was
`5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519`;
the compiled binary SHA256 was
`3d77fda2486e6adfee7eced30c9c382c0e03a21d66d739a188a7628a8edaba4b`.
The raw result stays off-repository with SHA256 commitment
`a621fcd7af6ccbc0dd235805aceba9feba56154c33e6e7481bd6374e4c1dfdec`.

All six visible checks passed: the four original public cases, the separate
batch-16 fixture, and a guarded CPU-oracle check launching with explicit HIP
stream handle `0`. The pinned host report SHA256 was
`0707955d447ea0511e38fb0ab57c024e74fdc405452790db87d8e97989205ceb`;
the scorer checked one MI350X, `gfx950`, and PCI `0x75a0`. Its own GPU PID was
visible at preflight and postflight, with no foreign active GPU PID at either
gate. No withheld case manifest was mounted.

| Public graph bucket | AITER median (us/call) | HIP median (us/call) | HIP/AITER | Largest MAD/median | 1.05 screen |
| --- | ---: | ---: | ---: | ---: | --- |
| `valid_slots` | 3.511 | 3.515 | 1.0011 | 0.142% | Pass |
| `strided_mixed` | 3.444 | 3.421 | 0.9935 | 0.201% | Pass |
| `large_valid_slots_candidate` | 8.648 | 8.990 | 1.0395 | 0.709% | Pass |

Each bucket used 32 stateful calls per graph, five direct and five graph
warmups, and 20 alternating pairs. All met the 5% relative-MAD screen and
the exploratory `<=1.05` non-inferiority screen. The geometric-mean ratio
was `1.0112`, so this seed did **not** meet the separately proposed `<=0.95`
improvement objective. This is one public-only smoke, not session-level
stability evidence, a scored task, an agent trial, or agent-error incidence.
The live feedback broker and adversarial withheld scoring boundary remain
unvalidated.
