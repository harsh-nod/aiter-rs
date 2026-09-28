# Quant r001: three unchanged performance replays

The submitted source is the immutable [r001 agent snapshot](agent-trials/quant-mxfp4-even-r001/README.md),
tree SHA256 `26d06c1b17d9fd353060e365fb472668cb994893094ba0e7cf07bf9b02c67382`.
The trusted `libcandidate.so` had raw SHA256
`757d0b4fd475a8c58ed4e2e1884ff233c1654be28245e82f79d5b0d0f2b0f7c3`.
All three replays used pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`,
harness `5abc535a9f54cb9c85e780abe9e3a9b3c2398f0b`, the frozen quant
specification, the same MI350X/gfx950 and ROCm image, and a fresh host GPU
report (raw SHA256 `c979c43a22f5c35701b637e1b209f329d171fb9fca0a58fe7f4017b02ffe16f0`).
No other active GPU process was found at either preflight or postflight.

The frozen rule requires byte-exact correctness on all seven cases, including
four withheld, and candidate median latency at most **1.05x** AITER in every
required bucket, with each side's median absolute deviation (MAD) at most
**5%** of its median. Outputs were preallocated for both implementations;
each replay used five warmups and 20 alternating-order paired GPU-event
measurements. The public medium and large bucket aggregates are below; the
additional withheld timing bucket passed its ratio and noise checks in all
three replays, but its case identity and measurements remain private.

| Replay | Correctness | Medium AITER / HIP (us) | Medium ratio / HIP MAD | Large AITER / HIP (us) | Large ratio | Performance / joint |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 001 | 7/7 | 24.941 / 16.341 | 0.655 / 3.18% | 25.861 / 16.500 | 0.638 | pass / pass |
| 002 | 7/7 | 25.360 / 15.860 | 0.625 / **5.30%** | 25.540 / 16.360 | 0.641 | **fail** / fail |
| 003 | 7/7 | 25.681 / 15.880 | 0.618 / 2.65% | 25.500 / 15.620 | 0.613 | pass / pass |

Replay 002 missed only the preregistered medium-candidate noise gate:
MAD/median was `0.052963`, above `0.05`. Its medium latency ratio was still
`0.625382`, and all other buckets passed. Do not reclassify this rejected
replay as a pass or relax the frozen gate after seeing it. The large speed
margin is observed consistently, but **noise qualification is not stable**
under this short GPU-event protocol. Two replays have `joint_pass=true`, one
has `joint_pass=false`; all three are measurements of **one** agent submission,
not three independent agent attempts. Type-level parity admission remains
pending a declared reproducibility rule or a newly frozen timing protocol.

The immutable raw scorer results and withheld inputs remain private on
`mi350-2`. Result-file SHA256 commitments, in replay order:

| Replay | Raw result SHA256 |
| --- | --- |
| 001 | `fd3abdd41a2781d4455d028eee5123d3ef25d7870286a9e5d09887a3ff294f42` |
| 002 | `3bebfa330f24432c25923d67ebc5303ab1fc35a30d979f29d01deec329429c83` |
| 003 | `3654aee761e5a056cfe7aeffdf7d89776ab470414a0ffc091a0b685e33cca64c` |
