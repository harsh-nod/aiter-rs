# OPUS persistent adversarial graph feasibility

**Unscored analyst control, one MI350X session.** The pinned AITER exact-kid
`opus_bmm` and its unchanged-source standalone HIP adapter used preallocated
BF16 inputs/outputs and the same callable boundary. Each case first checked
both outputs independently against Torch FP32 matmul and checked input/output
guards and immutability. It then replayed 32-call captured graphs with 20
alternating paired samples, checked outputs/guards again, and applied a
predeclared 5% relative-MAD noise screen. The host GPU identity/PID report
attested one MI350X/gfx950 (PCI `0x75a0`) and no foreign active GPU process.
Graph replay excludes ordinary Python enqueue gaps; event-only timings of
Python calls are not device-kernel evidence.

Eight full-K cases passed the oracle and guard checks before/after replay,
qualified on noise, and had adapter/AITER median ratios below 1.05. The
partial-K case failed **both** independent oracle checks before timing, so
there is no K=194 latency or parity result. The nine-case historical matrix
is therefore not admitted as a whole. No private cases or agent submissions
were involved, and `scored_eligible=false`.

| Public case | AITER us/call | Adapter us/call | Ratio | AITER/adapter relative MAD | Private raw SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| `aligned-nooob` | 73.394 | 74.636 | 1.0169 | 0.0028 / 0.0030 | `0f148f4d1cd06eea13ebcd808d6235fd019daae19c3d7a8a86e0349e402b369c` |
| `m-tail-oob` | 94.955 | 96.008 | 1.0111 | 0.0040 / 0.0028 | `fe8ce3e59604204731a48d927b6295a1c16557094bf509cf9757f0082ad19912` |
| `n-tail-16-aligned` | 108.247 | 107.935 | 0.9971 | 0.0042 / 0.0043 | `8ecb8e0d4d5dabe75b9c93dcf3659dd3144da3191d05a95eeda253ff5360110b` |
| `m-one-row-tail` | 93.367 | 92.929 | 0.9953 | 0.0038 / 0.0031 | `7e2d0082a27c5d464beb5d1ceaa735873fd15a8586aeb52b0e47f60c86aa10bb` |
| `n-first-vector-tail` | 81.849 | 83.847 | 1.0244 | 0.0023 / 0.0014 | `67886fa8ae5247e58b2098aa00c0881956dd6063ca942488e7eea1d3f7f74275` |
| `mn-combined-tail` | 80.539 | 81.563 | 1.0127 | 0.0023 / 0.0024 | `04c0677e4245709f3b62e5a222db9501875e45ae53cdd52e71b336a17835b218` |
| `k-min-even-loop` | 67.478 | 67.734 | 1.0038 | 0.0016 / 0.0025 | `a043c35d083e87820cb2d2d589a5c6f470fff178fad3798a3c57be5cdbd0dc1b` |
| `xcd-padded-grid` | 125.855 | 127.110 | 1.0100 | 0.0015 / 0.0019 | `eee99e0c339b8f1c14e7ce6a3a0124f3168f84b817d252e7f5b766594f5d44de` |
| `k-partial-final-tile` (K=194) | not timed | not timed | n/a | n/a | `265d2fbe11f16a8ad6a57969d516836711f35381ccd28cab1652125367ea51c6` |

For K=194 random BF16 input, each path was wrong on 32,350,699 of
33,554,432 output elements (maximum absolute error 46.115); both paths
were bitwise identical, and guards/inputs stayed intact. This is not an
agent-induced defect. The [fresh-input and padded-row control](K_TAIL_REPRO.md)
isolates the partial-K behavior further. A preflight import-path attempt
stopped before kernel launch and contributes no result.

Provenance: pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`;
public matrix raw SHA256 `e4a2346d96db18d6f071fae8b1caeffd327342dcc6976c762d47af37159fd3ed`;
unchanged adapter source/binary SHA256
`56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36` /
`356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50`;
graph driver raw SHA256
`9707aec738b58394e7a287f473f09589bb2b1f835114df3a82ad2687a7474a13`;
private host GPU report raw SHA256
`4b7b8c1d2bf204b465422b93c182fdb8cf4aa478bb47ecf2494380b05d364d4c`.
The private raw JSON files contain exact sample series. A second independent
session, profiler corroboration for the graph runs, and a revised supported
matrix are needed before any scored parity gate.
