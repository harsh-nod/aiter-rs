# MXFP4 Even r001 graph control

`quant_mxfp4_graph_control.py` is a **separate, unscored** measurement control
for the already admitted quant r001 `libcandidate.so`. It does not modify or
reinterpret the legacy scorer and does not consume the private withheld matrix.
It pins the public spec raw SHA256, the r001 binary's raw SHA256 and the
filename-inclusive scorer SHA256, and the AITER source revision. The target is
one host-attested MI350X (`0x75a0`, gfx950).

Correctness runs twice on all three public cases before timing. Each run
resets the preallocated packed/scales payloads to canaries, compares exact
packed and E8M0 bytes against the independent CPU oracle and pinned AITER,
checks input immutability and prefix/suffix guards, and validates output
shape/dtype/layout. The benchmark uses only the frozen `f16_medium` and
`bf16_large` buckets. Both sides receive the same guarded input and separate
preallocated guarded outputs. AITER calls the low-level `quant_mxfp4` binding
with arguments `(x, packed, scales, 32, 2, false, false, false, false)`;
the HIP candidate calls the frozen C ABI on the same current stream. Neither
timed closure allocates its outputs.

Each GPU graph captures 32 complete calls. After five warmups, 20 paired
alternating graph replays are timed with GPU events and divided by 32.
Preflight and postflight require the scorer's host PID to be visible in
`rocm-smi --showpids` and no foreign active GPU PIDs; run the container with
`--pid=host`. Output/oracle/guard checks repeat after timing. The frozen
noise gate is baseline relative MAD `<=0.05`, and the exploratory threshold
is HIP median/AITER median `<=1.05` **in both buckets**. Raw samples, median,
p95, MAD, ratio, exact hashes, and environment are written to an immutable
private result directory. Do not publish the raw host manifest.

Python-call GPU-event intervals can include CPU enqueue gaps, even though
they use GPU events. This graph control amortizes those gaps but remains an
unscored device-operation measurement, not a per-kernel profiler trace or a
new agent trial. `scored=false`, `scored_eligible=false`, and
`withheld_cases_evaluated=false` are invariant regardless of the outcome.

The remote invocation mounts a private code overlay and pinned AITER source
read-only, and an exclusive private output directory read-write. The module
is run as `python3 -m references.quant_mxfp4_graph_control` with `--spec`,
`--candidate`, `--aiter-source`, `--host-gpu-report`, `--output`, and
`--unscored-r001-control`.

## MI350X control result

The first public-only replay passed exact-byte candidate/AITER/CPU-oracle
comparisons in both correctness repetitions for all three cases. Input and
output guards passed before and after graph replay. The host GPU PID was
visible and no foreign active GPU PID was observed at the preflight or
postflight probes. The two benchmark buckets were noise-qualified:

| Public bucket | AITER median | HIP r001 median | HIP / AITER | Baseline relative MAD |
| --- | ---: | ---: | ---: | ---: |
| `f16_medium` | 2.304 us | 3.115 us | 1.352 | 0.163% |
| `bf16_large` | 2.340 us | 4.594 us | 1.963 | 0.134% |

These are per-call graph-replay intervals from 32 calls per graph and 20
alternating pairs, not Python-call event timings. The r001 binary misses the
exploratory `<=1.05` ratio threshold in both buckets. This is an analyst
control of an existing agent binary, not a new trial or an incidence data
point; it remains `scored_eligible=false` and no withheld cases were run.
It does not establish device-kernel attribution without a profiler trace.

Provenance: pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`;
public spec raw SHA256
`127e3bc54b2fbd312b5b34d3d95526bbc7603e148a854aa3ed5de9ec986a6d61`;
candidate binary raw SHA256
`757d0b4fd475a8c58ed4e2e1884ff233c1654be28245e82f79d5b0d0f2b0f7c3`;
candidate filename-inclusive scorer SHA256
`bcb1fa28074017435e3cb535623cae4d7d161f5a2591c4d5ff1a1890bf457d1b`.
The private raw result file's SHA256 is
`2bfd24c28443e282e019bd309c0f47a3720fee1323e033ab42b72cf4a09adcac`.
