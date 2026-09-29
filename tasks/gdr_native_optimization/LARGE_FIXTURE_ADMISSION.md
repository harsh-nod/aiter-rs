# GDR batch-16 public fixture: unscored admission

The [separate public fixture](large_valid_slots_candidate.json) proposes a
batch-16, pool-20, one-step workload with unique valid state slots and padded
slot strides. It was **not** added to the original four-case public spec or
the committed four-case private matrix. The trusted
[probe](large_fixture_probe.py) pins both raw hashes, the exact analyst HIP
source/binary, and pinned AITER; it remains `scored_eligible=false`.

Two unchanged MI350X/gfx950 runs completed. Before timing, both AITER and the
analyst HIP seed passed the existing four public cases plus the new fixture
against the independent CPU state oracle, guarded inputs/state/output, and
the existing BF16 tolerance (`rtol=0.01`, `atol=0.001`). Invalid sentinels
still require positive BF16 zero and untouched slots are checked bitwise.
Each performance bucket then used separate preallocated state/output, 32
captured stateful calls, state resets outside the timed interval, five direct
and five graph warmups, and 20 alternating paired samples. After replay, the
32-step CPU oracle and all guards passed. Host PID attestation showed no
foreign active GPU process at preflight/postflight and the scorer's own PID
at postflight. No withheld inputs or agent kernels were used.

| Run | Public bucket | AITER median us | HIP median us | HIP/AITER | Largest MAD |
| --- | --- | ---: | ---: | ---: | ---: |
| 01 | valid slots | 3.526 | 3.505 | 0.994 | 0.18% |
| 01 | strided mixed | 3.419 | 3.404 | 0.996 | 0.22% |
| 01 | batch-16 candidate | 8.675 | 8.866 | 1.022 | 0.81% |
| 02 | valid slots | 3.519 | 3.511 | 0.998 | 0.32% |
| 02 | strided mixed | 3.412 | 3.448 | 1.011 | 0.24% |
| 02 | batch-16 candidate | 8.675 | 8.933 | 1.030 | 0.83% |

Every bucket passed the exploratory `<=1.05` non-inferiority and 5% MAD
screens in both runs. The three-bucket geometric-mean ratios were 1.0038 and
1.0126, so the native-derived seed itself did **not** meet the proposed
`<=0.95` improvement objective. Across the two runs, the largest bucket
ratio shift was 1.51 percentage points and the largest within-run relative
MAD was 0.83%. A 5% change is above the variation observed here, so this
bounded control does not show that the objective is below measurement
resolution. Two runs are nevertheless insufficient to certify a stable 5%
scoring cutoff across sessions; repeat on separate host windows and inspect
kernel-level traces before freezing it.

Provenance: AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`; existing
public spec raw SHA256
`fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`;
new fixture raw SHA256
`10c53ca20475142a87cc515e917194ad07047bc54ef0dd8ca6e665038b023c0b`;
analyst source raw SHA256
`5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519`;
HIP binary raw SHA256
`8b66dd83364de6bbaed9c254e7d44a3183bc38445413e885ecfa86722870b709`.
The private immutable result-file SHA256 commitments are
`01a62dfd708abb652dfa0e71b8c37c51c67a6045815a763c352c5a0c18d7078a`
and `4d418b3a2b519cae161fa1a73e11a95f1905b2c52f766ca93f61f264f8430c80`.
Raw samples and host manifest remain off-repository.

This supports the batch-16 workload as a feasible **public candidate**.
It does not freeze a scored task: the larger fixture still needs an exact
device-kernel dispatch/profiler receipt, the 5% objective needs independent
session-level stability, and the agent/private-data sandbox boundary has
not been exercised. Graph replay amortizes host enqueue gaps but does not
identify per-kernel latency by itself.
