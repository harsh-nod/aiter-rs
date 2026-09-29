# OPUS full-K editable-header task candidate

Status: **unscored candidate**; `scored_eligible=false`, zero agent trials. The
task targets pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`
on one MI350X/gfx950. An agent may edit only
`csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh`
in a clean AITER overlay. The trusted runner, not the agent, copies the exact
pinned tree, verifies the base header SHA and only-header diff with
[`prepare_overlay.py`](../prepare_overlay.py), builds separate clean JIT
caches, and calls production `aiter.ops.opus.opus_bmm` with a preallocated Y,
explicit kid 300 or 1300, and `split_k=0`. Build scripts supplied by an
agent are not run. The starter snapshot should contain the pinned header and
read-only supporting AITER source; no hidden fixture, scorer, private report,
or trusted runtime path enters the agent workspace.

The [public matrix](public_matrix.json) contains exactly the eight full-K
shape/kid rows already covered by the guarded admission and graph controls.
BF16 A `[1,M,K]`, BF16 B `[1,N,K]` (not transposed in storage), and BF16 Y
`[1,M,N]` are contiguous; batch is one, with no bias or workspace. The
independent mathematical oracle is
`torch.matmul(A.float(), B.float().transpose(-1, -2))`. Every output element
must be finite and within `atol=0.125`, `rtol=0.02`; bitwise AITER/candidate
equality alone is insufficient. The six correctness steps, input/output
canaries, input immutability, inactive-output check, and phase-0 reproduction
are specified in [task.json](task.json) and implemented for the prior adapter
control in [`adversarial_admit.py`](../adversarial_admit.py). The new production
overlay scorer must implement the same checks on both separate AITER and
candidate processes, plus profiler-observed `gemm_a16w16_persistent_kernel`
with `ELb1` for kid 300 and `ELb0` for kid 1300. A guard perturbation with
unchanged output is not a memory-safety proof.

The withheld matrix is a mode-600 off-repository file, with its raw SHA256
committed in `task.json`; its generator is [`freeze.py`](freeze.py). It uses
fresh random BF16 inputs on all eight admitted shape/kid rows and four
full-K checkerboard/cancellation rows. Hidden correctness does not expand the
algorithmic contract or count as twelve additional kernel types. No
partial-K (including K=194, [known upstream failure](../K_TAIL_REPRO.md)),
N not divisible by 16, padded row strides, noncontiguous outputs, bias,
batch>1, or different kids are admitted. Such modes require their own
baseline, oracle, and performance admission before appearing in either
matrix.

## Performance and remaining gates

The target timing boundary is the same production preallocated `opus_bmm`
call for baseline and header overlay, excluding JIT compilation and Python
enqueue gaps. Capture 32 calls per graph; use five direct and two graph
warmups, then 20 alternating paired GPU-event replay samples per case.
Check outputs, guards, and inputs **read-only after replay**, without a fresh
operator call that could overwrite a bad graph result. Require an attested
MI350X (PCI `0x75a0`), exact dispatch, no foreign active GPU PID, and <=5%
relative MAD for each path. Every bucket must have candidate/AITER median
ratio <=1.05. Separately, optimization success requires geometric mean
ratio <=0.95. An unchanged header can satisfy non-inferiority but cannot
claim improvement. [`freeze.py`](freeze.py) validates these logical gates on
CPU; it is not a GPU scorer.

The first and [second](SECOND_SESSION.md) sessions compare pinned production
AITER against an unchanged-source standalone HIP adapter. They support
feasibility but **do not establish production editable-header parity**.
Before scored eligibility: validate a trusted production overlay
scorer; use two process-isolated `module_deepgemm_opus` builds with new,
disjoint `AITER_JIT_DIR` paths (the module name and Python import cache are
otherwise shared); attest source/header, image/hipcc, GPU and JIT artifacts;
run public and withheld correctness; obtain at least two uncontended
same-boundary performance sessions; and freeze the evaluation boundary.
The private matrix remains off-repo. Same-process native candidate code could
inspect a mounted hidden manifest, so strong confidentiality against a
malicious candidate also needs a separate restricted candidate process or
equivalent design. This draft does not claim that guarantee.

This kernel has eight waves per 512-thread workgroup and uses workgroup LDS,
barriers, wait counts, and persistent M tile iteration. Source anchors:
[wave/workgroup mapping and padded-grid return](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L103),
[outer-loop synchronization](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L198),
and [store drain/barrier](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L536).
This task can expose wave/LDS/barrier and scheduler-sensitive mistakes; it
does **not** test a cross-workgroup spin-wait, inter-rank protocol, or GPU
forward-progress theorem.

## Production control

[`dual_worker.py`](dual_worker.py) is a trusted, **one-case** production
control, not a complete task score. It launches separate Python main-process
workers for pinned AITER and the editable-header overlay. Each imports its
own AITER tree, builds `module_deepgemm_opus` in a fresh disjoint JIT cache,
checks the loaded module's origin/hash, exact kid/has-OOB profiler symbol,
six-step correctness, and identical hashed BF16 graph inputs. The parent
serializes alternating replay commands, accepts only the two owned GPU PIDs,
and inspects outputs without launching another operator. Raw outputs and
worker logs remain under a mode-700 private run directory. A worker timeout
or runtime exception is inconclusive, never an agent correctness miss.

The trusted host wrapper [`run_pair_host.sh`](run_pair_host.sh) prepares the
single-header overlay, uses the pinned image with network disabled and a
20-minute outer per-case watchdog, and cleans up only its named container.
It runs as the host UID with a private minimal read-only passwd/group mapping
from [`host_identity.py`](host_identity.py), because this host's LDAP UID is
not present in `/etc/passwd` inside the container.
It preserves an `attempt.json` and log hash when no production result is
written. Run it only in an exclusive MI350X window, with `AITERRS_STUDY_ROOT`
set, a trusted candidate-header snapshot, a public case ID, and a **new**
private run directory. It is not an agent-visible feedback command. The
trusted [`aggregate.py`](aggregate.py) refuses fewer than eight exact public
case results, compares per-bucket <=1.05 and the separate geomean <=.95
target, and optionally requires every committed withheld correctness case.
One case cannot imply task parity. Even after an all-case aggregate passes,
this task remains `scored_eligible=false` pending independent review,
repeat sessions, and any required hidden-data isolation.
The current pre/post GPU PID checks and MAD screen do not exclude a short-lived
foreign workload between samples; scored admission should add an external
contention monitor or explicitly retain that residual uncertainty.

The first [production seed smoke](PRODUCTION_SEED_SMOKE.md) passed **one**
unchanged-header public case. [`batch_host.py`](batch_host.py) is the next
trusted control: it runs all eight public cases sequentially, aggregates the
exact matrix, and opens the committed withheld manifest only after every
public case passes the <=1.05 and evidence gates. It then runs twelve
withheld correctness cases and writes per-case raw results, logs, hashes, and
an aggregate report under a new mode-700 private batch root. The candidate
header is frozen as a mode-400 snapshot there before any case launches and
re-hashed before and after every case. It stops on a
failed/incomplete case; no hidden stage is opened after a public failure.
Each case has a 20-minute Docker watchdog and 1300-second host watchdog; the
entire batch has a 90-minute budget. The seed case took about 98 seconds
including clean JIT, so an all-pass batch may take roughly 30-45 minutes.
These are operating limits, not proof that the remaining buckets pass.

Run only from a trusted host in an exclusive MI350X window, using a clean
checkout of the exact runner commit and the frozen candidate header:

```bash
PYTHONPATH="$CODE" python3 -m tasks.opus_a16w16_persistent.scored_candidate.batch_host \
  --candidate-header "$STUDY/aiter/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh" \
  --study-root "$STUDY" --code-root "$CODE" \
  --batch-root "$STUDY/private/opus-fullk-production-batch-001" \
  --withheld-matrix "$STUDY/private/opus-fullk-v1/withheld.json" \
  --host-gpu-report "$STUDY/private/opus-adversarial-20260929/host_gpu_report.txt"
```

The withheld path and raw report are private host inputs, never agent-visible.
The printed summary deliberately omits hidden IDs and per-case metrics.
This trusted replay is for nonadversarial research: native candidate code
runs in the scorer process while the hidden manifest is mounted, so it does
not establish confidentiality against malicious native code.

CPU preflight: `python3 -m unittest
tasks.opus_a16w16_persistent.scored_candidate.test_freeze -v`. Run
`python3 -m tasks.opus_a16w16_persistent.scored_candidate.freeze` to check
the public task. Before making an overlay, use `freeze --check-paths` with
`--aiter-source`, `--baseline-jit`, `--candidate-jit`, and `--overlay-tree`
pointing to four disjoint locations, with the latter three not yet existing.
