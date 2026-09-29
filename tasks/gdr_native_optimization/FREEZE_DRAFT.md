# GDR native-seed optimization: freeze draft

**Status: proposed, not admitted or scored.** No agent capture starts from this
note. This is an optimization task, not a from-spec kernel-writing task or an
agent-error incidence denominator: the initial HIP source is the AITER-derived
answer key in
[`analyst_native_gdr_decode_packed_bf16.hip`](../../references/analyst_native_gdr_decode_packed_bf16.hip).
Do not combine its outcomes with the earlier naive-GDR pilot or blind agent
trajectories.

## Frozen inputs proposed

| Item | Value |
| --- | --- |
| Starting HIP source raw SHA256 | `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519` |
| Pinned AITER SHA | `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` |
| Existing public GDR spec raw SHA256 | `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145` |
| Existing private four-case manifest raw SHA256 commitment | `4bb61ab49dab61bff62e124cc5554ec92af3a846f41341f77229f58e38e5c9d0` |
| Target | One host-attested MI350X (`gfx950`, PCI `0x75a0`) |
| ABI | `aiter_rs_gdr_decode_packed_bf16` in the existing C header; element strides, caller stream, asynchronous HIP launch |

Keep the existing task ID
`gfx950.gdr_decode_packed_bf16.stateful.v1` and assign a distinct immutable
`task_revision`/experiment ID for this optimization track. That preserves the
committed **4 public + 4 withheld** correctness matrix without rewriting the
private manifest. All eight cases must pass the independent CPU state oracle,
pinned AITER comparison, input/state/output guards, sentinel behavior, and
repeated-step checks. This is necessary but not sufficient for scored entry.
The analyst seed passed 8/8 in the
[parity-route control](../../results/2026-09-29-gdr-native-analyst-parity-route.md).

## Performance contract to admit

The first two public timing buckets remain `valid_slots` (batch 4) and
`strided_mixed` (batch 6). A larger public, supported workload is warranted:
the current two buckets are small and may overemphasize launch overhead.
The separate `large_valid_slots_candidate` fixture has batch 16, pool 20,
one step, unique valid indices, and padded state slots. It is a *proposed
performance fixture*, not part of the eight-case committed correctness
matrix. The [two-run admission](LARGE_FIXTURE_ADMISSION.md) passed the CPU
oracle, input/state/output guards, and graph checks with unchanged fixture
hash; an exact dispatch/profiler receipt is still missing. If later evidence
shows the fixture changes the supported domain, exclude it with a recorded
reason rather than silently adjusting it. The new public fixture is checked
for correctness at every timed replay, bringing the
total oracle-checked workload count to nine while retaining the eight-case
matrix commitment.

Use the trusted graph protocol: preallocated separate AITER/candidate
state/output, identical inputs, state reset outside each measured interval,
32 stateful calls captured per graph, five direct and five graph warmups, and
20 alternating paired samples per bucket. Check the 32-step final output and
state against the independent CPU oracle and all guards after replay. Require
one exact MI350X, host PID attribution, no active foreign GPU process,
both-side relative MAD/median `<=0.05`, and two independent unchanged runs.
Never substitute Python-call GPU-event intervals for graph replay; retain raw
samples and all source/binary/harness/spec/container hashes. A profiler check
is still needed to attribute per-kernel optimization.

The non-inferiority gate is candidate median/AITER median `<=1.05` in **every**
admitted bucket and both runs. A distinct optimization objective is geometric
mean(candidate/AITER) `<=0.95` across the admitted buckets in both runs, with
no bucket exceeding `1.05`. This 5% objective is proposed, not yet frozen;
freeze it only after session-level stability and a profiler control show it
is measurable. The batch-16 admission's two runs had at most 1.51 percentage
points of bucket-ratio shift and 0.83% relative MAD, below the proposed 5%
effect, but do not alone certify a scoring cutoff. The native-derived seed met
non-inferiority but did not meet that improvement objective.

## Trusted boundary

An agent receives only the exact hashed source snapshot, ABI, public cases,
and visible checks. It may edit allowed `.hip` sources, but cannot change the
task spec, scorer, CPU oracle, AITER revision, compiler flags, hidden manifest,
or measurement protocol. The private four-case manifest stays on `mi350-2`
outside any agent workspace; only its raw SHA256 commitment is public. After
the agent exits, a trusted runner freezes the final source-tree hash, compiles
it with pinned `hipcc -O3 -shared -fPIC --offload-arch=gfx950` without running
agent build scripts, records the binary hash, and scores the isolated binary.
No hidden inputs or raw host paths appear in agent-visible feedback. A
separate trust review must confirm that the agent cannot read the private
manifest through mounts or retained sessions.

The public analyst source is intentionally visible **for this optimization
task only**. Keep it out of any blind from-spec GDR study; exposure to it is
contamination for that separate question. Until the new fixture, objective,
sandbox boundary, and graph/profiler controls are admitted and frozen, mark
`scored_eligible=false` and run no agent captures.
