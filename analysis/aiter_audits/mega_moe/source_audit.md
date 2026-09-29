# MegaMoE V2 EP source audit (not a bug finding)

Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. This is a
source-only reading of the [MegaMoEV2 implementation](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31-L95),
related FlyDSL kernels, and its [multi-rank test](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L53-L65).
It follows [PR #4439](https://github.com/ROCm/aiter/pull/4439) but the pinned
tree, not the PR description, defines the current behavior. No GPU execution,
generated-code inspection, memory-model proof, or agent-authorship attribution
is implied.

## Scope and topology

`MegaMoEV2` accepts only A8W4, power-of-two maximum tokens per rank (MTPR),
and top-k in `[1, 16]`; it allocates rank-local and MORI symmetric/P2P state
and synchronizes peers in construction
([constructor](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31-L95),
[workspace](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L183-L295)).
The *V4-Pro workload* is not a separate V2 kernel: 384 experts / EP8 gives
48 experts per rank, selecting fixed-slot only when MTPR is at most 255;
other admitted geometries use compact
([test shape](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L24-L49),
[selection](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L377-L422)).
The class also configures compact for V4-Pro with larger MTPR. Source allows
world sizes 1..8, but **one visible MI350X can only exercise EP1/local
arithmetic and local CTA coordination**; it cannot test cross-rank P2P
publication or waits, or EP8's
fixed-slot branch. Minimum useful cross-rank compact gate is two peer-accessible
gfx95x GPUs with MORI SHMEM initialized on one EP communicator; the V4-Pro
fixed-slot gate is eight such GPUs with the model/test weight geometry and
matching MTPR. Confirm peer reachability and equal launch protocol on all ranks
before admission. The test's process-group/MORI setup and rank-invariant
configuration requirement are explicit
([setup](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L53-L78),
[skew policy](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L517-L569)).

## Mapping and protocol

* **CTA and wave mapping.** Stage1 requires gfx95x and `num_waves > 1`,
  sets `TOTAL_THREADS = num_waves * 64`, and has one communication wave plus
  grouping waves; its config commonly selects 4 or 8 waves, with a separate
  `waves_per_eu_hint`. This is not a one-wave-per-SIMD assumption
  ([Stage1 geometry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L104-L149),
  [wave ID](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L430-L468),
  [configs](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L197-L248)).
  Compact prepare uses 8 waves / 512 threads and assigns first-arrival
  tickets to one owner, `prepare_blocks` route producers, and optional quant
  producers; quant CTAs retire rather than enter the owner's epoch protocol
  ([prepare roles](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L54-L59),
  [tickets](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L101-L142)).
  Stage2 has 256-thread CTAs, four 64-lane waves, and optional persistent
  block loops ([launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L378-L389),
  [loops](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L549-L608)).
* **Compact producer-consumer chain.** The owner resets parity, queue tails,
  work heads and histogram before releasing its epoch gate; route producers
  then publish full source histograms to peers, signal `COUNT_DONE`, and the
  planner waits/acquires before deriving offsets. It publishes `PLAN_READY`
  after metadata; payload workers wait/acquire that flag, write remote payload,
  then release/increment `TILE_READY`. The last increment reserves a queue
  slot and release-stores `READY_TILE_EPOCH`; Stage1 consumer waits/acquires
  tile readiness or queue epoch before GEMM
  ([reset](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py#L143-L199),
  [histogram exchange](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L1066-L1094),
  [plan publication](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L1323-L1331),
  [payload wait](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L1626-L1645),
  [tile publication](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L396-L445),
  [tile consumption](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L475-L541),
  [queue consumption](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L583-L611)).
  These are explicit code-level *intended* synchronization edges, not a proof
  that all peer writes are visible on all ROCm/MORI topologies. Helpers lower
  system stores to release `one-as` stores and waits to repeated acquire loads
  ([helpers](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/communication_ops_utils.py#L253-L309)).
* **Fixed-slot V4-Pro chain.** An arrival-ticket owner first exchanges
  `LAUNCH_READY` epochs with peers; producers write per-expert remote slots,
  release and mark source completion. A finalizer waits/acquires all source
  completion flags, pads the local metadata, then release-publishes
  `PLAN_READY` to sources. Stage1 waits/acquires it before consuming
  ([entry handshake](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L272-L355),
  [payload completion](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L540-L570),
  [finalizer](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L606-L668),
  [consumer](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L475-L488)).
* **Padding and empty ranks.** Compact fills padded `SRCMAP` rows with
  `world_size * MTPR`; fixed-slot uses the same invalid-source sentinel.
  Stage2 checks decoded destination/slot validity before P2P scatter, using
  bounded out-of-range stores for inactive lanes
  ([compact metadata](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L304-L316),
  [fixed metadata](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/dispatch.py#L630-L659),
  [scatter mask](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L113-L140),
  [bounded stores](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py#L222-L280)).
  A zero-token rank still chooses a bundle and enters collectives; only its
  standalone quant launch is skipped
  ([bundle](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py#L154-L165),
  [quant guard](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L630-L635)).

## Forward-progress assumptions and bounded probes

The code comments say compact Stage1 launches one CTA per CU, so finite
producer work can drain before tile-waiting consumers occupy the device;
fixed-slot deliberately oversubscribes and uses first-arrival tickets for
owner/producers. This is a scheduling design, **not** a proved occupancy or
fairness guarantee: effective residency depends on registers/LDS, compiled
wave count, concurrent work, and GPU scheduling. The wait helper has no timeout
([Stage1 rationale](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py#L123-L147),
[spin loop](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/communication_ops_utils.py#L253-L259)).
The class documents one in-flight launch per instance, so concurrent reuse of
one instance is outside this audit's admitted contract
([class contract](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py#L31-L38)).

| Priority | Hypothesis and topology gate | Bounded probe and independent check |
| --- | --- | --- |
| P0 | Compact publication/progress, **EP2+ only**. A missing peer, stale epoch, or resident waiter could strand `COUNT_DONE`, `PLAN_READY`, or tile readiness. | Two peer-accessible gfx95x ranks, same MTPR/protocol; run one invocation with rank token counts `(0, 1)` then `(1, 0)`, then alternate bucket sizes for 3 sequential invocations. External process watchdog, no timing loop. Compare valid rows against a torch distributed FP32/BF16 reference; collect per-rank last reached phase if timeout. One MI350X cannot execute this gate. |
| P1 | Compact padding/fanout boundaries, **EP2+**; the existing top-k16 Kimi-K3 adversarial fixture is designed for **EP8**. Sentinel or tile-boundary mistakes could yield NaNs, wrong output, or misplaced writes. | Use separate 1-token padding and 32/33 routed-row-per-expert tile-boundary cases, route across first/last local expert, and poison output buffers before each invocation. Compare valid output and canaries; test `-1` route only if independently admitted by wrapper/reference. Existing test already exposes `--force-padding-boundary` and `--inject-invalid-route`; do not count their presence as failure evidence ([fixture](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L100-L144)). |
| P1 | Fixed-slot launch/epoch/finalize progress, **EP8 V4-Pro only**, MTPR 128. | Eight peer-accessible gfx95x ranks; run one uniform 128-token case, then three *sequential* invocations at 1/128/1 tokens with identical per-rank geometry. External watchdog; inspect source-done/plan-ready epochs on timeout. Compare valid rows against independent distributed reference and check overflow status. Do not run this on EP1 or label other geometries fixed-slot. |

The pinned test already has an independent reference, relative-L2 check,
CUDA-graph replay equality, rank-skew option, and burst option
([reference check](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L303-L369),
[burst](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/multigpu_tests/test_mega_moe_v2.py#L385-L393));
probes above are targeted additions, not claims those cases are untested.
PR #4439 itself includes earlier commits named
["fix deadlock"](https://github.com/ROCm/aiter/commit/35dfc3f43016775a6a478553a074a3610618b26e)
and ["bugfix for NaN from padding token"](https://github.com/ROCm/aiter/commit/92e0d8e3474e4d1df15f3a39a6743727f7f1847a).
Those historical fixes motivate regression probes; neither is a current
confirmed bug. Attribution to agent-written code would additionally require
provenance, not merely a failure in AITER.
