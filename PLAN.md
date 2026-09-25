# Study plan: agent mistakes during AMD kernel optimization

## The question

What mistakes do coding agents introduce while writing or improving
performance-critical AMD GPU kernels, especially when fusing stages into a
megakernel or modifying a persistent kernel? Which mistakes survive the tests
available to the agent, and which could fe2o3 detect early enough to matter?

The **unit of observation is an agent-produced candidate change**, not an
AITER PR. AITER supplies source-visible implementations, realistic workloads,
performance targets, and examples of hazards. It cannot tell us
how often agents make a mistake: most PRs are not labeled agent work, merged
diffs omit failed attempts, and a fix commit does not identify who or what
introduced the original defect.

The primary outcome is the distribution of errors in recorded agent
optimization trajectories. The secondary outcome is whether fe2o3 can detect
those observed errors, under a clearly stated model. Porting AITER to Rust is
not the primary discovery method. Primary tasks use the source's native AMD
kernel stack (such as HIP, Triton, or FlyDSL). Agent-written Rust ports are
valuable as a separately labeled task mode: they expose translation and
fe2o3 usability problems, but their error frequencies must not be pooled
with native-kernel optimization errors.

## A concrete experiment

1. Pin a source-visible AITER kernel that passes a broad correctness suite and
   has a reproducible latency profile on a specified AMD GPU. For a
   **from-spec** task, it is a withheld performance target; for an
   **optimization** task, it is the agent's correct starting implementation.
2. Give an agent a genuine performance objective: build a fast implementation
   from an operator contract, improve latency of existing code, extend a fast
   kernel to another shape regime, or fuse stages while preserving the full
   contract. Keep visible tests, time budget, and GPU access fixed. Analyze
   from-spec and optimization tasks separately.
3. Capture every candidate source snapshot, compiler result, visible test
   result, benchmark result, and repair attempt. The harness independently
   tests each candidate on withheld cases against the pinned baseline and an
   independent reference. Benchmark only correct candidates.
4. Reproduce and minimize each introduced regression. Classify its mechanism
   from evidence, not merely from the final symptom. Record whether the agent
   noticed, repaired, or submitted it.
5. After the error corpus exists, model representative failures in fe2o3.
   Measure both the bugs it rejects and the bugs or kernel features it cannot
   express or reason about.

For example, an agent may replace a communication store with a faster-looking
ordinary store. A hidden multi-block test may then hang. The evidence is the
agent's source diff, the reproducible hang, and the minimal synchronization
change. Only after that would we test whether fe2o3's memory-ordering model
can express and reject the same protocol error.

## Megakernels are a first-class track

At least half of pilot tasks and validation effort target fused or persistent
megakernels. Separate two cases:

- [AITER PR #4975](https://github.com/ROCm/aiter/pull/4975) is a valuable
  persistent-decoder integration example, but its public PR explicitly ships
  provider binaries **without the private kernel source**. We cannot inspect
  agent mistakes in that kernel or port it faithfully from this PR. It can
  inform ABI, checkpoint, cache-lifetime, and integration tests. A kernel
  source study requires an authorized source-visible target.
- [AITER PR #4439](https://github.com/ROCm/aiter/pull/4439) adds source-visible
  MegaMoE code. Its history includes an [explicit deadlock fix](https://github.com/ROCm/aiter/commit/35dfc3f43016775a6a478553a074a3610618b26e)
  involving a system-scope store and a [padding-token NaN fix](https://github.com/ROCm/aiter/commit/92e0d8e3474e4d1df15f3a39a6743727f7f1847a).
  These are candidate test ideas, **not evidence that an agent caused them**.
  Begin agent tasks from a tested, pinned baseline, not a known-buggy commit.

Megakernel tasks should cover stage fusion, persistent scheduling, or
shape/traffic extension, with an actual performance opportunity established
before agents start. The test matrix must vary, where supported, block and
wave configuration, occupancy, queue depth, token count, padding, expert
skew, scratch reuse, repeated invocation, and concurrent streams or ranks.
Check stage handoffs, memory scope and fences, barrier participation,
producer/consumer epochs, forward progress, state lifetime, quantization,
and numeric agreement. Hang-prone cases run in an isolated GPU worker with
a watchdog and a documented recovery procedure. No passing stress test is
reported as proof of forward progress.

## Pilot design

- Six performance challenges: three conventional kernels (one from-spec, two
  optimization) and three fused/persistent cases (one fusion from modular
  stages, two source-visible extensions). Include at least one multi-block
  communication or synchronization challenge.
- Two agent configurations, three independent seeds each: 36 trajectories.
  Fix model/version, prompt, tool access, time/token budget, GPU access, and
  starting revision. This is a feasibility pilot, not a population estimate.
- For each challenge, pin a correct AITER reference or starting
  implementation, target GPU, workload distribution, latency measurement
  procedure, public contract, public tests, hidden correctness matrix, and
  performance target.
  Include every run in the denominator, including timeouts and no-speedup
  outcomes.
- Test on gfx942 and gfx950 when the source and hardware support both. A
  gfx950-only MegaMoE result must not be generalized to gfx942. Record ROCm,
  compiler, model, GPU, clocks, and benchmark environment.
- Preserve code snapshots and diagnostics from first draft through final
  submission. Use fresh optimization objectives and hidden shapes to reduce
  memorized-PR leakage; disclose that public AITER code may be in training
  data.

Historical AITER incidents and PR histories form a **secondary hazard
catalog**, used to select edge cases and check that the test matrix is
realistic. Sample across operators and backends; do not review every PR or
turn historical fix counts into an agent-error frequency.

## Parallel agent work

The coordinator freezes the protocol and integrates work. Workers use
separate branches and owned paths. With four active agents, the coordinator
and three workers can run workstreams A, B, and C concurrently; D begins
when their interfaces are frozen; E and F can overlap with later trials.

| Workstream | Owned paths | Deliverable |
| --- | --- | --- |
| A. Source and task selection | tasks/, baselines/ | Six pinned performance challenges, split into from-spec and optimization modes, with measured headroom or performance targets; a secondary AITER hazard catalog. |
| B. Oracle and performance harness | harness/, references/ | Independent operator references, frozen visible/hidden tests, candidate replay, latency harness, environment manifest, and reliable result artifacts. |
| C. Megakernel protocol and stress tests | mega/ | Source-visible MegaMoE task(s), stage-handoff and progress invariants, shape/occupancy matrix, watchdog, and isolated recovery procedure. |
| D. Agent runner | runs/ | Reproducible agent invocations, complete candidate trajectories, and manifest for every trial cell. Start scored trials only after A-C freeze. |
| E. Failure adjudication | analysis/ | Minimized repros, root-cause taxonomy, independent review, detection timing, and per-task/architecture results. |
| F. fe2o3 coverage | cases/fe2o3/, proofs/ | Minimal faithful or explicitly abstracted Rust versions of observed agent failures, corrected counterparts, receipts, and an unsupported-feature inventory. |

The coordinator owns README.md, PLAN.md, protocol/, schema changes, task
freeze, and merges. Workers do not change another workstream's hidden tests
or rewrite a failed candidate after seeing its result. No public agent trace
is committed without credential/private-data review; third-party source
imports require license and provenance review.

## Evidence and labels

Each candidate record needs: task and run ID; agent/model/version and seed;
parent and candidate source hashes; source diff; prompt/tool budget; ordered
tool output; compiler status; visible and hidden test results; GPU/ROCm
environment; benchmark result if correct; elapsed time; and whether the agent
accepted, abandoned, repaired, or submitted it. Keep raw outputs immutable.

Label **mechanism**, **affected layer**, and **symptom** separately:

- Mechanism: indexing/tail/bounds, layout/stride, alias/race, memory
  scope/ordering, barrier or persistent progress, state/scratch lifetime,
  arithmetic/quantization, launch/config/dispatch, compiler/ISA assumption,
  performance-only regression, or specification misunderstanding.
- Layer: kernel source, host wrapper, inter-kernel interface, compiler,
  binary, or serving integration.
- Symptom: build failure, wrong answer, NaN, OOB/crash, nondeterminism,
  timeout/deadlock, architecture-specific failure, or latency regression.

Assign a primary root cause only after reproducing it. Record uncertain and
multi-cause cases honestly. Two reviewers independently classify a sample of
agent-induced regressions and resolve disagreements from the trace and
minimal repro. Historical AITER defects use the same vocabulary but remain
in a separate dataset with unknown agent attribution.

## Scoring and fe2o3 follow-up

Report first-candidate and final correctness; invalid candidates per
trajectory; fraction repaired within budget; time to first correct kernel
meeting its performance target; and latency gain among correct candidates.
Measure latency against the pinned AITER target or starting kernel on the
same hardware and workload, with warmup and repeated samples. Do not credit
a fast wrong kernel. Publish per-task and per-task-mode results, not only
aggregate percentages.

For each confirmed agent-induced error selected for fe2o3, first retain its
native repro. Port the smallest relevant property, labeling the translation
faithful, abstracted, or not representable. Show the buggy and corrected
variants, exact assumptions, verification command, diagnostic/receipt, and
native GPU result. Score detection, false rejection of the corrected case,
unsupported features, modeling time, and whether the result would have been
available before the agent's GPU debugging step.

fe2o3 currently documents bounds, race, barrier, and related checks in its
[kernel pipeline](https://github.com/powderluv/fe2o3/blob/main/docs/general-kernel-check-pipeline-v1.md).
Its [refinement receipt](https://github.com/powderluv/fe2o3/blob/main/docs/functional-refinement-receipt-v2.md)
does not establish compiled ISA, launch, runtime, or hardware correctness.
An abstract synchronization proof is useful only if its assumptions match
the native protocol; system-scope communication and forward progress may
remain outside the model. A later controlled Rust-with/without-verifier
experiment can test whether proof feedback changes agent outcomes. It is
not needed to discover native agent mistakes.

## Pilot exit criteria

The pilot is complete when all 36 planned trajectories have preserved
candidate histories or explicit failed/unsupported records; every scored
task has a pinned correct AITER reference or baseline and frozen independent
tests; all
reported regressions have reproducible evidence and layer/mechanism labels;
megakernel hangs were tested under a watchdog; and the fe2o3 report
distinguishes caught, missed, abstracted, and unrepresentable cases. The
next study should expand task and agent diversity based on the observed
failure distribution, not on a preset list of AITER PRs.
