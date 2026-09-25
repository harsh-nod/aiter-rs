# Study plan: agent-written HIP kernel mistakes

## The question

The primary empirical question is: what mistakes do coding agents introduce
while writing or improving performance-critical HIP kernels for AITER gfx950
workloads, especially megakernels, and which survive the tests available to
the agent?

The decision is which of those **agent-made mistakes** fe2o3 should catch.
A separate AITER audit asks what bugs or unsupported assumptions exist in
upstream kernels. It supplies realistic hazards and additional verifier
challenge cases, but does not set priorities for the agent-focused goal.

For the agent study, the **observed events are agent-produced candidate
changes**, not AITER PRs; independent task-runs are the incidence denominator.
For the independent audit, the unit is a pinned kernel entry point and its
tested configurations. AITER supplies source-visible
implementations, realistic workloads, performance targets, and examples of
hazards. It cannot tell us how often agents make a mistake: most PRs are not
labeled agent work, merged diffs omit failed attempts, and a fix commit does
not identify who or what
introduced the original defect.

Keep two separate corpora: **agent-induced HIP failures**, identified from
recorded agent trajectories, and **AITER audit findings**, identified by
examining and testing pinned upstream source. Their denominators and
attribution are different. All scored agent implementations are in HIP,
regardless of which AITER backend supplies the workload or comparison.
HIP is a practical AMD GPU programming proxy, not fe2o3's Rust semantics:
rank candidate fe2o3 checks using agent-produced HIP failures only after
marking each mechanism transferable, HIP-specific, or uncertain. Use AITER
audit cases to challenge the ranking and expose missing semantics. Rust
ports are a later, separately labeled fe2o3 coverage test, not the primary
bug-discovery method.

## A concrete experiment

1. Select an AITER gfx950 workload with a documented contract, independent
   correctness oracle, and reproducible latency profile on a specified gfx950
   GPU. For a **from-spec** task, AITER is a withheld performance comparison.
   For an **optimization** task, provide a tested HIP starting kernel; use
   AITER source directly only when that starting implementation is HIP.
2. Give an agent a genuine performance objective: build a fast implementation
   in HIP from an operator contract, improve latency of HIP code, extend a
   fast HIP kernel to another shape regime, or fuse stages while preserving
   the full contract. Keep visible tests, time budget, and GPU access fixed.
   Analyze from-spec and optimization tasks separately.
3. Capture every candidate source snapshot, compiler result, visible test
   result, benchmark result, and repair attempt. The harness independently
   tests each candidate on withheld cases against the pinned AITER operator
   and an independent reference. Benchmark only correct candidates, against
   AITER on the same device and workload. Record functionality, performance,
   and joint parity separately; failed attempts remain observations.
4. Reproduce and minimize each introduced regression. Classify its mechanism
   from evidence, not merely from the final symptom. Record whether the agent
   noticed, repaired, or submitted it.
5. After the HIP error corpus exists, map shared GPU mechanisms to minimal
   fe2o3 Rust examples. Measure both the bugs it rejects and the bugs or
   kernel features it cannot express or reason about. Keep HIP-only build,
   C++, library, and toolchain errors outside the fe2o3 priority ranking.

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
  An agent can implement this operator contract in HIP or optimize a separately
  vetted HIP version; the FlyDSL code is not itself a HIP starting kernel.
  Begin agent tasks from a tested baseline, not a known-buggy commit.

Megakernel tasks should cover stage fusion, persistent scheduling, or
shape/traffic extension, with a plausible HIP path to AITER performance
established before agents start. Compare the same fused operator boundary,
including relevant launches and communication, on the same GPU topology.
The test matrix must vary, where supported, block and
wave configuration, occupancy, queue depth, token count, padding, expert
skew, scratch reuse, repeated invocation, and concurrent streams or ranks.
Check stage handoffs, memory scope and fences, barrier participation,
producer/consumer epochs, forward progress, state lifetime, quantization,
and numeric agreement. Hang-prone cases run in an isolated GPU worker with
a watchdog and a documented recovery procedure. No passing stress test is
reported as proof of forward progress.

## Sampling AITER workloads

Pin an [AITER](https://github.com/ROCm/aiter) revision and census its gfx950
dispatchable operator paths from code, not README claims. Record operator
family, backend, source visibility, ABI, dtype/quantization, layout, shape
regime, launch geometry,
synchronization/communication protocol, fusion boundary, and topology.
Exclude unsupported paths from scored tasks. Binary-only operators may
supply a black-box parity target if their contract and benchmark are
reproducible, but not a source-level starting kernel. Do not count a wrapper,
tuned configuration, or shape row as a new kernel *type* by itself.

The scale target is **10,000 meaningfully distinct, AITER-grounded gfx950 HIP
challenge types**, with at least one independent agent trajectory per type.
A type is a frozen operator contract requiring a materially different
implementation algorithm or protocol: for example a different
quantization/layout contract, fusion or synchronization protocol, or shape
regime that changes algorithm or dispatch. Task mode, cosmetic prompts,
random seeds, small shape perturbations, and repeated trials of the same
specification do not increase the type count. Maintain a hierarchical
family/variant ID and a deduplication rationale; expert-review a stratified
sample for false distinctness. Count task types, agent runs, candidate
snapshots, and hidden
test cases separately. More test cases improve bug detection, but not the
agent-error sample size.

Do not assume AITER contains 10,000 distinct operator implementations. First
publish the gfx950 census and an achievable type-generation matrix, then
generate challenges from source-backed combinations of meaningful contracts
and algorithmic regimes. A challenge counts toward 10,000 only after its oracle,
AITER comparison, supported domain, and credible HIP parity path are vetted.
If the inventory cannot support 10,000 nonredundant types within AITER, report
the measured ceiling and coverage gaps rather than inflate the count or
silently broaden to unrelated kernels. Keep source backend and task mode as
metadata, but every scored agent submission is HIP. For non-HIP AITER paths,
provide a specification and independent oracle, or build a verified HIP
baseline before optimization. Oversample megakernel types deliberately and
report their rates separately.

## Gfx950 parity contract

Each task freezes the exact AITER entry point, revision, supported input
domain, required workload buckets, and operator boundary before agent runs.
**Functionality parity** requires the HIP result to satisfy an independent
oracle and agree with AITER across required shapes, strides/layouts, dtypes,
quantization scales, masks/tails/padding, side effects, and repeated-run
behavior under an operator-specific numerical tolerance. A disagreement
between AITER and the oracle is investigated as a baseline defect or contract
ambiguity, not automatically charged to the agent.

**Performance parity** compares only correct implementations to AITER on the
same exact gfx950 GPU SKU, ROCm/compiler environment, clocks, input data,
workload bucket, and measurement procedure. Report warmup, repeated latency
samples, variance, and throughput where relevant. Freeze a per-task
non-inferiority threshold before trials (default: HIP median latency no more
than 5% above AITER in *every* required bucket, after measurement noise is
qualified); do not let an aggregate hide a severe shape or tail regression.
For serving and persistent kernels, include p95 latency when relevant. MI350
and MI355X are separate performance strata even though both are gfx950.
Task selection must establish a plausible HIP parity route; unreachable
targets are recorded as exclusions, not quietly relaxed after agent results.
Report functional pass, performance pass, and joint pass separately. A fast
wrong kernel never passes; a correct but slower kernel remains in the error
and feasibility data.

## Supporting AITER audit

Inventory every gfx950-dispatchable source-visible AITER kernel module and
exposed entry point at the pinned upstream commit, marking uncertain
wrapper-to-kernel mappings.
Record backend, operation, supported shapes/dtypes/layouts, architecture,
launch/dispatch path, and existing test coverage. Rank the inventory by
risk: cross-block or cross-rank communication, persistent state,
barriers/fences/atomics, quantized layouts, shape-dependent dispatch, and
little edge-case coverage. The pilot deeply audits source-visible gfx950
MegaMoE and two other high-risk gfx950 families; the inventory lets later
rounds cover the rest. Binary-only providers such as PR #4975 receive
integration audits,
not kernel-source audits.

For each selected family, combine source review with independent differential
and metamorphic tests. Extend AITER's documented
[operator tests](https://github.com/ROCm/aiter/blob/main/CONTRIBUTE.md)
with boundary and tail shapes, noncontiguous views, layout-equivalent copies,
padding, quantization extremes, architecture-specific launch choices,
repeated runs, scratch reuse, and synchronization stress. For megakernels,
vary occupancy, stage timing, queue depth, and participating ranks where
supported. Use an independent mathematical reference or a separately
validated implementation; two paths sharing the same indexing or layout
assumption are not an independent oracle.

Log every audited entry point and tested configuration, including negative
results. A suspected defect becomes a confirmed finding only after a
reproducible input, pinned version and hardware, expected behavior under the
documented contract, and a minimized trigger are recorded. Distinguish a
kernel bug from unsupported input, wrapper/dispatch bug, flaky environment,
or an ambiguous contract. Check existing issues/PRs for duplicates, then
report new confirmed findings upstream with a safe reproducer. These findings
can strengthen future hidden tests, but cannot retroactively change frozen
agent trials, be labeled agent mistakes without trace evidence, or increase
a verifier priority score based on agent frequency.
If an audit finds a defect in an agent task's starting implementation, mark
that trial contaminated and rerun from a corrected, newly pinned baseline;
do not attribute the pre-existing defect to the agent.

## Pilot design

- The 36-run pilot validates task generation, parity measurement, failure
  capture, and GPU cost. It is not the final sample size.
- Six gfx950 HIP performance challenges: three conventional kernels (one
  from-spec, two optimization) and three fused/persistent cases (one fusion
  from modular stages, one optimization of a vetted HIP baseline, and one
  shape/config
  extension of a vetted HIP baseline). Include at least one multi-block
  communication or synchronization challenge.
- Two agent configurations, three independent seeds each: 36 trajectories.
  Fix model/version, prompt, tool access, time/token budget, GPU access, and
  starting revision. This is a feasibility pilot, not a population estimate.
- For each challenge, pin a gfx950 AITER comparison and, where needed, a
  correct HIP starting implementation, exact GPU SKU, workload distribution,
  latency measurement procedure, public contract, independent oracle, public
  tests, hidden correctness matrix, and AITER parity threshold.
  Include every run in the denominator, including timeouts and no-speedup
  outcomes.
- Score only gfx950. Pin and report the exact GPU SKU, ROCm, compiler, model,
  clocks, topology, and benchmark environment. Never pool MI350 and MI355X
  performance measurements as if they came from one device.
- Preserve code snapshots and diagnostics from first draft through final
  submission. Use fresh optimization objectives and hidden shapes to reduce
  memorized-PR leakage; disclose that public AITER code may be in training
  data.
- In parallel, inventory the pinned AITER source and deeply audit three
  high-risk families, led by source-visible MegaMoE. This is a separate,
  time-boxed validation track. Do not set a target number of bugs; record
  coverage and negative results.

Historical AITER incidents and PR histories form a **hazard catalog**, used
to select edge cases and check that the audit matrix is realistic. Sample
across operators and backends; do not review every PR or turn historical fix
counts into an agent-error frequency.

## Scale to 10,000 types

After the pilot, run a census/feasibility gate before scheduling scale trials.
Publish the number of eligible gfx950 source paths, generated challenge types
by family, rejected near-duplicates, unavailable or unbenchmarkable paths,
estimated agent/GPU-hours, and achievable parity coverage. Only promise the
10,000-type target if the audited generation matrix actually supports it.

Use staged enrollment: 36 pilot trajectories, then 500, 2,000, and up to
10,000 distinct validated challenge types. Every enrolled type receives at
least one independent agent implementation attempt; a stratified subset gets
repeat seeds and agent configurations to estimate within-type variance. Freeze
the task and hidden tests before its first scored run. Retain all nonbuilds,
timeouts, wrong answers, slower-but-correct kernels, and parity passes. Do
not replace difficult types with easy ones after seeing outcomes.

At each stage, audit deduplication, oracle independence, baseline correctness,
performance noise, and complete trace capture before expansion. Estimate
agent-failure rates within operator/hazard/megastructure strata with uncertainty
intervals that account for shared families and repeated runs. Adaptive
oversampling is useful for rare synchronization failures, but record sampling
probabilities and report stratum-specific rates; do not call an enriched
sample representative of all AITER work. If the 10,000-type census fails,
finish the feasible gfx950 population and report its actual size. Do not
substitute 10,000 test inputs or 10,000 candidate snapshots for 10,000 types.

## Parallel agent work

The coordinator freezes the protocol and integrates work. Workers use
separate branches and owned paths. With four active agents, the coordinator
and three workers can run workstreams A, B, and C concurrently; D and G begin
when the task/harness interfaces are frozen; E and F process findings as
they arrive.

| Workstream | Owned paths | Deliverable |
| --- | --- | --- |
| A. Source and task selection | tasks/, baselines/ | Gfx950 census, deduplicated type-generation matrix, six pilot tasks, and staged path to 10,000 vetted HIP challenges with credible AITER parity targets. |
| B. Oracle and performance harness | harness/, references/ | Independent operator references, frozen visible/hidden tests, candidate replay, per-bucket AITER parity harness, environment manifest, and reliable result artifacts. |
| C. Megakernel protocol and stress tests | mega/ | HIP MegaMoE/persistent tasks from documented contracts or vetted HIP baselines; stage-handoff and progress invariants, shape/occupancy matrix, watchdog, and recovery procedure. |
| D. Agent runner | runs/ | Reproducible agent invocations, complete candidate trajectories, and manifest for every trial cell. Start scored trials only after A-C freeze. |
| E. Failure adjudication | analysis/ | Minimized repros, root-cause taxonomy, independent review, detection timing, and family/variant-specific gfx950 results with uncertainty. |
| F. fe2o3 coverage | cases/fe2o3/, proofs/ | Minimal faithful or explicitly abstracted Rust versions of prioritized agent failures, plus a separate AITER-only holdout; corrected counterparts, receipts, and unsupported features. |
| G. AITER audit | audit/ | Gfx950 source-visible inventory and deep audits of MegaMoE plus two families; test/repro logs and confirmed upstream reports, separate from the agent priority score. |

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
agent-induced regressions and AITER audit findings, resolving disagreements
from the trace or minimal repro. Both corpora use the same vocabulary but
remain separate, with unknown agent attribution for upstream findings.

## Agent-first prioritization

Report first-candidate and final functionality parity; invalid candidates per
trajectory; fraction repaired within budget; time to first joint parity; and
per-bucket latency ratio among correct candidates. Measure performance against
the pinned AITER gfx950 target, never merely the HIP starting kernel, on the
same exact hardware and workload with warmup and repeated samples. Do not
credit a fast wrong kernel. Publish functionality, performance, and joint
parity rates by operator family, task mode, and megakernel stratum, not only
aggregate percentages.

The priority denominator is independent HIP agent task-runs, not commits,
candidate snapshots, or hidden test cases: one long repair loop must not
outweigh many separate runs. Track distinct challenge types separately from
repeated runs of those types. For each confirmed mechanism, report the fraction
of runs that introduced it, the fraction of final submissions that still
contain it,
severity (wrong output, OOB, hang, or only build/performance failure), and
whether ordinary visible tests caught it. Candidate counts and time to
repair remain useful secondary measures. Because megakernel tasks are
intentionally over-sampled, report their rates separately and do not pool
them into an overall frequency without a declared workload mix. Use
family-clustered uncertainty intervals; 10,000 related variants are not
10,000 independent operator families.

First label each HIP failure as a **shared GPU mechanism**, **HIP-specific**,
or **uncertain** for fe2o3. Rank fe2o3 work among the shared mechanisms by
observed agent incidence, then severity and likelihood of escaping normal
tests. Classify each high-priority mechanism as already caught, expressible
but missed, needing a new GPU/ISA/progress model, or outside the verifier's
intended scope. Record modeling and proof cost separately. Compiler-rejected
C++ syntax and HIP-library errors should not outrank silent wrong answers
just because agents produce many of them. AITER-only bugs can validate the
scope and suggest a safety watchlist, but cannot supply an agent-incidence
numerator.

For selected confirmed agent errors, retain the HIP repro and provenance.
Port the smallest relevant property to fe2o3 Rust, labeling it faithful,
abstracted, or not representable. Show buggy and corrected variants, exact
assumptions, verification command, diagnostic/receipt, and HIP GPU result.
Score detection, false rejection of the corrected case, unsupported features,
modeling time, and whether the result would have been available before GPU
debugging. Repeat this mapping for a small AITER-only holdout, reported in
a separate table to test whether the agent-derived priorities generalize.

fe2o3 currently documents bounds, race, barrier, and related checks in its
[kernel pipeline](https://github.com/powderluv/fe2o3/blob/main/docs/general-kernel-check-pipeline-v1.md).
Its [refinement receipt](https://github.com/powderluv/fe2o3/blob/main/docs/functional-refinement-receipt-v2.md)
does not establish compiled ISA, launch, runtime, or hardware correctness.
An abstract synchronization proof is useful only if its assumptions match
the native protocol; system-scope communication and forward progress may
remain outside the model. A later controlled Rust-with/without-verifier
experiment can test whether proof feedback changes agent outcomes. It is
not needed to discover HIP agent mistakes. [Current fe2o3](https://github.com/powderluv/fe2o3/blob/main/README.md)
is a developer preview without a supported general external Rust-kernel
compile/dispatch path or multi-GPU execution surface, so full AITER
megakernel ports are not a pilot requirement; unsupported mechanisms are
reported as roadmap gaps.

## Pilot exit criteria

The pilot is complete when all 36 planned trajectories have preserved
candidate histories or explicit failed/unsupported records; every scored
task has a pinned correct gfx950 AITER comparison, independent oracle, frozen
tests, and preregistered per-bucket parity threshold; all reported regressions
have reproducible evidence and layer/mechanism labels;
megakernel hangs were tested under a watchdog; and the fe2o3 report
distinguishes caught, missed, abstracted, and unrepresentable cases. The
main deliverable is a verifier backlog ranked by transferable HIP agent-run
incidence, severity, and test escape, with HIP-only exclusions, uncertainty,
and megakernel strata visible.
The separate AITER audit should inventory gfx950 source-visible modules and
entry points, log coverage and negative results for three deep-audited families,
and document confirmed findings without changing the agent-derived rank.
Before the scale study, publish the gfx950 census, nonredundant type count,
GPU/agent cost estimate, and whether 10,000 distinct AITER-grounded challenge
types are feasible. Scale-stage reports include enrolled types, independent
runs, functionality/parity rates, failure incidence with uncertainty, and
exclusions. If the source-backed population is smaller, report that ceiling
without claiming to have met the 10,000-type target.
