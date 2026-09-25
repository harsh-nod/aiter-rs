# Study plan

## Questions and claims

This repository will answer three different questions. Report them separately.

1. **Field failures:** What kinds of failures occur in real AMD GPU operator
   implementations? AITER issues, fixes, reviews, CI failures, and reverts are
   evidence here. They do not identify an *agent* error rate unless authorship
   and the agent's actual attempts are known.
2. **Agent failures:** What mistakes do agents make when asked to implement or
   modify AMD GPU kernels, including mistakes later repaired? This requires
   retained first attempts and full, consented run traces, not just merged PRs.
3. **Verifier value:** Which preserved failures does fe2o3 reject, which correct
   implementations can it express and accept, and does access to verification
   feedback improve an agent's final outcome at a fixed budget?

Maintain a claim ladder: a native reproducer establishes an observed failure;
an independent oracle establishes a tested discrepancy; a fe2o3 proof
establishes only the property and model named in its receipt; running the
compiled kernel on a target GPU adds hardware evidence. None implies
source-to-ISA or hardware correctness without an explicit refinement argument.
The current fe2o3 [refinement receipt](https://github.com/powderluv/fe2o3/blob/main/docs/functional-refinement-receipt-v2.md)
excludes lowering, ISA, launch, runtime, and hardware guarantees.

## Pilot scope

- Mine 24 documented AITER incidents, stratified by operator family (GEMM,
  attention, MoE, other), implementation layer (Triton, HIP/CK, ASM, host
  dispatch/config), and gfx942/gfx950 where available. Include unresolved
  reports; do not select only merged fixes.
- Define eight implementation tasks from operator contracts or workloads, not
  from a bug-fix diff. Reserve hidden shapes and metamorphic tests. Run three
  independent seeds per task per experimental arm. This is a feasibility pilot,
  not a prevalence estimate.
- Port six to eight minimal, independently reproduced failure patterns into
  fe2o3 Rust, plus a corrected counterpart for each. Record translation
  fidelity and unsupported features explicitly.
- Use existing [fe2o3-kernels](https://github.com/harsh-nod/fe2o3-kernels)
  examples to develop the harness, but exclude them from the scored holdout.
- Test on MI300X/gfx942 and MI350-class/gfx950 when available. If one is not
  available, mark that cell untested rather than treating results on the other
  architecture as equivalent.
- Freeze model versions, prompts, tool access, time/token budgets, repository
  revisions, compiler/ROCm versions, GPU model, and test seeds before comparing
  arms. Start with one agent configuration; add others only as a separate factor.

## Parallel workstreams

The coordinator freezes `protocol/` and owns merges. Workers use separate
branches/PRs and the paths below. No worker edits another workstream's files
or silently changes a frozen test or classification rule.

| Workstream | Owned paths | Start and deliverable |
| --- | --- | --- |
| A. Incident mining | `cases/aiter/` | Start immediately. Create 24 evidence cards with source links, failing revision, repro, fix/review status, and root-cause confidence. |
| B. Oracle and GPU harness | `harness/`, `references/` | Start immediately. Build independent references, deterministic input generators, result capture, GPU environment manifest, and hidden/edge-case tests. |
| C. Task and agent protocol | `tasks/`, `runs/` | Start immediately. Write eight leak-resistant task specs; configure repeatable agent runs and preserve first attempts plus repair iterations. Do not run scored trials until B's tests are frozen. |
| D. Rust/verifier cases | `cases/fe2o3/`, `proofs/` | Start after A provides reproduced cases and B provides an oracle. Port minimal buggy/correct pairs, recording semantic differences and exact proof scope. Can run alongside scored agent trials. |
| E. Adjudication and analysis | `analysis/` | Start after A-C establish schemas; blinded reviewers classify failures while D proceeds. Publish disagreement resolution, denominators, and results by layer/architecture. |

With four active agents, use one coordinator and three workers: run A, B, C in
parallel; then run D, scored C trials, and E in parallel. Avoid a global
"all PRs" review queue. The coordinator handles triage, issue/PR tracking,
schema changes, and integration; workers own their artifacts.

## Phase 0: freeze the protocol

The coordinator creates versioned schemas before evidence collection:

- `protocol/incident.schema.json`: case ID, AITER URLs and commit hashes,
  report/fix status, backend, operation, GPU/ROCm versions, minimal input,
  expected/observed behavior, reproducer command, evidence level, primary
  mechanism, secondary mechanisms, manifestation, affected layer, and reviewer
  confidence. Never infer that an AITER contributor used an agent.
- `protocol/task.schema.json`: operator contract, supported dtypes/layouts,
  numerical tolerance policy, legal launch configurations, public examples,
  hidden test matrix, allowed dependencies, and resource budget.
- `protocol/run.schema.json`: task/arm/seed IDs, agent/model/version, prompt and
  tool budget, ordered code snapshots, commands and diagnostics, build status,
  public/hidden test results, proof receipts, GPU environment, and elapsed
  time. Redact credentials and private paths before publication.
- A sampling log listing all screened incidents and inclusion/exclusion reasons.
  Record duplicates and unknown root causes rather than forcing a category.

Use two independent labels for each failure: **mechanism** (index/bounds,
layout/stride, aliasing/race, synchronization/progress, arithmetic/numerics,
quantization, launch/config/dispatch, compiler/ISA, performance, specification)
and **layer** (kernel source, host wrapper, compiler, binary, integration).
Record the observed symptom separately: compile failure, crash/OOB, wrong
answer, nondeterminism, hang, unsupported target, or regression. Allow
multiple mechanisms, with one evidence-supported primary label.

## Phase 1: AITER incident atlas

Search issues and PRs, including closed/unmerged/reverted work; inspect CI
logs and review comments where available. Sample across strata instead of
reading every PR. Pin each incident to an immutable revision and produce a
minimal native-language repro, or mark why one is unavailable. Validate the
reported bug on the stated architecture where possible. Keep issue authors'
hypotheses distinct from a confirmed root cause.

Useful initial leads are the reported [small-M split-K OOB](https://github.com/ROCm/aiter/issues/3737),
[FP4 MoE clone-sensitive output](https://github.com/ROCm/aiter/issues/2194),
and [gfx950 tuning/catalog mismatch](https://github.com/ROCm/aiter/issues/4735).
These deliberately span kernel memory behavior, representation-dependent
output, and host/runtime configuration. AITER's [contribution guide](https://github.com/ROCm/aiter/blob/main/CONTRIBUTE.md)
documents operator tests and MI300X/MI350X CI; existing coverage is a source
of test inputs, not an independent proof of correctness.

## Phase 2: independent tests and actual agent attempts

For each task, write a reference independently of the agent implementation.
Freeze tolerance per dtype and operation before results are seen. Test normal
and adversarial cases: zero/small/tail dimensions, non-power-of-two sizes,
strides and transposes, alignment, aliasing, scale layouts, masked lanes,
multi-wave/workgroup configurations, and relevant multi-kernel scratch reuse.
Use deterministic seeds, initialized canaries/sentinels, repeated launches,
metamorphic checks (such as equal-value cloned storage), and available ROCm
diagnostics. Differential tests and stress runs can expose failures; passing
them does not prove race freedom or forward progress.

Run these arms with matched tasks and budgets, randomized order, and at least
three seeds each:

1. **Native baseline:** an agent implements in the task's native AMD stack
   (for the pilot, HIP or Triton), with the same public functional tests.
2. **Rust without proof feedback:** an agent implements the representable
   task in fe2o3 Rust with functional tests, but receives no proof diagnostics
   during development. If the toolchain cannot support this arm, document the
   limitation before running trials.
3. **Rust with proof feedback:** the same Rust task and budget, with fe2o3
   checks and actionable diagnostics available during development.

Arms 2 vs. 3 estimate the effect of verification feedback while holding the
language more nearly constant. Arm 1 vs. Rust arms measures a broader workflow
tradeoff and must not be described as the isolated effect of a verifier. Keep
task exclusions and unsupported constructs in the denominator. Hide historical
PR links/fixes from agents and record possible training-data leakage; use fresh
task variants and withheld tests where feasible. Archive every first attempt,
not only a repaired submission.

## Phase 3: fe2o3 coverage and limits

For a selected AITER incident, first confirm the native behavior. Translate
the smallest relevant mechanism to fe2o3 and show that the Rust buggy/correct
pair produces the intended difference under an independent reference. Label
the port **faithful**, **abstracted**, or **not representable**, with reasons.
Do not claim detection of an AITER bug if the port removed the triggering
instruction, memory ordering, quantization rule, launch choice, or runtime
state. For each pair, record checker version, assumptions, proof command,
receipt, counterexample or rejection, runtime result, and whether a corrected
case passes. Test synchronization cases with varied waves/workgroups when the
model permits; document scheduling or progress assumptions that remain outside
it. Treat unsupported and inconclusive checks separately from successful
detection.

## Metrics and reporting

- **Field atlas:** counts by mechanism, layer, family, architecture, and
  evidence level. Do not generalize sampled proportions to all AITER PRs.
- **Agent study:** first-attempt and final compile/correctness rates, error
  categories, time to first correct result, proof effort, and performance only
  among correct outputs. Report per-task results and uncertainty intervals;
  include failed/timeout/unsupported runs in denominators.
- **Verifier study:** detected preserved bugs / representable confirmed bugs,
  valid counterparts accepted / attempted, false alarms, unsupported cases,
  translation fidelity, and proof scope. Separate source-model results from
  observed GPU behavior.
- **Utility:** matched Rust-arm difference in correctness, debugging time,
  and cost. A native/Rust comparison is a workflow comparison, not a proof
  that formal methods caused any difference.

Two reviewers independently classify a subset of incidents and agent runs.
Publish disagreements and adjudications. Preserve raw immutable outputs and
record all changes to hidden tests after the freeze; never tune the checker or
tasks against the scored holdout without starting a new evaluation version.

## Gates and next expansion

The pilot is ready for a larger study only when: (1) all 24 incident records
have provenance and screening decisions; (2) native failures selected for
porting reproduce or are marked unconfirmed; (3) the harness reruns the same
task deterministically on each available target; (4) all 72 planned agent
trial cells (8 tasks x 3 arms x 3 seeds) have complete traces or explicit
failure/unsupported records, with unsupported cells reported separately; and
(5) every verifier claim links a faithful/abstracted port, exact assumptions,
a buggy/correct pair, and a receipt. Then expand the sampled incident corpus
and task diversity based on measured coverage gaps.
