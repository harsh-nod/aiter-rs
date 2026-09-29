# BF16 sparse-prefill gfx950 admission

**Task ID:** `sparse-prefill-gfx950-bf16-hle32`

**AITER SHA:** `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`

**Status:** source-selected, CPU oracle tested, pinned gfx950 AITER public and
withheld correctness probes passed; HIP candidate parity and performance
baseline pending. **Not scored-eligible.** See the sanitized
[admission receipt](admission-receipt.json).

The proposed task is the BF16, `D=512`, `H<=32` branch of AITER's two-source
sparse prefill attention. On gfx950 the public wrapper reaches the OPUS HIP
`16mx1_16nx4` kernel, not a FlyDSL/ASM fallback or a gfx1250 code object;
see [source selection](../../inventory/next_conventional_task.md) for exact
pinned dispatch anchors. `H>32`, FP16, FP8, malformed CSR, and aliased output
are outside this task. Admission starts with `H=16,32`; the withheld `H=17`
tail case has now passed its AITER-only check.

The input/output contract is in
[`sparse_prefill_gfx950.py`](../../references/sparse_prefill_gfx950.py).
The independent CPU oracle computes one softmax over selected prefix and
extend rows; the FP32 per-head sink contributes only to its denominator.
Empty prefix and extend rows therefore produce zero. The oracle validates
CPU BF16 `Q/KV`, finite values, contiguous int32 CSR, monotone row pointers,
and every CSR index in bounds before any GPU run. These are *scorer input
guards*, not a claim that AITER's launcher checks every malformed CSR.

Public functional cases cover mixed prefix/extend, 63/64/65-row tile tails,
prefix-only, extend-only, and sink-only. CPU tests include a tiny hand-computed
two-source example, determinism, finiteness, and invalid-CSR rejection. The
trusted [`sparse_prefill_admission.py`](../../references/sparse_prefill_admission.py)
verifies the checkout SHA and actual gfx950 runtime, instruments the Python
route to the gfx950 binding, calls the public wrapper twice with a preallocated
output and canaries, compares valid BF16 outputs with the CPU oracle, checks
that inputs are unchanged, and records **no** latency. Its initial element
tolerance follows the pinned AITER test (`atol=rtol=1e-2`, at most 1%
mismatches), but must be calibrated on actual baseline outputs before scoring.

```sh
PYTHONPATH=. python3 -m pytest -q references/test_sparse_prefill_gfx950.py
PYTHONPATH=. python3 -m references.sparse_prefill_admission \
  --aiter-source /path/to/pinned/aiter \
  --output /path/outside/repo/public-correctness.json
```

Generate the withheld matrix **only outside the repository** on the remote
study host, in a mode-700 directory; the generator writes a new mode-600
manifest and prints only its SHA256 commitment and count. Do not publish its
case IDs, seeds, or matrix JSON. The private probe emits aggregate pass/fail
only, and its generator hash pins the probe revision:

```sh
PYTHONPATH=. python3 -m references.sparse_prefill_admission \
  --generate-private /path/outside/repo/private/sparse_prefill_gfx950/matrix.json
PYTHONPATH=. python3 -m references.sparse_prefill_admission \
  --aiter-source /path/to/pinned/aiter \
  --matrix /path/outside/repo/private/sparse_prefill_gfx950/matrix.json \
  --output /path/outside/repo/private/sparse_prefill_gfx950/probe.json
```

Before an agent HIP implementation is scored, require: (1) public and
withheld AITER oracle agreement and confirmed gfx950 dispatch; (2) a HIP ABI
that accepts the exact eight input pointers, scalar dimensions/scale,
preallocated disjoint output, and caller stream, launching asynchronously;
(3) candidate agreement on both matrices; and (4) same-device, same-buffer,
same-stream latency versus AITER's low-level binding at fixed nonempty CSR
histograms. Proposed public performance buckets are `N=64,H=16,P=E=256` and
`N=128,H=32,P=E=256`; record total nonzeros and row-length distribution.
Shape rows, CSR seeds, and head counts within this branch are test buckets,
not independent kernel types. No performance threshold is set before an
observed AITER baseline and a credible HIP parity attempt exist.

The candidate-facing C ABI is
[`sparse_prefill_abi.h`](../../references/sparse_prefill_abi.h); the
[agent prompt](agent_prompt.md) discloses the supported shape and numerical
contract. A trusted runner can perform a bounded correctness-only candidate
check with an **external process watchdog**:

```sh
timeout 180s python3 -m references.sparse_prefill_scorer \
  --candidate /path/to/candidate.so \
  --aiter-source /path/to/pinned/aiter \
  --output-dir /path/outside/repo/candidate-visible-001
```

The runner alone may add `--matrix` pointing to the sealed private manifest.
The scorer checks its SHA256 commitment from this pilot receipt, rejects
in-repo private matrices and output directories, writes hidden aggregate
results only, and always reports `performance: not_run` and
`scored_eligible: false`. Native candidate code is not sandboxed by the
scorer; deployment isolation and timeout are runner responsibilities. This
withheld matrix is a **correctness-pilot commitment**, not a frozen final
scoring policy.

At the pinned revision, the public probe passed all five cases twice; the
withheld probe passed all four cases twice. All public comparisons had zero
elements outside the proposed tolerance, with finite output, unchanged
inputs, intact canaries, and the instrumented gfx950 binding selected on
each call. This establishes only bounded AITER-to-CPU agreement, not
correctness for every CSR layout, a hardware memory-safety proof, or any
candidate HIP performance/functionality result.
