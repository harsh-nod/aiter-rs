# Stable long-row decode top-k: admission audit

Status: **excluded from the current HIP-agent pilot; zero agent trajectories
and zero scored types**. This is an operator-family probe, not a frozen agent
task. The correctness/dispatch gate passed, but the HIP adapter and credible
performance-comparison gates did not.

## Pinned target

- AITER revision: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public entry point: `aiter.ops.topk.top_k_per_row_decode` in
  `aiter/ops/topk.py`. The intended input is contiguous FP32 logits with
  physical shape `(4, 65536)`, contiguous INT32 `seqLens`, `next_n` 1 or 2,
  stable selection, `k=512`, and both INT32 indices and FP32 values outputs.
- Contract: use each row's effective length
  `seqLens[row // next_n] - next_n + row % next_n + 1`, select the largest
  float32 values, break equal-value ties by smallest index, emit selected
  indices in ascending order, pad a short row with index `-1` and value
  `-inf`. The CPU oracle ranks raw FP32 bits, distinguishing `-0` from `+0`.
- gfx950 dispatch: stable width 65536 and four rows satisfy the long-row gate
  in `aiter/ops/topk.py` (`_FLYDSL_TOPK_DECODE_GATES`). Valid contiguous
  tensors satisfy `aiter/ops/flydsl/topk/topk_per_row.py`'s FlyDSL
  preconditions. Since width exceeds 20000, the FlyDSL call uses its
  multi-workgroup path. The HIP fallback in
  `csrc/kernels/topk_per_row_kernels.cu` is one-block decode and is not the
  default performance target for this workload.

## Independent matrix

`references/topk_decode_gfx950.py` implements the CPU oracle without an
AITER import. The five public families are full random rows, heavy ties,
poisoned physical tails with effective lengths below and above `k`, a
`next_n=2` length transition, and signed-zero boundaries. The trusted
`references/topk_decode_admission.py` calls the public AITER wrapper twice
per case and counts actual FlyDSL-long, FlyDSL-one-block and HIP routes;
neither GPU latency nor an agent candidate is measured.

The private five-family matrix was generated once outside the repository and
stored on mi350-2 under its mode-700 private directory, with mode-600 file
permissions. It has random, nonpublic seeds and records the generator file
hash and pinned AITER SHA. Its SHA-256 commitment is
`32be7a04c9dd723126f5884bc4446fa0587bc8b72339e151f2d96da706a257aa`.
The generator SHA-256 is
`d74084f1f1ec7e8cea8a633259b3e4da4aa4b3b3da139a4cedbe86cddc114da6`.
No hidden seed or case data belongs in the agent workspace or public results.

## Correctness-only result

On AMD Instinct MI350X (`gfx950`, card model `0x75a0`), five public and five
private case instances each passed twice: indices and values' raw float32
bits matched the CPU oracle in all 20 wrapper calls. Instrumented dispatch
counted 20 calls to the long-row FlyDSL path, zero to the one-block FlyDSL
path, and zero to HIP. This is **10 case instances, not 20 independent cases**.
Performance was not run; there were no agent candidates. The container image
ID was `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`;
the host attestation SHA-256 was
`0707955d447ea0511e38fb0ab57c024e74fdc405452790db87d8e97989205ceb`.
The visible result SHA-256 was
`521dadeebe9128a95285c932c0b92c35e423a7bb8ffa3fe120c8392e72975cc2`.
The withheld result remains private; its status-only JSON has the same hash
because it records no seed, input value, or per-output data. See the
sanitized `admission-receipt.json` for machine-readable counts.

## Admission gates

1. The pinned AITER wrapper must select exactly the intended multi-workgroup
   FlyDSL branch and match the CPU oracle on the public and withheld matrices
   across repeated calls. A baseline mismatch is an AITER/contract issue, not
   an agent bug.
2. A task-specific HIP shared-library ABI and *trusted* adapter must cover
   the same inputs, outputs, effective lengths and stable bit-level ordering.
   It must invoke AITER's default FlyDSL branch as baseline, not silently
   force `AITER_DISABLE_FLYDSL_TOPK_DECODE=1` or time its slower HIP fallback.
3. Demonstrate a plausible HIP implementation path to that baseline, then
   preregister the exact performance buckets and the plan's default rule:
   candidate median latency at most 1.05 times AITER in **every** bucket,
   qualified for noise on the same idle MI350X. Required clocks, warmups,
   repetitions, output checking and GPU manifest must be frozen before agent
   capture.

Only gate 1 is satisfied. There is not yet a same-contract HIP shared-library
adapter or a qualified HIP-vs-default-FlyDSL performance feasibility check.
Until all three gates pass, this family is excluded from agent-error
incidence denominators. A later HIP-vs-HIP fallback experiment would answer
a different question and need its own task identity and baseline label.
