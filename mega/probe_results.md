# Bounded gfx950 probe results

AITER revision: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
Environment: one visible gfx950 MI350X, ROCm PyTorch/AITER JIT container,
2026-09-25. These are correctness-only invocations, not latency data. A
separate workload was using the GPU, so no performance conclusion is valid.
The AITER source tree was mounted read-only; only the JIT cache was writable.

## GDR empty batch: reproducible launch failure, contract unresolved

The local script was piped to the container's `python3 -` on stdin with a
150-second outer timeout. Equivalently, in an environment with AITER and this
repository available, run
`PYTHONPATH=/path/to/aiter python3 mega/probes/probe_gdr_empty_batch.py`.
The wrapper accepted all B=0 tensor shapes and dispatched to
the native function. The native launcher emitted:

```text
csrc/kernels/gdr_decode_packed_bf16.cu:333 hipGetLastError(): invalid configuration argument
```

The process exited with status 139; the exception handler in the Python
probe did not run. Source ties the failure to `grid(batch * kVHeads *
kVBlocks)` at [native launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L308)
with `batch=0`. The [wrapper docstring and validation](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L48)
state no positive-batch requirement, and the native checks also accept zero.
This establishes a wrapper-accepted invalid launch, **not** that empty batches
are contractually supported. If B=0 is outside the intended domain, classify
it as missing input validation or an undocumented precondition; if supported,
classify it as a kernel-launch bug. Confirm domain expectations with the AITER
maintainers before assigning severity. This is not an agent-authored bug.

## MHC fused post/pre tails: no mismatch observed

Run `PYTHONPATH=/path/to/aiter python3 mega/probes/probe_mhc_tail.py --m 17`
and repeat with `--m 257`, each with an outer timeout. The local scripts were
piped to the same container on stdin. `force_fused=True` selects the fused HIP
path for these shapes under
the [gfx950 dispatch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L859).
The script compares all four outputs to `mhc_post` followed by `mhc_pre` with
`torch.isclose(rtol=1e-2, atol=1e-2)` and reports the final-row mismatch ratio.

| M | Output | Mismatch fraction | Last-row mismatch | Max absolute error | Finite |
| ---: | --- | ---: | ---: | ---: | --- |
| 17 | post_mix | 0 | 0 | 0.000596881 | yes |
| 17 | comb_mix | 0 | 0 | 0.000379145 | yes |
| 17 | layer_input | 0 | 0 | 0.000488281 | yes |
| 17 | next_residual | 0 | 0 | 0 | yes |
| 257 | post_mix | 0 | 0 | 0.000141621 | yes |
| 257 | comb_mix | 0 | 0 | 0.000126630 | yes |
| 257 | layer_input | 0 | 0 | 0.000488281 | yes |
| 257 | next_residual | 0 | 0 | 0 | yes |

This is a **negative result for these two deterministic inputs**. It does
not prove all tail masks, split-K configurations, or cross-wave reductions
correct. The unfused path is a differential comparator, not an independent
mathematical oracle. No MegaMoE cross-rank test was attempted on one GPU.
