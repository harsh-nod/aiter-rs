# Sparse prefill HIP candidate contract

Implement one gfx950 HIP shared library exporting the C symbol in
[`sparse_prefill_abi.h`](../../references/sparse_prefill_abi.h). The contract
is BF16 two-source sparse prefill attention with `D=512`, nonempty `N`, and
`H<=32`; admitted correctness cases include `H=16`, `H=17`, and `H=32`.
Each token has one CSR row
over paged-prefix `unified_kv` and another over current-forward `kv`. For
every head, compute one softmax across **both** selected row sets. The FP32
`attn_sink[h]` contributes `exp(sink[h])` to the denominator, but no value to
the numerator. Write a BF16 output row; if both CSR rows are empty, write
zero. See the independent
[oracle](../../references/sparse_prefill_gfx950.py) and
[admission contract](ADMISSION.md).

Inputs, CSR arrays, sink, and output are contiguous; all CSR indices are
valid, and the supplied output is disjoint from all inputs. Preserve inputs.
Use the supplied HIP stream, return a HIP error code, and do not synchronize
the device or stream inside the entrypoint. Host allocations, hidden global
state, AITER calls, precomputed answers, and dispatching to another library
are outside the candidate contract. A candidate may use its own HIP kernels,
but must fill every valid output element on every invocation, including
back-to-back calls with different CSR patterns and a poisoned output buffer.

Build example:

```sh
hipcc --offload-arch=gfx950 -O3 -fPIC -shared candidate.hip -o candidate.so
```

The correctness scorer accepts only a `.so` with the exported symbol. It
checks public cases, and the trusted runner may provide a separately held
matrix for withheld cases. The scorer confirms the pinned AITER revision and
gfx950 GPU, checks output canaries and input immutability, and compares to a
CPU FP32 reference. Passing is **correctness-only**. No task is
scored-eligible until the same-boundary AITER performance baseline, threshold,
and withheld evaluation policy are frozen.
