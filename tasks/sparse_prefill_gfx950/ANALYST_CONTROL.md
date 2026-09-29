# Unscored analyst HIP control

[`analyst_candidate.hip`](analyst_candidate.hip) implements the frozen
[`sparse_prefill_bf16_gfx950`](../../references/sparse_prefill_abi.h) ABI as a
simple correctness control. It is **not an agent submission**, performance
baseline, or evidence about agent bug incidence. Its one-workgroup-per-token-
head design reduces each 512-element dot product and applies an online FP32
softmax to both CSR sources, starting with the sink in the denominator.
It writes BF16 output for every head dimension, including zero on sink-only
rows. Inputs and output are assumed contiguous and disjoint as in the task
contract; malformed CSR is not accepted.

Compile offline in a ROCm environment with the repo layout preserved:

```sh
hipcc --offload-arch=gfx950 -O3 -fPIC -shared \
  tasks/sparse_prefill_gfx950/analyst_candidate.hip \
  -o /path/outside/repo/sparse_prefill_analyst.so
```

This control has no AITER calls, lookup tables, host synchronization, or
runtime allocation. The `.so` must still pass the guarded public and withheld
correctness scorer on gfx950 before any numerical claim is made. Do not use
its latency as an agent score; a same-boundary AITER performance policy and
agent trial provenance would be separate work.

Offline build check on `mi350-2` (no GPU execution): `hipcc` exited 0 and
`nm -D` found `sparse_prefill_bf16_gfx950`. The candidate source SHA256 was
`7c9f60721e952f975e5490c0eda37ea4ced3232958b134f462e7890709f9d61a`;
the private compiled `.so` SHA256 was
`44894188eaf7134bfaf750c47499652d6fd4d5d2d7d9793fff04c755b7a01b4d`.
The binary is kept outside the repository under the private study directory.
