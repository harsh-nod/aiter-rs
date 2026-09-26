# GDR decode packed BF16 HIP kernel

Implement `aiter_rs_gdr_decode_packed_bf16` in `starter.hip` for AMD gfx950.
The ABI in `gdr_decode_packed_bf16_abi.h` is fixed. Launch asynchronously on
the supplied HIP stream, write the preallocated `out`, update `state` in place,
and return a HIP error code. The task is to match the functionality below;
there is no latency target for this pilot.

All tensor pointers address GPU storage. `mixed_qkv` is BF16 `[B,6144]` with
inner stride 1 and element row stride `qkv_row_stride`; the first 1024 values
are 8 query heads of width 128, the next 1024 are 8 key heads, and the last
4096 are 32 value heads of width 128. `a` and `b` are BF16 `[B,32]` with row
strides `a_row_stride` and `b_row_stride`. `dt_bias` is contiguous BF16 `[32]`;
`A_log` is contiguous FP32 `[32]`. `indices` is INT32 `[B]` with positive
element stride `indices_stride`. `state` is BF16 `[pool,32,128,128]` with
element strides `[state_slot_stride,16384,128,1]`. `out` is contiguous BF16
`[B,1,32,128]`. Pointers have aligned bases; row and slot strides may be
padded. Input tensors must not be modified.

For row `r`, let `slot = indices[r]`. If `slot` is negative or at least
`state_pool_size`, write **positive BF16 zero** to every output element for
that row and do not update state. Valid slots are unique within one launch;
the same slot can be reused in later, sequential launches. Do not assume state
starts at zero. Every output element must be written on every launch.

For a valid row and value head `h`, use query/key head `h/4`. Compute in FP32:

```text
q = query / sqrt(sum(query^2) + 1e-6) * scale
k = key   / sqrt(sum(key^2)   + 1e-6)
gate = softplus(float(a[r,h]) + float(dt_bias[h]))
decay = exp(-exp(A_log[h]) * gate)
beta = BF16(sigmoid(float(b[r,h]))), then promoted back to FP32
R[v,k] = float(state[slot,h,v,k]) * decay
u[v] = sum_k R[v,k] * k[k]
w[v] = sum_k R[v,k] * q[k]
t = sum_k k[k] * q[k]
z[v] = (float(value[h,v]) - u[v]) * beta
out[r,0,h,v] = BF16(w[v] + z[v] * t)
state[slot,h,v,k] = BF16(R[v,k] + z[v] * k[k])
```

`scale` is `1/sqrt(128)`. Round updated state to BF16 after each launch,
including when the same slot is used on subsequent launches. Untouched slots
must remain bitwise unchanged. The output and state may have guard storage
around them, including gaps between slots, which must not be written. The
correctness comparison allows `rtol=1e-2, atol=1e-3` for valid output and
updated state; invalid output zeros and untouched slots are checked bitwise.

No correctness tests are supplied in this workspace. You may compile locally
with `hipcc`, but there is no GPU execution feedback during this attempt.
