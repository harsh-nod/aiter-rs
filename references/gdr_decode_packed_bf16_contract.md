# gfx950 packed BF16 GDR decode task

Status: **AITER correctness-admission candidate**, not a scored agent task or
performance-parity target yet. Source is pinned to AITER commit
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.

## Supported domain

The operator decodes one token per batch row and updates a BF16 V-major state
pool in place. The [wrapper contract](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L38-L128)
requires one gfx950 GPU, `scale = 128^-1/2`, and these tensors on that device:

| Tensor | Shape | Type and layout |
| --- | --- | --- |
| `mixed_qkv` | `[B,6144]` | BF16; inner stride 1, 16-byte-aligned base and row stride |
| `a`, `b` | `[B,32]` | BF16; inner stride 1; row stride may be padded |
| `dt_bias` | `[32]` | contiguous BF16 |
| `A_log` | `[32]` | contiguous FP32 |
| `indices` | `[B]` | INT32; positive element stride |
| `state` | `[pool,32,128,128]` | BF16; inner strides `(16384,128,1)`, aligned base and slot stride |
| `out` | `[B,1,32,128]` | contiguous BF16, preallocated |

This pilot narrows the wrapper's accepted domain to positive `B` and `pool`,
nonoverlapping rows/slots, and no aliases between writable and input storage.
Every valid index `0 <= indices[r] < pool` must be **unique within one
invocation**. Duplicate valid indices are undefined because workgroups would
concurrently read-modify-write the same slot; negative sentinels may repeat.
Out-of-range positive indices are treated as invalid by the
[kernel branch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L128-L146).
For an invalid index, the row's entire output is positive BF16 zero and state
is unchanged. The public cases vary valid/invalid rows, padded row and slot
strides, index stride, and repeated calls. One invocation is one GPU launch;
the same state slot may be reused in later, sequential invocations.

## Mathematical oracle

For each valid row, split `mixed_qkv` into 8 query heads, 8 key heads, and 32
value heads. Query/key head `h//4` serves value head `h`. Normalize query and
key by `sqrt(sum(x^2) + 1e-6)`; multiply query by `128^-1/2`. For each value
head, set

```text
decay = exp(-exp(A_log) * softplus(a + dt_bias))
beta  = BF16(sigmoid(b)), then promoted to FP32
R     = FP32(state) * decay
u     = R @ key
w     = R @ query
t     = key dot query
z     = (value - u) * beta
out   = BF16(w + z * t)
state = BF16(R + outer(z, key))
```

The scorer's [CPU oracle](gdr_decode_packed_bf16.py) evaluates this
independently of AITER. Both output and changed state use `rtol=1e-2,
atol=1e-3`, matching the pinned
[upstream tests](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L14-L21).
State is rounded to BF16 **after every step**, not only after the sequence.
Untouched slots and invalid-row positive zeros are checked bitwise. Guards
surround output and state storage, with optional canaries between state slots;
padded input rows and index lanes must remain unchanged.

The [HIP ABI](gdr_decode_packed_bf16_abi.h) passes element strides and a
supplied stream with preallocated state/output buffers. The
[starter](gdr_decode_packed_bf16_starter.hip) is intentionally nonfunctional;
it is not a vetted HIP baseline. The
[candidate scorer](../harness/gdr_score.py) runs AITER and the HIP candidate
on independent guarded allocations against the same CPU oracle at every
step. It can report visible-only or visible-plus-withheld correctness, but
never times the candidate or marks it scored. The pinned AITER implementation
itself uses multiple workgroups over value heads and V tiles, but no inter-workgroup
communication inside this kernel. Unique valid slots prevent cross-row state
races; no forward-progress guarantee is inferred from a passing test.

## Admission limits

The visible matrix is in [the public spec](gdr_decode_packed_bf16_gfx950.json).
Four additional cases are kept in a remote private manifest; only its SHA256
commitment is public. No hidden values, identifiers, or raw results should be
copied into agent-visible workspaces or published during trials. A same-account
SSH credential could still read that remote manifest, so genuine hidden-test
isolation requires removing that credential from scored agent processes or a
separate inaccessible scoring account.

Pinned AITER has passed the independent oracle on both matrices. Before
**scored** agent trials, a separately vetted HIP baseline or credible parity
route must be established, and same-device performance buckets must be
measured without GPU contention. A correctness-only HIP pilot pass is not a
functionality/performance joint pass. No latency or parity result is claimed
by this task spec.
