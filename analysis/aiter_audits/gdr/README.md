# Pinned gfx950 GDR packed-BF16 decode audit

AITER source pin: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
This is a **source-only adversarial audit**, not a report of confirmed AITER
bugs or agent-authored mistakes. No GPU job was run for this audit. The older
[empty-batch probe](../../../mega/probe_results.md) observed a launch error,
but the intended B=0 contract remains unresolved.

## Contract and execution map

The public [wrapper](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L38)
requires gfx950, BF16 packed QKV `[B,6144]`, BF16 gates `[B,32]`,
FP32 `A_log[32]`, INT32 `indices[B]`, BF16 state
`[pool,32,128,128]`, and contiguous BF16 output `[B,1,32,128]`.
It allows aligned padded QKV/state rows and strided gate/index rows
([validation](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L71)).
The returned state is the caller's storage, updated in place
([return](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L117)).
The [upstream reference](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L102)
rounds both output and recurrent state to BF16; a multistep oracle must carry
the rounded state forward at every step.

The [native launch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L308)
uses 256 threads (four 64-lane waves) and `B * 32 * 4` workgroups, one per
`(row, V-head, 32-V-element tile)`. A workgroup reads one state index;
negative values become out of range by the unsigned comparison, write
positive-zero output and skip state
([index branch](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L128)).
Valid paths read state, compute normalized Q/K and gates, write one output
per V element, then write BF16 state
([stores](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L245)).
The K reduction uses an eight-lane DS-swizzle group
([source](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L57));
there is no cross-workgroup barrier or spin wait. Unique valid indices and
nonoverlapping state slots make workgroups' state writes disjoint. The source
does not assume one wave per SIMD or rely on another workgroup's forward
progress; repeated calls require normal stream ordering. This is a source
argument, not a formal hardware proof.

## Ranked probe matrix

Ranks order audit value, not proven severity. Run unsupported/ambiguous
aliasing cases only in isolated, watchdog-bounded processes **after** the
operator's intended input domain is decided. For supported-input cases,
compare both output and the *entire* state against an independent reference,
check untouched slots bitwise, and record per-row errors/nonfinite values.

| Rank | Probe and source basis | Expected behavior / classification gate |
| --- | --- | --- |
| P0: fixed-scale validation | Pass `scale=float("nan")` with otherwise valid `B=1`, unique index. The wrapper compares `abs(float(scale)-expected_scale) > 1e-12` ([L65-69](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L65)); NaN makes that test false, and the kernel multiplies Q normalization by `scale` ([L202](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L202)). | **High-confidence validation-bypass hypothesis**, not confirmed at runtime. The apparent fixed-scale contract implies rejection like `scale=1.0`; if maintainers intentionally allow NaN, define expected state/output behavior before calling it a bug. A launched NaN could poison in-place state. |
| P1: empty batch | Use B=0, nonempty state pool, empty indices/out, and a state snapshot. Neither wrapper shape check nor native checks demand B>0 ([wrapper L71-73](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L71), [native L291-311](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L291)); grid becomes zero. | The earlier bounded run observed `hipErrorInvalidConfiguration`, but no independent source specifies that B=0 must be a no-op. If B=0 is supported, expect empty output and unchanged state; otherwise require an explicit validation error/precondition. **Contract ambiguity, not a confirmed kernel bug.** |
| P1: physical state-slot overlap | Construct `state` with pool=2 and `stride(0)=0` (e.g. an expanded one-slot view), `indices=[0,1]`, B=2. The wrapper verifies inner strides and only 16-byte slot alignment, not slot separation ([L92-101](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L92)); native address is `state_idx * state_slot_stride` ([L143-154](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L143)). | Logical indices are unique but physical writes race. First decide whether mutable state must have nonoverlapping slots. If yes, wrapper should reject such views; if aliasing is supported, specify deterministic semantics. **Input-validation/contract candidate**, not an established supported-input wrong answer. Do not use its raced output as an oracle. |
| P1: supported strided mixed batch | For B=1,3,5 and pool>B, mix unique valid indices with repeated `-1`/`INT_MIN`; use padded QKV/state slot strides, broadcast or padded read-only gate rows, positive-stride indices, canary gaps. The wrapper allows these layouts ([L74-101](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L74)); native uses supplied row/slot strides ([L326-331](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L326)). | **Supported-input regression probe.** Compare oracle outputs and updated slots; invalid rows must output positive zero, untouched slots and padding canaries must remain bitwise unchanged. Odd B is not a partial CTA: the grid is exactly `128*B`. Existing tests cover B=2,4,6, one strided B=6 case ([L179-243](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L179)). |
| P1: empty state pool | Use `pool=0`, B=1 or 3, and only negative sentinel indices. The docstring permits negative sentinels ([L52-55](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L52)); the unsigned `state_idx >= pool` branch occurs before any state load ([L132-154](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L132)). | **Supported-by-docstring boundary hypothesis**, subject to confirming that a zero-sized pool is permitted. Expect all positive-zero outputs and no state access; a failing empty tensor/layout check would signal a contract ambiguity, not automatically a device bug. |
| P1: repeated state evolution | Alternate valid/sentinel rows and permute unique slots over 64 steps on one stream, carrying BF16 state. The kernel reads the old state then stores a rounded update ([L148-154](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L148), [L253-263](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L253)); current test uses 16 fixed-index steps ([L246-265](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L246)). | **Supported-input regression probe.** Check each step, not only final state. Test independent state on two streams separately; sharing state across streams without an event is a caller race, not a kernel failure. |
| P2: positive out-of-pool index | Try `pool`, `pool+1`, `INT_MAX` amid valid rows. The [docstring L52-55](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L52) requires every non-negative index to be in range, yet [the native branch L132-140](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L132) returns zero and [upstream tests L204-215](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/op_tests/test_gdr_decode_packed_bf16.py#L204) assert that behavior. | **Documented-unsupported / tested-behavior conflict.** Treat zero/no-state-update as a regression expectation only after the maintainers resolve the public contract; do not score an agent wrong for choosing either interpretation before task freeze. |
| P2: tensor aliasing | Arrange `out` to overlap `mixed_qkv` or `state`, or two state slots to partially overlap via an aligned small slot stride. Wrapper checks shape/contiguity/alignment, not storage overlap ([L74-108](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L74)); native pointers are `__restrict__` ([L96-104](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L96)). | **Likely unsupported aliasing.** First define non-overlap requirement and reject invalid views; a raced result is not evidence of a supported-input bug. Distinguish this from the *required* return aliases (`returned_state is state`, `returned_out is out`). |
| P3: heterogeneous-device guard | On a future mixed-architecture multi-GPU host, put all tensors on one device while a different device is current. Wrapper checks *current* device architecture ([L60-63](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L60)) but native guard switches to `mixed_qkv.device_id` ([L308](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L308)). | **Deferred wrapper-guard hypothesis**; impossible to exercise on the present one-gfx950 setup and not a kernel arithmetic finding. |
| Exclude: duplicate valid index | `indices=[0,0]` for B=2. [Wrapper docstring L52-55](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/gdr_decode_packed_bf16.py#L52) expressly calls concurrent updates undefined; [kernel comment L129-131](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/gdr_decode_packed_bf16.cu#L129) agrees. | **Documented unsupported input.** Do not include in correctness scoring or bug incidence. It may be used only to study how validators communicate preconditions, never as a wrong-answer repro. |

The independent expected-behavior gate matters especially for P0, B=0,
and aliasing. A source-visible branch, a failed launch, or a racy input alone
does not establish a supported-domain AITER bug. Any confirmed baseline
finding should be separately minimized and attributed to AITER; it cannot
be counted as an error made by an agent in a later HIP trial.
