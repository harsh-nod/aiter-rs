# GDR packed-BF16 scored optimization with public feedback

Optimize only `kernel.hip` for MI350X/gfx950. Preserve the exact C ABI in
`gdr_decode_packed_bf16_abi.h`, supported strides, invalid-index behavior,
state updates, and the caller-provided HIP stream. The starter is the pinned
AITER-derived HIP implementation. Do not add files, alter the header or public
cases, or depend on a workspace build script.

Preserve exact output and state-update functionality on every supported case.
The scored non-inferiority target is candidate/AITER latency ratio at most
1.05 in each of three graph-replay buckets. The separate optimization target
is a ratio geomean at most 0.95 across those buckets after non-inferiority
passes. Each bucket must also qualify on measurement noise (relative MAD at
most 0.05 on both AITER and candidate samples).

You may request public-only feedback up to three times from the trusted host
broker:

```sh
python3 /public-feedback.py correctness
python3 /public-feedback.py benchmark
```

The `benchmark` request also runs public correctness before the three public
graph buckets. The helper returns only public case pass/fail values and bucket
ratios. Each request snapshots the current `kernel.hip`; make the edit you want
tested before requesting feedback. A request may take several minutes. Hidden
cases are never available during this session, and a local compile or a
successful agent exit is not a correctness or performance result.

The final immutable source snapshot is scored after this session against
separate public and withheld cases. Keep the final source as the version you
intend to submit, including if all three feedback requests are spent.
