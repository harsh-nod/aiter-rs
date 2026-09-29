# Pinned AITER GDR fixed-scale NaN bypass

On 2026-09-29, the bounded [GDR probe](../analysis/aiter_audits/gdr/probe_scale_nan.py)
ran on one MI350X/gfx950 using pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. The source SHA256 was
`834a3d7c75ea9c35f7591cb9f64418a33fcf00cc82f96dfdaa4d2a50630e050f`.
The isolated no-network ROCm container image ID was
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
The probe verified the pinned wrapper/kernel source before import, used only
ephemeral tensors, and finished under a 150-second watchdog.

| Call | Wrapper outcome | Output | Recurrent state |
| --- | --- | --- | --- |
| `scale=1.0` | `ValueError` | unchanged | unchanged |
| default fixed scale | returned | 4,096/4,096 finite | 1,048,576/1,048,576 finite |
| `scale=float("nan")` | **returned** | **4,096/4,096 NaN** | all finite; bitwise equal to default-scale control |

The wrapper's fixed-scale check is `abs(float(scale) - expected_scale) > 1e-12`;
NaN makes that comparison false. The device kernel multiplies normalized Q by
the supplied scale, so it produces NaN output. The state update is Q-scale
independent on this path and was not poisoned. The finite control returned
the caller's output/state allocations and left an untouched state slot
unchanged; the NaN call did the same.

**Classification:** reproducible AITER Python-wrapper validation bypass for
the stated fixed-scale policy. It is not a defect in the GDR kernel's
synchronization or state-update path. The expected contract is to reject a
nonfinite scale before launch; this classification is separate from the
agent-authored HIP-error incidence denominator. Public GitHub issue/PR
searches for this specific GDR NaN-scale case found no match on this date.
The same wrapper comparison was present on AITER `main` at
`2b6ff3d6b5bd20ea7197bfb6d7307e4a235904a2` when checked.
Reported upstream as [ROCm/aiter#5951](https://github.com/ROCm/aiter/issues/5951).
