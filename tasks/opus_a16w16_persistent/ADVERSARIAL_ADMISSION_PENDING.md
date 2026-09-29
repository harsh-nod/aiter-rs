# OPUS adversarial matrix: pending-admission receipt

**Status: offline proposal only.** This is not a GPU execution receipt, not
a parity admission, and not an agent trial. The three original random-input
anchors passed their earlier pinned AITER/analyst-adapter control in
[`ADMISSION.md`](ADMISSION.md) and [`FEASIBILITY.md`](FEASIBILITY.md). No row
has yet passed the new two-phase/reuse protocol; the six new rows have not
been launched on MI350X at all.

| Gate | Current evidence |
| --- | --- |
| Candidate public matrix and CPU contract validation | 9/9 locally valid against source-derived constraints |
| Independent Torch FP32 oracle patterns | CPU unit tests pass on miniature shapes; full-size GPU oracle pending |
| Pinned AITER exact-kid dispatch on new rows | not run |
| Standalone adapter correctness on new rows | not run |
| Two-phase repeated output/scratch-reuse sequence | not run on any row |
| Input/output canaries and guard-value OOB-read control | not run on new rows |
| Profiler identity/geometry on new rows | not run |
| Private withheld matrix and trust boundary | not created/admitted |
| Same-boundary graph parity on proposed new rows | not run |
| `scored_eligible` | **false** |

The pretrial `N=2177` failure remains an explicit tuner-domain exclusion.
`k-partial-final-tile` is a particularly important admission question:
the pinned tuner and adapter's host validator allow even `K=194` with four
ceil-div 64-wide loops, but that is not proof the compiled kernel handles
its partial K tile correctly. If AITER fails the oracle there, record the
case as a pretrial source-domain/implementation investigation and remove it
from the scored matrix before freeze; do not call it an agent error. The
same rule applies to any other proposed row that fails either exact dispatch
or independent oracle. No case is silently repaired after agents start.

Raw SHA256 commitments for the intended baseline artifacts:

| Artifact | SHA256 |
| --- | --- |
| Pinned AITER revision | `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` |
| Existing three-case public matrix | `1372ef7f29034b9b3c2f4a43da47ad7618bdbd64df8c05d9903f56cc8c6f794b` |
| Proposed nine-case public matrix | `e4a2346d96db18d6f071fae8b1caeffd327342dcc6976c762d47af37159fd3ed` |
| CPU contract/Torch oracle module | `4fa89a0c61e04797ded6db2f38a7618d6ea6794fa2d78d40725fe8e0f02d85d4` |
| Source-equivalent standalone adapter | `56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36` |
| Pinned persistent device header | `bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b` |

The earlier adapter `.so` hash
`356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50`
identifies that **earlier** three-case control only. A future nine-case
admission must record its own binary, compiler, container, raw result, and
profiler hashes. Private inputs, raw samples and host paths must remain
off-repository; publish only sanitized aggregate dispositions after review.
