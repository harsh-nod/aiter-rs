# aiter-rs

Reproducible studies of mistakes agents introduce while writing or optimizing
performant AMD GPU kernels, especially fused and persistent megakernels, and the
practical value of Rust-based verification with
[fe2o3](https://github.com/powderluv/fe2o3).

Agents will implement kernels from specifications or optimize pinned,
source-visible kernels while we retain all candidate changes and test them
independently. AITER supplies realistic starting points, workloads, and
performance targets; historical PRs alone cannot reveal an agent error rate.
See [PLAN.md](PLAN.md) for the parallel workstreams, megakernel track,
evidence requirements, and pilot exit criteria.

No AITER source code or agent traces have been imported yet. Any future
third-party source import requires a license and provenance review; public
agent traces must be checked for credentials and private data before commit.
