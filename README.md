# aiter-rs

Reproducible studies of AMD GPU kernel failures, agent-written kernels, and
the practical value of Rust-based verification with
[fe2o3](https://github.com/powderluv/fe2o3).

The project begins with a study protocol, not a claim that porting a kernel to
Rust proves the behavior of its original AITER implementation or its compiled
GPU binary. See [PLAN.md](PLAN.md) for the parallel workstreams, experiment
design, evidence requirements, and pilot exit criteria.

No AITER source code or agent traces have been imported yet. Any future
third-party source import requires a license and provenance review; public
agent traces must be checked for credentials and private data before commit.
