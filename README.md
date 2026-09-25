# aiter-rs

Agent-first study of mistakes introduced while writing or optimizing
gfx950 AITER workloads in HIP, especially megakernels, and the practical
value of Rust-based verification with
[fe2o3](https://github.com/powderluv/fe2o3).

Agents will implement HIP kernels from specifications or optimize vetted HIP
baselines while we retain all candidate changes and test them independently.
Joint success requires matching the pinned AITER gfx950 operator's functionality
and performance on the same GPU and workload; failed attempts remain in the
error corpus. Correctness also uses an independent oracle. After a pilot, the
study aims for 10,000 meaningfully distinct AITER-grounded challenge types,
each with an independent agent run, subject to a source-inventory and resource
feasibility gate. A separate AITER source audit supplies validation cases,
but verifier priorities come from observed, transferable agent failures.
See [PLAN.md](PLAN.md) for the protocol and parallel workstreams.

No AITER source code or agent traces have been imported yet. Any future
third-party source import requires a license and provenance review; public
agent traces must be checked for credentials and private data before commit.
