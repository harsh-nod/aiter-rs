# GDR native-seed optimization boundary (prototype)

**No agent has been launched for this revision.** The task is frozen as
`task_mode=no_feedback` and `scored_eligible=false`. Its AITER-derived HIP
starter is intentionally visible only in this optimization track, not in
blind from-spec GDR trials. An optimization outcome is not an agent-error
incidence sample.

## Frozen material

- Pinned AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Starter `kernel.hip`: SHA256
  `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519`.
- Immutable ABI header: SHA256
  `69cd59acd3ecaae6bbcb57ab67e1369d9d4f0f67315a4adda259ee3b370d87dc`.
- Four-case public spec: SHA256
  `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
- Separate visible batch-16 fixture: SHA256
  `10c53ca20475142a87cc515e917194ad07047bc54ef0dd8ca6e665038b023c0b`.
- Public feedback contract: SHA256
  `73f97de9dfe39535ebebfb4f36824629c7b2e2cfb29a2d13dcbec9e193ccb5d8`.
- ROCm image ID: `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`;
  `/opt/rocm/bin/hipcc` SHA256
  `9a0fa4bc274155e7add34dc1897bfc08f2f5d6732ad8b9102b7d7a86c0e3d781`.

The starter has only one editable source file, `kernel.hip`; it also contains
the pinned ABI and visible JSON cases. The existing `runs.runner` task freeze
matches [`task.freeze.json`](task.freeze.json). The trusted compiler argv is
exactly `hipcc -O3 -shared -fPIC --offload-arch=gfx950 kernel.hip -o
libcandidate.so`. Workspace build scripts and extra source files are rejected
and never passed to a shell or compiler invocation.

## Public feedback interface

[`public_feedback.json`](public_feedback.json) freezes the five visible
correctness cases, three public graph-timing buckets, 32 captured stateful
calls, five direct and five graph warmups, 20 alternating pairs, 1.05
per-bucket non-inferiority screen, and 5% relative-MAD limit. Its 0.95
geometric-mean improvement target is explicitly **proposed**, not scored.

An implementable broker protocol is in
[`boundary.py`](../../gdr_native_optimization/boundary.py):

1. A future agent-visible helper writes a bounded request with the current
   `kernel.hip` SHA256 under `.feedback/requests/0001.json`.
2. A trusted host watcher verifies the frozen task, exact source allowlist,
   request hash, and three-request cap; it copies immutable source/public
   inputs to a private snapshot and records its hash.
3. The host verifies the image ID and uses a network-disabled public-only
   Docker command with a fresh empty 0700 output directory.
   [`public_score.py`](../../gdr_native_optimization/public_score.py)
   compiles with fixed argv, runs the independent CPU oracle and pinned AITER
   comparison, then measures only the visible graph buckets. No withheld
   manifest is mounted for this public call.
4. Raw results stay outside the repository and agent workspace. The broker
   returns only allowlisted public case booleans and bucket ratios/pass flags,
   with request/source/snapshot/response/raw-result hashes in its private
   event log. Every source snapshot used for feedback remains preserved.

Only a CPU fake-scorer test of this broker exists now. The live watcher and
agent-visible helper have **not** been connected or exercised, so the task
must remain `no_feedback`. The public scorer itself has not had a GPU smoke
run in this task revision. Before changing the task mode, test the full
request/response path and validate that the trusted checkout mounted in the
public container contains no private research artifacts.

## Hidden replay boundary

The four-case withheld manifest stays outside the repository and outside any
agent capture workspace. After an agent session ends, the trusted runner must
freeze the final source snapshot before a network-disabled, private-output
withheld replay; no raw hidden cases, seeds, traces, or host paths return to
that session. The existing `runs.remote_score` **does not** establish strong
confidentiality against malicious native submissions: it mounts the private
manifest while loading candidate `.so` code in the same scoring process.
That limitation is separate from the public-only mount test here. A stronger
adversarial guarantee would require a separate restricted candidate process
and no private manifest in its address space or mount namespace. For a
nonadversarial offline study, record this threat-model limitation rather
than claiming the hidden data are cryptographically or sandbox-protected.

CPU boundary checks: `PYTHONPATH=. python3 -m unittest
runs.gdr_native_optimization.test_boundary -v`. These inspect the constructed
mount policy; they do not prove runtime isolation from GPU drivers or arbitrary
malicious native code.
