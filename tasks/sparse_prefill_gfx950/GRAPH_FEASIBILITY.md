# Unscored graph-replay feasibility gate

[`perf_probe.py`](perf_probe.py) compares the correct analyst HIP control with
the pinned AITER **low-level gfx950 forward call**, using already allocated
BF16 inputs and separate guarded outputs. This is not an agent trial or a
scored performance result. The two public buckets are `mixed_sources` and
`tile_tail_63_64_65`; no private case is used for timing.

For each bucket, the probe checks both implementations against the independent
CPU oracle and verifies output guards and unchanged inputs with a direct
preflight call and a **read-only** inspection of the output left by the final
graph replay. It captures 32 calls per graph, runs five graph warmup replays, then
collects 20 alternating AITER/HIP pairs. ROCm disallows external timing-event
nodes inside captured graphs, so GPU events bracket each graph replay. This
boundary can include host enqueue delay; the 32 captured calls amortize it,
but it remains a limitation of the feasibility estimate. The
per-call median HIP/AITER ratio must be at most 1.05 in **each** bucket, and
each implementation's median absolute deviation divided by its median must
be at most 0.05. A noisy bucket is inconclusive, not a pass. Raw sample arrays
and detailed checks stay in a private result outside the repository.

Run inside the pinned ROCm container with `PYTHONPATH` including both the
study code and pinned AITER source:

```sh
python3 -m tasks.sparse_prefill_gfx950.perf_probe \
  --candidate /workspace/private/analyst_candidate.so \
  --candidate-sha256 44894188eaf7134bfaf750c47499652d6fd4d5d2d7d9793fff04c755b7a01b4d \
  --candidate-source /workspace/code/tasks/sparse_prefill_gfx950/analyst_candidate.hip \
  --candidate-source-sha256 7c9f60721e952f975e5490c0eda37ea4ced3232958b134f462e7890709f9d61a \
  --aiter-source /workspace/aiter \
  --output-dir /workspace/private/analyst-graph-feasibility-NNN
```

Use an external watchdog. The candidate is native code, and this probe does
not sandbox it. The result is feasibility evidence only; it does not replace
a frozen scored-task performance policy, multiple independent runs, or agent
submission provenance. In particular, the graph boundary excludes Python
dispatch and host-side allocation for **both** implementations.
