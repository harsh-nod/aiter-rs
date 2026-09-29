# MXFP4 Even: high-reasoning replication tranche

This preregistration freezes three fresh `gpt-5.5` **high** reasoning,
no-feedback captures (`r004`-`r006`) against the existing scored-eligible
quant task. Each attempt gets a new bubblewrap workspace and 900 seconds.
The task freeze, prompt/starter bytes, pinned AITER and harness, private-case
commitment, gfx950 SKU, and 1.05 per-bucket performance rule are unchanged
from r001-r003. No candidate sees the withheld matrix or a GPU during capture.

The agent CLI changed from `codex-cli 0.157.0` in r001-r003 to the frozen
`codex-cli 0.159.0` here. That is a comparability caveat: a result difference
cannot be attributed to reasoning effort alone. The CLI does not expose a
controlled sampling seed, so independence means fresh sessions, not seeded
replicates.

Capture exit status is not a correctness verdict. Preserve all source
snapshots, including non-builds and timeouts, then export candidate sources
for trusted correctness-only replay with the pinned clean AITER/harness and
private manifest on mi350-2. Do not sample latency while that GPU is busy.
Only reviewed, sanitized aggregate results should be published; raw events
must be screened for credentials and private data first. No incidence or
joint-parity claim follows from this configuration freeze alone.
