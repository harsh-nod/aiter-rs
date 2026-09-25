# AITER gfx950 source census (first pass)

The checked-in index pins AITER at
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Reproduce it with:

```sh
git clone --filter=blob:none --depth=1 https://github.com/ROCm/aiter.git /tmp/aiter
git -C /tmp/aiter fetch --depth=1 origin 868ccf62a0bcad3aa47f92728340ccb37ed4fb39
git -C /tmp/aiter checkout --detach 868ccf62a0bcad3aa47f92728340ccb37ed4fb39
python3 inventory/census_gfx950.py /tmp/aiter --output /tmp/gfx950_source_hints.json
cmp inventory/gfx950_source_hints.json /tmp/gfx950_source_hints.json
```

The script refuses a dirty AITER tree. It indexes git-tracked source files under `aiter/`
and `csrc/` whose path or contents mention `gfx950`, and records hit lines.
At this revision there are **319 source hints** (203 Python, 116 native; 63
paths mention gfx950). This is **not** a kernel count, dispatch census, or
evidence for 10,000 task types: a hit may be a comment, architecture guard,
shared helper, or dead code. Conversely, generic kernels that run on gfx950
but never name it are not in this index. Tests, configs, assembly blobs, and
binary-only providers are outside this first-pass index.

For each scored task, trace its public wrapper to the selected gfx950 backend
on the target machine, pin a particular supported contract and baseline, and
record an independent oracle. The [pilot shortlist](../tasks/pilot_candidates.md)
is the first manual validation pass, with unresolved feasibility marked.
Count task types only after those checks, not from this file count.
