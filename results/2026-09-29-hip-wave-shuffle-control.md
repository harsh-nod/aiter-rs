# gfx950 HIP wave-shuffle control

The [one-wave source](../cases/hip/wave_shuffle_participation.hip) was compiled
and run on `mi350-2` on 2026-09-29 at approximately 16:06 UTC. Remote
hostname: `asrock-1w300-g2-2b`; device architecture reported by HIP:
`gfx950:sramecc+:xnack-`. Compiler invocation:

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 \
  /tmp/aiter-rs-wave-shuffle-participation.hip \
  -o /tmp/aiter-rs-wave-shuffle-participation
/tmp/aiter-rs-wave-shuffle-participation
```

HIP version `7.2.53150-b958ce88c2`, AMD clang `22.0.0git` (LLVM project
revision `9b0f9aa3067a907226b0ebb8bee1fe51d1f0fca5`), installed under
`/opt/rocm-7.3.0`. Source SHA256:
`3332ab917065ed813330ce8186c8aed8600379babd8cb371958cd0ef6d89c861`.
Executable SHA256:
`30af00606ebf0291054f6c9e9eeaf1ea29f88c0257074d5948aee4664d015dc9`.

The program reported `divergent_mismatches=32/32` and
`repaired_mismatches=0/32`. Every divergent output was zero while the expected
odd-lane values were 2, 4, ..., 64. The source-lane participation change
alone explains this controlled difference on the tested compiler/device. It
also matches the quant r002 failure and its one-line analyst repair. This is
not a claim about all gfx950 systems or instruction-level semantics, and it
does not add to the independent agent-run denominator.
