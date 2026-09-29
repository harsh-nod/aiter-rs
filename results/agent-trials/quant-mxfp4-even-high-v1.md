# MXFP4 Even high-configuration tranche

The [preregistered configuration](../../runs/tranches/quant_mxfp4_even_gpt55_high_v1.md)
produced three fresh, scored-eligible, no-feedback agent captures. All used
the same frozen quant contract and starter; each had its own isolated
workspace and immutable source history. Trusted correctness-only replay on
MI350X gave final-source passes for [r004](quant-mxfp4-even-r004/README.md)
and [r005](quant-mxfp4-even-r005/README.md), and a packed-output failure for
[r006](quant-mxfp4-even-r006/README.md). Candidate case counts were 7/7,
7/7, and 1/7 respectively; pinned AITER passed 7/7 each time.

These are **three sessions on one kernel task**, not three kernel types.
Performance was not sampled because the old event-only procedure can include
unequal Python-wrapper host gaps; no joint-parity or performance comparison
is available. The existing medium-reasoning r001-r003 used CLI `0.157.0`,
whereas this high-reasoning tranche used `0.159.0`. This is a bundled
configuration comparison, not evidence for a causal reasoning-effort effect.
The incidence denominator remains pending cohort admission.
