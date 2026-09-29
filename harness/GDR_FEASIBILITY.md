# GDR performance feasibility probe

`harness.gdr_perf_probe` compares the existing **unscored**, correctness-first
HIP pilot against pinned AITER on two visible one-step GDR cases:
`valid_slots` and `strided_mixed`. It is not an agent trial or a scored parity
result. The selected case IDs, five warmups, twenty alternating-order paired
GPU-event samples, 1.05 per-bucket ratio and 5% MAD noise gate are fixed in
the probe before timing. Correctness is rerun first on all four public cases.

Each invocation mutates recurrent state. Before every timed call, the probe
restores that implementation's state and clears its output outside the event
window, then synchronizes. AITER and HIP get separate preallocated guarded
state/output buffers and identical read-only inputs. The event covers the
complete operator launch on the current stream, not reset/copy or allocation.
After timing, the last result for each side is checked against the CPU oracle,
including state, output, input guards, and untouched slots. The probe rejects
GPU contention observed at preflight or postflight and stores raw samples
outside the repository.

This is only a feasibility screen for the naive pilot. A slow result excludes
that implementation as a vetted HIP starting point; it does not prove that
no HIP kernel can reach AITER's speed. A fast result does not admit a scored
task either: hidden tests, repeated-state timing buckets, candidate sandbox
and a frozen task contract must still be established.

The [first two gfx950 runs](../results/2026-09-29-gdr-performance-feasibility.md)
passed public correctness and showed a roughly 0.52-0.54 HIP/AITER latency
ratio, but one unchanged replay failed the frozen 5% MAD qualification. The
result remains a feasibility signal, not scored performance parity.
