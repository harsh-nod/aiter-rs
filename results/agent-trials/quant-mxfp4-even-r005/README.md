# MXFP4 Even no-feedback trial r005

Fresh scored-eligible capture under the preregistered
[`gpt-5.5` high configuration](../../../runs/tranches/quant_mxfp4_even_gpt55_high_v1.md).
Codex CLI `0.159.0`, high reasoning, 900-second wall limit; completed in
219.444 seconds with three immutable source states and no snapshot errors.
The final source SHA-256 is
`b8d26ad9ba46d692159043f1ef3e24e04630d6a811eba10e8e21e52f1e8cd479`.

The [sanitized replay](remote-correctness-summary.json) passed 7/7 correctness
cases against the pinned AITER baseline, including withheld cases. Performance
was not run, so joint parity is unknown. The final source evaluates its
partner-lane shuffle before the conditional packed-byte store.

Raw prompt, events, snapshots, diffs, blobs, and final source are preserved
here after credential/private-path review. No private test data or raw scorer
result is published. The agent sandbox did make its temporary auth file
readable to agent shell commands; this is not a total secret boundary.
