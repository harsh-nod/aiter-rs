# MXFP4 Even no-feedback trial r004

Fresh scored-eligible capture under the preregistered
[`gpt-5.5` high configuration](../../../runs/tranches/quant_mxfp4_even_gpt55_high_v1.md).
Codex CLI `0.159.0`, high reasoning, 900-second wall limit; completed in
338.966 seconds with five immutable source states and no snapshot errors.
The final source SHA-256 is
`9b840e8869b64fd2e3b1d2e8f74fe4f04343e9e83acbb716a0bbaab66aa56c63`.

The [sanitized replay](remote-correctness-summary.json) passed 7/7 correctness
cases against the pinned AITER baseline, including withheld cases. Performance
was not run, so joint parity is unknown. The first remote deployment failed
before evaluation due to the older trusted runner's missing Git
`safe.directory` setting; a new exclusive replay with the corrected pinned
runner produced the reported result. This was infrastructure, not a candidate
failure.

Raw prompt, events, snapshots, diffs, blobs, and final source are preserved
here after credential/private-path review. No private test data or raw scorer
result is published. The agent sandbox did make its temporary auth file
readable to agent shell commands; this is not a total secret boundary.
