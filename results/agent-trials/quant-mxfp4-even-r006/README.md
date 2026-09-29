# MXFP4 Even no-feedback trial r006

Fresh scored-eligible capture under the preregistered
[`gpt-5.5` high configuration](../../../runs/tranches/quant_mxfp4_even_gpt55_high_v1.md).
Codex CLI `0.159.0`, high reasoning, 900-second wall limit; completed in
208.849 seconds with four immutable source states and no snapshot errors.
The final source SHA-256 is
`6a167b9986cc39fcbb6c6fb9312c4ce779a4536459381729fe18f02b1977c058`.

The [sanitized replay](remote-correctness-summary.json) failed final-source
correctness: candidate 1/7, AITER 7/7. Six cases had packed-output mismatches;
scales and input integrity passed. The source places `__shfl_down` inside an
even-lane branch while drawing its high nibble from an odd partner lane. This
is the same source pattern seen in earlier failures. A separate
[one-line analyst control](../../analyst-controls/quant-mxfp4-even-r006-shuffle/README.md)
moved only that shuffle before the branch and passed 7/7 on the same matrix,
supporting lane participation as the cause of the tested failures. The failed
agent source remains unchanged. Performance was not run; joint parity is false.

Raw prompt, events, snapshots, diffs, blobs, and final source are preserved
here after credential/private-path review. No private test data or raw scorer
result is published. The agent sandbox did make its temporary auth file
readable to agent shell commands; this is not a total secret boundary.
