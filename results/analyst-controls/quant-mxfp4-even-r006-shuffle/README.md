# r006 subgroup-participation analyst control

This is **not an agent trial**. It changes a separate copy of the immutable
[r006 final source](../../agent-trials/quant-mxfp4-even-r006/workspace/starter.hip)
by moving only `__shfl_down(nibble, 1)` above the even-lane branch. The
packed-byte store remains conditional. The [exact diff](change.patch) is one
statement moved, and the original trial source still hashes to
`6a167b9986cc39fcbb6c6fb9312c4ce779a4536459381729fe18f02b1977c058`.
The analyst copy hashes to
`9dd1a8205565ed78edfbc682e84f20e5fbef044330026356296f30812dd96665`.

Compiled in the pinned ROCm image with the task's frozen gfx950 flags, then
replayed against the pinned clean AITER/harness and public plus withheld
correctness matrix. The control passed **7/7**; the unchanged final r006 agent
source passed **1/7** on the same matrix, and AITER passed **7/7**. The scorer
ran in unscored `reference` mode, with `scored_agent_candidate=false`, no
performance samples, and no joint-parity claim. The [sanitized receipt](result-summary.json)
records binary/result hashes without publishing private cases or raw logs.

Moving this one statement makes the odd partner lanes execute the shuffle
while only even lanes store. The 1/7-to-7/7 change supports lane participation
as the mechanism of the observed failures. It does not establish correctness
for other inputs or prove broader GPU scheduling semantics.
