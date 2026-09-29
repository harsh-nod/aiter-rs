# fe2o3 wave-participation IR admission check

At fe2o3 commit `182e989a279454346953b990ce14aceeb1ebbb70`, the
[executable case](../cases/fe2o3/README.md) constructs a wave64 program whose
`LaneId & 1 == 0` branch executes `ShuffleIndex` only on even lanes. Each
reading even lane requests its odd neighbor. The shuffle's result is returned
by the even branch. Its metadata nevertheless asserts `active_lanes = 64` and
uniform subgroup convergence.

`bash cases/fe2o3/run-wave-ir-gap.sh` tests the pinned fe2o3 crate from a
clean temporary worktree. The test outcomes are:

| IR mutation | `verify_module` | Interpretation |
| --- | --- | --- |
| Branch-local shuffle, declared 64 active | accepts | Convergence claim is not authenticated by this IR check. |
| Branch-local shuffle, declared 32 active | rejects (`InvalidWaveOperation`) | The explicit active-lane field is checked. |
| Shuffle before lane-varying branch, declared 64 active | accepts | Correct participation structure is admitted. |

This is a concrete **missed bug at the core IR check** if an untrusted frontend
can supply false convergence metadata. It is not evidence that the current
authenticated Rust source path admits such a program, or that fe2o3 can prove
the HIP `__shfl_down` semantics on gfx950. The existing gfx942 collective
frontend documents a separate compiler-authenticated convergence obligation;
this case has not traversed that path. Crediting fe2o3 with catching the
agent's r002/r003 failures needs a source-level negative/positive pair, then
lowering and hardware/ISA evidence under the target profile.
