# GDR native-seed optimization previews

Three independent Codex sessions optimized the same frozen GDR HIP seed. These
are **unscored, no-feedback previews**, not three kernel types and not entries
in the agent-error incidence denominator. No GPU, live test broker, correctness
scorer, or performance scorer ran during capture. A successful agent exit says
nothing about kernel correctness or speed.

The task was launched from main `2ce835e6e2d47ce031a1bac02ec1823f4c58bd70`
with [task freeze](../../runs/tasks/gdr_native_optimization_v1/task.freeze.json)
SHA256 `67d70c443118f1c3d9161e0c950616bdf5e0b50b2f82e702d3018ed7c1b1be4c`.
Its pinned AITER revision is `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
Each session used an isolated workspace, `codex-cli 0.159.0` (binary SHA256
`d2752c52353401f7f6efbfcea68796f4f7a3d3e4769f5d1da53fa49d4856b72f`),
`gpt-5.5` with high reasoning effort, a 900-second wall limit, and a fresh
ephemeral session. A sampling seed was not supported. All manifests say
`run_purpose=unscored_preview`, `incidence_eligible=false`; all results say
`scored=false`.

| Replicate | Exit / elapsed | Source snapshots | Final tree SHA256 | Final `kernel.hip` SHA256 | Private archive SHA256 |
| --- | --- | ---: | --- | --- | --- |
| `preview01` | 0 / 281.513 s | 2 | `6be5db5376c854b23461fc9ff8b06887fe5e09fc56a3ebfeeb218102684d3437` | `42501664399f223db5d02cc1bd761953c397026cc8f47b4550d1856ed8c1c0dd` | `c693e1436b0ffa38af0256e04a1b9e73efb9575d330ebe0ff6e8ff942fa7aa89` |
| `preview02` | 0 / 375.545 s | 3 | `f191caa48bb4f6ba37d7aa5eb6b07aef94e20d5f297405a6ab05836b83f61eba` | `fccad350421d8857fd34ee6364860a92352aaa2cd84c4832702a2a6f2cf7188f` | `cdfc7909443053e12224e6b965e9f1b72b598f41b3253da8f7add39e88a8ec78` |
| `preview03` | 0 / 373.269 s | 3 | `2bb4d82d48b8ab5571067cb2592e62eda677dd36b521a0b32a9a1726a59a81bb` | `e6b6c12b008325280647f68f41e2fb504ac5fdfb9fd2c74dfbc90b0ff2d03935` | `b1f036f91a9ea310a9f4195a72fc2920e38cfec5394b48a520c489664d3df289` |

The trusted post-agent `validate_capture` accepted all three completed,
contiguous source histories and their pinned immutable files. Among files in
each final workspace, only `kernel.hip` differed from the starter. Preview 3
also left four **empty** directories (`.agents`, `.aws`, `.codex`, `.git`) in its
workspace; its snapshot and source allowlist remained valid. Its transcript
records a rejected cleanup command targeting temporary files. Neither event
is a kernel correctness finding, and both remain in the private capture.

Raw events, blobs, diffs, workspaces, and any future scorer outputs remain
outside this repository. A credential-pattern scan and exact-value comparison
against the current Codex auth profile found no matches in the three captures;
this is a review check, not proof against every possible disclosure. Full
private archives were transferred to `mi350-2`; each archive SHA256 and
extracted source SHA256 matched at both ends. No candidate has a scored
incidence or performance result from this capture step.
