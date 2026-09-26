# GDR decode BF16 unscored HIP pilot 01

This is one isolated, **unscored** Codex attempt on the frozen
[`gdr_decode_packed_bf16_pilot` task](../../../runs/tasks/gdr_decode_packed_bf16_pilot/task.json).
It is outside the agent-error incidence denominator. The agent received the
HIP ABI, nonfunctional starter, and functional contract, but no supplied
tests, GPU execution feedback, AITER source, scorer, or private cases.

- Codex CLI `0.157.0`, model `gpt-5.5`, medium reasoning; completed in
  147.797 seconds with four source snapshots and no capture errors.
- Final source: `workspace/starter.hip`, raw SHA256
  `74525d5e18833f71e53d6d332fafa7953746bb5c9538cb548298efbc72d37bff`.
  Its source-tree SHA256 is
  `3ffad208fbde25ca3659a2f0ca5c927959b878a3e809819a94e3e4cecda23888`.
- The final snapshot built with `hipcc -O3 -shared -fPIC --offload-arch=gfx950`
  in the pinned ROCm container. Library raw SHA256:
  `3b7ed6d207f441c7a337279ff916ca80950eac81c2ef56f963284cb58407f60d`.
- Trusted visible-only replay on MI350X: candidate and pinned AITER each
  passed all four public cases; the candidate passed all seven sequential
  steps. The [sanitized summary](visible-correctness-summary.json) records
  the pinned scorer, AITER, host-report, and image hashes.

The scorer used the independent CPU oracle and guards around state, output,
and inputs. It did **not** receive the withheld manifest or measure latency;
`joint_pass=false`. A visible-only correctness pass does not show that this
kernel is performant or that it covers all supported AITER behavior.

Raw `events.jsonl`, source blobs, snapshots, diffs, manifest, and final files
were reviewed before publication. The review found no API-key, GitHub-token,
private-key, bearer-token, host SSH, remote private-path, or withheld-case
content. The manifest mentions only the sandbox's temporary Codex auth path,
not auth bytes. The sandbox protects host SSH/private inputs from the agent,
but agent commands can read the temporary Codex auth JSON; this is not total
credential isolation.
