# OPUS production batch interruption

Status on 2026-09-29: **incomplete evidence, unscored**. The trusted
eight-public-then-twelve-withheld unchanged-header batch was launched on
mi350-2 from clean detached aiter-rs main
`1ab1a6d8e0d86354289f158816e4632ea9e5cc64` with pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. The batch driver SHA256
was `53e1339070c14644ba161abf9f3b34d20bdb65cd6c59110edee70c66e82b59eb`;
the per-case production scorer SHA256 was
`bc8c41e223cc53cc856198efa7460f5716b009e98001c9439c795f9c6985cc91`;
task JSON SHA256 was
`e94d748a44228639fc0e056ed5077e9a789fc6d2ebc03237124de9cf35b5b13c`.
The image ID, AITER/header hashes, withheld-manifest commitment, host GPU
attestation, and clean GPU PID preflight matched the frozen task before
launch. The private raw run root is
`/home/harmenon/aiter-rs-study/private/opus-fullk-production-batch-002`.

Before the connection was lost, read-only inspection of remote per-case
`result.json` files returned:

| Public case | Reported status | Candidate/AITER graph ratio |
| --- | --- | ---: |
| `aligned-nooob` | `single_bucket_pass` | 0.9907216344620488 |
| `m-tail-oob` | `single_bucket_pass` | 1.002328586712671 |

These are observed per-case fields, **not** a verified eight-case aggregate.
Their raw result hashes were not captured before the connection was lost.
The parent SSH exec session then disappeared (`Unknown process id 93375`),
and a new read-only SSH attempt failed before connection with hostname DNS
resolution error. No local copy of the attempt-002 raw result or batch report
was found. The final public count, withheld-stage status, batch exit code,
Docker/container state, and GPU PID state are therefore unknown. Do not
restart the batch or release/claim the GPU based on this receipt. Reconcile
the remote process and raw artifacts when access returns, preserving any
existing result files and their hashes.

Attempt 001 was a separate pre-kernel infrastructure-invalid run: snapshot
placement made the overlay a descendant of the candidate directory, which
`prepare_overlay.py` rejected. It produced 0 public and 0 withheld results;
private batch-report SHA256 was
`75e169ef8141e32814b186163955b7bf1aeb14630976ca1fbee9ca913b726b7c`.
The corrected sibling `source/` snapshot layout in main `1ab1a6d` was used
for attempt 002. Neither attempt establishes all-case parity, improvement,
scored eligibility, or agent-error incidence. `scored_eligible=false`.
