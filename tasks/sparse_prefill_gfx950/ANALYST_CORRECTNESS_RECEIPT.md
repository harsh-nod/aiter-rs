# Analyst control correctness receipt

On 2026-09-29, the unscored HIP analyst control was run on one gfx950 GPU
against the guarded, correctness-only scorer. The AITER checkout was pinned at
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. The candidate source SHA256
was `7c9f60721e952f975e5490c0eda37ea4ced3232958b134f462e7890709f9d61a`,
the compiled `.so` SHA256 was
`44894188eaf7134bfaf750c47499652d6fd4d5d2d7d9793fff04c755b7a01b4d`,
and the scorer source SHA256 was
`c74321942af8959a14010c86878deefd72966f4a760fa7eef15fd135083ffa51`.

| Matrix | Result | Aggregate checks | Private result SHA256 |
| --- | --- | --- | --- |
| Public | 5/5 passed, two invocations per case | All outputs finite; no guard or input mutation; zero mismatched elements at the scorer tolerance | `2c687920ed5da95eb7e0d7e3628be607c90cc2bc879c6c14a86e56c452c12bde` |
| Withheld | 4/4 passed | Aggregate only; matrix commitment `70feede754e3f5c18d0f1cbbf3f6434b6602cd97cab4e3d6b6bc11073bfcdebd` | `94241f21c698dca5a7847ea49f690fcb575c576045b6045d8260c8cd60f1fd2a` |

Both scorer results report `performance: not_run` and `scored_eligible: false`.
The private matrix and detailed result files remain outside the repository.
This control is not an agent submission, a performance comparison, or evidence
about agent bug incidence. It establishes only correctness on these committed
cases under the current CPU oracle and scorer contract.
