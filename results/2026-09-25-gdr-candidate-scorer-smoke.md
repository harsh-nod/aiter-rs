# gfx950 GDR candidate scorer smoke

Status: **correctness-only scorer validation**, not an agent trial or
performance-parity result.

- Hardware: host-attested AMD Instinct MI350X (`gfx950`, PCI model `0x75a0`).
  Pinned AITER source: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- Public spec raw-file SHA256:
  `fb5fdbfd031fa3d9fbf64a716b067b260d63ef29911e468e98a8279301a53145`.
- The trusted compiler command built the deliberately nonfunctional GDR HIP
  starter for `gfx950`. In the visible-only candidate scorer, AITER passed
  **4/4** cases and all **7** sequential steps against the independent CPU
  oracle. The starter returned HIP error `801` in every case, so candidate
  correctness was **0/4** and overall status was `fail`, not an inherited
  AITER admission pass.
- After input boundary canaries and bitwise snapshots were added for all six
  input tensors, pinned AITER also passed **8/8** visible-plus-withheld cases
  and **28/28** steps in a separate admission run. The withheld case identifiers
  and raw result remain private.
- Both runs were correctness-only: `performance=not_run`, `joint_pass=false`.
  No agent candidate was run on the withheld matrix. This does not admit the
  GDR task for scored trials or contribute to an agent-error denominator.
