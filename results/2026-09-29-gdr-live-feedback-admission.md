# GDR live public-feedback prototype admission

This receipt covers a **public-only seed-source protocol smoke**, not an agent
trial. Task revision `native-opt-live-feedback-prototype-v2` remained
`scored_eligible=false`; no attempt enters an error-incidence denominator.
No benchmark request or withheld case ran.

## Frozen boundary

- Trusted code revision: `982e7456069f35a0e6cb12dc6dcad2e09575acd6`.
- Clean remote task checkout: `b5dc96fc0f665607b001a0bcbda52dd1819637cd`.
- Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`.
- MI350X/gfx950 host-report SHA256:
  `0707955d447ea0511e38fb0ab57c024e74fdc405452790db87d8e97989205ceb`.
- ROCm image ID:
  `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
- Seed kernel SHA256:
  `5a70279fb1611eb768eb3c0fddc92e35aa94b6b425c839d7a5e280342929e519`.

The local trusted broker froze source, transferred it with a tree hash, ran a
remote network-disabled, public-only Docker scorer as the non-root host UID,
checked the returned raw-result SHA, and sent only the sanitized response to
the helper. The remote source tree and pinned AITER checkout were clean. The
source snapshot tree SHA256 was
`a70e75fabc65944ad487bd3d539d49c1ada1a081605299431ca433251ce3fe84`.

## Bounded results

1. Initial smoke, on earlier trusted code `59c9cb63cbfa689a71fbce26c27c697dc254f129`:
   raw public scorer returned `complete` and 6/6 visible passes, but broker
   sanitized to generic `error`. Its extra assertion required the container's
   `gpu_name` string to equal the SKU. Non-root container `rocm-smi get_name`
   returned empty even though the pinned host report, gfx950, and PCI `0x75a0`
   passed the scorer's attestation. This was a broker validation defect, **not**
   a candidate kernel failure. The private event-log SHA256 was
   `bcaec7361582171d13de94657aa40547d7d5adf9a9e24a153afb71f57b3f3011`.
2. Fresh repeat after the exact regression fix: one `correctness` request,
   helper exit 0, broker `complete`, public response `complete`, **6/6 visible
   passes**. Compile return code was 0. Pre/post GPU PID probes saw the scorer
   itself and zero foreign active PIDs. Raw result SHA256 was
   `edbfeeee4d3acbe668825db90038875b39bdfb26cd1320c2736ffc6d27cc719d`;
   private event-log SHA256 was
   `c04614b43f87b1178b8ae217ec90605c0fc9026b7438b23fb17329b45572be38`;
   response SHA256 was
   `de23183e37db5fd49a384c9a345dabb762a093704041035173e27ad81ebb8304`.

All raw JSON, transfer logs, and remote paths remain in private storage.
The first result is retained as an infrastructure/protocol failure; it is not
silently replaced by the repeat. CPU fake-broker tests and historical GDR
boundary/replay tests pass 24/24. A local Codex sandbox dry-run validated the
frozen helper mount and `codex-cli 0.159.1`; it did not invoke an agent.

## Remaining gates

- Run a fresh **unscored** real-agent session in the isolated local sandbox,
  with the v2 helper and bounded public requests, preserving every source
  snapshot and raw trace privately. A seed smoke and CPU fake capture do not
  prove the live agent loop works.
- Validate that completed capture in a clean trusted checkout and replay the
  frozen final source first against public, then against the separately mounted
  withheld manifest. Private replay is post-agent only; native candidate code
  shares a process with the hidden scorer, so this is a nonadversarial
  confidentiality boundary, not strong secrecy against malicious code.
- Exercise a public benchmark request under an exclusive, attested GPU window
  and review noise qualification separately from correctness. No performance
  parity is implied by this correctness-only smoke.
- Freeze a **new** scored-eligible task revision only after those gates and the
  threat model are reviewed. Do not relabel v1 previews, this v2 prototype,
  or CPU fake runs as scored incidence evidence.
