# Paired Rust source control for divergent Wave64 use

## Question

The observed quant HIP agent error placed `__shfl_down` in an even-lane branch
while requesting data from an odd lane. The analyst repair moved the shuffle
before the branch and kept the byte store conditional. See
[`analysis/quant-wave-participation.md`](../../../analysis/quant-wave-participation.md)
and the separate [one-wave HIP control](../../hip/README.md). Does fe2o3
reject the corresponding *source* pattern through an authenticated frontend?

These paired sources test a narrower, expressible analogue on fe2o3's `gfx942`
safe API. `bad_divergent_collective.rs` calls the safe Wave64 sum from only
even lanes. `repaired_uniform_collective.rs` calls it from all lanes and
branches only on the returned value. The public API does not offer a safe raw
shuffle or an FP4 pack operation, so these are not literal quant ports.

## Pinned source check

Run `./cases/fe2o3/source_wave_participation/run-typecheck.sh` from this repo.
The script makes a disposable worktree from fe2o3 commit
`4fb8ae500e38ad22475861aae7c3b3ef238068f6`, substitutes each source into
the existing `gfx942_wave_lds_v1` compiler fixture, and executes
`cargo check --locked` for both. It does not modify the live fe2o3 checkout.
The run uses the pinned `nightly-2026-04-03` specified by that worktree's
`rust-toolchain.toml`. The result on 2026-09-29 was:

| Source | Ordinary Rust typecheck |
| --- | --- |
| Divergent collective | Accepted |
| Uniform repair | Accepted |

Both results are expected: Rust typechecking alone does not prove GPU
convergence. In the pinned fe2o3 API,
`crates/fe2o3-device/src/collective.rs` documents that the compiler must prove
all 64 physical lanes execute the reduction convergently; its raw
`__fe2o3_wave64_shuffle_index` operation is hidden and unsafe. API
inexpressibility for the literal safe-Rust shuffle is **not** a proof that an
agent's HIP shuffle would be rejected.

Source SHA-256: divergent
`a0bf05a222e0542be069c2aa8b2fe597d91cfdf19f6f66f4a7d72abde8abd12a`;
uniform repair
`59e6bce9d7f2e737f4594d601de0655c202b305ac1ba3e48c8cc6f5180d14e83`.
An earlier attempt at fe2o3 `182e989a279454346953b990ce14aceeb1ebbb70`
stopped before a source result: its debug backend was 575,147,184 bytes,
above the 536,870,912-byte gate. Retrying with `CARGO_PROFILE_DEV_DEBUG=0`
cleared that infrastructure gate but the older API lacked
`Gfx942Collectives::current()`. Neither outcome is classified as a verifier
rejection, semantic pass, or kernel result; no fe2o3 source was edited.

## Authenticated boundary

The pinned fe2o3
`examples/wave64_collectives_v1/README.md` explicitly says current
source-to-Kernel-IR qualification is unsupported: the historical template
profile was retired and no present source-to-KIR receipt exists. The production
source-capture test in
`crates/rustc-codegen-fe2o3/src/production_rustc_driver_wave64_capture_source_v1_tests.rs`
observes safe wrapper calls in inert semantic MIR and then expects a
pre-ranked-materialization or ranked-projection refusal. Its direct raw-shuffle
case expects a **source-safety** refusal for a user-provided `unsafe` block.
Neither is a demonstrated semantic rejection of this divergent safe-wrapper
source, and neither yields an executable positive proof for the repair. We did
not bypass that gate with hand-authored IR or claim that the ignored capture
test ran for these two fixtures.

The older [IR-only case](../README.md) separately shows that an untruthful
full-wave convergence annotation can make a divergent shuffle pass the IR
verifier. That observation cannot be promoted to authenticated Rust behavior.
**Conclusion:** on this pinned production path, fe2o3 cannot yet be credited
with catching the quant wave-participation agent error end to end. This is an
unsupported source-to-KIR path, not a measured semantic rejection or a proof
of correctness. All source controls here target `gfx942`; no `gfx950`
lowering, ISA semantics, or MI350 execution is established.
