# GDR native optimization: trusted post-agent replay

This is an **exploratory, unscored** replay of three completed no-feedback
previews described in [the capture receipt](gdr-native-optimization-previews.md).
It is one task with three agent attempts, not three kernel types or an
agent-error incidence sample. The trusted adapter was main
`5bd520355e0b34306fd61cfa76521bfcda651e2b` against AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. It accepted each completed
capture freeze/final-tree hash, restored only the final `kernel.hip`, and used
the pinned ABI, public spec, and fixture from the trusted task.

The MI350X host report SHA256 was
`0707955d447ea0511e38fb0ab57c024e74fdc405452790db87d8e97989205ceb`;
the Docker image ID was
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`;
the pinned `hipcc` SHA256 was
`9a0fa4bc274155e7add34dc1897bfc08f2f5d6732ad8b9102b7d7a86c0e3d781`.
The withheld manifest matched the public commitment
`4bb61ab49dab61bff62e124cc5554ec92af3a846f41341f77229f58e38e5c9d0`.
Raw JSON, logs, candidate binaries, and the manifest remain private; only
aggregate outcomes and cryptographic commitments appear here.

## Outcome

| Run | Visible | Withheld | Public gate | Final tree SHA256 | Candidate binary SHA256 |
| --- | ---: | ---: | --- | --- | --- |
| Pinned seed control | 6/6 | Not run | 3/3 <=1.05 | Not an agent capture | `3d77fda2486e6adfee7eced30c9c382c0e03a21d66d739a188a7628a8edaba4b` |
| `preview01` | 6/6 | 4/4 | 0/3 <=1.05 | `6be5db5376c854b23461fc9ff8b06887fe5e09fc56a3ebfeeb218102684d3437` | `9f7af885997777c2842ffbaf3a91daa3201d4265a4ee1ec016405e192de93623` |
| `preview02` | 6/6 | 4/4 | 0/3 <=1.05 | `f191caa48bb4f6ba37d7aa5eb6b07aef94e20d5f297405a6ab05836b83f61eba` | `d34eac29db8267c0ce03c923b5528b7107b08b5f83aaf12b2fe4642b9421df50` |
| `preview03` | 6/6 | 4/4 | 2/3 <=1.05 | `2bb4d82d48b8ab5571067cb2592e62eda677dd36b521a0b32a9a1726a59a81bb` | `b5bf97b10dd7ad2e6fe9da650c99f499d3cb8a02597b35d35e9f0612bf6228e9` |

The withheld scorer runs stateful correctness only; it does not benchmark
withheld inputs. Both public and withheld stages attested the pinned AITER SHA
and MI350X/gfx950/PCI `0x75a0`. Public pre/post GPU PID probes saw the scorer
itself and zero foreign active GPU processes in every completed run. Every
bucket below was noise-qualified under the 5% relative-MAD cap. The protocol
used 32 captured calls, five direct plus five graph warmups, and 20 alternating
pairs. Ratios are candidate/AITER, lower is better; displayed metrics are
rounded to six decimals and full-precision raw results are privately committed
by the SHA256 values below.

| Run | Public bucket | Ratio | AITER MAD | Candidate MAD | Noise-qualified |
| --- | --- | ---: | ---: | ---: | --- |
| Seed | Valid slots | 0.996981 | 0.001421 | 0.002137 | Yes |
| Seed | Strided mixed | 0.998540 | 0.001455 | 0.002004 | Yes |
| Seed | Batch 16 | 1.035432 | 0.008716 | 0.006189 | Yes |
| `preview01` | Valid slots | 1.276437 | 0.001238 | 0.001940 | Yes |
| `preview01` | Strided mixed | 1.270797 | 0.002184 | 0.002291 | Yes |
| `preview01` | Batch 16 | 1.087302 | 0.004536 | 0.003904 | Yes |
| `preview02` | Valid slots | 1.200599 | 0.000536 | 0.002807 | Yes |
| `preview02` | Strided mixed | 1.221799 | 0.002725 | 0.002234 | Yes |
| `preview02` | Batch 16 | 1.098378 | 0.007921 | 0.020839 | Yes |
| `preview03` | Valid slots | 1.002299 | 0.001772 | 0.000888 | Yes |
| `preview03` | Strided mixed | 1.003089 | 0.001819 | 0.001265 | Yes |
| `preview03` | Batch 16 | 1.062332 | 0.004965 | 0.006570 | Yes |

The public spec proposes <=1.05 per bucket and <=0.95 geometric-mean
improvement. All three previews pass the tested functionality but miss the
per-bucket performance boundary in this session; `preview01` and `preview02`
show clear measured regressions. `preview03` misses only the Batch 16 bucket
by 1.2 percentage points beyond the threshold, so a fresh independent session
is needed before treating that narrow miss as stable. These outcomes do not
establish joint parity, agent-error incidence, or the absence of other bugs.

## Commitments

| Run | Public raw SHA256 | Withheld raw SHA256 | Sanitized private summary SHA256 |
| --- | --- | --- | --- |
| Seed | `5069006f1181255d4256f84a35fbb16db01ad10e282d6ca3fd9eb177286cc124` | Not run | Not applicable |
| `preview01` | `85e3a9b621cab88f78df2774ab086b708573904469c31376b19c49967b2c2a61` | `ae938eed04885cc58bff8d1f7adc13cc75934e1d1661daf9dd2e0dd55c073d94` | `3faee5d9b5c5a3b3c404e1f72df3b45a80c54420fda8ea7ed5d1d2bf0b2bee0e` |
| `preview02` | `05e3b4bb2ac9ca4f7b29cb7001c3d57c1c296aadd69e581edfee85d2a7ab97a8` | `2e643ac2b8dbb893ba58ebd86d6ff0d52c9debff8fe299c171881417d6c8e57b` | `fc355fa14702abb8c2dfe6780e2fc01e123fa550f46d03162c29eadc3c25d47d` |
| `preview03` | `83395b591717c4f147d8a8d8990143b661a67a63f32adbb99ac22fdb25cfe858` | `f925e2bc4ce3eada345ba751593f14f59917b3f26184d1c67f97a6572b5150f3` | `45cffd742024859eade8f2b2e009280dafeab50798d2478f12a7bbaa1ea736f5` |

## Infrastructure History And Limits

`preview01` first ran under adapter `a2c639d`. Its public scorer wrote a
root-owned mode-0600 result, unreadable by the trusted host adapter, so the
attempt was **infrastructure-invalid** and the withheld stage did not open.
That private public raw file committed as
`ad929d9e8f1ffb1dcb8781a9469ddfdb28a59307e30a7582288264b637e4ef9b`;
its apparent public 6/6 is not used as this receipt's replay verdict. Two
pinned-seed setup attempts then found that the image lacks a passwd entry for
the host UID and that its default `/aiter-jit-cache` is not writable by that
UID. Their private raw SHAs were
`ce0772e422f84bda6485d1c76646924a72572bff5a5a8d832b846ef30495de74`
and `27fc47fd6e3e75a80ca1ff4bf63e9fcfeb08fb7d509830657271a87d96fa9f52`.
Neither is a candidate result.

The final adapter ran as the trusted host UID with a minimal read-only
passwd/group mount and set `HOME=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`, and
`AITER_JIT_DIR=/tmp/aiter-jit-cache` instead of the image default
`/aiter-jit-cache`. The pinned image, source, compiler SHA, and HIP flags did
not change. JIT cache contents were ephemeral and no artifact hash was
retained; these timings therefore belong to this **same-run environment**,
not the earlier root-run seed environment or a frozen scored stratum.

The private manifest was absent from the agent workspace and public scorer.
However, the withheld scorer loads candidate native code in the same process
as the private manifest: this is **not** a strong confidentiality boundary
against malicious native code. The task still had no agent-visible feedback,
and no scored incidence or parity claim is made.
