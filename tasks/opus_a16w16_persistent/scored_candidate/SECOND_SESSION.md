# OPUS full-K second graph session

**Unscored source-equivalence control, not editable-header parity.** On
2026-09-29, a fresh MI350X process/container session repeated all eight
full-K public shapes with pinned AITER `opus_bmm` versus the unchanged-source
standalone HIP adapter. The graph driver used 32 calls per capture, five
direct and two graph warmups, 20 alternating paired GPU-event replay samples,
and read-only output/guard/input checks after graph replay. For each case,
the driver reported its two pre-graph independent Torch FP32 oracle checks,
post-graph oracle/guards, noise qualification, and self-attributed GPU PID
gate as passing. No private fixture or agent candidate was evaluated.

| Case | First ratio | Second ratio | Second AITER/adapter relative MAD | Second raw SHA256 |
| --- | ---: | ---: | ---: | --- |
| aligned-nooob | 1.0169 | 1.00235 | .00358 / .01012 | `af833b133ab7ff1da8d88e28abdeb6b6557af8890b39ab5d334c083c8f2a0faa` |
| m-tail-oob | 1.0111 | .99206 | .00145 / .00283 | `8d049c393adfce9ba8746462f718028592e423df308c78be049a961608bc62fe` |
| n-tail-16-aligned | .9971 | 1.00124 | .00434 / .00408 | `c913fa70c13834638e1040eded8d2027cb26122357f8030ddcf3571e6921841d` |
| m-one-row-tail | .9953 | .99794 | .00183 / .00461 | `63d0e7a06f3cfe57eeb5d22fccc53a859517f57e9c5480d2db3ae35859f07a75` |
| n-first-vector-tail | 1.0244 | .97739 | .00210 / .00212 | `eab3cd45b25d1f44a3c3cbf7dbb12e2ab4564fc3c409ebfbbac54f6907e5b8e8` |
| mn-combined-tail | 1.0127 | 1.01087 | .00232 / .00201 | `27480bcb73243d46fefba06f3d90c7d57511ff23c8efdc5559b3d2613e6b8886` |
| k-min-even-loop | 1.0038 | 1.00145 | .00208 / .00236 | `8c71b669056a2dbab2f889f940ff10657502b060409561698a3cc404bb657263` |
| xcd-padded-grid | 1.0100 | .99747 | .00167 / .00166 | `e04f003f0747aeebd5186e57036e381eb0c53a906ac574ff519a23f8f44ad79a` |

All second-session ratios were <=1.05 and both relative MADs were <=.05.
The first-session values and raw commitments are in
[`ADVERSARIAL_GRAPH_FEASIBILITY.md`](../ADVERSARIAL_GRAPH_FEASIBILITY.md).
The second-session inputs for graph timing were seeded random BF16 for every
shape, whereas the separate six-step admission used each public matrix
pattern. The graph driver itself does not profiler-check exact kernel symbol;
that was done in the earlier correctness-only admission. This repeat used
the same established AITER JIT cache, not the proposed two clean production
JIT caches. It therefore cannot validate candidate-header compilation,
same-boundary production parity, or an optimization gain.

Provenance: pinned AITER SHA
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`; public historical matrix
SHA256 `e4a2346d96db18d6f071fae8b1caeffd327342dcc6976c762d47af37159fd3ed`;
adapter source/binary SHA256
`56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36` /
`356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50`;
driver SHA256 `9707aec738b58394e7a287f473f09589bb2b1f835114df3a82ad2687a7474a13`;
host GPU attestation SHA256
`4b7b8c1d2bf204b465422b93c182fdb8cf4aa478bb47ecf2494380b05d364d4c`;
compiler image ID
`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`.
The second-session raw JSON and logs remain private off-repo. An initial
attempt stopped before any kernel because its private driver dependency was
not staged; its preserved log SHA256 is
`75c0696398eab53cd3eec2e02f95fe661d6d9bbc78f4c0bef64a068885c685fb`.
It is infrastructure-invalid, not an operator result. At release, all
owned containers had exited; `rocm-smi --showpids` showed only the
preexisting zero-VRAM GPU service. `scored_eligible=false`.
