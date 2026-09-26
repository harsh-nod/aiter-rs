# Selected gfx950 dispatch tranche

The machine-readable [registry](gfx950_dispatch_tranche.json) records six
hand-selected AITER operator families at pinned revision
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Its source anchors trace
Python entry, dispatch branch, binding where applicable, and implementation
path. Recheck them with:

```sh
python3 inventory/validate_dispatch_tranche.py /tmp/aiter
```

The validator requires a clean AITER checkout at the exact SHA and verifies
each path, line, and symbol fragment. It cannot prove a branch is taken at
runtime, a compiled code object matches source, or that a HIP implementation
can meet AITER performance. `source_traced` means **static path trace only**;
`partial` means even the selected final kernel artifact varies or remains
untraced. No registry entry is `scored_eligible`.

| Family | Static trace | Provisional regimes | Distinction used |
| --- | --- | ---: | --- |
| MXFP4 quantization | Source-traced | 2 | Even versus Ceil scale semantics, both no-shuffle |
| Sparse paged prefill A16W16 | Source-traced | 2 | `H<=32` and `H>32` choose different OPUS HIP kernels |
| Per-row decode TopK | Source-traced | 3 | Stable short one-workgroup, stable long multiblock, unstable long multiblock |
| Packed BF16 GDR decode | Source-traced | 1 | Stateful update contract, including negative-sentinel input cases |
| MHC post/pre | Source-traced | 2 | `M<=1024` fused path versus large-M split path under `force_fused` |
| MegaMoE V2 | Partial | 2 | Compact bounded cohort versus fixed-slot arrival-ticket protocol |

The source-backed selected-tranche totals are **six families, five static
source traces, one partial trace, 12 provisional regimes** (10 under the
static traces and two under the partial trace), and **zero parity-admitted
task types**. These are neither a complete AITER inventory nor an incidence
denominator. The [hint index](gfx950_source_hints.json) finds 319 source
files naming gfx950, but includes comments/helpers and omits generic code
that also runs on gfx950. It cannot be used as a kernel count.

## Deduplication and exclusions

- The MXFP4 `M,K` grid, RNG seeds, and FP4 threshold cases are tests for one
  regime unless scale mode or output layout changes. The selected registry
  includes Even and Ceil; other round modes and shuffled layouts need their
  own source and oracle review before they count.
- Within a sparse-prefill head regime, CSR row lengths, token counts, and
  empty rows are tests. The `H=32/33` boundary is countable provisionally
  because [`mla_v4_prefill_opus_kernels.cu`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/py_itfs_cu/mla_v4_prefill_opus_kernels.cu#L130)
  selects different kernel templates.
- TopK widths/k values inside a selected one-workgroup or multiblock path
  are workload buckets. Stable versus unstable selection changes tie/order
  semantics, and the short/long boundary changes the underlying algorithm.
  The wrapper's older docstring says "always one-block," but its
  [actual branches](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/topk.py#L637)
  choose FlyDSL and HIP routes; code controls the registry.
- GDR negative sentinels, strided input rows, batch sizes, and repeated
  invocations all probe the same in-place state contract. A duplicated valid
  index is explicitly unsupported, not a new type.
- MHC `M=32,256,1024` are buckets of one fused path. Its large-M branch is
  a separate algorithmic regime; packed-BF16 and optional RMSNorm are left
  out until separately traced and vetted.
- MegaMoE token buckets, route skew, padding, and seed changes do not each
  become types. Compact and fixed-slot have materially different scheduling
  and communication protocols. A world-size-one smoke test does not exercise
  either cross-rank protocol and is excluded from their scoring.
- Gfx1250 prebuilt prefill, binary-only persistent-decoder provider PR #4975,
  and A4W6 ASM without a credible HIP parity route are not counted in this
  gfx950 HIP source-study tranche. They may support separate integration or
  feasibility work.

## 10,000-type feasibility gate

This tranche is deliberately high-risk and non-random, so multiplying its
12 regimes by a file count would be statistically meaningless. At this
revision, there is **no evidence-backed route to 10,000 genuine AITER-grounded
gfx950 HIP types**: only 12 provisional regimes have been traced here, and
none has passed oracle and parity admission. This does not prove a global
upper bound or impossibility. It does mean the study should not schedule
10,000 agents or call shape rows "types" yet. Reaching that target would
require a much wider static and runtime dispatch census, explicit contract
and algorithm distinctions for each proposed new type, independent oracles,
and a credible HIP parity route on the same gfx950 device. Report the
measured ceiling if that work finds fewer than 10,000.
