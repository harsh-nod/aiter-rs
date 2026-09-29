# Gfx950 kernel-type census: evidence and 10,000-type gate

Pinned AITER: `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. This report is
source-only and does not claim that every syntactic kernel can launch on
gfx950. Reproduce the [source-entry index](source_entry_hints.json) with:

```sh
python3 inventory/census_source_entries.py /tmp/aiter \
  --output /tmp/source_entry_hints.json
cmp inventory/source_entry_hints.json /tmp/source_entry_hints.json
python3 inventory/validate_dispatch_tranche.py /tmp/aiter
```

| Layer | Measured at the pin | What it means |
| --- | ---: | --- |
| Python module-level symbols and private binding stubs | 2,421 | Importable names and `@compile_ops` stubs, including helpers; [AST census](python_route_census.md) |
| Reviewed Python-to-backend paths | 8 | Static traces, not runtime-proven dispatch or eight task types; [reviews](python_route_reviews.json) |
| HIP/C++ `__global__` lexical tokens | 399 | Includes possible comments, macros, headers, inactive branches, and templates |
| Triton / Gluon `@jit` functions | 572 / 319 | May be device helpers rather than launch entries |
| FlyDSL `@kernel` / `@jit` functions | 137 / 235 | `@jit` includes helpers; `@kernel` may be unreachable on gfx950 |
| All five source-entry hints combined | 1,662 | 192 explicit-gfx950-path, 1,162 generic-path/unverified, 308 explicit-other-arch-path |
| Prebuilt gfx950 `.co` files | 1,483 | Compiled code objects, including layout/tile/tuning variants; 878 are under `fmoe/` |
| Gfx950 HSA CSV files / data rows | 55 / 1,095 | Config/code-object selections, not contracts |
| Selected dispatch tranche | 6 families, 12 provisional regimes | Five static source traces and one partial trace; zero `scored_eligible` in the [registry](gfx950_dispatch_tranche.json) |

The first three syntactic counts are deliberately **not additive** with the
Python-symbol count: a Python wrapper can call several device entries, and a
device entry can be reused by many wrappers. The 1,662 source-entry hints
are an optimistic audit queue, not 1,662 dispatchable kernels. In particular,
only a path name identifies 192 as gfx950-specific; the 1,162 generic-path
hints are neither included nor excluded by a real dispatch trace. The
prebuilt `.co` files are outside a source-visible HIP authoring study unless
their operator contract, provenance, and plausible HIP parity route are
established separately.

## Deduplication examples

- A gfx950 [model configuration](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/configs/model_configs/a8w8_blockscale_untuned_fmoe_glm5_3_flash_shared_gfx950.csv#L1)
  has 30 data rows; its first seven vary `token=1,2,4,8,16,32,64` while
  holding model dimensions, expert count, top-k, activation, dtype, and
  quantization fixed. Those are workload shape buckets, not seven new
  semantic types.
- The [gfx950 BF16 GEMM code-object CSV](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv#L1)
  distinguishes `tileM`, `tileN`, prefetch and split-K fields. Tile and
  prefetch choices are tuning variants; a preshuffled-B layout may change
  the input contract and needs review rather than blanket deduplication.
- [`rms_norm` and `rmsnorm2d_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/rmsnorm.py#L401)
  both call `_rms_norm_fwd_dispatch`; do not count two algorithms merely from
  two public aliases. Its HIP-common versus OPUS fallback is a real backend
  boundary, but still lacks an admitted task/oracle.
- [MHC's gfx950 policy](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/mhc.py#L366)
  picks different split-K/tile tuples at `M=32,256,1024` for the same fused
  post/pre contract. These are necessary performance/correctness buckets,
  not three independent task types. The large-M post/pre branch is a
  different algorithmic path and is kept separate in the selected tranche.
- The 878 code objects under `hsa/gfx950/fmoe/` span real quantization/activation
  distinctions **and** many tile/layout/codegen variants. Neither one file
  nor one CSV row implies one source-visible HIP challenge; each proposed
  distinction must be traced through its dispatcher and output contract.

## Feasibility bound and uncertainty

The defensible **current evidence bound** is 12 *provisional* semantic
regimes in the six selected families, with no scored-eligible type in that
registry. This is not a global lower or upper bound on all AITER operators:
the selection is deliberate and non-random, and only eight Python entries
have static traces. This census supports **neither** a global upper bound on
distinct gfx950 HIP task types **nor** a claim that 10,000 are reachable.

There is one useful *conditional* scale calculation. The scanner finds 1,354
source-entry hints that are not explicitly named for another architecture
(192 gfx950-path plus 1,162 generic-path). If these hints exhausted the
source-visible gfx950 study and every one were dispatchable, 10,000 distinct
contracts would require at least **eight semantic regimes per hint on
average** (`ceil(10000/1354)`). Both premises are unverified; helpers and
inactive code make this denominator too generous, while alias decorators,
generated kernels, and other source forms may be missed. It is a planning
pressure test, **not** a proof that 10,000 is impossible or reachable.

For an actual 10,000-type claim, enumerate runtime-selected wrapper-to-device
paths on the target SKU; deduplicate aliases, shape rows, and tuning variants;
write a distinct input/output/side-effect contract for each algorithmic
regime; then require an independent oracle and same-boundary HIP performance
parity. Count admitted types only after those gates. If the audited corpus
contains fewer than 10,000, publish that measured ceiling and the exclusions
instead of manufacturing extra types from shape grids.
