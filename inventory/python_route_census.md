# Python route discovery and review coverage

This [machine-readable index](gfx950_python_routes.json), described by its
[JSON Schema](python_route_schema.json), is generated from
git-tracked `aiter/ops/**/*.py` at AITER revision
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`:

```sh
python3 inventory/census_python_routes.py /tmp/aiter \
  --output /tmp/gfx950_python_routes.json
cmp inventory/gfx950_python_routes.json /tmp/gfx950_python_routes.json
```

The scanner parses every module with Python AST, records module-level public
functions/classes plus private `@compile_ops` binding stubs, and attaches
**hints** for FlyDSL/Triton paths, native binding decorators, backend-named
calls, and `gfx950` text. A `compile_ops` row records its requested module and
export name, but does not prove the compiled native symbol resolves. Public
symbols include helpers and configuration factories; private nondecorated
helpers and class methods are outside this index. The MegaMoE class review
explicitly covers its `forward` method. This is a discovery queue, **not** a
dispatchable-kernel or task-type count.

| Coverage measure | Count |
| --- | ---: |
| Python modules parsed / parse failures | 666 / 0 |
| Indexed symbols | 2,421 |
| Public symbols / private native-binding stubs | 2,325 / 96 |
| Direct `aiter/ops/*.py` symbols / nested-module symbols | 715 / 1,706 |
| Reviewed static routes / unreviewed symbols | 8 / 2,413 |
| Symbols whose body contains `gfx950` text | 132 |

Backend hints overlap and are **not** confirmed backend counts: 335
`compile_ops` decorators, 779 FlyDSL-path symbols, 871 Triton-path symbols,
and 438 symbols with backend-named calls. Path and name clues can describe a
helper or code that never dispatches on gfx950. Conversely, generic routes
may run there without a `gfx950` literal. Every unreviewed row therefore has
`trace_confidence=none` and `source_visibility=unknown` even when hints exist.

The eight [review records](python_route_reviews.json) reuse the six
[selected-tranche](gfx950_dispatch_tranche.json) traces and add two generic
routes missed by a gfx950-only text census:

- [`rms_norm`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/rmsnorm.py#L401)
  dispatches common 2-D FP16/BF16, hidden-size <=8192, non-model-sensitive
  inputs to the [visible HIP kernel](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/rmsnorm_quant_kernels.cu#L620)
  and other supported inputs to the
  [visible OPUS entry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/rmsnorm/rmsnorm_opus_norm_entry.cu#L25).
  This is a meaningful two-backend review target, not yet two admitted task
  types.
- [`rope_fwd`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/rope.py#L365)
  calls a generic JIT native binding and
  [visible HIP entry](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/kernels/rope/general_1c_uncached_fwd_kernels.cu#L7).
  The nearby cached, positions, offsets, 2c, 2-D, backward, and in-place
  wrappers are separate review candidates only when their contracts or
  algorithms differ, not merely because they have separate function names.

All eight reviews are **static source traces** (MegaMoE partial); none of
the new generic routes has a runtime dispatch observation, independent
oracle, or HIP performance-parity admission. The selected-tranche total
remains 12 provisional regimes and zero scored-eligible task types. The
full 2,421-row index has not been manually inspected, so it cannot bound
the number of genuine gfx950 algorithms above or below 10,000. In particular,
one entry can dispatch several algorithms, while many entries are helpers or
aliases. There is no defensible global 10,000-type upper bound yet. The
defensible **evidence bound today** is eight reviewed routes, 12 previously
proposed regimes, and zero admitted types. Shapes, seeds, dtype instances,
and tuned configurations do not increase those counts by themselves.

For the next tranche, prioritize the 715 facade-module symbols, then trace
public callers into nested FlyDSL/Triton modules and native bindings.
RMSNorm's add/quant variants, RoPE's genuinely different layout/state
contracts, attention/GEMM/MoE dispatchers, and persistent stateful operators
are credible families to audit. Each proposed regime needs a source-backed
branch or semantic distinction, an executable runtime-dispatch check on
gfx950, and an oracle/performance route before it becomes a scored task.
