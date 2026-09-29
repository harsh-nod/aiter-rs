# AITER OPUS persistent partial-K reproducer

This is a standalone, AITER-only gfx950 reproducer for the BF16 persistent
`opus_bmm` kid 300 path. It uses public random operands at
`(batch,M,N)=(1,8192,4096)` and K=194, 254, 256, compares each complete
output with Torch FP32 matmul, and reports mismatches under AITER's public
test tolerance (`atol=0.5`, `rtol=0.03`). It then repeats K=194 with the
*same logical operands* in rows of physical stride 256, first with zero
padding and then with one padding. All calls check 512-element outer
input/output canaries, input immutability, and padding immutability. The
script writes no files or private host information and has no `aiter-rs`
adapter, matrix, or harness dependency.

Run inside a ROCm/PyTorch environment on gfx950 with a clean AITER checkout:

```sh
PYTHONPATH=/path/to/aiter AITER_JIT_DIR=/writable/cache \
  python3 analysis/aiter_audits/opus/probe_partial_k.py \
  --aiter-checkout /path/to/aiter
```

The script prints JSON containing the imported checkout SHA, mismatch
counts, maximum absolute errors, tensor strides, guards, and a final
`partial_k_padding_dependence_observed` boolean. Exit status 0 means the
probe completed without guard/input corruption, **not** that outputs were
correct; the JSON distinguishes reproduced, fixed, and inconclusive runs.
Use a clean, writable JIT cache outside the checkout and an uncontended GPU.
Run static checks without Torch/GPU using
`python3 -m unittest discover -s analysis/aiter_audits/opus -p 'test_*.py'`.

The [persistent tuner](https://github.com/ROCm/aiter/blob/9ef0d08fc20c81755c30c4f867b2f39664c75702/csrc/opus_gemm/opus_gemm_tune.py#L698)
accepts kid 300 with even `ceil_div(K,64)` and 16-aligned N, without a
`K % 64 == 0` guard. The
[pipeline](https://github.com/ROCm/aiter/blob/9ef0d08fc20c81755c30c4f867b2f39664c75702/csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh#L196)
uses ceil-div K loops and async A/B loads; the
[wrapper](https://github.com/ROCm/aiter/blob/9ef0d08fc20c81755c30c4f867b2f39664c75702/aiter/ops/opus/gemm_op_a16w16.py#L89)
accepts padded A/B row strides. The physical-padding intervention can
show that the partial-K path consumes bytes beyond logical K, but it does
not by itself prove a particular GPU instruction is responsible. No
upstream issue has been filed from this study yet.

## Pinned validation

One correctness-only run on MI350X/gfx950 with AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39` and the script above
reported:

| Input | Mismatches / 33,554,432 | Max absolute error |
| --- | ---: | ---: |
| Contiguous K=194 | 30,689,090 | 47.108 |
| Contiguous K=254 | 14,448,196 | 18.077 |
| Contiguous K=256 | 0 | 0.250 |
| K=194, row stride 256, zero padding | 0 | 0.249 |
| K=194, row stride 256, one padding | 33,554,432 | 62.452 |

The padded-row calls used identical logical K=194 values; only their
physical padding changed. Every input/output guard and logical-input and
padding immutability check passed. The raw JSON remains off-repository,
committed by SHA256
`43740d5266a808d14948adc02f863128e4f3e5434bbcd8f5d0c7bcbfc5ed7ea0`.
The executed script matched this repository file's raw SHA256
`936aeec77907bc58747bf3151189e3187748324fc987b6b77ac48710db6e8f57`.
This observation is specific to the pinned build; identical relevant
source files on current main suggest, but do not prove, current runtime
behavior.
