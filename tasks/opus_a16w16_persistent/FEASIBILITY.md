# Source-equivalent OPUS persistent adapter control

This is an **unscored analyst control**, not an agent kernel or a claim that
an independently written HIP kernel matches AITER. The standalone
[`source_adapter.hip`](source_adapter.hip) includes the **unchanged** pinned
OPUS persistent device header and reproduces its exact-kid 300/1300 launch
arguments at a contiguous BF16 `[1,M,K] x [1,N,K] -> [1,M,N]` pointer/stream
boundary. It has no workspace or bias. The pinned AITER revision is
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`; the source header's SHA-256
is `bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b`.

The adapter compiled as an independent HIP shared library with gfx950 and
the relevant GPU flags from AITER's JIT `build.ninja`. `build_source_adapter.sh`
checks the source revision/cleanliness and requires the output outside both
source trees. A CPU-only ABI preflight then called the compiled library with
the valid default HIP stream `0` on all three shapes and verified invalid
N/kid rejection. The first preflight attempt had wrongly rejected stream 0;
it stopped before a candidate GPU launch and produced no timing result.
That host-adapter error is not an agent trial or OPUS kernel defect.

On `mi350-2` GPU 0 (MI350X, PCI `0x75a0`, gfx950), the final library and
pinned AITER each passed the independent Torch FP32 oracle twice on every
frozen [`cases.json`](cases.json) case. Outputs were bitwise identical between
the two implementations; input/output canaries and input immutability passed
before and after graph replay. No hidden cases were used.

| Case / kid | AITER median us | Adapter median us | Adapter / AITER | Relative MAD (AITER / adapter) |
| --- | ---: | ---: | ---: | ---: |
| aligned, `1300` | 73.769 | 73.285 | 0.9934 | 0.0028 / 0.0048 |
| M tail, `300` | 95.514 | 95.527 | 1.0001 | 0.0028 / 0.0029 |
| N tail, `300` | 109.331 | 107.001 | 0.9787 | 0.0022 / 0.0019 |

Each row used 20 alternating paired samples of 32-call graph replays; the
reported value is GPU-event elapsed time divided by 32. Both sides passed
the predeclared 5% relative-MAD noise screen on every row. These graph
measurements amortize Python enqueue gaps, but they are one session on one
device, not a profiler-validated production dispatch distribution or a
performance threshold for agent-written changes. The small differences are
consistent with compilation/measurement variation; they are not an
optimization gain.

Private raw artifacts stay off-repo. Public provenance:

| Artifact | SHA-256 |
| --- | --- |
| Public cases | `1372ef7f29034b9b3c2f4a43da47ad7618bdbd64df8c05d9903f56cc8c6f794b` |
| Adapter HIP source | `56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36` |
| Standalone `.so` | `356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50` |
| `feasibility.py` | `47643df538d52d0c079c2efbbb5b71c2bbaa20a609df72f5d4f3b8e70105b388` |
| Private raw result | `f650bbd2191c7ebbc0244cceabb4a42141c1d199fe4f2c538b28cde4b37c694c` |
| ROCm PyTorch image | `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b` |

## Agent-task route

The promising task is **source-preserving optimization in place**, not
from-scratch GEMM. Freeze AITER, kid IDs, shapes and hidden matrix before
agents work. Expose only the one HIP device header
`csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_persistent_gfx950.cuh`
as an editable candidate. `prepare_overlay.py` checks a clean pinned base,
the exact base-header hash, a regular candidate file/size cap, and verifies
that a newly copied trusted checkout differs only at that allowlisted path.
The agent must not run the trusted scorer or see hidden inputs. Build each
overlay with a unique JIT cache, then score the same exact-kid AITER callable
and full output/guard contract outside the agent sandbox. This route best
preserves AITER's host policy, compiler flags, and device semantics, though
the JIT may recompile a broad OPUS module per candidate.

The standalone adapter is a useful compilation/parity **control** and could
score an overlaid header against pinned AITER at the same device kernel
boundary. It is narrower than production `gemm_a16w16_opus` (no tuned
selection, bias, FP32 output, batch>1, or arbitrary strides), and its separate
host ABI/compiler build makes it a weaker scored route than the clean-JIT
overlay. Neither route yet has a frozen hidden matrix or an agent trajectory;
the current eligible-trial count remains zero.
