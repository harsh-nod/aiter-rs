# OPUS persistent A16W16: nine-case public admission

**Historical, pattern-specific unscored analyst control.** The nine public cases in the separate
[adversarial matrix](adversarial_matrix_candidate.json) passed a bounded
correctness-only admission on one host-attested MI350X (`gfx950`, PCI
`0x75a0`). Pinned AITER's exact-kid `opus_bmm` and the unchanged-source
standalone HIP adapter were compared independently against
`torch.matmul(A.float(), B.float().transpose(-1,-2))`, with BF16 output
checked at `atol=0.125, rtol=0.02`. This is **not** a hidden-test result,
agent trial, optimization gain, performance-parity admission, or memory
safety proof. `scored_eligible=false`.

**Subsequent random-input testing invalidated general K=194 admission.**
The K=194 row here used a last-K one-hot pattern and passed only that
fixture. Both pinned AITER and the bitwise-identical adapter fail the same
shape with fresh seeded random BF16 values; see the
[K-tail causal receipt](K_TAIL_REPRO.md). The nine-case matrix remains an
immutable historical artifact, but its K=194 row is **excluded** from any
supported/scored domain. The separate [graph receipt](ADVERSARIAL_GRAPH_FEASIBILITY.md)
timed only the eight full-K rows after correctness checks.

All nine cases resolved to their requested persistent kid (`300`, OOB, or
`1300`, no-OOB) without workspace. Each case ran six steps: initial inputs,
same-input repeat, changed-input reuse, alternate output allocation, restored
inputs, and an outer input-guard perturbation. That is 54 checked steps and
108 AITER/adapter launches before the profiling calls. Every step had zero
bad elements on both paths, finite BF16 output, input immutability, intact
input/output canaries, unchanged inactive outputs, and bitwise-identical
AITER/adapter output. Restored phase-0 results were bitwise reproducible;
changing only the outer guard values did not change output. A negative guard
perturbation is not a proof against all OOB reads. The current ABI has no
external scratch pointer: reuse checks cover internal launch state, input
and output allocation reuse, not an unexposed workspace contract.

PyTorch's GPU profiler then saw the exact
`gemm_a16w16_persistent_kernel` specialization for each path and kid. Each
path's profiler name set had one A16W16 GEMM name, with `ELb1` for kid 300
and `ELb0` for kid 1300; AITER and adapter names were identical. Outputs,
guards, and inputs were rechecked after profiling. This in-memory event-name
check establishes runtime kernel identity, **not** a device-instruction proof
or a timed profiler trace. No separate profiler CSV was generated; each
private raw result file commits its recorded event names, and the distinct
UTF-8 kernel-name hashes are:

| Kid | AITER and adapter profiler kernel-name SHA256 |
| --- | --- |
| `1300`, no-OOB | `9b95d44473721c5af77092a2f48bbfc5150b241676fc0d1ec8889a586c0972c9` |
| `300`, OOB | `d0c27cb8f0d1cacd6b4eba09ce1bbf7e5340ad3616190ee233f9aed0d4295958` |

| Public case | `(M,N,K)`, kid | Six steps / oracle | Private raw result SHA256 |
| --- | --- | --- | --- |
| `aligned-nooob` | `(8192,4096,256)`, 1300 | 6/6, zero bad | `5e146262ff8de388f7d0d5164921a6ae4f9f24de6a3b9880da27cc73e14e0f5e` |
| `m-tail-oob` | `(12287,4096,256)`, 300 | 6/6, zero bad | `7c2bf30faa6716a0f58fc8a83b9551b5c728cdb39b5a404bb49595376394836e` |
| `n-tail-16-aligned` | `(16384,2192,512)`, 300 | 6/6, zero bad | `a8f175e1b76bb5a02651a0861a35006351419f07163cd5f892dd701d4e2b7ff4` |
| `m-one-row-tail` | `(12033,4096,256)`, 300 | 6/6, zero bad | `fdeff145d500f681877bbd253aac434c42c7eca1f9835dd162e6fa54d7b7249a` |
| `n-first-vector-tail` | `(16384,2064,256)`, 300 | 6/6, zero bad | `47e6da414d8cdc2e0ba6e843df2764c1ac7a04785f848b66dc9485a2711795c2` |
| `mn-combined-tail` | `(16383,2064,256)`, 300 | 6/6, zero bad | `8dcb3cc74061289a86c8d7586636b5141eb79f74cf24680d2d3ba00358928c0a` |
| `k-partial-final-tile` | `(8192,4096,194)`, 300 | 6/6, zero bad | `28b63a8890e96615066113365e4d8c854491e44cdd2b27b5a422c112fb373c29` |
| `k-min-even-loop` | `(8192,4096,128)`, 1300 | 6/6, zero bad | `218a05661f1e4549c04277652c062d642bb9f42cd725a1595b1550e2b7df2ded` |
| `xcd-padded-grid` | `(4096,16384,128)`, 1300 | 6/6, zero bad | `e0d33e7b1ebe752e187d68083d3e2af80faf14d43518a588b25a78589b989e7d` |

The partial `K=194` case passed this particular one-hot fixture only; it is
now excluded after the random-input repro. The earlier `N=2177` pretrial
remains excluded by the pinned tuner's `N % 16` rule and is not counted as
an agent or upstream-kernel bug. Padded A/B row strides remain outside the
standalone adapter ABI even though the pinned AITER wrapper supports them;
they require a separate ABI and admission study.

## Provenance and limits

| Artifact | Raw SHA256 |
| --- | --- |
| Pinned AITER revision | `868ccf62a0bcad3aa47f92728340ccb37ed4fb39` |
| Nine-case public matrix | `e4a2346d96db18d6f071fae8b1caeffd327342dcc6976c762d47af37159fd3ed` |
| Oracle/domain module | `4fa89a0c61e04797ded6db2f38a7618d6ea6794fa2d78d40725fe8e0f02d85d4` |
| Pinned persistent device header | `bf621cfe97d2b38a89ca668c57b32a10094aeef92b4fdbb723e530e0abd6946b` |
| Standalone adapter source | `56556edb068aa1b761b23d554b4b54223c627de69710ce04da38beeffea5fa36` |
| Standalone adapter binary | `356cf5ae803a7a4d30634e5a3876fd5d85f816c44405890db9fca55f55b79e50` |
| Per-case GPU admission driver | `f5cd2aee4760ddcbdc9aa64ccab3aa0aee68f40d8a9e34a8de2bba3093036a5f` |
| Host watchdog wrapper | `68409b80db532860f55c188205e6e17dc07248afeb3181b0686c2992f23b81ef` |
| Private host GPU identity report | `4b7b8c1d2bf204b465422b93c182fdb8cf4aa478bb47ecf2494380b05d364d4c` |
| ROCm PyTorch image | `sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b` |

The image reported PyTorch `2.11.0+gitd0c8b1f` and HIP `7.2.53211`.
PyTorch's in-container GPU name was blank and in-container `rocm-smi`
reported product `N/A` due libdrm, so SKU admission required **all** of:
Torch `gfx950`, container PCI `0x75a0`, host MI350X/PCI `0x75a0`, and the
same GPU GUID in host and container reports. Docker used the host PID
namespace; every valid result attributed its own active GPU PID and found
no foreign active GPU process. After all nine per-case containers exited,
the host PID table contained only its preexisting zero-VRAM service.

Five pre-kernel infrastructure attempts were excluded from kernel evidence:
two invocations used the image's default `vllm` entrypoint, one hit Git's
read-only-bind safe-directory check, one overrequired a nonempty PyTorch GPU
name, and one overrequired an in-container ROCm product name. The latter
two preserved private logs have raw SHA256
`140417db0ce92459989ca7eb0bf9cbaea2bb8d15b5616482482e0af7580aa5ed`
and `ec48e5ac3c5795276bd5295e91d28b9ac2c911ef3500393031de169aff8cbf03`.
Each valid case used a separate container with a 300-second watchdog and
forced owned-container cleanup; none timed out. Raw JSON and logs remain
off-repository. This receipt itself ran no latency comparison; the later
graph feasibility control is documented separately. No private cases were
used, and the eligible agent-trial count remains zero.
