# OPUS persistent A16W16: adversarial task freeze candidate

**Proposed public matrix, not GPU-admitted, not scored, zero agent trials.**
This is a source-preserving optimization study of pinned AITER
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39` on one MI350X/gfx950. It
is not a from-scratch GEMM task or evidence that an agent made any error.
The three existing cases in [`cases.json`](cases.json) passed AITER dispatch
and the source-equivalent HIP adapter's guarded oracle and graph control;
the six additions in
[`adversarial_matrix_candidate.json`](adversarial_matrix_candidate.json)
have **not** been run on the GPU. The CPU-side
[`adversarial_matrix.py`](adversarial_matrix.py) validates their proposed
domain and provides a separate Torch FP32 mathematical oracle. A source
contract is not a runtime admission result.

## Exact proposed boundary

| Property | Candidate freeze |
| --- | --- |
| Operation | `Y[0,m,n] = sum_k A[0,m,k] * B[0,n,k]` |
| Types | BF16 A/B and BF16 Y; Torch FP32 accumulation reference |
| Batch, bias, split-K, workspace | `batch=1`, no bias, `split_k=0`, no workspace |
| Current standalone ABI | [`source_adapter.hip`](source_adapter.hip): `aiter_rs_opus_a16w16_persistent(a,b,y,m,n,k,kid,stream_ptr)` |
| Layout | Contiguous A `[1,M,K]`, B `[1,N,K]`, Y `[1,M,N]`; B is **not** stored transposed |
| Kernels | Exact kid `300` with OOB predicates; kid `1300` without OOB predicates |
| Golden output | `torch.matmul(A.float(), B.float().transpose(-1,-2))`; all elements checked at `atol=0.125`, `rtol=0.02` |

The pinned [tuner domain](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/opus_gemm_tune.py#L698)
requires at least two, **even** 64-wide K loops and `N % 16 == 0` even for
kid 300. Kid 1300 also needs full 256x256x64 M/N/K tiles. The pinned
[launcher](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/codegen/gen_instances_gfx950.py#L636)
uses a 512-thread block, 256 CU/8 XCD persistent-grid formula and `M` tile
loop. Every proposed row has at least two M tiles per workgroup and dense
strides fit signed 32-bit integers. Admission must still check exact-kid
resolution, no workspace route, profiler-visible persistent symbol and
expected HAS_OOB specialization on pinned AITER **and** adapter. The source
header/adapter control is documented in [`FEASIBILITY.md`](FEASIBILITY.md).

| Public case | `(M,N,K)` | Kid | M tiles/WG | Adversarial intent | Status |
| --- | --- | ---: | ---: | --- | --- |
| `aligned-nooob` | `(8192,4096,256)` | 1300 | 2 | full-tile anchor | admitted anchor |
| `m-tail-oob` | `(12287,4096,256)` | 300 | 3 | last-row M tail | admitted anchor |
| `n-tail-16-aligned` | `(16384,2192,512)` | 300 | 2 | partial N tile | admitted anchor |
| `m-one-row-tail` | `(12033,4096,256)` | 300 | 3 | only one valid row in final M tile | proposed |
| `n-first-vector-tail` | `(16384,2064,256)` | 300 | 2 | first 16 valid columns in final N tile | proposed |
| `mn-combined-tail` | `(16383,2064,256)` | 300 | 2 | simultaneous M and N masks | proposed |
| `k-partial-final-tile` | `(8192,4096,194)` | 300 | 2 | last K contribution at element 193; 4 ceil-div loops | proposed, runtime support unverified |
| `k-min-even-loop` | `(8192,4096,128)` | 1300 | 2 | two-loop prologue/epilogue boundary | proposed |
| `xcd-padded-grid` | `(4096,16384,128)` | 1300 | 4 | `split_m=4`, grid-Y padded to 8, overshoot WGs | proposed |

The generated BF16 patterns include random, alternating signed
checkerboard, exact cancellation, and a last-K one-hot. Every case has two
input phases whose FP32 references differ. These are public fixtures only;
do not put withheld shapes, seeds, or raw private GPU results in this repo.

## Correctness admission protocol

1. Pin the exact AITER SHA, clean source header hash, adapter source/binary
   hashes, image, compiler flags, and MI350X SKU; verify both exact-kid
   dispatches before treating a case as supported. Build the adapter from
   trusted pinned source, never an agent-provided build command.
2. For each row, allocate A/B and separate AITER/adapter Y buffers with
   leading/trailing canaries. Generate phase 0, poison both outputs, launch
   each path twice on its supplied stream, synchronize, and check **every**
   BF16 output against the independent Torch FP32 oracle. Also require
   finite outputs, unchanged A/B, intact input/output canaries, and matching
   launch/return status. Bitwise AITER/adapter equality is useful supporting
   evidence, never a replacement for the oracle.
3. Without reallocating A/B or the first Y pair, overwrite inputs with
   phase 1. Reinvoke on the same outputs **without clearing them**; also
   alternate to a second guarded Y pair and verify the inactive pair stays
   untouched. Restore phase 0 and repeat. Compare against the reference
   after every step, including first/last M/N/K edges. The current ABI has
   no external scratch pointer, so this tests internal per-launch state,
   output reuse, allocator reuse and possible stale data, not a workspace
   contract it does not expose.
4. On a separate copy of each case, vary only the outer input guard values
   while logical A/B stay identical; output changes are evidence of an OOB
   read. Unchanged output is **not** a proof of memory safety. Record any
   memory fault or invalid row as a pretrial exclusion until its domain is
   established, then freeze supported cases and a private withheld matrix.

The existing `N=2177` pretrial failed with nonfinite output but is explicitly
excluded by the pinned tuner's `N % 16` rule; it is an unsupported-domain
observation, **not** an agent mistake or established AITER bug. Likewise,
batch>1, bias, split-K, noncontiguous Y, K-transposed B, and padded A/B rows
are outside the **standalone adapter** ABI. The pinned AITER
[wrapper](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/aiter/ops/opus/gemm_op_a16w16.py#L89)
does support K-contiguous A/B with padded row strides, but those are a
different route needing an extended trusted ABI and fresh parity admission;
do not silently include them in this matrix or misclassify their rejection.

## Agent-task freeze gates

The preferred future route lets an agent edit only the pinned persistent
device header in a clean AITER overlay, then the trusted runner JIT-builds
and checks the exact-kid `opus_bmm` boundary. The standalone adapter is a
separate source-equivalent feasibility/control path, not by itself proof of
production wrapper parity. Before any scored capture: run both paths on all
proposed public cases, exclude or repair unsupported cases **before** task
freeze, create an off-repo private matrix with only its hash published, test
guard/oracle/reuse behavior, confirm dispatch with a device profiler, and
establish multiple uncontended graph-replay sessions at the same operator
boundary. Predeclare per-bucket `<=1.05` AITER non-inferiority and a separate
optimization objective; an unchanged seed is parity, not an optimization.
Do not count the six proposed cases or the analyst adapter control as agent
error incidence. Until these gates and sandbox isolation pass,
`scored_eligible=false`.
