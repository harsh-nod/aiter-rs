# Exact-kid OPUS A16W16 persistent admission

This is a correctness/dispatch admission probe, **not** an agent task or a
performance-parity result. Its public matrix is [`cases.json`](cases.json);
the target is pinned AITER `868ccf62a0bcad3aa47f92728340ccb37ed4fb39`
on gfx950. The three batch-1 BF16/BF16 cases use kid `1300` (aligned
no-OOB) or kid `300` (partial M or N tile). All have at least two persistent
M tiles per workgroup. The scripted grid formula mirrors the pinned
[`gen_instances_gfx950.py`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/codegen/gen_instances_gfx950.py#L636).

The independent oracle is `torch.matmul` on FP32 casts of BF16 A and B, with
the physical `[B,N,K]` B transposed. The frozen elementwise gate
is `abs(actual-reference) <= 0.125 + 0.02*abs(reference)`. The probe checks
all output elements twice, finite output, unchanged A/B, and 512-element
canaries around each input/output allocation. It also checks the Python
exact-kid plan and captures the real GPU kernel symbol with PyTorch profiler.
Passing is not a memory-safety proof; canaries do not cover every OOB read.

The persistent tuner [rejects `N % 16 != 0`](https://github.com/ROCm/aiter/blob/868ccf62a0bcad3aa47f92728340ccb37ed4fb39/csrc/opus_gemm/opus_gemm_tune.py#L698),
even for OOB-capable kid 300: vector stores can straddle rows. A pretrial
matrix mistakenly used `N=2177`; it passed the explicit-kid launch planner
but failed correctness with nonfinite output. That result is retained as
**excluded tuner-domain evidence**, not labeled an upstream bug or agent
error. The replacement `N=2192` still exercises the N tail while satisfying
the documented 16-element alignment. No-OOB kid 1300 additionally requires
full M/N/K tiles. The Python exact-kid planner is less restrictive than the
tuner, so these rules are enforced in `admit.py` before any GPU call.

Run CPU-only validation with:

```sh
python3 -m unittest discover -s tasks/opus_a16w16_persistent -p 'test_*.py' -v
```

Run the GPU probe only inside the pinned ROCm PyTorch image with a clean,
read-only pinned AITER checkout, writable JIT cache, exclusive off-repo
output path, and exclusive GPU window. For example, inside such a container:

```sh
PYTHONPATH=/workspace/aiter AITER_JIT_DIR=/workspace/jit \
  python3 /workspace/probe/admit.py \
  --spec /workspace/probe/cases.json \
  --aiter-source /workspace/aiter \
  --output /workspace/private/opus-a16w16-admission-result.json
```

`--no-profile` is for development and cannot yield admission pass. Raw
results, JIT artifacts, and profiler data stay off-repo. No latency is
measured. Before any later parity study, verify the same exact-kid branch
and whole callable boundary in graph/profiler timings under noise controls;
source visibility alone does not establish standalone HIP parity.

The later [source-equivalent adapter control](FEASIBILITY.md) checks an
independent HIP shared-library build against this exact-kid boundary using
guarded correctness and paired graph replay. It also documents the stricter
single-header clean-JIT overlay route for future agent optimization tasks.
