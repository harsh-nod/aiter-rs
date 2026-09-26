# gfx950 parity scorer

`python3 -m harness.run` scores an agent-produced HIP shared library, not its
source directory. The trusted runner compiles an immutable HIP source snapshot
with a frozen `hipcc` command. The library must export the C ABI in
[`references/quant_mxfp4_abi.h`](../references/quant_mxfp4_abi.h). It launches
asynchronously on the supplied stream into preallocated output tensors. The
scorer owns output allocation, deterministic inputs, correctness checks, and
GPU-event timing. An agent may not provide the timing code or a Python adapter.

The initial task is `gfx950.quant_mxfp4.even.noshuffle.v1` at AITER commit
`868ccf62a0bcad3aa47f92728340ccb37ed4fb39`. Its reference is the pinned
low-level `aiter.ops.quant.quant_mxfp4` binding with Even round mode (2), group
size 32, and no shuffles. This deliberately bypasses the high-level RoundUp
fast-path dispatch. Functionality requires **byte-exact** packed FP4 and E8M0
outputs against a separate scalar CPU oracle. The required performance buckets
use the same operator boundary with preallocated outputs and paired GPU-event
measurements. Every bucket must stay within the frozen 1.05 latency ratio and
pass the 5% median-absolute-deviation noise gate. Under GPU contention,
performance is `invalid`, never a pass.

The public JSON lists visible cases only and commits to a private withheld
manifest by raw-file SHA256. The admission runner keeps that manifest outside
the repository and agent workspace, passes it with `--withheld-spec`, and
verifies the hash before scored runs. A visible-only smoke without that flag
reports `visible_pass`, never full correctness parity. Scored results and the
private case manifest remain runner-only until the trial is closed and reviewed
for safe release; stdout contains summary statuses, not hidden inputs.

Before starting the container, the trusted host runner captures an exact GPU
report with `python3 -m harness.host_probe --output <new-private-path>`. Pass
that read-only file to the scorer with `--host-gpu-report`. This is required
when a container reports `Card Series: N/A`; the scorer still cross-checks
the container's PCI model and `gfx950` architecture. The report must stay in
runner-owned storage, not the agent workspace.

From the repository root inside a ROCm container with PyTorch and the pinned
AITER source mounted:

```sh
hipcc -O3 -shared -fPIC --offload-arch=gfx950 -I references \
  references/quant_mxfp4_baseline.hip -o /tmp/libquant-baseline.so
python3 -m harness.run \
  --spec references/quant_mxfp4_gfx950.json \
  --candidate /tmp/libquant-baseline.so \
  --aiter-source /workspace/aiter \
  --host-gpu-report /workspace/host-gpu-report.json \
  --output /tmp/quant-baseline-smoke-001 \
  --correctness-only --unscored-reference
```

For `mi350-2`, the prepared pinned source is
`/home/harmenon/aiter-rs-study/aiter`. Mount it at `/workspace/aiter` with
Docker `--device=/dev/kfd --device=/dev/dri --group-add video`, and mount the
`aiter-rs` checkout at `/workspace/aiter-rs` with that as the working
directory. Use Docker `--pid=host` for timing so ROCm's host process ID can be
matched to the scorer; if the scorer's own host PID is not visible,
performance is invalid. Expose exactly one MI350X GPU. The existing
`vllm-aiter-layout-contract:hipblaslt-2ad56d2-aiter-deps` image has PyTorch
and AITER dependencies. Set `PYTHONPATH=/workspace/aiter-rs:/workspace/aiter`.

The scorer creates an exclusive output directory and a read-only `result.json`.
It records AITER SHA, spec and binary hashes, runner-supplied task/source-tree
hashes, GPU SKU/architecture, ROCm environment, each correctness case, raw
latency samples, noise qualification, and separate functionality/performance/
joint status. `--task-freeze-sha256` and `--final-tree-sha256` preserve the
runner's frozen task and agent-source provenance; they are not substitutes for
the scorer's binary hash. For scored attempts, also supply
`--withheld-spec /workspace/private/quant_mxfp4_withheld.json` from a runner-only,
read-only mount. `--unscored-reference` labels the supplied HIP
baseline correctly. The raw result contains machine details and withheld case
IDs; publish only a sanitized summary until the private matrix is retired.
