# GDR packed-BF16 native-seed optimization

Optimize `kernel.hip` for MI350X/gfx950 while preserving the exact C ABI in
`gdr_decode_packed_bf16_abi.h`, all supported strided and stateful behavior,
and the caller-provided HIP stream. The initial source is intentionally an
AITER-derived native HIP implementation; this is an optimization study, not
a blind from-spec kernel-writing trial or an agent-error incidence sample.

Only `kernel.hip` may change. Keep the header unchanged and do not add build
scripts, helper source files, or runtime dependencies. The trusted runner
compiles the frozen source directly with pinned `hipcc -O3 -shared -fPIC
--offload-arch=gfx950`; it never executes a workspace build script. The
public correctness set covers valid slots, invalid sentinels, noncontiguous
strides, repeated state updates, and a separate batch-16 fixture. The
proposed performance buckets are valid slots, strided mixed, and batch 16.
The public correctness contract also checks an explicit HIP stream handle
`0` on a supported valid-slot update. Treat it as a host-adapter requirement,
not a separate performance workload.

No interactive GPU feedback is connected for this prototype. Do not infer
success from local compilation alone. A trusted public-only feedback broker
and separate withheld replay are specified but not active for this task
revision. Do not attempt to access hidden cases or private host paths.
