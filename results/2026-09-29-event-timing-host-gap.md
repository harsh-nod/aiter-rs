# GPU-event timing includes host enqueue gaps

On `mi350-2` with one MI350X/gfx950, in the same no-network ROCm image used
for the AITER probes (`sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b`),
this control ran on 2026-09-29:

```python
import time
import torch

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
time.sleep(0.05)
end.record()
end.synchronize()
print(start.elapsed_time(end))
```

Observed elapsed time: **50.434 ms**. There was no kernel between the events.
The GPU executed the start event while the CPU slept before enqueueing the
end event, so the event interval included the host-side gap. This agrees with
[PyTorch's CUDA/ROCm stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
and its discussion of CPU launch overhead and graph replay.

The current AITER comparison harness records a start event, calls a Python
AITER wrapper or a direct HIP candidate ABI, then records an end event. For
short kernels, unequal host wrapper/dispatch work can therefore influence the
reported event interval. The previously published quant and GDR ratios remain
**stream-interval observations**, not proven device-kernel speedups or valid
same-boundary HIP performance parity. Correctness results are unaffected.

Before any type receives performance admission, measure the actual device
work via a controlled graph-replay or kernel-profiler route and separately
report end-to-end host/API latency if that is the intended metric. Freeze the
new measurement protocol before new scored trials; do not retroactively
reinterpret old frozen score gates as passes.
