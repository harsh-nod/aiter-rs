# One-wave shuffle participation control

`wave_shuffle_participation.hip` distills quant trial r002's agent-authored
error to one wave64. Even lanes request the next odd lane's value, but odd
lanes do not execute the first shuffle. The second shuffle is executed by
every lane before the even-lane store. The two paths otherwise use the same
input and expected output.

On `mi350-2` with a gfx950 MI350X:

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx950 -O2 \
  cases/hip/wave_shuffle_participation.hip -o /tmp/wave_shuffle_participation
/tmp/wave_shuffle_participation
```

The [September 29 receipt](../../results/2026-09-29-hip-wave-shuffle-control.md)
records 32/32 wrong values for the divergent path and 0/32 wrong values for
the repaired path. This is a mechanism control, not an additional agent trial
or a performance measurement. Do not infer that the divergent path must
produce zero on every compiler or GPU; the crucial error is reading a source
lane that did not participate in the shuffle.
