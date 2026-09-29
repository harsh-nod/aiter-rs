# GDR native-seed post-agent replay (exploratory)

`post_agent.py` is a trusted-host adapter for completed `unscored_preview` runs of
`runs/tasks/gdr_native_optimization_v1`. It is not an agent-visible feedback
broker or an incidence/scored-parity pipeline. Do not mount a withheld manifest
in the agent workspace.

## Inputs and execution

Invoke from a clean trusted `aiter-rs` checkout after the agent process exits:

```bash
python3 -m runs.gdr_native_optimization.post_agent \
  --run-dir /private/agent-runs/<run-id> \
  --repo /trusted/aiter-rs \
  --aiter-source /trusted/aiter \
  --withheld-spec /private/gdr-withheld.json \
  --host-gpu-report /private/mi350x-host-report.json \
  --output /private/gdr-post-agent/<new-run-directory> \
  --image <pinned-rocm-image-name>
```

The output path must not already exist; its parent and both input manifests must
be outside the agent run and source checkouts. The runner manifest/result must
have a completed, zero-exit, unscored capture with the task freeze matching the
current frozen task. Final snapshot blobs and the live workspace are checked;
only `kernel.hip` may differ from the starter. The ABI, public spec, and public
large fixture are copied from the frozen trusted task, never from the agent.

The adapter requires a clean pinned AITER checkout and clean trusted scorer
files. It checks the frozen compiler image ID and HIP compiler hash through the
public scorer, and the host report SHA, MI350X name, gfx950 architecture, and
PCI `0x75a0` through the private correctness scorer. The committed four-case
withheld manifest is mounted only in a second, network-disabled Docker run,
after public-only compile/correctness/benchmark completes. The public scorer
has no withheld mount. The second scorer reuses the exact public-stage `.so`;
it does not benchmark hidden cases.

Each Docker stage runs under the trusted host UID/GID, with a writable cache
under `/tmp`, so private result files remain readable by the trusted adapter.
A minimal trusted passwd/group pair under the private replay directory is
mounted read-only because the pinned image does not contain the host numeric
UID; the host's full account database is not mounted.
Compared with the earlier root-run public seed control, this exploratory
adapter sets `HOME=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`, and
`AITER_JIT_DIR=/tmp/aiter-jit-cache`: the image default `/aiter-jit-cache`
is not writable by the host UID. The compiler image, HIP flags, source SHA,
and task/manifest commitments remain pinned, but timings under these user/cache
settings are not the originally frozen scored environment.
Each stage also has a replay-specific name, label, and host CID file. The
trusted adapter checks the ownership tuple and removes only that exact
container if the Docker client times out or exits while the container remains.
If Docker state cannot be verified, the replay stops and requires manual GPU
PID/container inspection before another run.

Keep the output tree private. It contains raw public/withheld JSON and process
logs. `summary.json` and stdout contain only aggregate counts, public bucket
ratios, pinned source/binary/result hashes, and explicit unscored flags. A
withheld correctness failure is recorded as a failure, not discarded merely
because the scorer exits nonzero. A public compile/correctness failure writes a
separate sanitized receipt and does not open the withheld stage. Public
environment/setup failures are labeled separately. A missing result,
attestation mismatch, or timeout is a replay error, not a correctness result.

## Limits

- This is exploratory evidence for a native-seed agent preview. It does not
  establish agent bug incidence, reproducibility, or AITER performance parity.
- Same-process native candidate code can inspect the withheld manifest or
  scorer memory during private scoring. Network isolation and post-capture
  timing do not provide strong adversarial hidden-data confidentiality.
- Public benchmark ratios are environmental measurements, not a scored gate.
  Run only with an exclusive, attested MI350X window; the public scorer checks
  pre/post GPU PID contention.
- The frozen task remains `no_feedback`; the CPU fake tests do not make visible
  GPU feedback available to the agent.

CPU-only checks:

```bash
python3 -m unittest runs.gdr_native_optimization.test_post_agent \
  runs.gdr_native_optimization.test_boundary -v
```
