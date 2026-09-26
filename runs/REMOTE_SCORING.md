# Trusted mi350-2 scoring handoff

`runs.remote_score` has two stages. `export` verifies a completed capture's
final tree against its blobs and writes only source plus a provenance manifest.
It does not export Codex events, prompts, credentials, or the agent workspace.
`score` is run by the trusted operator on the GPU host, outside the agent
sandbox. It validates source hashes, the exact Docker image ID, clean pinned
harness/AITER Git trees, public spec hash, private withheld commitment, and
host GPU attestation before starting the ROCm container. The container compiles
with the frozen `hipcc` flags and calls `harness.run --correctness-only`.

Example for the existing unscored quant preview (substitute an exclusive run
name for each attempt):

```sh
# On the capture host, from aiter-rs root. Transfer this source-only directory
# to mi350-2; never transfer raw events or private manifests to an agent host.
python3 -m runs.remote_score export \
  --run-dir results/agent-previews/quant-mxfp4-even-preview02 \
  --output /tmp/quant-preview02-source-bundle
scp -r /tmp/quant-preview02-source-bundle \
  mi350-2:/home/harmenon/aiter-rs-study/previews/quant-preview02-source-bundle

# On mi350-2, using a clean aiter-rs checkout containing runs.remote_score.
# The private paths below never enter the agent sandbox or public repository.
cd /home/harmenon/aiter-rs-study/trusted-runner-2cc1667
python3 -m runs.remote_score score \
  --bundle /home/harmenon/aiter-rs-study/previews/quant-preview02-source-bundle \
  --harness-root "$PWD" \
  --harness-revision "$(git rev-parse HEAD)" \
  --aiter-source /home/harmenon/aiter-rs-study/aiter \
  --spec "$PWD/references/quant_mxfp4_gfx950.json" \
  --spec-sha256 "$(sha256sum references/quant_mxfp4_gfx950.json | cut -d' ' -f1)" \
  --withheld-spec /home/harmenon/aiter-rs-study/private/quant_mxfp4_withheld.json \
  --host-gpu-report /home/harmenon/aiter-rs-study/private/host_gpu_report_20260925.json \
  --output /home/harmenon/aiter-rs-study/scores/quant-preview02-correctness-NEW \
  --image vllm-aiter-layout-contract:hipblaslt-2ad56d2-aiter-deps \
  --image-id sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b \
  --gpu-index 0
```

The example's `--harness-revision` and `--spec-sha256` are shell-derived only
for manual smoke runs. Admission trials must freeze those values *before* agent
launch and compare them to the captured task freeze. The source bundle's
`run_purpose` controls whether `--unscored-preview` is passed. Preview output
has `candidate_kind=agent_preview`, `joint_pass=false`, and cannot enter
incidence denominators, even if correctness passes. Docker has no network and
sees one GPU (`HIP_VISIBLE_DEVICES=0` within this container); no latency sampling occurs while the
host has other GPU processes. The raw `scores/` tree includes withheld case
results and must remain private. Publish only a reviewed summary.

This isolation prevents the *agent process* from accessing hidden tests or
host SSH credentials. It is not a malicious-native-code boundary: an agent's
compiled `.so` is loaded into the scorer process and could read the mounted
withheld manifest or its in-process inputs. The container has no network, but
candidate output and private logs still need review. A future adversarial
evaluation needs a separate-process or hardware-enforced candidate boundary.
