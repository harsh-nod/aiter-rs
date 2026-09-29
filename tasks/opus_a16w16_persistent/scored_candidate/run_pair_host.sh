#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 3 ]]; then
  printf 'usage: %s CANDIDATE_HEADER PUBLIC_CASE_ID NEW_PRIVATE_RUN_ROOT\n' "$0" >&2
  exit 2
fi
: "${AITERRS_STUDY_ROOT:?set the trusted host study root}"
study_root=$(realpath "$AITERRS_STUDY_ROOT")
code_root=$(realpath "${AITERRS_CODE_ROOT:-$study_root/aiter-rs}")
candidate=$(realpath "$1")
case_id=$2
run_root=$(realpath -m "$3")
report=$(realpath "${OPUS_HOST_GPU_REPORT:-$study_root/private/opus-adversarial-20260929/host_gpu_report.txt}")
image=vllm-aiter-layout-contract:hipblaslt-2ad56d2-aiter-deps
expected_image=sha256:90885f811fc53d8d03fb6ab6d05b5f0a2c88e277f26f56d19e6b990663a5626b

case "$case_id" in
  aligned-nooob|m-tail-oob|n-tail-16-aligned|m-one-row-tail|n-first-vector-tail|mn-combined-tail|k-min-even-loop|xcd-padded-grid) ;;
  *) printf 'unsupported public full-K case: %s\n' "$case_id" >&2; exit 2 ;;
esac
if [[ "$run_root" != "$study_root/private/"* || -e "$run_root" ]]; then
  printf 'run root must be new and under the trusted private study root\n' >&2
  exit 2
fi
if [[ $(docker image inspect "$image" --format '{{.Id}}') != "$expected_image" ]]; then
  printf 'compiler image ID changed\n' >&2
  exit 2
fi
candidate_sha=$(sha256sum "$candidate" | cut -d' ' -f1)
install -d -m 700 "$run_root"
install -m 600 "$report" "$run_root/host_gpu_report.txt"
record_no_result() {
  python3 - "$run_root" "$1" "$candidate_sha" "$2" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = int(sys.argv[2])
stage = sys.argv[4]
log = root / ("prepare.log" if stage == "overlay_preflight" else "docker.log")
attempt = {
    "schema": "aiter-rs-opus-production-host-attempt-v1",
    "status": "timeout_or_hang_inconclusive" if status == 124 else "infrastructure_or_execution_inconclusive",
    "stage": stage,
    "exit_code": status,
    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "candidate_header_sha256": sys.argv[3],
    "production_result_written": False,
}
descriptor = os.open(root / "attempt.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as output:
    json.dump(attempt, output, indent=2, sort_keys=True)
    output.write("\n")
PY
}
set +e
python3 "$code_root/tasks/opus_a16w16_persistent/prepare_overlay.py" \
  --aiter-source "$study_root/aiter" --candidate-header "$candidate" \
  --output-tree "$run_root/overlay" >"$run_root/overlay_manifest.json" 2>"$run_root/prepare.log"
prepare_status=$?
set -e
if [[ $prepare_status -ne 0 ]]; then
  record_no_result "$prepare_status" overlay_preflight
  printf 'OPUS overlay preflight stopped; private root: %s\n' "$run_root" >&2
  exit "$prepare_status"
fi

container="opus-production-${case_id}-$$"
cleanup() {
  timeout 30s docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT
set +e
timeout --foreground --signal=TERM --kill-after=30s 1200s \
  docker run --rm --name "$container" --entrypoint python3 --network=none --pid=host \
    --user "$(id -u):$(id -g)" --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render -e HOME=/tmp \
    -e PYTHONPATH=/workspace/code \
    -e GIT_CONFIG_COUNT=2 \
    -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace/aiter \
    -e GIT_CONFIG_KEY_1=safe.directory -e GIT_CONFIG_VALUE_1=/workspace/private/overlay \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    -v "$study_root/aiter:/workspace/aiter:ro" \
    -v "$code_root:/workspace/code:ro" \
    -v "$run_root:/workspace/private:rw" \
    -v "$run_root/overlay:/workspace/private/overlay:ro" \
    -w /workspace/code "$image" \
    -m tasks.opus_a16w16_persistent.scored_candidate.dual_worker \
    --aiter-source /workspace/aiter --overlay-tree /workspace/private/overlay \
    --jit-baseline /workspace/private/jit-baseline \
    --jit-candidate /workspace/private/jit-candidate \
    --matrix /workspace/code/tasks/opus_a16w16_persistent/scored_candidate/public_matrix.json \
    --host-gpu-report /workspace/private/host_gpu_report.txt \
    --case "$case_id" --candidate-header-sha256 "$candidate_sha" \
    --compiler-image-id "$expected_image" \
    --output /workspace/private/result.json >"$run_root/docker.log" 2>&1
status=$?
set -e
if [[ ! -e "$run_root/result.json" ]]; then
  record_no_result "$status" docker_execution
fi
printf 'OPUS production case %s exit=%d; private root: %s\n' "$case_id" "$status" "$run_root"
exit "$status"
