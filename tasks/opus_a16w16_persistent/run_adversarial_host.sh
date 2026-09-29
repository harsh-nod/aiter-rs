#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s PUBLIC_CASE_ID\n' "$0" >&2
  exit 2
fi
: "${AITERRS_STUDY_ROOT:?set AITERRS_STUDY_ROOT to the host study directory}"
: "${ADVERSARIAL_PRIVATE_ROOT:?set ADVERSARIAL_PRIVATE_ROOT to the private output directory}"

case_id=$1
study_root=$(realpath "$AITERRS_STUDY_ROOT")
private_root=$(realpath "$ADVERSARIAL_PRIVATE_ROOT")
case "$case_id" in
  aligned-nooob|m-tail-oob|n-tail-16-aligned|m-one-row-tail|n-first-vector-tail|mn-combined-tail|k-partial-final-tile|k-min-even-loop|xcd-padded-grid) ;;
  *) printf 'not a public proposed case: %s\n' "$case_id" >&2; exit 2 ;;
esac

output="$private_root/results/$case_id.json"
log="$private_root/logs/$case_id.log"
if [[ -e $output || -e $log ]]; then
  printf 'immutable case output/log already exists: %s\n' "$case_id" >&2
  exit 2
fi
mkdir -p "$private_root/results" "$private_root/logs"
container="opus-adversarial-$case_id-$$"
cleanup() {
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

set +e
timeout --foreground --signal=TERM --kill-after=20s 300s \
  docker run --rm --name "$container" --entrypoint python3 --pid=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    -e PYTHONPATH=/workspace/aiter-rs:/workspace/aiter \
    -e AITER_JIT_DIR=/workspace/jit \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory \
    -e GIT_CONFIG_VALUE_0=/workspace/aiter \
    -v "$study_root/aiter:/workspace/aiter:ro" \
    -v "$study_root/aiter-rs:/workspace/aiter-rs:ro" \
    -v "$study_root/private/opus-a16w16-jit:/workspace/jit:rw" \
    -v "$private_root:/workspace/private:rw" \
    -v "$study_root/private/libopus_persistent_jitflags_v2.so:/workspace/adapter.so:ro" \
    -w /workspace/aiter-rs \
    vllm-aiter-layout-contract:hipblaslt-2ad56d2-aiter-deps \
    /workspace/private/overlay/tasks/opus_a16w16_persistent/adversarial_admit.py \
    --matrix /workspace/aiter-rs/tasks/opus_a16w16_persistent/adversarial_matrix_candidate.json \
    --anchors /workspace/aiter-rs/tasks/opus_a16w16_persistent/cases.json \
    --aiter-source /workspace/aiter \
    --adapter-source /workspace/aiter-rs/tasks/opus_a16w16_persistent/source_adapter.hip \
    --adapter /workspace/adapter.so \
    --host-gpu-report /workspace/private/host_gpu_report.txt --case "$case_id" \
    --output "/workspace/private/results/$case_id.json" >"$log" 2>&1
status=$?
set -e
if [[ $status -ne 0 ]]; then
  printf 'case %s stopped with status %d; private log: %s\n' "$case_id" "$status" "$log" >&2
  tail -n 18 "$log" >&2
  exit "$status"
fi
printf 'case %s completed; private result: %s\n' "$case_id" "$output"
