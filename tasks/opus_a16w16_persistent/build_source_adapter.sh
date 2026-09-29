#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s PINNED_AITER_SOURCE ADAPTER_SOURCE OUTPUT_SO\n' "$0" >&2
  exit 2
fi

aiter_source=$(realpath "$1")
adapter_source=$(realpath "$2")
output=$(realpath -m "$3")
expected=868ccf62a0bcad3aa47f92728340ccb37ed4fb39
if [[ $(git -C "$aiter_source" rev-parse HEAD) != "$expected" ]] ||
   [[ -n $(git -C "$aiter_source" status --porcelain) ]]; then
  printf 'AITER source must be clean at %s\n' "$expected" >&2
  exit 2
fi
if [[ $output == "$aiter_source"/* || $output == "$(dirname "$adapter_source")"/* ]]; then
  printf 'output must be outside both source directories\n' >&2
  exit 2
fi

hipcc -O3 -std=c++20 -fPIC -shared --offload-arch=gfx950 \
  -I"$aiter_source/csrc/include" -I"$aiter_source/csrc/opus_gemm/include" \
  -fgpu-flush-denormals-to-zero -fno-offload-uniform-block \
  -mllvm --amdgpu-kernarg-preload-count=32 \
  -mllvm --amdgpu-mfma-vgpr-form \
  -mllvm --lsr-drop-solution=1 \
  -mllvm -amdgpu-early-inline-all=true \
  -mllvm -amdgpu-function-calls=false \
  -mllvm -enable-post-misched=0 \
  -fno-gpu-rdc \
  "$adapter_source" -o "$output"
