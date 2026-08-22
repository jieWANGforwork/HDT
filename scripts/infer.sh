#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train.yaml}"
INPUT="${INPUT:-1,2,3,4,5}"
TOP_K="${TOP_K:-10}"
CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

PYTHONPATH=src python -m hdt.inference.infer --config "${CONFIG}" "${CHECKPOINT_ARGS[@]}" --input "${INPUT}" --top-k "${TOP_K}"
