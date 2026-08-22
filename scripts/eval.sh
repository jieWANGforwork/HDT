#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/eval.yaml}"
CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT:-}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi
ARGS=(--config "${CONFIG}" "${CHECKPOINT_ARGS[@]}")
if [[ -n "${MAX_BATCHES:-}" ]]; then
  ARGS+=(--max-batches "${MAX_BATCHES}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

PYTHONPATH=src python -m hdt.evaluation.evaluate "${ARGS[@]}"
