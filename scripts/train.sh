#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train.yaml}"
ARGS=(--config "${CONFIG}")
if [[ -n "${MAX_STEPS:-}" ]]; then
  ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

PYTHONPATH=src python -m hdt.training.train "${ARGS[@]}"
