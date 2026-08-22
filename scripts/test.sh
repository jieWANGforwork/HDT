#!/usr/bin/env bash
set -euo pipefail

if PYTHONPATH=src python -c "import pytest" >/dev/null 2>&1; then
  PYTHONPATH=src python -m pytest -q tests
else
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -c "from tests.test_imports import test_import_core_modules, test_config_loads_default; test_import_core_modules(); test_config_loads_default(); print('import tests passed')"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -c "from tests.test_smoke import test_reward_helpers, test_synthetic_forward; test_reward_helpers(); test_synthetic_forward(); print('smoke tests passed')"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -c "from tests.test_data_assets import test_release_metadata_files_are_present, test_item_metadata_is_readable, test_configs_can_infer_item_num_from_release_metadata, test_real_replay_buffers_are_present; test_release_metadata_files_are_present(); test_item_metadata_is_readable(); test_configs_can_infer_item_num_from_release_metadata(); test_real_replay_buffers_are_present(); print('data asset tests passed')"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -c "from tests.test_release_paths import test_release_directories_exist; test_release_directories_exist(); print('release path tests passed')"
fi
