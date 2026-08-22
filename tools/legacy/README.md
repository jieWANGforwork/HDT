# Legacy Mapping

Files intentionally not copied as runnable modules:

- `HDT/test_*.py`, `HDT_lfm/test_*.py`: experiment/evaluation scripts with
  hard-coded paths and wandb side effects.
- `HDT/train_step/`, `HDT_lfm/train_step/`: older duplicate training snapshots.
- `HDT/baselines/`, `HDT_lfm/baselines/`: baseline methods not required for the
  minimal HDT release.
- `HDT/data/preprocess_reddit/**`, `HDT_lfm/data/**/train_test/*.py`: useful
  historical preprocessing references, but they contain local paths and write
  directly into old data directories.

Core source files cross-checked:

- `HDT/DUTST.py` and `HDT_lfm/DUTST.py`: identical.
- `HDT/DUTDT.py` and `HDT_lfm/DUTDT.py`: identical.
- `HDT/DUTDT_ND.py` and `HDT_lfm/DUTDT_ND.py`: differ in LFM-specific defaults,
  data file names, hidden size, and one attention residual path.
- `HDT/src/dataset*.py` and `HDT_lfm/src/dataset*.py`: mostly shared schema,
  with LFM-specific file names.
