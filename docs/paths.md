# Project Paths

This release keeps all expected local paths in place so the repository can be
used directly on a server, while large data and model artifacts can be mounted,
downloaded, or copied later.

## Data

- `data/raw/<dataset>/`: optional raw exports before preprocessing.
- `data/interim/<dataset>/`: intermediate preprocessing outputs.
- `data/processed/<dataset>/`: metadata, small release assets, and final replay
  buffers used by training/evaluation.
- `data/external/<dataset>/`: externally hosted files before they are moved into
  the expected processed layout.
- `data/release_links/`: notes for dataset or checkpoint download links.

Dataset names currently prepared:

- `reddit`
- `xing`
- `lfm`
- `lfmalbum`

## Model Artifacts

- `checkpoints/reddit/`
- `checkpoints/xing/`
- `checkpoints/lfm/`
- `checkpoints/lfmalbum/`
- `checkpoints/pretrained/`

Checkpoint files are ignored by git. Put released weights or downloaded weights
in these folders.

## Runtime Outputs

- `outputs/<dataset>/`: evaluation and inference outputs.
- `outputs/demo/`: demo predictions.
- `logs/train/`, `logs/eval/`, `logs/infer/`: text logs.
- `logs/tensorboard/`: TensorBoard event files.
- `runs/<dataset>/`: optional run-level artifacts.

## Reports

- `reports/metrics/`: result tables exported from evaluation.
- `reports/tables/`: paper-style tables.
- `reports/figures/`: figures and plots.

These folders are placeholders for local use and release preparation. Large
generated files should stay out of git unless they are intentionally published.
