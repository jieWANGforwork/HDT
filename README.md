# HDT: Hierarchical Decision Transformers for Multi-Objective Recommendation

[![Paper](https://img.shields.io/badge/CIKM%202024-Paper-0A66C2)](https://doi.org/10.1145/3627673.3679533)
[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-D22128)](LICENSE)

Official implementation of **Sparks of Surprise: Multi-objective Recommendations
with Hierarchical Decision Transformers for Diversity, Novelty, and
Serendipity**, published at CIKM 2024.

HDT formulates personalized session-based recommendation as hierarchical
sequence modeling. A user-level component carries preference information across
sessions, while an intra-session Decision Transformer conditions next-item
prediction on return-to-go signals for objectives such as novelty and diversity.

> **Release status.** This repository contains the cleaned research code,
> experiment configurations, lightweight metadata, and synthetic smoke tests.
> Large replay buffers and trained checkpoints are not stored in Git. Paper-scale
> training and evaluation require those artifacts to be supplied separately; see
> [Data and checkpoints](#data-and-checkpoints).

## Paper

- **Authors:** Jie Wang, Alexandros Karatzoglou, Ioannis Arapakis, Xin Xin,
  Xuri Ge, and Joemon M. Jose
- **Venue:** 33rd ACM International Conference on Information and Knowledge
  Management (CIKM 2024)
- **Paper:** [ACM Digital Library / DOI](https://doi.org/10.1145/3627673.3679533)
- **Supplementary materials:** [Google Drive](https://drive.google.com/drive/folders/1AnN7dyf3_-i79ZPfvQe-cSjlJqcujvBU?usp=sharing)

## Repository layout

```text
configs/          Default, training, evaluation, and paper experiment configs
data/             Data schemas, lightweight metadata, and expected paths
docs/             Detailed data, path, usage, and reproduction notes
scripts/          Shell entry points for preparation, training, and evaluation
src/hdt/          Installable HDT package
tests/            Synthetic, import, path, and data-availability checks
checkpoints/      Expected location for downloaded or trained weights
outputs/          Local evaluation and inference outputs
```

## Installation

HDT requires Python 3.10 or newer. PyTorch will use CUDA when a compatible build
and GPU are available; otherwise the configuration falls back to CPU.

```bash
git clone https://github.com/jieWANGforwork/HDT.git
cd HDT

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "transformers>=4.35,<4.53"
python -m pip install -e .
```

The upper bound is required by the bundled GPT-2 implementation, which imports
interfaces moved in Transformers 4.53. The smoke tests below were verified with
Transformers 4.52.4 on Python 3.12.

For a CUDA installation, select the PyTorch build appropriate for your system
from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
before installing the remaining requirements.

## Quick start: no research data required

Use the small synthetic configuration to verify the installation without a
dataset or checkpoint:

```bash
PYTHONPATH=src python -m hdt.training.train \
  --config configs/default.yaml \
  --dry-run
PYTHONPATH=src python -m hdt.evaluation.evaluate \
  --config configs/default.yaml \
  --dry-run
python -m pytest -q \
  tests/test_imports.py \
  tests/test_smoke.py \
  tests/test_release_paths.py
```

These commands check imports, configuration loading, model construction, a
forward pass, metrics, and the expected directory layout. They do **not**
reproduce paper results. The complete `bash scripts/test.sh` suite additionally
checks for the external replay-buffer assets described below.

## Data and checkpoints

The tracked repository contains lightweight item metadata but excludes large
replay buffers, similarity matrices, and model weights. Arrange prepared data
under `data/processed/<dataset>/` and checkpoints under
`checkpoints/<dataset>/`.

| Dataset | Experiment configuration | Processed-data directory |
| --- | --- | --- |
| Reddit | `configs/experiments/reddit_main.yaml` | `data/processed/reddit/` |
| XING | `configs/experiments/xing_main.yaml` | `data/processed/xing/` |
| Last.fm | `configs/experiments/lastfm_main.yaml` | `data/processed/lfm/` |
| Last.fm albums | `configs/experiments/lastfm_album_main.yaml` | `data/processed/lfmalbum/` |

The loaders consume pandas replay-buffer files. Depending on the objective, the
records use the following fields:

- session identifiers and sequences: `userID`, `sessionID`, `itemsID`,
  `actionsID`;
- novelty signals: `user_nov_rtgs`, `nov_reward`, `nov_intra`;
- diversity signals: `user_div_rtgs`, `div_scores`, `div_rtgs`.

See [`data/README.md`](data/README.md) for the expected file layout,
[`data/processed/MANIFEST.md`](data/processed/MANIFEST.md) for the config-to-file
mapping, and [`docs/data.md`](docs/data.md) for the schema notes. The repository
does not download or redistribute the source datasets automatically; users are
responsible for complying with the original datasets' terms.

## Training

After placing the required processed files, run the Reddit multi-objective
configuration with:

```bash
CONFIG=configs/experiments/reddit_main.yaml bash scripts/train.sh
```

To confirm the real-data path with a short run before starting the full job:

```bash
CONFIG=configs/experiments/reddit_main.yaml MAX_STEPS=1 bash scripts/train.sh
```

Checkpoints are written to the directory specified by
`training.checkpoint_dir`. Change `CONFIG` to one of the configurations in the
table above to run another dataset or objective.

## Evaluation

Evaluate a trained checkpoint on one batch:

```bash
CONFIG=configs/experiments/reddit_main.yaml \
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_1steps.pt \
MAX_BATCHES=1 \
bash scripts/eval.sh
```

Remove `MAX_BATCHES` for the full evaluation replay buffer. Reported metrics are
Hit Rate and NDCG at the `evaluation.top_k` value in the selected configuration.

## Inference

Pass a comma-separated interaction history or a path to a JSON/text sequence:

```bash
CONFIG=configs/experiments/reddit_main.yaml \
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_1steps.pt \
INPUT="1,2,3,4" \
TOP_K=10 \
bash scripts/infer.sh
```

Item IDs must follow the indexing used by the selected dataset's metadata. A
checkpoint should be provided for meaningful recommendations; omitting it uses
randomly initialized weights.

## Reproduction notes

The cleaned implementation exposes three model modes:

- `multi_objective`: hierarchical novelty-and-diversity conditioning;
- `single_objective`: hierarchical conditioning for one auxiliary objective;
- `inter_only`: user/inter-session ablation without the intra-session Decision
  Transformer objective.

The experiment mapping and paper-consistency checklist are documented in
[`docs/reproduction.md`](docs/reproduction.md). For a reproducible report, record
the configuration file, data version, checkpoint, random seed, hardware, and
exact commit hash. Paper-scale claims should not be made from the synthetic
smoke test.

## Citation

If this repository supports your work, please cite:

```bibtex
@inproceedings{wang2024sparks,
  title     = {Sparks of Surprise: Multi-objective Recommendations with Hierarchical
               Decision Transformers for Diversity, Novelty, and Serendipity},
  author    = {Wang, Jie and Karatzoglou, Alexandros and Arapakis, Ioannis and
               Xin, Xin and Ge, Xuri and Jose, Joemon M.},
  booktitle = {Proceedings of the 33rd ACM International Conference on
               Information and Knowledge Management},
  pages     = {2358--2368},
  year      = {2024},
  doi       = {10.1145/3627673.3679533}
}
```

## License

Released under the [Apache License 2.0](LICENSE). The local GPT-2 components are
derived from Hugging Face Transformers; retain the applicable notices when
redistributing modified versions.
