# HDT: Hierarchical Decision Transformers for Multi-Objective Recommendation

**Paper:** [Sparks of Surprise: Multi-objective Recommendations with Hierarchical Decision Transformers for Diversity, Novelty, and Serendipity](https://doi.org/10.1145/3627673.3679533)

## 📖 Overview

HDT is a hierarchical decision-transformer framework for personalized
session-based recommendation. It models user preferences across sessions and
uses return-conditioned sequence modeling within each session to optimize
recommendation accuracy together with objectives such as novelty and diversity.

The framework contains two complementary levels:

- **Inter-session modeling** captures the evolution of a user's preferences
  across multiple sessions.
- **Intra-session modeling** conditions next-item prediction on interaction
  sequences and objective-specific returns-to-go.
- **Multi-objective recommendation** jointly represents accuracy, novelty, and
  diversity signals in a unified architecture.

## 🛠️ Environment Setup

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later
- CUDA-compatible GPU (optional)

### Installation

```bash
git clone https://github.com/jieWANGforwork/HDT.git
cd HDT

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt "transformers>=4.35,<4.53"
python -m pip install -e .
```

## 📊 Data Preparation

Prepare each dataset as a pandas replay-buffer file and place it under
`data/processed/<dataset>/`. The experiment configurations use the following
locations:

| Dataset | Configuration | Data directory |
| --- | --- | --- |
| Reddit | `configs/experiments/reddit_main.yaml` | `data/processed/reddit/` |
| XING | `configs/experiments/xing_main.yaml` | `data/processed/xing/` |
| Last.fm | `configs/experiments/lastfm_main.yaml` | `data/processed/lfm/` |
| Last.fm albums | `configs/experiments/lastfm_album_main.yaml` | `data/processed/lfmalbum/` |

### Data Format

The replay buffers use fixed-length interaction sequences and reward signals.
The principal fields are:

- `userID`: user identifier
- `sessionID`: session identifier
- `itemsID`: item interaction sequence
- `actionsID`: next-item target sequence
- `user_nov_rtgs`, `nov_reward`, `nov_intra`: novelty signals
- `user_div_rtgs`, `div_scores`, `div_rtgs`: diversity signals

Item information is stored in the `item_data.csv` file specified by the
selected configuration. Dataset paths, filenames, sequence length, batch size,
and model hyperparameters can be changed directly in the YAML configuration.

## 🚀 Training

Train HDT on Reddit with the multi-objective configuration:

```bash
CONFIG=configs/experiments/reddit_main.yaml bash scripts/train.sh
```

To train on another dataset, select the corresponding configuration:

```bash
CONFIG=configs/experiments/xing_main.yaml bash scripts/train.sh
CONFIG=configs/experiments/lastfm_main.yaml bash scripts/train.sh
CONFIG=configs/experiments/lastfm_album_main.yaml bash scripts/train.sh
```

The trained model is saved in the `training.checkpoint_dir` defined by the
configuration file.

## 🧪 Evaluation

Evaluate a trained model by supplying its configuration and checkpoint:

```bash
CONFIG=configs/experiments/reddit_main.yaml \
CHECKPOINT=/path/to/checkpoint.pt \
bash scripts/eval.sh
```

The evaluation reports Hit Rate and NDCG at the cutoff specified by
`evaluation.top_k` in the configuration file.

## 🔍 Inference

Generate top-*K* recommendations from a comma-separated item history:

```bash
CONFIG=configs/experiments/reddit_main.yaml \
CHECKPOINT=/path/to/checkpoint.pt \
INPUT="1,2,3,4" \
TOP_K=10 \
bash scripts/infer.sh
```

`INPUT` may also be a JSON list or the path to a text file containing item IDs.
The item IDs should follow the indexing used by the selected dataset.

## 📁 Repository Structure

```text
HDT/
├── configs/                 # Training, evaluation, and experiment settings
├── data/                    # Dataset directories and data documentation
├── scripts/                 # Training, evaluation, and inference commands
├── src/hdt/
│   ├── config/              # Configuration loading
│   ├── datasets/            # Replay-buffer loaders and reward functions
│   ├── evaluation/          # Ranking metrics and evaluation pipeline
│   ├── inference/           # Recommendation inference
│   ├── models/              # HDT and Decision Transformer modules
│   └── training/            # Model training pipeline
├── checkpoints/             # Trained model output directory
├── outputs/                 # Evaluation and inference outputs
├── requirements.txt
└── pyproject.toml
```

## 📝 Citation

If you use this code in your research, please cite:

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

## 📄 License

This project is released under the [Apache License 2.0](LICENSE).
