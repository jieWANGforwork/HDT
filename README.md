# HDT

Official code release for **Sparks of Surprise: Multi-objective
Recommendations with Hierarchical Decision Transformers for Diversity,
Novelty, and Serendipity**.

HDT is a Hierarchical Decision Transformer framework for multi-objective
personalized session-based recommendation. This public release was cleaned and
reorganized from the historical `HDT/` and `HDT_lfm/` experiment directories,
with private paths, W&B keys, caches, logs, and checkpoints removed.

## Paper and Code

- Paper: [Sparks of Surprise: Multi-objective Recommendations with Hierarchical
  Decision Transformers for Diversity, Novelty, and
  Serendipity](https://doi.org/10.1145/3627673.3679533)
- Authors: Jie Wang, Alexandros Karatzoglou, Ioannis Arapakis, Xin Xin, Xuri Ge,
  and Joemon M. Jose
- Venue: CIKM 2024
- Code: [github.com/jieWANGforwork/HDT](https://github.com/jieWANGforwork/HDT)
- Supplementary folder: [Google Drive](https://drive.google.com/drive/folders/1AnN7dyf3_-i79ZPfvQe-cSjlJqcujvBU?usp=sharing)

## Method

HDT models recommendation at two levels:

- Inter-session/user level: learns user preference state from user return and
  session state.
- Intra-session level: uses a Decision Transformer to condition next-item
  prediction on return-to-go signals such as novelty and diversity.

## Structure

```text
configs/          experiment configs
data/             lightweight data assets and prepared data paths
docs/             usage, data, path, reproduction notes
scripts/          shell entry points
src/hdt/          importable package
tests/            import and smoke tests
tools/legacy/     mapping to old project files
```

## Install

```bash
pip install -e .
```

For CUDA, install the PyTorch build matching your driver from the official
PyTorch instructions, then install this package.

## Quick Check

```bash
bash scripts/train.sh
bash scripts/eval.sh
bash scripts/infer.sh
bash scripts/test.sh
```

`scripts/train.sh`, `scripts/eval.sh`, and `scripts/infer.sh` use real data by
default. Use `DRY_RUN=1` only when you explicitly want a synthetic installation
check.

## Data

The server-side release includes real data under `data/processed/`, including
the replay-buffer files needed for training and evaluation. Large data files are
ignored by git so they can be distributed separately when publishing the public
GitHub repository.
See `data/README.md`, `data/processed/MANIFEST.md`, `docs/data.md`, and
`docs/paths.md`.

## Training

```bash
bash scripts/train.sh
```

## Evaluation

```bash
CHECKPOINT=checkpoints/reddit/hdt_multi_objective_1steps.pt MAX_BATCHES=1 bash scripts/eval.sh
```

Omit `MAX_BATCHES` for full test replay-buffer evaluation.

## Inference

```bash
PYTHONPATH=src python -m hdt.inference.infer \
  --config configs/train.yaml \
  --checkpoint checkpoints/reddit/hdt_multi_objective_1steps.pt \
  --input "1,2,3,4" \
  --top-k 10
```

## Checkpoints

Put released checkpoints in `checkpoints/`. Weight files are ignored by git.

## Citation

```bibtex
@inproceedings{wang2024sparks,
  title = {Sparks of Surprise: Multi-objective Recommendations with Hierarchical
           Decision Transformers for Diversity, Novelty, and Serendipity},
  author = {Wang, Jie and Karatzoglou, Alexandros and Arapakis, Ioannis and
            Xin, Xin and Ge, Xuri and Jose, Joemon M.},
  booktitle = {Proceedings of the 33rd ACM International Conference on
               Information and Knowledge Management},
  pages = {2358--2368},
  year = {2024},
  doi = {10.1145/3627673.3679533}
}
```

## License

Apache-2.0. The local GPT-2 implementation is derived from Hugging Face
Transformers and remains Apache-2.0 compatible.
