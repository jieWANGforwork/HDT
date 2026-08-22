# Reproduction Notes

The old `HDT/readme.nd` maps the paper method variants as follows:

- `DUTST.py`, `DUTST_div.py`: inter-level only ablations.
- `DUTDT.py`, `DUTDT_div_return.py`: inter + intra single-objective variants.
- `DUTDT_ND.py`: novelty + diversity multi-objective HDT variant.

The clean project maps these to:

- `model.mode: inter_only`
- `model.mode: single_objective`
- `model.mode: multi_objective`

Recommended configs:

- `configs/experiments/reddit_main.yaml`: main multi-objective setting.
- `configs/experiments/xing_main.yaml`: single-objective XING setting.
- `configs/experiments/lastfm_main.yaml`: LFM branch cross-check.
- `configs/experiments/lastfm_album_main.yaml`: LFM-album branch cross-check.
- `configs/experiments/inter_only_ablation.yaml`: inter-only ablation.

## Paper Consistency Checklist

Before reporting final numbers, confirm these against the manuscript:

- Objective set: accuracy, novelty, diversity.
- Hierarchy: inter-session/user state and intra-session action transformer.
- Sequence length: 20.
- Hidden size: 100 for Reddit/XING, 64 for LFM branch where used.
- Main loss weights: supervised session loss, user-level loss, intra-session
  novelty loss, optional diversity classification loss, and diversity return
  regression.

The exact paper PDF was not found in the local workspace, and public web search
did not locate the title. Treat the checklist above as code-derived until the
paper source is provided.
