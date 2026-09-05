# Data Preparation

HDT uses session-level replay-buffer files generated from sequential
recommendation logs. The public repository includes lightweight metadata but
does not include the large `.df` replay buffers, similarity matrices, raw
exports, or checkpoints required for paper-scale experiments.

The expected local assets are summarized in `data/processed/MANIFEST.md`.

## Required Columns

The training loaders expect pandas pickle files with these columns:

- Common: `userID`, `sessionID`, `itemsID`, `actionsID`
- Novelty objective: `user_nov_rtgs`, `nov_reward`, `nov_intra`
- Diversity objective: `user_div_rtgs`, `div_scores`, `div_rtgs`
- Multi-objective HDT: all novelty and diversity columns above

`itemsID` and `actionsID` are fixed-length item-id sequences. The padding item
id is `item_num`, where `item_num = len(item_data.csv)`.

## Source Cross-Check

The old workspace has two relevant data/code sources:

- `HDT/data/reddit` and `HDT/data/xing`: main HDT experiments.
- `HDT_lfm/data/lfm` and `HDT_lfm/data/lfmalbum`: LastFM/LFM branch.

The HDT release code and data assumptions are extracted only from `HDT/` and
`HDT_lfm/`. Other project directories in the same workspace are not used as
references because their code and data schemas are separate from HDT.

## Audited Research Data

The authors' original research workspace used the following assets. This list
documents the provenance check; it does not indicate that the assets are tracked
in the public repository.

- Reddit: real replay buffers, item metadata, split CSV files, popularity lists,
  less-popular item lists, TF-IDF goal features, and `embs_sims.npy`.
- XING: real replay buffers, raw/split CSV files, item metadata, popularity
  lists, and less-popular item lists.
- LFM and LFM-album: real replay buffers, SAS/reward/diversity variants,
  metadata, popularity lists, and less-popular item lists.

When placed in the expected paths, these files support real-data training and
evaluation with the prepared configs. Large files are ignored by Git and must be
distributed separately with the applicable dataset terms.

## Excluded Files

- Old project checkpoints.
- Logs, wandb runs, generated outputs, and Python caches.

## Reward Construction

The clean implementation exposes the core reward logic in
`hdt.datasets.rewards`:

- `compute_novelty_returns`: extracted from `return_nov.py`.
- `compute_diversity_returns`: extracted from `return_div.py`.

These functions are pure and path-free. Full raw-to-replay preprocessing from
the historical scripts should be re-run only after replacing hard-coded paths
with config values.
