# Processed Data Manifest

This server-side `HDT_release` directory contains real HDT data copied from
`HDT/` and `HDT_lfm/`. Checkpoints are not copied from the old projects.

The large data files are intentionally ignored by git, so the repository can be
published with code first and data links added later.

## Included Datasets

| Dataset | Release path | Source path |
| --- | --- | --- |
| reddit | `data/processed/reddit/` | `HDT/data/reddit/` |
| xing | `data/processed/xing/` | `HDT/data/xing/` |
| lfm | `data/processed/lfm/` | `HDT_lfm/data/lfm/` |
| lfmalbum | `data/processed/lfmalbum/` | `HDT_lfm/data/lfmalbum/` |

## Key Files Used By Configs

| Config | Train replay buffer | Test replay buffer |
| --- | --- | --- |
| `configs/experiments/reddit_main.yaml` | `reddit/train_test/train_session_df_replay_buffer_nov2_user.df` | `reddit/train_test/test_session_df_replay_buffer_nov2_users_h.df` |
| `configs/experiments/inter_only_ablation.yaml` | `reddit/train_test/train_session_df_replay_buffer_nov1_user.df` | `reddit/train_test/test_session_df_replay_buffer_nov1_users_h.df` |
| `configs/experiments/xing_main.yaml` | `xing/train_session_df_replay_buffer_nov.df` | `xing/test_session_df_replay_buffer.df` |
| `configs/experiments/lastfm_main.yaml` | `lfm/train_test/train_session_df_replay_buffer_nov_users.df` | `lfm/train_test/test_session_df_replay_buffer_nov_users_h.df` |
| `configs/experiments/lastfm_album_main.yaml` | `lfmalbum/train_test/train_session_df_replay_buffer_nov3_user.df` | `lfmalbum/train_test/test_session_df_replay_buffer_nov_users_h3.df` |

## Also Present

- Item metadata files such as `item_data.csv`.
- Raw/split CSV files from the HDT data directories.
- Popular and less-popular item lists.
- Replay-buffer CSV exports.
- Reddit `embs_sims.npy`.
- LFM/LFM-album SAS/reward/diversity replay buffers that were present in
  `HDT_lfm/data/...`.

## Excluded

- Old project checkpoints: `*.pth`, `*.pt`, `*.ckpt`.
- Python caches.
- Logs, wandb runs, and generated experiment outputs.
