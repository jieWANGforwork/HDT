# Expected Processed Data Manifest

This manifest records the processed-data layout used by the HDT configurations.
The public repository includes only lightweight metadata and placeholders.
Replay buffers, large matrices, raw exports, and checkpoints are intentionally
ignored by Git and are not available in a fresh clone.

## Dataset Layouts

| Dataset | Expected local path | Audited research source |
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

## Companion Assets

- Lightweight item metadata and popularity lists are tracked where permitted.
- Full research runs additionally require the replay buffers listed above.
- Some configurations also use raw/split CSV files, replay-buffer CSV exports,
  Reddit `embs_sims.npy`, and LFM/LFM-album SAS/reward/diversity variants.
- These large companion assets were present in the audited research workspace;
  they are not implied to be present in the public repository.

## Excluded From Git

- Large processed datasets and replay buffers.
- Similarity matrices and raw data exports.
- Old project checkpoints: `*.pth`, `*.pt`, `*.ckpt`.
- Python caches.
- Logs, wandb runs, and generated experiment outputs.
