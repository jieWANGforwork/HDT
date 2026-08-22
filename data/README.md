# Data

This server-side release includes real HDT data under `data/processed/`.
The data was copied from `HDT/data/...` and `HDT_lfm/data/...`; old checkpoints
were not copied.

See `data/processed/MANIFEST.md` for the key included files and config mapping.

Expected processed layouts:

```text
data/processed/reddit/
  item_data.csv
  pop_items.npy
  less_items.npy
  goal_tfidf.npy
  train_test/
    pop_items.npy
    less_items.npy
    train_session_df_replay_buffer_nov2_user.df
    test_session_df_replay_buffer_nov2_users_h.df
    embs_sims.npy

data/processed/xing/
  item_data.csv
  train_data.csv
  valid_data.csv
  test_data.csv
  pop_items.npy
  less_items.npy
  train_session_df_replay_buffer_nov.df
  test_session_df_replay_buffer.df

data/processed/lfm/
  train_test/
    item_data.csv
    pop_items.npy
    less_items.npy
    train_session_df_replay_buffer_nov_users.df
    test_session_df_replay_buffer_nov_users_h.df

data/processed/lfmalbum/
  train_test/
    item_data.csv
    pop_items.npy
    less_items.npy
    train_session_df_replay_buffer_nov3_user.df
    test_session_df_replay_buffer_nov_users_h3.df
```

The old workspace sources used for cross-checking were `HDT/data/...` and
`HDT_lfm/data/...`. For public GitHub release, keep large data out of git and
publish it separately as links or external artifacts.
