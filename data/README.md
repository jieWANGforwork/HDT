# Data

The public repository tracks lightweight item metadata and directory
placeholders under `data/processed/`. Large replay buffers, similarity matrices,
raw exports, and checkpoints are excluded from Git and must be obtained or
prepared separately.

See `data/processed/MANIFEST.md` for the expected files and config mapping.

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

The layouts were cross-checked against the authors' original `HDT/data/...` and
`HDT_lfm/data/...` research directories. Their presence in that audited local
workspace does not mean the files are included in a public clone. Keep large
data out of Git and distribute it separately with provenance, terms, and
checksums.
