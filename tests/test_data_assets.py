from pathlib import Path

import numpy as np
import pandas as pd

from hdt.config import load_config
from hdt.training.common import infer_item_num


def test_release_metadata_files_are_present():
    root = Path("data/processed")
    expected = [
        root / "reddit/item_data.csv",
        root / "reddit/pop_items.npy",
        root / "reddit/less_items.npy",
        root / "xing/item_data.csv",
        root / "xing/train_data.csv",
        root / "lfm/train_test/item_data.csv",
        root / "lfmalbum/train_test/item_data.csv",
    ]
    for path in expected:
        assert path.exists(), path


def test_item_metadata_is_readable():
    reddit_items = pd.read_csv("data/processed/reddit/item_data.csv")
    lfm_items = pd.read_csv("data/processed/lfm/train_test/item_data.csv")
    less_items = np.load("data/processed/reddit/less_items.npy")

    assert list(reddit_items.columns) == ["item_id", "item_idx"]
    assert list(lfm_items.columns) == ["item_id", "item_idx"]
    assert len(reddit_items) == 14369
    assert len(lfm_items) == 111857
    assert less_items.ndim == 1


def test_configs_can_infer_item_num_from_release_metadata():
    for config_path in [
        "configs/experiments/reddit_main.yaml",
        "configs/experiments/xing_main.yaml",
        "configs/experiments/lastfm_main.yaml",
        "configs/experiments/lastfm_album_main.yaml",
    ]:
        config = load_config(config_path)
        assert infer_item_num(config) > 0


def test_real_replay_buffers_are_present():
    required = [
        "data/processed/reddit/train_test/train_session_df_replay_buffer_nov2_user.df",
        "data/processed/reddit/train_test/test_session_df_replay_buffer_nov2_users_h.df",
        "data/processed/reddit/train_test/embs_sims.npy",
        "data/processed/xing/train_session_df_replay_buffer_nov.df",
        "data/processed/xing/test_session_df_replay_buffer.df",
        "data/processed/lfm/train_test/train_session_df_replay_buffer_nov_users.df",
        "data/processed/lfm/train_test/test_session_df_replay_buffer_nov_users_h.df",
        "data/processed/lfmalbum/train_test/train_session_df_replay_buffer_nov3_user.df",
        "data/processed/lfmalbum/train_test/test_session_df_replay_buffer_nov_users_h3.df",
    ]
    for value in required:
        path = Path(value)
        assert path.exists(), value
        assert path.stat().st_size > 0, value
