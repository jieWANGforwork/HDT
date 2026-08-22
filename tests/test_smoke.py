import numpy as np


def test_reward_helpers():
    from hdt.datasets.rewards import compute_diversity_returns, compute_novelty_returns

    novelty, novelty_returns = compute_novelty_returns(
        history=[1, 2, 3],
        actions=[2, 4, 5],
        less_popular_items={4, 5},
        pad_item=99,
        state_size=5,
    )
    assert len(novelty) == 5
    assert len(novelty_returns) == 5

    sims = np.eye(100)
    diversity, diversity_returns = compute_diversity_returns(
        history=[1, 2, 3],
        actions=[2, 4, 5],
        similarity_matrix=sims,
        pad_item=99,
        state_size=5,
    )
    assert len(diversity) == 5
    assert len(diversity_returns) == 5


def test_synthetic_forward():
    from hdt.config import load_config
    from hdt.training.common import run_synthetic_forward

    config = load_config("configs/default.yaml")
    result = run_synthetic_forward(config)
    assert result["mode"] == "multi_objective"
    assert result["num_outputs"] == 8
    assert result["first_shape"][0] == config["training"]["batch_size"]
