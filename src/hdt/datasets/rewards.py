from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pad_to_length(values: Sequence[float], length: int, pad_value: float) -> list[float]:
    values = list(values)
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


def compute_novelty_returns(
    history: Sequence[int],
    actions: Sequence[int],
    less_popular_items: set[int] | Sequence[int],
    pad_item: int,
    state_size: int = 20,
) -> tuple[list[int], list[float]]:
    """Compute the novelty position reward and intra-session return.

    This follows the logic in `HDT/data/preprocess_reddit/train_test/return_nov.py`:
    prepend the first historical item, mark less-popular items as novel, then use
    the difference to the final average novelty as return-to-go.
    """

    less_popular = set(less_popular_items)
    sequence = [history[0]] + list(actions) if history else list(actions)
    novelty_positions = [1 if item in less_popular else 0 for item in sequence]
    valid_len = sum(1 for item in sequence if item != pad_item)
    if valid_len <= 0:
        return pad_to_length(novelty_positions, state_size, 0), [0.0] * state_size

    running_avg = [
        sum(novelty_positions[: j + 1]) / float(j + 1)
        for j in range(len(novelty_positions))
    ]
    final_avg = running_avg[valid_len - 1]
    returns = [final_avg - running_avg[k - 1] for k in range(1, valid_len)]
    return (
        pad_to_length(novelty_positions, state_size, 0),
        pad_to_length(returns, state_size, 0.0),
    )


def compute_diversity_returns(
    history: Sequence[int],
    actions: Sequence[int],
    similarity_matrix: np.ndarray,
    pad_item: int,
    state_size: int = 20,
) -> tuple[list[float], list[float]]:
    """Compute diversity scores and return-to-go from item similarities.

    This follows `HDT/data/preprocess_reddit/train_test/return_div.py`: diversity
    is one minus the average pairwise similarity among prefix items.
    """

    sequence = list(history)
    valid_actions = [item for item in actions if item != pad_item]
    if valid_actions:
        sequence = sequence[: len(valid_actions)]
        sequence.append(valid_actions[-1])
    valid_len = sum(1 for item in sequence if item != pad_item)
    if valid_len <= 1:
        return [-1.0] * state_size, [-1.0] * state_size

    sims = similarity_matrix.take(sequence[:valid_len], 0).take(sequence[:valid_len], 1)
    scores: list[float] = []
    for j in range(1, valid_len):
        prefix_sum = sims.take(range(j + 1), 0).take(range(j + 1), 1).sum() - (j + 1)
        avg_sim = prefix_sum / float(j * (j + 1))
        scores.append(float(1 - avg_sim))

    returns = [scores[valid_len - 2]]
    for j in range(1, valid_len - 1):
        returns.append(scores[valid_len - 2] - scores[j - 1])

    return (
        pad_to_length(scores, state_size, -1.0),
        pad_to_length(returns, state_size, -1.0),
    )
