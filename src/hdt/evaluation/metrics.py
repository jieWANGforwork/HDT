from __future__ import annotations

import math

import torch


def hit_rate_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=-1).indices
    hits = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return float(hits.mean().item())


def ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=-1).indices
    matches = topk.eq(targets.unsqueeze(-1))
    if not matches.any():
        return 0.0
    scores = []
    for row in matches:
        positions = torch.nonzero(row, as_tuple=False)
        if positions.numel() == 0:
            scores.append(0.0)
        else:
            rank = int(positions[0].item()) + 1
            scores.append(1.0 / math.log2(rank + 1))
    return float(sum(scores) / len(scores))
