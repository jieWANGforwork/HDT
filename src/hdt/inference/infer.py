from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from hdt.config import load_config
from hdt.evaluation.evaluate import load_checkpoint
from hdt.training.common import build_model, get_device


def parse_items(value: str) -> list[int]:
    path = Path(value)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = value
    text = text.strip()
    if text.startswith("["):
        return [int(x) for x in json.loads(text)]
    return [int(x) for x in text.replace(",", " ").split()]


def pad_or_trim(items: Sequence[int], state_size: int, pad_item: int) -> list[int]:
    values = list(items)[-state_size:]
    if len(values) < state_size:
        values = values + [pad_item] * (state_size - len(values))
    return values


def infer(config_path: str, checkpoint: str | None, items: Sequence[int], top_k: int) -> dict:
    config = load_config(config_path)
    device = get_device(config)
    model = build_model(config, device=device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    state_size = int(config.get("model", {}).get("state_size", 20))
    item_num = model.item_num
    states = torch.tensor([pad_or_trim(items, state_size, item_num)], device=device).long()
    actions = states.clone()
    user_mask = torch.zeros(1, 1, device=device)
    reward = torch.zeros(1, 1, device=device)
    rtg = torch.zeros(1, state_size, device=device)
    hidden = model.init_hidden(1).unsqueeze(1)
    with torch.no_grad():
        outputs = model(states, actions, hidden, hidden, user_mask, reward, rtg, reward, rtg)
    mode = config.get("model", {}).get("mode", "multi_objective")
    if mode == "multi_objective":
        logits = outputs[2]
    elif mode == "single_objective":
        logits = outputs[2]
    else:
        logits = outputs[0]
    scores = logits[-1]
    top = scores.topk(top_k)
    return {
        "items": list(items),
        "top_k": [{"item_id": int(idx), "score": float(score)} for score, idx in zip(top.values, top.indices)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input", required=True, help="JSON list, comma-separated item ids, or a text file.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    result = infer(args.config, args.checkpoint, parse_items(args.input), args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
