from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from hdt.datasets import session_dataset, session_dataset_div, session_dataset_nd
from hdt.config import load_config
from hdt.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from hdt.training.common import build_model, get_device, make_loader_args, run_synthetic_forward, synthetic_batch


def select_dataset_module(config: Dict[str, Any]):
    objective = config.get("data", {}).get("objective", "novelty")
    mode = config.get("model", {}).get("mode", "multi_objective")
    if mode == "multi_objective":
        return session_dataset_nd
    if objective == "diversity":
        return session_dataset_div
    return session_dataset


def load_checkpoint(model: torch.nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return
    checkpoint_path = Path(checkpoint)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict)


def evaluate_synthetic(config: Dict[str, Any], checkpoint: str | None = None) -> Dict[str, float]:
    device = get_device(config)
    model = build_model(config, device=device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    batch = synthetic_batch(config, device)
    hidden = model.init_hidden(batch["inputs"].shape[0]).unsqueeze(1)
    with torch.no_grad():
        outputs = model(
            batch["inputs"],
            batch["targets"],
            hidden,
            hidden,
            batch["user_change"],
            batch["sess_reward"],
            batch["poss_rtg"],
            batch["sess_reward_nov"],
            batch["poss_rtg_nov"],
        )
    mode = config.get("model", {}).get("mode", "multi_objective")
    if mode == "multi_objective":
        logits, targets = outputs[2], outputs[4]
    elif mode == "single_objective":
        logits, targets = outputs[2], outputs[3]
    else:
        logits, targets = outputs[0], outputs[1]
    k = int(config.get("evaluation", {}).get("top_k", 10))
    return {
        f"hit@{k}": hit_rate_at_k(logits, targets, k),
        f"ndcg@{k}": ndcg_at_k(logits, targets, k),
    }


def evaluate_real(config: Dict[str, Any], checkpoint: str | None = None, max_batches: int = 0) -> Dict[str, float]:
    device = get_device(config)
    model = build_model(config, device=device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    dataset_module = select_dataset_module(config)
    transforms = [dataset_module.ToTensorReward(device)]
    loader = dataset_module.TestDataLoaderUDT(make_loader_args(config), type="test", transforms=transforms)

    k = int(config.get("evaluation", {}).get("top_k", 10))
    user_state = model.init_hidden(loader.batch_size)
    action_state = model.init_hidden(loader.batch_size)
    hit_sum = 0.0
    ndcg_sum = 0.0
    batches = 0

    with torch.no_grad():
        for sample in loader:
            states = sample["inputs"].to(device).long()
            actions = sample["targets"].to(device).long()
            user_mask = sample["user_change"].to(device)
            sess_reward = sample["sess_reward"].unsqueeze(1).to(device)
            poss_rtg = sample["poss_rtg"].to(device)

            user_state = user_state.detach() * (1 - user_mask) + model.mask_zeros(user_state) * user_mask
            action_state = action_state.detach() * (1 - user_mask) + model.mask_zeros(action_state) * user_mask

            if model.mode == "multi_objective":
                sess_reward_nov = sample["sess_reward_nov"].unsqueeze(1).to(device)
                poss_rtg_nov = sample["poss_rtg_nov"].to(device)
                outputs = model(
                    states,
                    actions,
                    user_state.unsqueeze(1),
                    action_state.unsqueeze(1),
                    user_mask,
                    sess_reward,
                    poss_rtg,
                    sess_reward_nov,
                    poss_rtg_nov,
                )
                logits, targets, state_hidden = outputs[2], outputs[4], outputs[5]
            elif model.mode == "single_objective":
                outputs = model(
                    states,
                    actions,
                    user_state.unsqueeze(1),
                    action_state.unsqueeze(1),
                    user_mask,
                    sess_reward,
                    poss_rtg,
                )
                logits, targets, state_hidden = outputs[2], outputs[3], outputs[4]
            else:
                outputs = model(
                    states,
                    actions,
                    user_state.unsqueeze(1),
                    action_state.unsqueeze(1),
                    user_mask,
                    sess_reward,
                )
                logits, targets, state_hidden = outputs[0], outputs[1], outputs[3]

            hit_sum += hit_rate_at_k(logits, targets, k)
            ndcg_sum += ndcg_at_k(logits, targets, k)
            batches += 1
            user_state = state_hidden.detach()
            action_state = state_hidden.detach()
            if max_batches and batches >= max_batches:
                break

    if batches == 0:
        raise RuntimeError("No evaluation batches were produced from the real replay buffer.")
    return {
        "batches": batches,
        f"hit@{k}": hit_sum / batches,
        f"ndcg@{k}": ndcg_sum / batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Evaluate a synthetic batch.")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit real-data evaluation batches; 0 means full.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        print(run_synthetic_forward(config))
        print(evaluate_synthetic(config, args.checkpoint))
        return
    print(evaluate_real(config, args.checkpoint, max_batches=args.max_batches))


if __name__ == "__main__":
    main()
