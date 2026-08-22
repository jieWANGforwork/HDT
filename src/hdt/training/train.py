from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from tqdm import tqdm

from hdt.config import load_config
from hdt.datasets import session_dataset, session_dataset_div, session_dataset_nd
from hdt.training.common import build_model, get_device, make_loader_args, run_synthetic_forward


def select_dataset_module(config: Dict[str, Any]):
    objective = config.get("data", {}).get("objective", "novelty")
    mode = config.get("model", {}).get("mode", "multi_objective")
    if mode == "multi_objective":
        return session_dataset_nd
    if objective == "diversity":
        return session_dataset_div
    return session_dataset


def train(config: Dict[str, Any]) -> Path:
    device = get_device(config)
    model = build_model(config, device=device)
    args = make_loader_args(config)
    dataset_module = select_dataset_module(config)
    transforms = [dataset_module.ToTensorReward(device)]
    loader = dataset_module.DataLoaderUDT(args, type=None, transforms=transforms)

    training_cfg = config.get("training", {})
    output_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    output_dir.mkdir(parents=True, exist_ok=True)
    lr = float(training_cfg.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    max_steps = int(training_cfg.get("max_steps", 0))
    epochs = int(training_cfg.get("epochs", 1))

    weights = training_cfg.get("loss_weights", {})
    w_supervised = float(weights.get("supervised", 1.0))
    w_user = float(weights.get("user", 1.0))
    w_session = float(weights.get("session", 0.5))
    w_diversity = float(weights.get("diversity", 0.0))
    default_reward_weight = 0.2 if model.mode == "multi_objective" else 0.0
    w_reward = float(weights.get("reward", default_reward_weight))

    total_step = 0
    for epoch in range(epochs):
        model.train()
        user_state = model.init_hidden(args.batch_size)
        action_state = model.init_hidden(args.batch_size)
        for sample in tqdm(loader, desc=f"epoch {epoch}"):
            states = sample["inputs"].to(device).long()
            actions = sample["targets"].to(device).long()
            user_mask = sample["user_change"].to(device)
            sess_reward = sample["sess_reward"].unsqueeze(1).to(device)
            poss_rtg = sample["poss_rtg"].to(device)
            user_state = user_state.detach() * (1 - user_mask) + model.mask_zeros(user_state) * user_mask
            action_state = action_state.detach() * (1 - user_mask) + model.mask_zeros(action_state) * user_mask

            optimizer.zero_grad()
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
                (
                    user_preds,
                    action_preds0,
                    action_preds_nov,
                    action_preds_div,
                    action_target,
                    state_hidden,
                    reward_preds,
                    reward_target,
                ) = outputs
                loss = (
                    w_supervised * ce_loss(action_preds0, action_target)
                    + w_user * ce_loss(user_preds, states[:, 0])
                    + w_session * ce_loss(action_preds_nov, action_target)
                    + w_diversity * ce_loss(action_preds_div, action_target)
                    + w_reward * mse_loss(reward_preds, reward_target)
                )
                user_state = state_hidden.detach()
                action_state = state_hidden.detach()
            elif model.mode == "single_objective":
                user_preds, action_preds0, action_preds, action_target, state_hidden, reward_preds, reward_target = model(
                    states,
                    actions,
                    user_state.unsqueeze(1),
                    action_state.unsqueeze(1),
                    user_mask,
                    sess_reward,
                    poss_rtg,
                )
                loss = (
                    w_supervised * ce_loss(action_preds0, action_target)
                    + w_user * ce_loss(user_preds, states[:, 0])
                    + w_session * ce_loss(action_preds, action_target)
                    + w_reward * mse_loss(reward_preds, reward_target)
                )
                user_state = state_hidden.detach()
                action_state = state_hidden.detach()
            else:
                action_preds, action_target, user_state_out, action_state_out, reward_preds, user_preds = model(
                    states,
                    actions,
                    user_state.unsqueeze(1),
                    action_state.unsqueeze(1),
                    user_mask,
                    sess_reward,
                )
                loss = (
                    w_supervised * ce_loss(action_preds, action_target)
                    + w_user * ce_loss(user_preds, states[:, 0])
                    + w_reward * mse_loss(reward_preds.squeeze(-1), sess_reward.squeeze(-1))
                )
                user_state = user_state_out.detach()
                action_state = action_state_out.detach()

            loss.backward()
            optimizer.step()
            total_step += 1
            if max_steps and total_step >= max_steps:
                break
        if max_steps and total_step >= max_steps:
            break

    checkpoint = output_dir / f"hdt_{model.mode}_{total_step}steps.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run a synthetic forward pass only.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps; 0 means full config run.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.dry_run:
        print(run_synthetic_forward(config))
        return
    print(f"saved checkpoint: {train(config)}")


if __name__ == "__main__":
    main()
