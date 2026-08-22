from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import torch

from hdt.models import HDT, TransformerVariant


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (root or project_root()) / value


def get_device(config: Dict[str, Any]) -> torch.device:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def infer_item_num(config: Dict[str, Any]) -> int:
    model_cfg = config.get("model", {})
    if model_cfg.get("item_num"):
        return int(model_cfg["item_num"])
    data_cfg = config.get("data", {})
    data_root = resolve_path(data_cfg.get("root", "data/processed/reddit"))
    item_meta = data_cfg.get("item_metadata", "item_data.csv")
    item_path = data_root / item_meta
    if not item_path.exists():
        raise FileNotFoundError(
            f"Cannot infer item_num because item metadata is missing: {item_path}. "
            "Set model.item_num in the config for smoke tests or prepare the data first."
        )
    return len(pd.read_csv(item_path))


def build_model(config: Dict[str, Any], device: torch.device | None = None) -> HDT:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    device = device or get_device(config)
    hidden_size = int(model_cfg.get("hidden_size", model_cfg.get("hidden_factor", 100)))
    variant = TransformerVariant(
        embed_dim=int(model_cfg.get("embed_dim", hidden_size)),
        n_layer=int(model_cfg.get("n_layer", 1)),
        n_head=int(model_cfg.get("n_head", 1)),
        activation_function=str(model_cfg.get("activation_function", "relu")),
        dropout=float(model_cfg.get("dropout", model_cfg.get("dropout_rate", 0.1))),
    )
    model = HDT(
        hidden_size=hidden_size,
        item_num=infer_item_num(config),
        state_size=int(model_cfg.get("state_size", 20)),
        dropout=float(model_cfg.get("dropout", model_cfg.get("dropout_rate", 0.1))),
        discount=float(training_cfg.get("discount", 0.5)),
        device=device,
        mode=str(model_cfg.get("mode", "multi_objective")),
        variant=variant,
        num_heads=int(model_cfg.get("num_heads", model_cfg.get("n_head", 1))),
        action_embedding_source=str(model_cfg.get("action_embedding_source", "actions")),
    )
    return model.to(device)


def make_loader_args(config: Dict[str, Any]) -> SimpleNamespace:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    data_root = resolve_path(data_cfg.get("root", "data/processed/reddit"))
    return SimpleNamespace(
        data_path=str(data_root) + "/",
        batch_size=int(training_cfg.get("batch_size", 200)),
        train_file=data_cfg.get("expected_train"),
        valid_file=data_cfg.get("expected_valid"),
        test_file=data_cfg.get("expected_test"),
    )


def synthetic_batch(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 4))
    state_size = int(model_cfg.get("state_size", 20))
    item_num = int(model_cfg.get("item_num", 100))
    states = torch.randint(0, item_num, (batch_size, state_size), device=device)
    actions = torch.randint(0, item_num, (batch_size, state_size), device=device)
    user_change = torch.zeros(batch_size, 1, device=device)
    rewards = torch.rand(batch_size, 1, device=device)
    position_rewards = torch.rand(batch_size, state_size, device=device)
    return {
        "inputs": states,
        "targets": actions,
        "user_change": user_change,
        "sess_reward": rewards,
        "poss_rtg": position_rewards,
        "sess_reward_nov": rewards,
        "poss_rtg_nov": position_rewards,
    }


def run_synthetic_forward(config: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device(config)
    model = build_model(config, device=device)
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
    return {
        "mode": config.get("model", {}).get("mode", "multi_objective"),
        "num_outputs": len(outputs),
        "first_shape": tuple(outputs[0].shape),
    }
