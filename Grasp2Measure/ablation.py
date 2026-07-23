from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.loader import load_tactile_split_table
from data.normalize import StandardScaler
from models.mlp_multitask import TactileContainerClassifier
from train import load_config, make_dataset, run_epoch
from utils.wandb_utils import init_wandb


def run_one(config: dict, interpolation: str, wandb_enabled: bool = False) -> dict[str, float]:
    config = copy.deepcopy(config)
    config["data"]["interpolation"] = interpolation
    train_table = load_tactile_split_table(config, str(config["data"]["train_split"]))
    val_table = load_tactile_split_table(config, str(config["data"]["val_split"]))
    scaler = StandardScaler().fit(train_table["x"])
    train_table["x"] = scaler.transform(train_table["x"])
    val_table["x"] = scaler.transform(val_table["x"])

    train_loader = DataLoader(
        make_dataset(train_table),
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(make_dataset(val_table), batch_size=int(config["train"]["batch_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TactileContainerClassifier(
        train_table["x"].shape[1],
        len(train_table["class_order"]),
        max(32, int(config["model"]["hidden_dim"]) * 8),
        float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    best = None
    wandb_run = init_wandb(
        wandb_enabled,
        config,
        run_name=f"ablation_{interpolation}",
        tags=["tactile", "ablation", interpolation],
    )
    if wandb_run is not None:
        wandb_run.summary["input_dim"] = int(train_table["x"].shape[1])
        wandb_run.summary["num_train_trials"] = int(len(train_table["y"]))
        wandb_run.summary["num_val_trials"] = int(len(val_table["y"]))

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        metrics = run_epoch(model, val_loader, None, device)
        if best is None or metrics["loss"] < best["loss"]:
            best = metrics
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/acc": train_metrics["acc"],
                    "val/loss": metrics["loss"],
                    "val/acc": metrics["acc"],
                },
                step=epoch,
            )
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = float((best or {}).get("loss", 0.0))
        wandb_run.summary["best_val_acc"] = float((best or {}).get("acc", 0.0))
        wandb_run.finish()
    return best or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", nargs="+", default=["base", "mean", "idw4"])
    parser.add_argument("--wandb", action="store_true", help="Log each ablation variant to Weights & Biases.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config((project_root / args.config).resolve())
    if args.wandb_mode is not None:
        config.setdefault("wandb", {})["mode"] = args.wandb_mode
    config.setdefault("wandb", {})["job_type"] = "ablation"
    results = {variant: run_one(config, variant, args.wandb) for variant in args.variants}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
