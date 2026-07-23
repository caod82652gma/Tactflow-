from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from data.loader import TactileTrialDataset, load_tactile_split_table
from data.normalize import StandardScaler
from models.mlp_multitask import TactileContainerClassifier
from utils.cv_metrics import classification_metrics, summarize_classification_folds, write_summary_files
from utils.wandb_utils import init_wandb, log_wandb_artifact


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/default.yaml")
    return yaml.safe_load(text)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_dataset(table: dict[str, Any]) -> TactileTrialDataset:
    return TactileTrialDataset(table["x"], table["y"])


def validate_compatible_tables(train_table: dict[str, Any], test_table: dict[str, Any]) -> None:
    if train_table["class_order"] != test_table["class_order"]:
        raise ValueError("Train and test class_order differ")
    if train_table["x"].shape[1] != test_table["x"].shape[1]:
        raise ValueError(
            "Train and test feature dimensions differ: "
            f"{train_table['x'].shape[1]} vs {test_table['x'].shape[1]}"
        )
    if train_table["tactile_columns"] != test_table["tactile_columns"]:
        raise ValueError("Train and test tactile columns differ")


def run_epoch(
    model: TactileContainerClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()
    total = 0
    sums = {"loss": 0.0, "acc": 0.0}

    for x, target_class in loader:
        x = x.to(device)
        target_class = target_class.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = ce(logits, target_class)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total += batch_size
        sums["loss"] += float(loss.detach().cpu()) * batch_size
        sums["acc"] += float((logits.argmax(dim=1) == target_class).float().sum().cpu())

    return {
        "loss": sums["loss"] / total,
        "acc": sums["acc"] / total,
    }


def predict_classes(
    model: TactileContainerClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predicted: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, target_class in loader:
            logits = model(x.to(device))
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(target_class.numpy())
    return np.concatenate(targets), np.concatenate(predicted)


def save_failure_csv(
    output_path: Path,
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_order: list[str],
) -> None:
    failed = y_true != y_pred
    pd.DataFrame(
        {
            "SourceFile": np.asarray(sample_ids, dtype=object)[failed],
            "True": [class_order[int(idx)] for idx in y_true[failed]],
            "Pred": [class_order[int(idx)] for idx in y_pred[failed]],
            "TrueIndex": y_true[failed],
            "PredIndex": y_pred[failed],
        }
    ).to_csv(output_path, index=False)


def train_once(
    config: dict[str, Any],
    project_root: Path,
    args: argparse.Namespace,
    run_name: str,
) -> dict[str, Any]:
    train_table = load_tactile_split_table(config, str(config["data"]["train_split"]))
    val_table = load_tactile_split_table(config, str(config["data"]["val_split"]))
    test_table = load_tactile_split_table(config, str(config["data"]["test_split"]))
    validate_compatible_tables(train_table, val_table)
    validate_compatible_tables(train_table, test_table)

    scaler = StandardScaler().fit(train_table["x"])
    train_table["x"] = scaler.transform(train_table["x"])
    val_table["x"] = scaler.transform(val_table["x"])
    test_table["x"] = scaler.transform(test_table["x"])

    train_ds = make_dataset(train_table)
    val_ds = make_dataset(val_table)
    test_ds = TactileTrialDataset(test_table["x"], test_table["y"])
    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = choose_device(str(config["train"]["device"]))
    model = TactileContainerClassifier(
        input_dim=train_table["x"].shape[1],
        num_classes=len(train_table["class_order"]),
        hidden_dim=max(32, int(config["model"]["hidden_dim"]) * 8),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    output_dir = (project_root / config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = output_dir / "best.pt"
    wandb_run = init_wandb(
        args.wandb,
        config,
        run_name=run_name,
        tags=["tactile", "container-classification", str(config["data"]["interpolation"])],
    )
    if wandb_run is not None:
        wandb_run.summary["input_dim"] = int(train_table["x"].shape[1])
        wandb_run.summary["num_train_trials"] = int(len(train_table["y"]))
        wandb_run.summary["num_val_trials"] = int(len(val_table["y"]))
        wandb_run.summary["num_test_trials"] = int(len(test_table["y"]))
        wandb_run.summary["num_classes"] = int(len(train_table["class_order"]))

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "class_order": train_table["class_order"],
                    "input_dim": train_table["x"].shape[1],
                    "x_mean": scaler.mean,
                    "x_std": scaler.std,
                },
                best_path,
            )
            if wandb_run is not None:
                wandb_run.summary["best_val_loss"] = float(best_val)
                wandb_run.summary["best_val_acc"] = float(val_metrics["acc"])
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/acc": train_metrics["acc"],
                    "val/loss": val_metrics["loss"],
                    "val/acc": val_metrics["acc"],
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f}"
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, None, device)
    y_true, y_pred = predict_classes(model, test_loader, device)
    cls_metrics = classification_metrics(y_true, y_pred, len(train_table["class_order"]))
    failure_path = output_dir / "failure.csv"
    save_failure_csv(
        failure_path,
        list(test_table["trial_ids"]),
        y_true,
        y_pred,
        list(train_table["class_order"]),
    )
    result = {
        "best_val_loss": best_val,
        "test": test_metrics,
        **cls_metrics,
        "tactile_root": train_table["root"],
        "splits": {
            "train": train_table["split"],
            "val": val_table["split"],
            "test": test_table["split"],
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log(
            {
                "test/loss": test_metrics["loss"],
                "test/acc": test_metrics["acc"],
            },
            step=int(config["train"]["epochs"]) + 1,
        )
        wandb_run.summary["test_loss"] = float(test_metrics["loss"])
        wandb_run.summary["test_acc"] = float(test_metrics["acc"])
        artifact_prefix = f"tactile-{config['data']['interpolation']}"
        log_wandb_artifact(wandb_run, best_path, f"{artifact_prefix}-best", "model")
        log_wandb_artifact(wandb_run, metrics_path, f"{artifact_prefix}-metrics", "metrics")
        wandb_run.finish()
    print(f"saved={best_path}")
    print(f"failure_csv={failure_path}")
    print(json.dumps({"test": test_metrics, "acc": cls_metrics["acc"], "macro_f1": cls_metrics["macro_f1"]}, indent=2))
    if args.eval:
        subprocess.run(
            [
                sys.executable,
                str(project_root / "eval.py"),
                "--checkpoint",
                str(best_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=project_root,
            check=True,
        )
    return result


def with_tactile_fold_config(base_config: dict[str, Any], fold: int, folds: int, output_root: str) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["data"]["split_root"] = f"splits/tactile/cv{folds}/fold_{fold}"
    config["data"]["train_split"] = "train"
    config["data"]["val_split"] = "val"
    config["data"]["test_split"] = "test"
    config["train"]["output_dir"] = f"{output_root}/fold_{fold}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--interpolation", choices=["base", "mean", "idw4"], default=None)
    parser.add_argument("--input-mode", choices=["tactile_summary", "pressure_16"], default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eval", action="store_true", help="Run eval.py after training finishes.")
    parser.add_argument("--fold", type=int, default=None, help="Run one CV fold from splits/tactile/cvK/fold_N.")
    parser.add_argument("--cv-folds", type=int, default=0, help="Run all folds from splits/tactile/cvK and write cv_summary.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config((project_root / args.config).resolve())
    if args.interpolation is not None:
        config["data"]["interpolation"] = args.interpolation
    if args.input_mode is not None:
        config.setdefault("features", {})["input_mode"] = args.input_mode
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.wandb_mode is not None:
        config.setdefault("wandb", {})["mode"] = args.wandb_mode
    if args.fold is not None and args.cv_folds <= 0:
        raise SystemExit("--fold requires --cv-folds")
    if args.cv_folds and args.cv_folds < 2:
        raise SystemExit("--cv-folds must be at least 2")

    run_name = args.run_name or f"tactile_{config['data']['interpolation']}"
    if args.cv_folds:
        output_root = str(config["train"]["output_dir"])
        folds = [args.fold] if args.fold is not None else list(range(args.cv_folds))
        fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            print(f"cv{args.cv_folds} fold={fold}")
            fold_config = with_tactile_fold_config(config, fold, args.cv_folds, output_root)
            result = train_once(fold_config, project_root, args, f"{run_name}_fold_{fold}")
            result["fold"] = fold
            fold_metrics.append(result)
        if args.fold is None:
            class_order = list(config["labels"]["class_order"])
            summary = summarize_classification_folds(fold_metrics, class_order)
            summary["fold_metrics"] = fold_metrics
            write_summary_files((project_root / output_root).resolve(), summary)
            print(json.dumps(summary, indent=2))
        return

    train_once(config, project_root, args, run_name)


if __name__ == "__main__":
    main()
