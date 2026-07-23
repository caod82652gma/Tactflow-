from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.normalize import StandardScaler
from data.thermal_loader import ThermalLevelDataset, load_thermal_level_table
from models.thermal_level import ThermalLevelMLP
from train import choose_device, load_config
from utils.cv_metrics import summarize_classification_folds, write_summary_files


def _make_dataset(x: np.ndarray, y: np.ndarray) -> ThermalLevelDataset:
    return ThermalLevelDataset(x, y)


def _run_epoch(
    model: ThermalLevelMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()
    total = 0
    loss_sum = 0.0
    acc_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = ce(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += x.shape[0]
        loss_sum += float(loss.detach().cpu()) * x.shape[0]
        acc_sum += float((logits.argmax(dim=1) == y).float().sum().cpu())
    return {"loss": loss_sum / total, "acc": acc_sum / total}


def _predict(model: ThermalLevelMLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    matrix = _confusion(y_true, y_pred, num_classes)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1_scores: list[float] = []
    per_class_acc: list[float] = []
    for idx in range(num_classes):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
        per_class_acc.append(tp / max(float(matrix[idx, :].sum()), 1.0))
    return {
        "acc": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_acc": per_class_acc,
        "confusion": matrix.tolist(),
    }


def _save_confusion(matrix: np.ndarray, class_order: list[str], output_dir: Path, prefix: str) -> None:
    counts = pd.DataFrame(matrix, index=class_order, columns=class_order)
    counts.index.name = "true"
    counts.columns.name = "pred"
    counts.to_csv(output_dir / f"{prefix}_confusion_counts.csv")
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float32), where=row_sums > 0)
    pd.DataFrame(normalized, index=class_order, columns=class_order).to_csv(
        output_dir / f"{prefix}_confusion_normalized.csv",
        float_format="%.6f",
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_order)), labels=class_order)
    ax.set_yticks(np.arange(len(class_order)), labels=class_order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(prefix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{normalized[i, j]:.2f}\n({matrix[i, j]})", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_heatmap.png", dpi=220)
    plt.close(fig)


def _save_failure_csv(
    output_path: Path,
    source_files: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_order: list[str],
) -> None:
    failed = y_true != y_pred
    pd.DataFrame(
        {
            "SourceFile": source_files[failed],
            "True": [class_order[int(idx)] for idx in y_true[failed]],
            "Pred": [class_order[int(idx)] for idx in y_pred[failed]],
            "TrueIndex": y_true[failed],
            "PredIndex": y_pred[failed],
        }
    ).to_csv(output_path, index=False)


def _b0_predictions(table: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    delta = table["delta_t"]
    ambient = table["ambient_c"]
    liquid = table["liquid_c"]
    denom = np.maximum(liquid - ambient, 1e-6)[:, None]
    theta_max = (delta / denom).max(axis=1)
    k = float(config["b0"]["sigmoid_k"])
    l_value = float(config["b0"]["sigmoid_l"])
    x0 = float(config["b0"]["sigmoid_x0"])
    y0 = float(config["b0"]["sigmoid_y0"])
    eps = 1e-6
    theta_clamped = np.clip(theta_max, y0 + eps, y0 + l_value - eps)
    height = x0 - (1.0 / k) * np.log(l_value / (theta_clamped - y0) - 1.0)
    t1, t2 = [float(value) for value in config["b0"]["level_thresholds_mm"]]
    pred = np.zeros_like(height, dtype=np.int64)
    pred[height >= t1] = 1
    pred[height >= t2] = 2
    return pred, height.astype(np.float32)


def _train_branch(
    branch: str,
    train_table: dict[str, Any],
    val_table: dict[str, Any],
    test_table: dict[str, Any],
    feature_key: str,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(train_table[feature_key])
    train_x = scaler.transform(train_table[feature_key])
    val_x = scaler.transform(val_table[feature_key])
    test_x = scaler.transform(test_table[feature_key])
    input_dim = train_table[feature_key].shape[1]
    hidden_dim = 16 if branch.startswith("b1") else 32
    if branch.startswith("b1") and input_dim != 8:
        raise ValueError(f"{branch} expected 8 temperature features, got {input_dim}")

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(_make_dataset(train_x, train_table["y"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_make_dataset(val_x, val_table["y"]), batch_size=batch_size)
    test_loader = DataLoader(_make_dataset(test_x, test_table["y"]), batch_size=batch_size)

    model = ThermalLevelMLP(input_dim, hidden_dim, len(config["thermal"]["level_order"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    branch_dir = output_dir / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = branch_dir / "best.pt"
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        val_metrics = _run_epoch(model, val_loader, None, device)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "branch": branch,
                    "input_dim": input_dim,
                    "hidden_dim": hidden_dim,
                    "x_mean": scaler.mean,
                    "x_std": scaler.std,
                },
                best_path,
            )
        if epoch == 1 or epoch % 25 == 0 or epoch == int(config["train"]["epochs"]):
            print(
                f"{branch} epoch={epoch:03d} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f}"
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = _run_epoch(model, test_loader, None, device)
    y_pred = _predict(model, test_loader, device)
    cls_metrics = _classification_metrics(test_table["y"], y_pred, len(config["thermal"]["level_order"]))
    matrix = np.asarray(cls_metrics["confusion"], dtype=np.int64)
    _save_confusion(matrix, list(config["thermal"]["level_order"]), branch_dir, "test")
    _save_failure_csv(
        branch_dir / "failure.csv",
        np.asarray(test_table["source_files"], dtype=object),
        test_table["y"],
        y_pred,
        list(config["thermal"]["level_order"]),
    )
    result = {
        "best_val_loss": best_val,
        "test_loss": test_metrics["loss"],
        **cls_metrics,
    }
    (branch_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_thermal_once(
    config: dict[str, Any],
    project_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_table = load_thermal_level_table(config, "train", args.tactile_checkpoint)
    val_table = load_thermal_level_table(config, "val", args.tactile_checkpoint)
    test_table = load_thermal_level_table(config, "test", args.tactile_checkpoint)
    print(
        f"samples train={len(train_table['y'])} "
        f"val={len(val_table['y'])} test={len(test_table['y'])}"
    )

    output_dir = (project_root / config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(str(config["train"]["device"]))

    b0_pred, b0_height = _b0_predictions(test_table, config)
    b0_metrics = _classification_metrics(test_table["y"], b0_pred, len(config["thermal"]["level_order"]))
    b0_dir = output_dir / "b0_physics"
    b0_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion(np.asarray(b0_metrics["confusion"], dtype=np.int64), list(config["thermal"]["level_order"]), b0_dir, "test")
    test_source_files = np.asarray(test_table["source_files"], dtype=object)
    _save_failure_csv(
        b0_dir / "failure.csv",
        test_source_files,
        test_table["y"],
        b0_pred,
        list(config["thermal"]["level_order"]),
    )
    pd.DataFrame(
        {
            "SourceFile": test_source_files,
            "True": [config["thermal"]["level_order"][idx] for idx in test_table["y"]],
            "Pred": [config["thermal"]["level_order"][idx] for idx in b0_pred],
            "Height_hat_mm": b0_height,
        }
    ).to_csv(b0_dir / "predictions.csv", index=False)
    (b0_dir / "metrics.json").write_text(json.dumps(b0_metrics, indent=2), encoding="utf-8")

    branch_specs = [
        ("b1_temp_only", "b1_x"),
        ("b2_weighted_temp_only", "b2_x"),
        ("b3_no_container_one_hot", "b3_x"),
        ("b4_no_tau_hat", "b4_x"),
        ("b5_full_fusion", "b5_x"),
        ("b6_direct_pressure16_temp", "b6_x"),
    ]
    summary = {"b0_physics": b0_metrics}
    for branch, feature_key in branch_specs:
        summary[branch] = _train_branch(
            branch,
            train_table,
            val_table,
            test_table,
            feature_key,
            config,
            output_dir,
            device,
        )
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def with_thermal_fold_config(base_config: dict[str, Any], fold: int, folds: int, output_root: str) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["thermal"]["split_root"] = f"splits/thermal/cv{folds}/fold_{fold}"
    config["thermal"]["train_split"] = "train.csv"
    config["thermal"]["val_split"] = "val.csv"
    config["thermal"]["test_split"] = "test.csv"
    config["train"]["output_dir"] = f"{output_root}/fold_{fold}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/thermal.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--tactile-checkpoint",
        default=None,
        help="Checkpoint for predicting B2 container codes from tactile inputs.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Run one CV fold from splits/thermal/cvK/fold_N.")
    parser.add_argument("--cv-folds", type=int, default=0, help="Run all folds from splits/thermal/cvK and write cv_summary files.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config((project_root / args.config).resolve())
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
    if args.tactile_checkpoint is not None:
        config["thermal"]["tactile_classifier_checkpoint"] = args.tactile_checkpoint
    if args.fold is not None and args.cv_folds <= 0:
        raise SystemExit("--fold requires --cv-folds")
    if args.cv_folds and args.cv_folds < 2:
        raise SystemExit("--cv-folds must be at least 2")

    if args.cv_folds:
        output_root = str(config["train"]["output_dir"])
        folds = [args.fold] if args.fold is not None else list(range(args.cv_folds))
        fold_summaries: list[dict[str, Any]] = []
        for fold in folds:
            print(f"cv{args.cv_folds} fold={fold}")
            fold_config = with_thermal_fold_config(config, fold, args.cv_folds, output_root)
            result = train_thermal_once(fold_config, project_root, args)
            result["fold"] = fold
            fold_summaries.append(result)
        if args.fold is None:
            class_order = list(config["thermal"]["level_order"])
            cv_summary: dict[str, Any] = {"folds": args.cv_folds, "branches": {}, "fold_metrics": fold_summaries}
            for branch in fold_summaries[0]:
                if branch == "fold":
                    continue
                branch_summary = summarize_classification_folds(
                    [fold_summary[branch] for fold_summary in fold_summaries],
                    class_order,
                )
                cv_summary["branches"][branch] = branch_summary
                write_summary_files((project_root / output_root / branch).resolve(), branch_summary)
            output_dir = (project_root / output_root).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
            print(json.dumps(cv_summary, indent=2))
        return

    train_thermal_once(config, project_root, args)


if __name__ == "__main__":
    main()
