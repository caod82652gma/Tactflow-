from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.loader import TactileTrialDataset, load_tactile_split_table
from data.normalize import StandardScaler
from models.mlp_multitask import TactileContainerClassifier
from train import load_config, run_epoch, save_failure_csv


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


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def save_confusion_outputs(
    matrix: np.ndarray,
    class_order: list[str],
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = pd.DataFrame(matrix, index=class_order, columns=class_order)
    counts.index.name = "true"
    counts.columns.name = "pred"
    counts.to_csv(output_dir / f"{prefix}_confusion_counts.csv")

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float32), where=row_sums > 0)
    normalized_df = pd.DataFrame(normalized, index=class_order, columns=class_order)
    normalized_df.index.name = "true"
    normalized_df.columns.name = "pred"
    normalized_df.to_csv(output_dir / f"{prefix}_confusion_normalized.csv", float_format="%.6f")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(class_order)), labels=class_order)
    ax.set_yticks(np.arange(len(class_order)), labels=class_order)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = normalized[i, j]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(j, i, f"{value:.2f}\n({matrix[i, j]})", ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/default/best.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefix", default="test")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    checkpoint_path = (project_root / args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_config((project_root / args.config).resolve()) if args.config else checkpoint["config"]

    table = load_tactile_split_table(config, str(config["data"]["test_split"]))
    scaler = StandardScaler(checkpoint["x_mean"], checkpoint["x_std"])
    x = scaler.transform(table["x"])
    dataset = TactileTrialDataset(x, table["y"])
    loader = DataLoader(dataset, batch_size=int(config["train"]["batch_size"]))
    model = TactileContainerClassifier(
        int(checkpoint["input_dim"]),
        len(checkpoint["class_order"]),
        max(32, int(config["model"]["hidden_dim"]) * 8),
        float(config["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_epoch(
        model,
        loader,
        None,
        torch.device("cpu"),
    )
    y_true, y_pred = predict_classes(model, loader, torch.device("cpu"))
    matrix = confusion_matrix(y_true, y_pred, len(checkpoint["class_order"]))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    save_confusion_outputs(matrix, list(checkpoint["class_order"]), output_dir, args.prefix)
    save_failure_csv(
        output_dir / "failure.csv",
        list(table["trial_ids"]),
        y_true,
        y_pred,
        list(checkpoint["class_order"]),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
