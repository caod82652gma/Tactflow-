from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1_scores: list[float] = []
    per_class_acc: list[float] = []
    for idx in range(num_classes):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
        per_class_acc.append(tp / max(float(matrix[idx, :].sum()), 1.0))
    return {
        "acc": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_acc": per_class_acc,
        "confusion": matrix.tolist(),
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "fold_values": [float(value) for value in array],
        "mean": float(array.mean()) if len(array) else 0.0,
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize_classification_folds(
    fold_metrics: list[dict[str, Any]],
    class_order: list[str],
) -> dict[str, Any]:
    matrices = [np.asarray(metrics["confusion"], dtype=np.int64) for metrics in fold_metrics]
    pooled = np.sum(matrices, axis=0) if matrices else np.zeros((len(class_order), len(class_order)), dtype=np.int64)
    correct = int(np.trace(pooled))
    total = int(pooled.sum())
    summary: dict[str, Any] = {
        "folds": len(fold_metrics),
        "class_order": class_order,
        "acc": _mean_std([float(metrics["acc"]) for metrics in fold_metrics]),
        "macro_f1": _mean_std([float(metrics["macro_f1"]) for metrics in fold_metrics]),
        "pooled_confusion": pooled.tolist(),
        "pooled_correct": correct,
        "pooled_total": total,
        "pooled_acc": correct / total if total else 0.0,
        "pooled_acc_wilson_95_ci": wilson_interval(correct, total),
        "per_class_acc": {},
    }
    for idx, label in enumerate(class_order):
        fold_values = [float(metrics["per_class_acc"][idx]) for metrics in fold_metrics]
        class_total = int(pooled[idx, :].sum())
        class_correct = int(pooled[idx, idx])
        summary["per_class_acc"][label] = {
            **_mean_std(fold_values),
            "pooled_correct": class_correct,
            "pooled_total": class_total,
            "pooled": class_correct / class_total if class_total else 0.0,
            "wilson_95_ci": wilson_interval(class_correct, class_total),
        }
    return summary


def write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for name in ("acc", "macro_f1"):
        metric = summary[name]
        rows.append(
            {
                "metric": name,
                "mean": metric["mean"],
                "std": metric["std"],
                "pooled": summary.get(f"pooled_{name}", ""),
                "wilson95_low": summary.get(f"pooled_{name}_wilson_95_ci", ["", ""])[0],
                "wilson95_high": summary.get(f"pooled_{name}_wilson_95_ci", ["", ""])[1],
            }
        )
    for label, metric in summary.get("per_class_acc", {}).items():
        rows.append(
            {
                "metric": f"per_class_acc/{label}",
                "mean": metric["mean"],
                "std": metric["std"],
                "pooled": metric["pooled"],
                "wilson95_low": metric["wilson_95_ci"][0],
                "wilson95_high": metric["wilson_95_ci"][1],
            }
        )
    with (output_dir / "cv_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["metric", "mean", "std", "pooled", "wilson95_low", "wilson95_high"])
        writer.writeheader()
        writer.writerows(rows)
