from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.thermal_sequence_loader import (
    SequenceStandardScaler,
    ThermalSequenceDataset,
    load_thermal_sequence_table,
)
from models.thermal_sequence import ThermalSequenceClassifier
from thermal_train import _classification_metrics, _save_confusion, _save_failure_csv
from train import choose_device, load_config
from utils.cv_metrics import summarize_classification_folds, write_summary_files


BRANCH_LABELS = {
    "b1_temp_only": "B1",
    "b2_weighted_temp_only": "B2",
    "b3_no_container_one_hot": "B3",
    "b4_no_tau_hat": "B4",
    "b5_full_fusion": "B5",
    "b6_direct_pressure16_temp": "B6",
}


def _make_dataset(x: np.ndarray, y: np.ndarray) -> ThermalSequenceDataset:
    return ThermalSequenceDataset(x, y)


def _run_epoch(
    model: ThermalSequenceClassifier,
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


def _predict_sequence(
    model: ThermalSequenceClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


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
    scaler = SequenceStandardScaler().fit(train_table[feature_key])
    train_x = scaler.transform(train_table[feature_key])
    val_x = scaler.transform(val_table[feature_key])
    test_x = scaler.transform(test_table[feature_key])
    input_dim = train_table[feature_key].shape[-1]
    seq_cfg = config["thermal_sequence"]

    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(_make_dataset(train_x, train_table["y"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_make_dataset(val_x, val_table["y"]), batch_size=batch_size)
    test_loader = DataLoader(_make_dataset(test_x, test_table["y"]), batch_size=batch_size)

    model = ThermalSequenceClassifier(
        input_dim=input_dim,
        hidden_dim=int(seq_cfg["hidden_dim"]),
        num_classes=len(seq_cfg["level_order"]),
        encoder=str(seq_cfg.get("encoder", "gru")),
        num_layers=int(seq_cfg.get("num_layers", 1)),
        dropout=float(seq_cfg.get("dropout", 0.0)),
    ).to(device)
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
                    "sequence_length": int(train_table[feature_key].shape[1]),
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
    y_pred = _predict_sequence(model, test_loader, device)
    cls_metrics = _classification_metrics(test_table["y"], y_pred, len(seq_cfg["level_order"]))
    matrix = np.asarray(cls_metrics["confusion"], dtype=np.int64)
    _save_confusion(matrix, list(seq_cfg["level_order"]), branch_dir, "test")
    _save_failure_csv(
        branch_dir / "failure.csv",
        np.asarray(test_table["source_files"], dtype=object),
        test_table["y"],
        y_pred,
        list(seq_cfg["level_order"]),
    )
    result = {
        "best_val_loss": best_val,
        "test_loss": test_metrics["loss"],
        **cls_metrics,
    }
    (branch_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def train_thermal_sequence_once(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    train_table = load_thermal_sequence_table(config, "train")
    val_table = load_thermal_sequence_table(config, "val")
    test_table = load_thermal_sequence_table(config, "test")
    print(
        f"samples train={len(train_table['y'])} "
        f"val={len(val_table['y'])} test={len(test_table['y'])}"
    )

    output_dir = (project_root / config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(str(config["train"]["device"]))
    encoder = str(config["thermal_sequence"].get("encoder", "gru"))
    branch_specs = [
        (f"b1_temp_only_{encoder}", "b1_seq"),
        (f"b2_weighted_temp_only_{encoder}", "b2_seq"),
        (f"b3_no_container_one_hot_{encoder}", "b3_seq"),
        (f"b4_no_tau_hat_{encoder}", "b4_seq"),
        (f"b5_full_fusion_{encoder}", "b5_seq"),
        (f"b6_direct_pressure16_temp_{encoder}", "b6_seq"),
    ]
    summary = {}
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


def with_thermal_sequence_fold_config(
    base_config: dict[str, Any],
    fold: int,
    folds: int,
    output_root: str,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["thermal_sequence"]["split_root"] = f"splits/thermal/cv{folds}/fold_{fold}"
    config["thermal_sequence"]["train_split"] = "train.csv"
    config["thermal_sequence"]["val_split"] = "val.csv"
    config["thermal_sequence"]["test_split"] = "test.csv"
    config["train"]["output_dir"] = f"{output_root}/fold_{fold}"
    return config


def _branch_label(branch: str) -> str:
    for prefix, label in BRANCH_LABELS.items():
        if branch.startswith(prefix):
            return label
    return branch


def _percent_mean_std(metric: dict[str, Any]) -> str:
    return f"{float(metric['mean']) * 100.0:.2f} ± {float(metric['std']) * 100.0:.2f}"


def _summary_rows(cv_summary: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    branches = list(cv_summary["branches"])
    labels = [_branch_label(branch) for branch in branches]
    class_order = list(next(iter(cv_summary["branches"].values()))["class_order"])
    row_specs: list[tuple[str, str, str | None]] = [
        ("Accuracy (%)", "acc", None),
        ("Macro F1 (%)", "macro_f1", None),
    ]
    row_specs.extend((f"Per-class Accuracy {label} (%)", "per_class_acc", label) for label in class_order)

    rows: list[dict[str, str]] = []
    for row_name, metric_name, class_label in row_specs:
        row = {"Metric": row_name}
        for branch, label in zip(branches, labels, strict=True):
            branch_summary = cv_summary["branches"][branch]
            if class_label is None:
                row[label] = _percent_mean_std(branch_summary[metric_name])
            else:
                row[label] = _percent_mean_std(branch_summary[metric_name][class_label])
        rows.append(row)
    return labels, rows


def _write_markdown_table(path: Path, labels: list[str], rows: list[dict[str, str]]) -> None:
    lines = [
        f"| Metric | {' | '.join(labels)} |",
        f"|---|{'|'.join(['---:'] * len(labels))}|",
    ]
    for row in rows:
        lines.append(f"| {row['Metric']} | {' | '.join(row[label] for label in labels)} |")
    lines.append("")
    lines.append("Cells are fold mean ± sample standard deviation in percent.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_accuracy_svg(path: Path, cv_summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = list(cv_summary["branches"])
    labels = [_branch_label(branch) for branch in branches]
    means = np.asarray([cv_summary["branches"][branch]["acc"]["mean"] * 100.0 for branch in branches], dtype=np.float32)
    lows = np.asarray([cv_summary["branches"][branch]["pooled_acc_wilson_95_ci"][0] * 100.0 for branch in branches], dtype=np.float32)
    highs = np.asarray([cv_summary["branches"][branch]["pooled_acc_wilson_95_ci"][1] * 100.0 for branch in branches], dtype=np.float32)
    lower_err = np.maximum(means - lows, 0.0)
    upper_err = np.maximum(highs - means, 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(labels))
    ax.bar(x, means, color="#4C78A8", width=0.65)
    ax.errorbar(x, means, yerr=np.vstack([lower_err, upper_err]), fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("B1-B6 Sequence Accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_cv_summary_folder(output_dir: Path, cv_summary: dict[str, Any]) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    labels, rows = _summary_rows(cv_summary)
    with (summary_dir / "B_metrics_summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["Metric", *labels])
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown_table(summary_dir / "B_metrics_summary.md", labels, rows)
    _write_accuracy_svg(summary_dir / "B1_B6_accuracy_wilson95.svg", cv_summary)


def summarize_existing_run(output_dir: Path) -> None:
    cv_summary_path = output_dir / "cv_summary.json"
    if not cv_summary_path.exists():
        raise SystemExit(f"Missing cv_summary.json: {cv_summary_path}")
    cv_summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    write_cv_summary_folder(output_dir, cv_summary)
    print(f"summary_dir={output_dir / 'summary'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/thermal_sequence.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--encoder", choices=["gru", "lstm", "cnn", "transformer"], default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Run one CV fold from splits/thermal/cvK/fold_N.")
    parser.add_argument("--cv-folds", type=int, default=0, help="Run all folds from splits/thermal/cvK and write cv_summary files.")
    parser.add_argument("--summarize-only", action="store_true", help="Create summary/ from an existing output_dir/cv_summary.json.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config((project_root / args.config).resolve())
    encoder = str(args.encoder or config["thermal_sequence"].get("encoder", "gru"))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["learning_rate"] = args.lr
    if args.output_dir is not None:
        config["train"]["output_dir"] = args.output_dir
    elif args.encoder is not None:
        config["train"]["output_dir"] = f"runs/thermal_sequence_ablation_{encoder}"
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.encoder is not None:
        config["thermal_sequence"]["encoder"] = args.encoder
    if args.sequence_length is not None:
        config["thermal_sequence"]["sequence_length"] = args.sequence_length
    if args.summarize_only:
        summarize_existing_run((project_root / config["train"]["output_dir"]).resolve())
        return
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
            fold_config = with_thermal_sequence_fold_config(config, fold, args.cv_folds, output_root)
            result = train_thermal_sequence_once(fold_config, project_root)
            result["fold"] = fold
            fold_summaries.append(result)
        if args.fold is None:
            class_order = list(config["thermal_sequence"]["level_order"])
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
            write_cv_summary_folder(output_dir, cv_summary)
            print(json.dumps(cv_summary, indent=2))
        return

    train_thermal_sequence_once(config, project_root)


if __name__ == "__main__":
    main()
