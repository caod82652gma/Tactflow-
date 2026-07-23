from __future__ import annotations

# Usage examples:
#   python scripts/pca_tactile_trials.py --interpolation idw4
#   python scripts/pca_tactile_trials.py --interpolation mean
#   python scripts/pca_tactile_trials.py --interpolation base
# Defaults:
#   config: configs/default.yaml
#   tactile root: data.tactile_root -> ../Workspace/C_model_training/tactile
#   splits: tactile/splits/train.csv, val.csv, test.csv
#   tactile summary: features.tactile_summary from config, default stable_frame
#   output dir: runs/cluster_tactile/pca_analysis

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("matplotlib is required: python -m pip install matplotlib") from exc

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required: python -m pip install PyYAML") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import load_tactile_split_table  # noqa: E402
from data.normalize import StandardScaler  # noqa: E402


PRESSURE_FEATURE_NAMES = [
    "left_n_active",
    "left_x_center",
    "left_y_center",
    "left_sigma_x2",
    "left_sigma_y2",
    "left_sigma_xy",
    "left_peak_abs_mv",
    "left_mean_abs_mv",
    "right_n_active",
    "right_x_center",
    "right_y_center",
    "right_sigma_x2",
    "right_sigma_y2",
    "right_sigma_xy",
    "right_peak_abs_mv",
    "right_mean_abs_mv",
]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    denom = max(1, x.shape[0] - 1)
    explained = (singular_values**2) / denom
    ratio = explained[:2] / max(explained.sum(), 1e-12)
    return coords.astype(np.float32), ratio.astype(np.float32), vt[:2].astype(np.float32)


def with_feature_mode(config: dict[str, Any], input_mode: str) -> dict[str, Any]:
    mode_config = dict(config)
    mode_config["features"] = dict(config["features"])
    mode_config["features"]["input_mode"] = input_mode
    return mode_config


def load_dataset_tables(config: dict[str, Any], input_mode: str) -> list[dict[str, Any]]:
    mode_config = with_feature_mode(config, input_mode)
    tables = []
    for split_key in ("train_split", "val_split", "test_split"):
        split_name = str(mode_config["data"][split_key])
        table = load_tactile_split_table(mode_config, split_name)
        table["dataset"] = split_name
        tables.append(table)
    return tables


def combine_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    class_order = list(tables[0]["class_order"])
    for table in tables[1:]:
        if list(table["class_order"]) != class_order:
            raise ValueError(f"class_order mismatch in {table['dataset']}")

    x = np.concatenate([table["x"] for table in tables], axis=0)
    y = np.concatenate([table["y"] for table in tables], axis=0)
    dataset = np.concatenate(
        [np.full(len(table["y"]), table["dataset"], dtype=object) for table in tables],
        axis=0,
    )
    trial_ids = [trial_id for table in tables for trial_id in table["trial_ids"]]
    selected_frame_indices = np.concatenate([table["selected_frame_indices"] for table in tables], axis=0)
    stability_scores = np.concatenate([table["stability_scores"] for table in tables], axis=0)

    return {
        "x": x,
        "y": y,
        "dataset": dataset,
        "trial_ids": trial_ids,
        "selected_frame_indices": selected_frame_indices,
        "stability_scores": stability_scores,
        "class_order": class_order,
    }


def group_labels(table: dict[str, Any]) -> np.ndarray:
    class_order = table["class_order"]
    return np.asarray(
        [f"{dataset}_{class_order[int(label)]}" for dataset, label in zip(table["dataset"], table["y"])],
        dtype=object,
    )


def save_pca_points(
    path: Path,
    table: dict[str, Any],
    coords: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_order = table["class_order"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "class",
                "group",
                "trial_id",
                "selected_frame_index",
                "stability_score",
                "pc1",
                "pc2",
            ]
        )
        labels = group_labels(table)
        for idx in range(len(table["y"])):
            writer.writerow(
                [
                    table["dataset"][idx],
                    class_order[int(table["y"][idx])],
                    labels[idx],
                    table["trial_ids"][idx],
                    int(table["selected_frame_indices"][idx]),
                    f"{float(table['stability_scores'][idx]):.6f}",
                    f"{float(coords[idx, 0]):.6f}",
                    f"{float(coords[idx, 1]):.6f}",
                ]
            )


def save_pca_summary(path: Path, explained_ratio: np.ndarray, components: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "explained_variance_ratio"])
        for idx, ratio in enumerate(explained_ratio, start=1):
            writer.writerow([f"PC{idx}", f"{float(ratio):.8f}"])
        writer.writerow([])
        writer.writerow(["component", "feature_index", "loading"])
        for component_idx, row in enumerate(components, start=1):
            for feature_idx, loading in enumerate(row):
                writer.writerow([f"PC{component_idx}", feature_idx, f"{float(loading):.8f}"])


def plot_pca(path: Path, table: dict[str, Any], coords: np.ndarray, explained_ratio: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = group_labels(table)
    unique_labels = sorted(set(labels), key=lambda value: (value.split("_")[0], value.split("_")[-1]))

    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    markers = {"train": "o", "val": "s", "test": "^"}
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        dataset = str(label).rsplit("_", maxsplit=1)[0]
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=30,
            alpha=0.82,
            marker=markers.get(dataset, "o"),
            color=cmap(idx % 10),
            label=label,
        )

    axis.set_title(title)
    axis.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% var.)")
    axis.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% var.)")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8, ncols=2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_group_means(path: Path, table: dict[str, Any], feature_names: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_order = table["class_order"]
    n_features = table["x"].shape[1]
    names = feature_names if feature_names and len(feature_names) == n_features else [f"feature_{idx}" for idx in range(n_features)]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "class", "sample_count", *names])
        for class_idx, class_name in enumerate(class_order):
            mask = table["y"] == class_idx
            if not mask.any():
                continue
            mean_values = table["x"][mask].mean(axis=0)
            writer.writerow(
                [
                    table["dataset"],
                    class_name,
                    int(mask.sum()),
                    *[f"{float(value):.6f}" for value in mean_values],
                ]
            )


def run_pca_analysis(
    config: dict[str, Any],
    input_mode: str,
    output_dir: Path,
    interpolation: str,
) -> None:
    tables = load_dataset_tables(config, input_mode)
    combined = combine_tables(tables)
    scaled = StandardScaler().fit_transform(combined["x"])
    coords, explained_ratio, components = pca_2d(scaled)

    stem = f"{interpolation}_{input_mode}"
    save_pca_points(output_dir / f"{stem}_pca_points.csv", combined, coords)
    save_pca_summary(output_dir / f"{stem}_pca_summary.csv", explained_ratio, components)
    plot_pca(
        output_dir / f"{stem}_pca.png",
        combined,
        coords,
        explained_ratio,
        title=f"{input_mode} PCA ({interpolation})",
    )

    if input_mode == "pressure_16":
        for table in tables:
            save_group_means(
                output_dir / f"{interpolation}_{table['dataset']}_pressure16_feature_means.csv",
                table,
                PRESSURE_FEATURE_NAMES,
            )

    print(
        f"{input_mode}: samples={len(combined['y'])}, features={combined['x'].shape[1]}, "
        f"pc1={explained_ratio[0] * 100:.2f}%, pc2={explained_ratio[1] * 100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA analysis for tactile split trial data.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--interpolation", choices=["base", "mean", "idw4"], default="idw4")
    parser.add_argument("--output-dir", default="runs/cluster_tactile/pca_analysis")
    args = parser.parse_args()

    config = load_config((PROJECT_ROOT / args.config).resolve())
    config["data"]["interpolation"] = args.interpolation

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    run_pca_analysis(config, "tactile_summary", output_dir, args.interpolation)
    run_pca_analysis(config, "pressure_16", output_dir, args.interpolation)

    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
