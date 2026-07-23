from __future__ import annotations

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


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    denom = max(1, x.shape[0] - 1)
    explained = (singular_values**2) / denom
    ratio = explained[:2] / max(explained.sum(), 1e-12)
    return coords.astype(np.float32), ratio.astype(np.float32)


def kmeans(x: np.ndarray, n_clusters: int, seed: int, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if n_clusters > len(x):
        raise ValueError(f"n_clusters={n_clusters} is larger than sample count={len(x)}")
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int64)

    for _ in range(max_iter):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = x[mask].mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, len(x))]

    return labels, centers


def class_distance_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(x), dtype=np.float32)
    for class_id in np.unique(y):
        mask = y == class_id
        class_x = x[mask]
        centroid = class_x.mean(axis=0)
        distances = np.linalg.norm(class_x - centroid, axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        scale = 1.4826 * mad if mad > 1e-12 else float(distances.std() + 1e-6)
        scores[mask] = (distances - median) / max(scale, 1e-6)
    return scores


def cluster_majority_labels(cluster_labels: np.ndarray, y: np.ndarray, n_clusters: int) -> dict[int, int]:
    result: dict[int, int] = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if not mask.any():
            result[cluster_id] = -1
            continue
        counts = np.bincount(y[mask])
        result[cluster_id] = int(counts.argmax())
    return result


def save_report(
    path: Path,
    table: dict[str, Any],
    split_name: np.ndarray,
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    majority: dict[int, int],
    outlier_scores: np.ndarray,
) -> None:
    class_order = table["class_order"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "trial_id",
                "split",
                "class",
                "cluster",
                "cluster_majority_class",
                "cluster_mismatch",
                "outlier_score",
                "selected_frame_index",
                "stability_score",
                "pc1",
                "pc2",
            ]
        )
        order = np.argsort(-outlier_scores)
        y = table["y"]
        for rank, row_idx in enumerate(order, start=1):
            cluster_id = int(cluster_labels[row_idx])
            majority_id = majority[cluster_id]
            writer.writerow(
                [
                    rank,
                    table["trial_ids"][row_idx],
                    split_name[row_idx],
                    class_order[int(y[row_idx])],
                    cluster_id,
                    class_order[majority_id] if majority_id >= 0 else "",
                    int(majority_id != int(y[row_idx])),
                    f"{float(outlier_scores[row_idx]):.6f}",
                    int(table["selected_frame_indices"][row_idx]),
                    f"{float(table['stability_scores'][row_idx]):.6f}",
                    f"{float(coords[row_idx, 0]):.6f}",
                    f"{float(coords[row_idx, 1]):.6f}",
                ]
            )


def plot_clusters(
    path: Path,
    table: dict[str, Any],
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    outlier_scores: np.ndarray,
    explained_ratio: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = table["y"]
    class_order = table["class_order"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for class_id, label in enumerate(class_order):
        mask = y == class_id
        axes[0].scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.85, label=label)
    axes[0].set_title("PCA by true container class")
    axes[0].legend(loc="best", fontsize=8)

    scatter = axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        c=cluster_labels,
        s=28 + np.clip(outlier_scores, 0, 6) * 10,
        cmap="tab10",
        alpha=0.85,
    )
    axes[1].set_title("KMeans cluster; larger points are farther from class center")
    fig.colorbar(scatter, ax=axes[1], label="cluster")

    x_label = f"PC1 ({explained_ratio[0] * 100:.1f}% var.)"
    y_label = f"PC2 ({explained_ratio[1] * 100:.1f}% var.)"
    for axis in axes:
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def combine_split_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    class_order = tables[0]["class_order"]
    return {
        "x": np.concatenate([table["x"] for table in tables], axis=0),
        "y": np.concatenate([table["y"] for table in tables], axis=0),
        "trial_ids": [trial_id for table in tables for trial_id in table["trial_ids"]],
        "selected_frame_indices": np.concatenate(
            [table["selected_frame_indices"] for table in tables],
            axis=0,
        ),
        "stability_scores": np.concatenate([table["stability_scores"] for table in tables], axis=0),
        "class_order": class_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster and visualize tactile trial-level features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--interpolation", choices=["base", "mean", "idw4"], default=None)
    parser.add_argument("--clusters", type=int, default=None, help="Default: number of configured classes.")
    parser.add_argument("--output-dir", default="runs/cluster_tactile")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config((PROJECT_ROOT / args.config).resolve())
    if args.interpolation is not None:
        config["data"]["interpolation"] = args.interpolation
    seed = int(args.seed)

    split_names = [
        str(config["data"]["train_split"]),
        str(config["data"]["val_split"]),
        str(config["data"]["test_split"]),
    ]
    split_tables = [load_tactile_split_table(config, name) for name in split_names]
    table = combine_split_tables(split_tables)
    row_split_names = np.concatenate(
        [np.full(len(table_part["y"]), name, dtype=object) for table_part, name in zip(split_tables, split_names)],
        axis=0,
    )
    x = StandardScaler().fit(table["x"]).transform(table["x"])
    n_clusters = args.clusters or len(table["class_order"])

    coords, explained_ratio = pca_2d(x)
    cluster_labels, _ = kmeans(x, n_clusters=n_clusters, seed=seed)
    majority = cluster_majority_labels(cluster_labels, table["y"], n_clusters)
    outlier_scores = class_distance_scores(x, table["y"])
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    stem = str(config["data"]["interpolation"])
    report_path = output_dir / f"{stem}_cluster_report.csv"
    figure_path = output_dir / f"{stem}_cluster_pca.png"
    save_report(report_path, table, row_split_names, coords, cluster_labels, majority, outlier_scores)
    plot_clusters(figure_path, table, coords, cluster_labels, outlier_scores, explained_ratio)

    mismatches = sum(majority[int(cluster_labels[i])] != int(table["y"][i]) for i in range(len(table["y"])))
    print(f"samples={len(table['y'])}")
    print(f"classes={table['class_order']}")
    print(f"clusters={n_clusters}")
    print(f"cluster_mismatches={mismatches}")
    print(f"figure={figure_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
