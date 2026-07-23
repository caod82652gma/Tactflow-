from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
OUTPUT_ROOT = RUNS_ROOT / "A_results_cn"

CLASS_LABELS_CN = {
    "15ml": "15 ml",
    "50ml": "50 ml",
    "100ml": "100 ml",
    "200ml": "200 ml",
    "Low": "低液位",
    "Mid": "中液位",
    "High": "高液位",
}

def configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_heatmap(counts_path: Path, output_path: Path) -> None:
    counts_df = pd.read_csv(counts_path, index_col=0)
    class_order = [str(value) for value in counts_df.index]
    labels = [CLASS_LABELS_CN.get(label, label) for label in class_order]
    matrix = counts_df.to_numpy(dtype=np.int64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=row_sums > 0,
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = normalized[i, j]
            text_color = "white" if value >= 0.55 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}\n({matrix[i, j]})",
                ha="center",
                va="center",
                color=text_color,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    configure_chinese_font()
    generated = 0
    for counts_path in sorted(RUNS_ROOT.rglob("test_confusion_counts.csv")):
        if OUTPUT_ROOT in counts_path.parents:
            continue
        relative_dir = counts_path.parent.relative_to(RUNS_ROOT)
        output_path = OUTPUT_ROOT / relative_dir / "test_confusion_heatmap.png"
        save_heatmap(counts_path, output_path)
        generated += 1
        print(output_path)
    print(f"generated={generated}")


if __name__ == "__main__":
    main()
