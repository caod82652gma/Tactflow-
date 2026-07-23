#!/usr/bin/env python
# coding: utf-8

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from plot_style import COLORS, SAVE_DIRS, add_subplot_label, apply_measurement_style, clear_experiment_outputs, cn_label, save_figure

apply_measurement_style()
SAVE_DIR = SAVE_DIRS[4]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(SAVE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
clear_experiment_outputs((SCRIPT_DIR, CSV_DIR), SAVE_DIR, ("04_", "04a_", "04b_"))

V_REF = 5.0
LSB = V_REF / (2**15)

BASE_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "..",
    "Workspace",
    "A_calibration",
    "A04_hysteresis",
)

FORCE_CONFIG = {
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
    "10": 10.0,
}

CYCLE_FOLDERS = [
    ("cycle1", 1),
    ("cycle2", 2),
    ("cycle3", 3),
    ("cycle4", 4),
    ("cycle5", 5),
    ("cycle6", 6),
    ("cycle7", 7),
    ("cycle8", 8),
]


def load_sensor_data(file_path):
    """Load one sensor CSV and normalize column names used by old/new files."""
    df = pd.read_csv(file_path)
    if len(df.columns) < 3:
        raise ValueError(f"{file_path} must contain at least 3 columns.")
    df = df.iloc[:, :3].copy()
    df.columns = ["Time_ms", "RawValue", "RawGround"]
    return df


def extract_stable_mean(df, start_ratio=0.2, end_ratio=0.8):
    """Return stable-window mean/std in raw ADC codes."""
    n = len(df)
    start_idx = int(n * start_ratio)
    end_idx = int(n * end_ratio)
    stable_data = df["RawValue"].iloc[start_idx:end_idx]
    return float(np.mean(stable_data)), float(np.std(stable_data, ddof=1))


def get_csv_files_sorted_by_timestamp(folder_path):
    """Return CSV data files sorted by timestamp in the filename."""
    files = [
        path
        for path in glob.glob(os.path.join(folder_path, "*.csv"))
        if not os.path.basename(path).startswith("04_hysteresis_")
    ]

    def extract_timestamp(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r"(\d{8})_(\d{6})", basename)
        if match:
            return match.group(1) + match.group(2)
        return basename

    files.sort(key=extract_timestamp)
    return files


def load_loading_unloading_data(folder_path):
    """Load loading/unloading stable means for one force folder."""
    csv_files = get_csv_files_sorted_by_timestamp(folder_path)
    if len(csv_files) < 2:
        return None

    loading_file, unloading_file = csv_files[0], csv_files[1]
    loading_mean, loading_std = extract_stable_mean(load_sensor_data(loading_file))
    unloading_mean, unloading_std = extract_stable_mean(load_sensor_data(unloading_file))

    return {
        "loading_file": loading_file,
        "unloading_file": unloading_file,
        "loading_lsb": loading_mean,
        "unloading_lsb": unloading_mean,
        "loading_std_lsb": loading_std,
        "unloading_std_lsb": unloading_std,
        "loading_abs_v": abs(loading_mean * LSB),
        "unloading_abs_v": abs(unloading_mean * LSB),
        "loading_std_v": loading_std * LSB,
        "unloading_std_v": unloading_std * LSB,
    }


def load_cycle_data(cycle_folder, cycle_number):
    """Load all configured force points for one cycle."""
    cycle_path = os.path.join(BASE_PATH, cycle_folder)
    if not os.path.isdir(cycle_path):
        print(f"Missing cycle folder: {cycle_path}")
        return None

    rows = []
    print(f"\nCycle {cycle_number}: {cycle_path}")
    for force_folder, force_value in sorted(FORCE_CONFIG.items(), key=lambda item: item[1]):
        force_path = os.path.join(cycle_path, force_folder)
        if not os.path.isdir(force_path):
            print(f"  - Missing force folder: {force_folder}")
            continue

        point = load_loading_unloading_data(force_path)
        if point is None:
            print(f"  - Force {force_value:.0f} N has fewer than two CSV files.")
            continue

        point.update(
            {
                "Cycle": cycle_number,
                "CycleFolder": cycle_folder,
                "ForceFolder": force_folder,
                "Force_N": force_value,
            }
        )
        rows.append(point)
        print(
            f"  + {force_value:>4.1f} N: "
            f"load={point['loading_abs_v']:.6f} V, "
            f"unload={point['unloading_abs_v']:.6f} V"
        )

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("Force_N").reset_index(drop=True)


def require_complete_force_grid(cycle_frames):
    """Keep only force points present in every loaded cycle."""
    force_sets = [set(frame["Force_N"]) for frame in cycle_frames]
    common_forces = sorted(set.intersection(*force_sets))
    if not common_forces:
        raise RuntimeError("No common force points found across cycles.")

    filtered = []
    for frame in cycle_frames:
        filtered.append(frame[frame["Force_N"].isin(common_forces)].sort_values("Force_N").reset_index(drop=True))
    return common_forces, filtered


def summarize_cycles(cycle_frames):
    """Build per-point, per-cycle, and mean/std summaries."""
    enriched_frames = []
    rows = []

    for frame in cycle_frames:
        frame = frame.copy()
        cycle = int(frame["Cycle"].iloc[0])
        full_scale_v = float(frame["loading_abs_v"].max() - frame["loading_abs_v"].min())
        if full_scale_v == 0:
            errors_percent = np.full(len(frame), np.nan)
            max_hysteresis = np.nan
            avg_hysteresis = np.nan
        else:
            errors_v = np.abs(frame["unloading_abs_v"].values - frame["loading_abs_v"].values)
            errors_percent = errors_v / abs(full_scale_v) * 100.0
            max_hysteresis = float(np.nanmax(errors_percent))
            avg_hysteresis = float(np.nanmean(errors_percent))

        frame["Hysteresis_percentFS"] = errors_percent
        enriched_frames.append(frame)

        rows.append(
            {
                "Cycle": cycle,
                "Full_Scale_V": full_scale_v,
                "Max_Hysteresis_percentFS": max_hysteresis,
                "Avg_Hysteresis_percentFS": avg_hysteresis,
                "Num_Test_Points": len(frame),
            }
        )

    all_points = pd.concat(enriched_frames, ignore_index=True)

    force_summary = (
        all_points.groupby("Force_N")
        .agg(
            Loading_absV_mean=("loading_abs_v", "mean"),
            Loading_absV_std=("loading_abs_v", "std"),
            Unloading_absV_mean=("unloading_abs_v", "mean"),
            Unloading_absV_std=("unloading_abs_v", "std"),
            Loading_LSB_mean=("loading_lsb", "mean"),
            Loading_LSB_std=("loading_lsb", "std"),
            Unloading_LSB_mean=("unloading_lsb", "mean"),
            Unloading_LSB_std=("unloading_lsb", "std"),
            Hysteresis_percentFS_mean=("Hysteresis_percentFS", "mean"),
            Hysteresis_percentFS_std=("Hysteresis_percentFS", "std"),
        )
        .reset_index()
        .sort_values("Force_N")
    )

    cycle_summary = pd.DataFrame(rows).sort_values("Cycle").reset_index(drop=True)
    return all_points, force_summary, cycle_summary


def add_hysteresis_annotation(ax, force_summary, stats):
    """Mark the force point where the maximum hysteresis was observed."""
    if not stats:
        return

    annotation_color = "#3A3A3A"
    force = float(stats["force_n"])
    idx = int(np.abs(force_summary["Force_N"].values - force).argmin())
    row = force_summary.iloc[idx]
    x = float(row["Force_N"])
    y_load = float(row["Loading_absV_mean"])
    y_unload = float(row["Unloading_absV_mean"])
    y_low, y_high = sorted((y_load, y_unload))
    y_mid = (y_low + y_high) / 2.0

    ax.plot(
        [x, x],
        [y_low, y_high],
        color=annotation_color,
        linewidth=1.0,
        solid_capstyle="round",
        zorder=5,
    )

    ax.text(
        x + 0.18,
        y_mid,
        f"max @ {x:g} N",
        ha="left",
        va="center",
        fontsize=8,
        color=annotation_color,
    )


def add_hysteresis_legend(ax, max_percent=None, avg_percent=None):
    handles, labels = ax.get_legend_handles_labels()
    if max_percent is not None and avg_percent is not None:
        note_handle = Line2D([], [], linestyle="none", marker="", color="none")
        handles = list(handles) + [note_handle]
        labels = list(labels) + [
            cn_label(
                f"Max hysteresis = {max_percent:.3f}%FS\nAvg hysteresis = {avg_percent:.3f}%FS"
            )
        ]
    ax.legend(handles, labels, frameon=False, fontsize=9)


def plot_main_panel(ax, cycle_frames, force_summary, simple=False, hysteresis_stats=None):
    """Main hysteresis panel."""
    forces = force_summary["Force_N"].values
    load_mean = force_summary["Loading_absV_mean"].values
    unload_mean = force_summary["Unloading_absV_mean"].values
    load_std = force_summary["Loading_absV_std"].fillna(0).values
    unload_std = force_summary["Unloading_absV_std"].fillna(0).values

    if not simple:
        for i, frame in enumerate(cycle_frames):
            label_loading = "Individual loading cycles" if i == 0 else None
            label_unloading = "Individual unloading cycles" if i == 0 else None
            ax.plot(
                frame["Force_N"],
                frame["loading_abs_v"],
                color=COLORS["gray"],
                alpha=0.30,
                linewidth=1.0,
                marker="o",
                markersize=3,
                label=label_loading,
            )
            ax.plot(
                frame["Force_N"],
                frame["unloading_abs_v"],
                color=COLORS["gray"],
                alpha=0.30,
                linewidth=1.0,
                linestyle="--",
                marker="s",
                markersize=3,
                label=label_unloading,
            )

        ax.fill_between(
            forces,
            load_mean - load_std,
            load_mean + load_std,
            color=COLORS["loading"],
            alpha=0.14,
            linewidth=0,
            label="Loading +/-1 sigma",
        )
        ax.fill_between(
            forces,
            unload_mean - unload_std,
            unload_mean + unload_std,
            color=COLORS["unloading"],
            alpha=0.14,
            linewidth=0,
            label="Unloading +/-1 sigma",
        )

    ax.plot(
        forces,
        load_mean,
        color=COLORS["loading"],
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Loading",
    )
    ax.plot(
        forces,
        unload_mean,
        color=COLORS["unloading"],
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Unloading",
    )

    if hysteresis_stats:
        add_hysteresis_annotation(ax, force_summary, hysteresis_stats)

    ax.set_xlabel("Force (N)")
    ax.set_ylabel(r"$|V_{total}|$ (V)")
    add_hysteresis_legend(
        ax,
        hysteresis_stats["max_percent"] if hysteresis_stats else None,
        hysteresis_stats["avg_percent"] if hysteresis_stats else None,
    )
    if not simple:
        add_subplot_label(ax, "(a)")


def plot_cycle_bar_panel(ax, cycle_summary):
    """Side panel: maximum hysteresis per cycle and break-in comparison."""
    cycles = cycle_summary["Cycle"].astype(int).values
    hmax = cycle_summary["Max_Hysteresis_percentFS"].values
    x = np.arange(len(cycles))
    colors = [COLORS["orange"] if cycle <= 3 else COLORS["blue"] for cycle in cycles]

    ax.bar(x, hmax, color=colors, edgecolor="none", alpha=0.86, width=0.66)

    first3 = float(np.nanmean(hmax[cycles <= 3]))
    last5 = float(np.nanmean(hmax[cycles >= 4]))
    delta = first3 - last5

    ax.axhline(
        first3,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.1,
        label=f"Mean cycles 1-3 = {first3:.3f}%FS",
    )
    ax.axhline(
        last5,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.1,
        label=f"Mean cycles 4-8 = {last5:.3f}%FS",
    )

    if len(cycles) >= 2 and np.all(np.isfinite(hmax)):
        coef = np.polyfit(cycles, hmax, 1)
        fit = np.poly1d(coef)(cycles)
        ax.plot(x, fit, color=COLORS["red"], linewidth=1.2, marker=None, label=f"Trend {coef[0]:+.3f}%FS/cycle")

    ax.text(
        0.98,
        0.96,
        f"Break-in delta\n{delta:+.3f}%FS",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.86),
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(cycle) for cycle in cycles])
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Max hysteresis (%FS)")
    ax.legend(frameon=False, fontsize=8)
    add_subplot_label(ax, "(b)")


def main():
    print(f"Base path: {BASE_PATH}")
    print(f"Configured force points: {len(FORCE_CONFIG)}")
    print(f"Configured cycles: {', '.join(name for name, _ in CYCLE_FOLDERS)}")

    loaded_cycles = []
    for cycle_folder, cycle_number in CYCLE_FOLDERS:
        frame = load_cycle_data(cycle_folder, cycle_number)
        if frame is not None:
            loaded_cycles.append(frame)

    if len(loaded_cycles) < 2:
        raise RuntimeError("At least two complete cycles are required for cycle-to-cycle hysteresis analysis.")

    common_forces, cycle_frames = require_complete_force_grid(loaded_cycles)
    print(f"\nLoaded cycles: {len(cycle_frames)}")
    print(f"Common force points: {', '.join(f'{force:g} N' for force in common_forces)}")

    all_points, force_summary, cycle_summary = summarize_cycles(cycle_frames)

    max_row = cycle_summary.loc[cycle_summary["Max_Hysteresis_percentFS"].idxmax()]
    first3_mean = float(cycle_summary.loc[cycle_summary["Cycle"] <= 3, "Max_Hysteresis_percentFS"].mean())
    last5_mean = float(cycle_summary.loc[cycle_summary["Cycle"] >= 4, "Max_Hysteresis_percentFS"].mean())
    first3_std = float(cycle_summary.loc[cycle_summary["Cycle"] <= 3, "Max_Hysteresis_percentFS"].std(ddof=1))
    last5_std = float(cycle_summary.loc[cycle_summary["Cycle"] >= 4, "Max_Hysteresis_percentFS"].std(ddof=1))
    hmax_force_row = force_summary.loc[force_summary["Hysteresis_percentFS_mean"].idxmax()]
    hmax_mean = float(hmax_force_row["Hysteresis_percentFS_mean"])
    hmax_std = float(hmax_force_row["Hysteresis_percentFS_std"])
    hmax_force = float(hmax_force_row["Force_N"])
    overall_hmax = float(max_row["Max_Hysteresis_percentFS"])
    overall_hmax_force = float(
        all_points.loc[
            all_points["Hysteresis_percentFS"].idxmax(),
            "Force_N",
        ]
    )
    overall_hmax_force_std = float(
        force_summary.loc[
            np.isclose(force_summary["Force_N"], overall_hmax_force),
            "Hysteresis_percentFS_std",
        ].iloc[0]
    )
    avg_force_hysteresis = float(max_row["Avg_Hysteresis_percentFS"])
    hysteresis_stats = {
        "force_n": overall_hmax_force,
        "max_percent": overall_hmax,
        "avg_percent": avg_force_hysteresis,
    }

    print("\nPer-cycle maximum hysteresis:")
    print(cycle_summary.to_string(index=False))
    print("\nForce-point mean and cycle-to-cycle sigma:")
    print(force_summary.to_string(index=False))
    print(
        f"\nMaximum cycle hysteresis: {max_row['Max_Hysteresis_percentFS']:.4f} %FS "
        f"(cycle {int(max_row['Cycle'])})"
    )
    print(f"Break-in mean cycles 1-3: {first3_mean:.4f} %FS")
    print(f"Steady mean cycles 4-8: {last5_mean:.4f} %FS")
    print(f"Mean difference (1-3 minus 4-8): {first3_mean - last5_mean:+.4f} %FS")
    print(
        f"8-cycle max force-point hysteresis: {hmax_mean:.4f} +/- {hmax_std:.4f} %FS "
        f"@ {hmax_force:g} N"
    )
    print(f"11-force-point average hysteresis: {avg_force_hysteresis:.4f} %FS")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [2.15, 1]})
    plot_main_panel(axes[0], cycle_frames, force_summary, hysteresis_stats=hysteresis_stats)
    plot_cycle_bar_panel(axes[1], cycle_summary)
    fig.suptitle("Hysteresis Across 8 Loading-Unloading Cycles", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.close(fig)

    fig_main, ax_main = plt.subplots(figsize=(6, 4.5))
    plot_main_panel(ax_main, cycle_frames, force_summary, simple=True, hysteresis_stats=hysteresis_stats)
    plt.tight_layout()
    p = save_figure(fig_main, SAVE_DIR, "04a_hysteresis_8cycle_main")
    print(f"Saved: {p}")
    plt.close(fig_main)

    fig_bar, ax_bar = plt.subplots(figsize=(6.0, 4.2))
    plot_cycle_bar_panel(ax_bar, cycle_summary)
    plt.tight_layout()
    p = save_figure(fig_bar, SAVE_DIR, "04b_hysteresis_cycle_max_bar")
    print(f"Saved: {p}")
    plt.close(fig_bar)

    params = {
        "Num_Cycles": len(cycle_frames),
        "Num_Test_Points": len(common_forces),
        "Eight_Cycle_Epsilon_h_Max_percentFS": overall_hmax,
        "Eight_Cycle_Epsilon_h_Max_Std_at_Force_percentFS": overall_hmax_force_std,
        "Eight_Cycle_Epsilon_h_Max_Force_N": overall_hmax_force,
        "Avg_11Force_Epsilon_h_percentFS": avg_force_hysteresis,
        "BreakIn_Mean_Cycle1_3_percentFS": first3_mean,
        "Steady_Mean_Cycle4_8_percentFS": last5_mean,
        "BreakIn_Delta_percentFS": first3_mean - last5_mean,
    }
    pd.DataFrame([params]).to_csv(os.path.join(CSV_DIR, "04_hysteresis_params.csv"), index=False)

    # Compatibility output: mean loading/unloading curve by force point.
    legacy_like = pd.DataFrame(
        {
            "Force_N": force_summary["Force_N"],
            "Loading_absV_mean": force_summary["Loading_absV_mean"],
            "Unloading_absV_mean": force_summary["Unloading_absV_mean"],
            "Loading_absV_std_cycle": force_summary["Loading_absV_std"],
            "Unloading_absV_std_cycle": force_summary["Unloading_absV_std"],
            "Hysteresis_percentFS_mean": force_summary["Hysteresis_percentFS_mean"],
            "Hysteresis_percentFS_std": force_summary["Hysteresis_percentFS_std"],
            "Error_absV_mean": np.abs(
                force_summary["Unloading_absV_mean"] - force_summary["Loading_absV_mean"]
            ),
        }
    )
    legacy_like["Error_percentFS_mean_curve"] = (
        legacy_like["Error_absV_mean"]
        / abs(legacy_like["Loading_absV_mean"].max() - legacy_like["Loading_absV_mean"].min())
        * 100.0
    )
    legacy_like.to_csv(os.path.join(CSV_DIR, "04_hysteresis_data.csv"), index=False)

    print("\nSaved data tables:")
    print(f"  {os.path.join(CSV_DIR, '04_hysteresis_data.csv')}")
    print(f"  {os.path.join(CSV_DIR, '04_hysteresis_params.csv')}")


if __name__ == "__main__":
    main()
