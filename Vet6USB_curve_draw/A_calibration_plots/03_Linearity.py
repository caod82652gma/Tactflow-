#!/usr/bin/env python
# coding: utf-8

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from plot_style import (
    COLORS,
    SAVE_DIRS,
    add_subplot_label,
    apply_measurement_style,
    clear_experiment_outputs,
    save_figure,
)

apply_measurement_style()
SAVE_DIR = SAVE_DIRS[3]
CSV_DIR = os.path.join(SAVE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
clear_experiment_outputs((SCRIPT_DIR, CSV_DIR), SAVE_DIR, ("03_", "03a_", "03b_", "test3summary"))

V_REF = 5.0
LSB = V_REF / (2**15)

WORKSPACE_TEST3 = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "Workspace", "A_calibration", "A03_linearity")
)
BASE_PATH = os.path.join(WORKSPACE_TEST3, "mean")

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


def load_sensor_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ["Time_ms", "RawValue", "RawGround"]
    return df


def extract_stable_mean(df, start_ratio=0.2, end_ratio=0.8):
    n = len(df)
    start_idx = int(n * start_ratio)
    end_idx = int(n * end_ratio)
    stable_data = df["RawValue"].iloc[start_idx:end_idx]
    return float(np.mean(stable_data)), float(np.std(stable_data))


def get_csv_files_in_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    files.sort()
    return files


def load_folder_data(folder_path):
    csv_files = get_csv_files_in_folder(folder_path)
    if not csv_files:
        return None, None

    means = []
    stds = []
    for file_path in csv_files:
        df = load_sensor_data(file_path)
        mean_val, std_val = extract_stable_mean(df)
        means.append(mean_val)
        stds.append(std_val)

    return float(np.mean(means)), float(np.mean(stds))


def load_calibration_points(base_path, force_config, verbose=True):
    forces = []
    outputs = []
    stds = []
    loaded_folders = []

    for folder_name, force_value in sorted(force_config.items(), key=lambda x: x[1]):
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            if verbose:
                print(f"Warning: folder not found - {folder_path}")
            continue

        mean_val, std_val = load_folder_data(folder_path)
        if mean_val is None:
            if verbose:
                print(f"Warning: no csv files in folder - {folder_path}")
            continue

        forces.append(force_value)
        outputs.append(mean_val)
        stds.append(std_val)
        loaded_folders.append(folder_name)

        if verbose:
            csv_count = len(get_csv_files_in_folder(folder_path))
            print(
                f"Folder {folder_name} ({force_value:.1f} N): "
                f"{csv_count} file(s), mean={mean_val:.2f} LSB"
            )

    return (
        np.array(forces, dtype=float),
        np.array(outputs, dtype=float),
        np.array(stds, dtype=float),
        loaded_folders,
    )


def fit_cubic_model(force_values, output_values):
    coeffs = np.polyfit(force_values, output_values, 3)
    poly = np.poly1d(coeffs)
    predicted = poly(force_values)
    residuals = output_values - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((output_values - np.mean(output_values)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    rms = float(np.sqrt(np.mean(residuals**2)))
    return coeffs, poly, predicted, residuals, r2, rms


def load_cycle_fit_results(workspace_test3_path, force_config):
    cycle_results = []
    for cycle_idx in range(1, 4):
        cycle_name = f"cycle{cycle_idx}"
        cycle_path = os.path.join(workspace_test3_path, cycle_name)
        if not os.path.isdir(cycle_path):
            print(f"Warning: {cycle_name} not found - {cycle_path}")
            continue

        cycle_forces, cycle_outputs, _, _ = load_calibration_points(
            cycle_path, force_config, verbose=False
        )
        if len(cycle_forces) < 4:
            print(f"Warning: {cycle_name} has fewer than 4 points; cubic fit skipped")
            continue

        coeffs, _, _, _, cycle_r2, _ = fit_cubic_model(cycle_forces, cycle_outputs)
        cycle_results.append(
            {
                "Cycle": cycle_name,
                "a3_LSB_per_N3": coeffs[0],
                "a2_LSB_per_N2": coeffs[1],
                "a1_LSB_per_N": coeffs[2],
                "a0_LSB": coeffs[3],
                "R2": cycle_r2,
            }
        )
    return cycle_results


print(f"Base path: {BASE_PATH}")
print(f"Configured force points: {len(FORCE_CONFIG)}")

forces, outputs, stds, loaded_folders = load_calibration_points(BASE_PATH, FORCE_CONFIG)
if len(forces) < 4:
    raise RuntimeError("At least 4 valid force points are required for cubic fitting.")

print(f"\nLoaded {len(forces)} test points")
print("Calibration data:")
print("-" * 60)
print(f"{'Folder':<10} {'Force (N)':<12} {'Output (LSB)':<15} {'Std (LSB)':<12}")
print("-" * 60)
for folder, force, output, std in zip(loaded_folders, forces, outputs, stds):
    print(f"{folder:<10} {force:<12.2f} {output:<15.2f} {std:<12.3f}")

coeffs, poly3, predicted, residuals, r2, rms_residual = fit_cubic_model(forces, outputs)
a3, a2, a1, a0 = coeffs

full_scale = outputs[-1] - outputs[0]
max_abs_residual = float(np.max(np.abs(residuals)))
nonlinearity_fs = max_abs_residual / abs(full_scale) * 100 if full_scale else np.nan

zero_mask = np.isclose(forces, 0.0)
residual_at_0n = float(residuals[zero_mask][0]) if np.any(zero_mask) else np.nan

cycle_results = load_cycle_fit_results(WORKSPACE_TEST3, FORCE_CONFIG)
cycle_a1 = np.array([row["a1_LSB_per_N"] for row in cycle_results], dtype=float)
cycle_a1_by_name = {row["Cycle"]: row["a1_LSB_per_N"] for row in cycle_results}
a1_1 = cycle_a1_by_name.get("cycle1", np.nan)
a1_2 = cycle_a1_by_name.get("cycle2", np.nan)
a1_3 = cycle_a1_by_name.get("cycle3", np.nan)
a1_mean = float(np.nanmean([a1_1, a1_2, a1_3]))
if len(cycle_a1) > 0:
    a1_scan_mean = float(np.mean(cycle_a1))
    a1_scan_std = float(np.std(cycle_a1, ddof=1)) if len(cycle_a1) > 1 else 0.0
    a1_rel_std_percent = (
        a1_scan_std / abs(a1_scan_mean) * 100 if a1_scan_mean != 0 else np.nan
    )
else:
    a1_scan_mean = np.nan
    a1_scan_std = np.nan
    a1_rel_std_percent = np.nan

print("=" * 60)
print("Cubic fit result")
print("=" * 60)
print(
    "V_mean = "
    f"{a3:.6g} F_total^3 + {a2:.6g} F_total^2 + "
    f"{a1:.6g} F_total + {a0:.6g}"
)
print(f"a3: {a3:.6f} LSB/N^3")
print(f"a2: {a2:.6f} LSB/N^2")
print(f"a1: {a1:.6f} LSB/N")
print(f"a0: {a0:.6f} LSB")
print(f"R2: {r2:.6f}")
print(f"LS nonlinearity: {nonlinearity_fs:.6f} %FS")
print(f"max|r|: {max_abs_residual:.6f} LSB")
print(f"RMS(r): {rms_residual:.6f} LSB")
print(f"0 N residual: {residual_at_0n:.6f} LSB")
print(f"Cycle1 a1: {a1_1:.2f}")
print(f"Cycle2 a1: {a1_2:.2f}")
print(f"Cycle3 a1: {a1_3:.2f}")
print(f"Mean a1: {a1_mean:.2f}")
print(f"a1 scan mean: {a1_scan_mean:.6f} LSB/N")
print(f"sigma(a1)/mean(a1): {a1_rel_std_percent:.6f} %")

# Bootstrap 95% CI for the cubic fit curve.
n_boot = 500
force_fine = np.linspace(forces.min(), forces.max(), 200)
boot_preds = np.zeros((n_boot, len(force_fine)))
rng = np.random.default_rng(42)
for boot_idx in range(n_boot):
    sample_idx = rng.integers(0, len(forces), len(forces))
    boot_coeffs = np.polyfit(forces[sample_idx], outputs[sample_idx], 3)
    boot_preds[boot_idx] = np.poly1d(boot_coeffs)(force_fine)
ci_lo = np.percentile(boot_preds, 2.5, axis=0)
ci_hi = np.percentile(boot_preds, 97.5, axis=0)

force_fit = np.linspace(forces.min(), forces.max(), 200)
x_pos = np.arange(len(forces))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.errorbar(
    forces,
    outputs,
    yerr=3 * stds,
    fmt="o",
    color=COLORS["data"],
    capsize=3,
    elinewidth=0.8,
    markersize=5,
    label="Measured (mean +/- 3 sigma)",
)
ax.plot(
    force_fit,
    poly3(force_fit),
    color=COLORS["fit"],
    linewidth=1.5,
    label=f"Cubic fit  R2={r2:.4f}",
)
ax.fill_between(force_fine, ci_lo, ci_hi, color=COLORS["fit"], alpha=0.15, label="95% CI")
ax.set_xlabel("Force (N)")
ax.set_ylabel("ADC Output (LSB)")
ax.text(
    0.97,
    0.05,
    f"R2 = {r2:.4f}\nRMS = {rms_residual:.1f} LSB\nMax |r| = {max_abs_residual:.1f} LSB",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
)
ax.legend(frameon=False, fontsize=8)
add_subplot_label(ax, "(a)")

ax = axes[1]
ax.bar(x_pos, residuals, width=0.55, color=COLORS["residual"], alpha=0.8, edgecolor="none")
ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{force:.0f}" for force in forces])
ax.set_xlabel("Force (N)")
ax.set_ylabel("Residual (LSB)")
if full_scale:
    ax2b = ax.twinx()
    ax2b.set_ylabel("Residual (%FS)")
    ax2b.set_ylim(ax.get_ylim()[0] / abs(full_scale) * 100, ax.get_ylim()[1] / abs(full_scale) * 100)
    ax2b.spines["top"].set_visible(False)
add_subplot_label(ax, "(b)")

fig.suptitle("Linearity & Sensitivity Calibration", fontsize=11, y=1.01)
plt.tight_layout()
plt.close(fig)

for stem in ["03a_calibration", "03b_residuals"]:
    fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
    if stem == "03a_calibration":
        ax_s.errorbar(
            forces,
            outputs,
            yerr=3 * stds,
            fmt="o",
            color=COLORS["data"],
            capsize=3,
            elinewidth=0.8,
            markersize=5,
        )
        ax_s.plot(force_fit, poly3(force_fit), color=COLORS["fit"], linewidth=1.5)
        ax_s.fill_between(force_fine, ci_lo, ci_hi, color=COLORS["fit"], alpha=0.15)
        ax_s.set_xlabel("Force (N)")
        ax_s.set_ylabel("ADC Output (LSB)")
        ax_s.text(
            0.97,
            0.05,
            f"R2 = {r2:.4f}\nRMS = {rms_residual:.1f} LSB\nMax |r| = {max_abs_residual:.1f} LSB",
            transform=ax_s.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )
    else:
        ax_s.bar(
            x_pos,
            residuals,
            width=0.55,
            color=COLORS["residual"],
            alpha=0.8,
            edgecolor="none",
        )
        ax_s.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
        ax_s.set_xticks(x_pos)
        ax_s.set_xticklabels([f"{force:.0f}" for force in forces])
        ax_s.set_xlabel("Force (N)")
        ax_s.set_ylabel("Residual (LSB)")
        if full_scale:
            ax2s = ax_s.twinx()
            ax2s.set_ylabel("Residual (%FS)")
            ax2s.set_ylim(
                ax_s.get_ylim()[0] / abs(full_scale) * 100,
                ax_s.get_ylim()[1] / abs(full_scale) * 100,
            )
            ax2s.spines["top"].set_visible(False)
    plt.tight_layout()
    path = save_figure(fig_s, SAVE_DIR, stem)
    print(f"Saved: {path}")
    plt.close(fig_s)

print("Residuals by force point:")
print("-" * 78)
print(
    f"{'Force (N)':<12} {'Measured':<12} {'Predicted':<12} "
    f"{'Residual (LSB)':<16} {'Residual (%FS)':<14}"
)
print("-" * 78)
for force, output, pred, residual in zip(forces, outputs, predicted, residuals):
    residual_fs = residual / abs(full_scale) * 100 if full_scale else np.nan
    print(f"{force:<12.1f} {output:<12.2f} {pred:<12.2f} {residual:<16.3f} {residual_fs:<14.4f}")

calibration_params = {
    "a3_LSB_per_N3": a3,
    "a2_LSB_per_N2": a2,
    "a1_LSB_per_N": a1,
    "a0_LSB": a0,
    "a1_scan_mean_LSB_per_N": a1_scan_mean,
    "a1_scan_std_LSB_per_N": a1_scan_std,
    "a1_scan_rel_std_percent": a1_rel_std_percent,
    "Full_Scale_LSB": full_scale,
    "R2": r2,
    "RMS_residual_LSB": rms_residual,
    "Max_abs_residual_LSB": max_abs_residual,
    "Nonlinearity_LS_percentFS": nonlinearity_fs,
    "Residual_at_0N_LSB": residual_at_0n,
}
pd.DataFrame([calibration_params]).to_csv(
    os.path.join(CSV_DIR, "03_calibration_params.csv"), index=False
)

df_cal = pd.DataFrame(
    {
        "Folder": loaded_folders,
        "Force_N": forces,
        "Output_LSB": outputs,
        "Std_LSB": stds,
        "Predicted_Cubic_LSB": predicted,
        "Residual_Cubic_LSB": residuals,
    }
)
df_cal.to_csv(os.path.join(CSV_DIR, "03_calibration_data.csv"), index=False)

df_residuals = pd.DataFrame(
    {
        "Force_N": forces,
        "V_mean_LSB": outputs,
        "V_mean_hat_LSB": predicted,
        "Residual_LSB": residuals,
        "Residual_percentFS": residuals / abs(full_scale) * 100 if full_scale else np.nan,
    }
)
df_residuals.to_csv(os.path.join(CSV_DIR, "03b_residuals.csv"), index=False)

summary_rows = [
    ("a3", a3, "LSB/N^3"),
    ("a2", a2, "LSB/N^2"),
    ("a1", a1, "LSB/N"),
    ("a0", a0, "LSB"),
    ("R2", r2, ""),
    ("LS_nonlinearity", nonlinearity_fs, "%FS"),
    ("max_abs_r", max_abs_residual, "LSB"),
    ("RMS_r", rms_residual, "LSB"),
    ("residual_at_0N", residual_at_0n, "LSB"),
    ("sigma_a1_over_mean_a1", a1_rel_std_percent, "%"),
]
pd.DataFrame(summary_rows, columns=["Item", "Value", "Unit"]).to_csv(
    os.path.join(CSV_DIR, "test3summary.csv"), index=False
)

print("=" * 60)
print("Experiment 03 summary")
print("=" * 60)
print(f"Saved CSV: 03_calibration_data.csv")
print(f"Saved CSV: 03_calibration_params.csv")
print(f"Saved CSV: 03b_residuals.csv")
print(f"Saved CSV: test3summary.csv")
