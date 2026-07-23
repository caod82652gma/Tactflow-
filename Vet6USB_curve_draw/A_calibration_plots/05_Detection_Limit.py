#!/usr/bin/env python
# coding: utf-8

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
SAVE_DIR = SAVE_DIRS[5]
CSV_DIR = os.path.join(SAVE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
clear_experiment_outputs((SCRIPT_DIR, CSV_DIR), SAVE_DIR, ("05_", "05a_", "05b_"))

V_REF = 5.0
LSB = V_REF / (2**15)
VPP_NOISE_THRESHOLD_LSB = 7.0

BASE_PATH = os.path.join(
    SCRIPT_DIR,
    "..",
    "..",
    "Workspace",
    "A_calibration",
    "A05_detection_limit",
)

FORCE_CONFIG = {
    "0": 0.0,     # zero-load baseline
    "1": 0.05,    # 50 mN
    "2": 0.10,    # 100 mN
    "3": 0.20,    # 200 mN
    "4": 0.50,    # 500 mN
}


def load_sensor_data(file_path):
    df = pd.read_csv(file_path)
    if len(df.columns) < 3:
        raise ValueError(f"{file_path} must contain at least three columns.")
    df = df.iloc[:, :3].copy()
    df.columns = ["Time_ms", "RawValue", "RawGround"]
    return df


def extract_stable_stats(df, start_ratio=0.2, end_ratio=0.8):
    n = len(df)
    start_idx = int(n * start_ratio)
    end_idx = int(n * end_ratio)
    stable_data = df["RawValue"].iloc[start_idx:end_idx].to_numpy(dtype=float)
    return {
        "mean": float(np.mean(stable_data)),
        "std": float(np.std(stable_data, ddof=1)),
        "min": float(np.min(stable_data)),
        "max": float(np.max(stable_data)),
        "data": stable_data,
    }


def get_csv_file_in_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    files.sort()
    return files[0] if files else None


def cycle_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"cycle(\d+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else name


def get_cycle_paths(base_path):
    cycle_paths = [
        path for path in glob.glob(os.path.join(base_path, "cycle*"))
        if os.path.isdir(path)
    ]
    cycle_paths.sort(key=cycle_sort_key)
    if cycle_paths:
        return cycle_paths
    return [base_path]


print(f"主路径: {BASE_PATH}")
print(f"配置的测试点数量: {len(FORCE_CONFIG)} (含零点)")
print(f"检测阈值: Vpp,noise = {VPP_NOISE_THRESHOLD_LSB:.1f} LSB")

cycle_rows = []
zero_segments = []
force_segments = {force: [] for force in FORCE_CONFIG.values() if force > 0}

print("数据加载详情:")
print("=" * 70)

for cycle_path in get_cycle_paths(BASE_PATH):
    cycle_name = os.path.basename(cycle_path)
    cycle_stats = {}

    for folder_name, force_value in sorted(FORCE_CONFIG.items(), key=lambda item: item[1]):
        folder_path = os.path.join(cycle_path, folder_name)
        if not os.path.isdir(folder_path):
            print(f"✗ {cycle_name}/{folder_name}: 文件夹不存在")
            continue

        csv_file = get_csv_file_in_folder(folder_path)
        if csv_file is None:
            print(f"✗ {cycle_name}/{folder_name}: 没有CSV文件")
            continue

        stats_data = extract_stable_stats(load_sensor_data(csv_file))
        cycle_stats[force_value] = {
            "folder": folder_name,
            "file": csv_file,
            "stats": stats_data,
        }
        print(
            f"✓ {cycle_name}/{folder_name} ({force_value*1000:.0f} mN): "
            f"{os.path.basename(csv_file)}  mean={stats_data['mean']:.2f} LSB, "
            f"σ={stats_data['std']:.3f} LSB"
        )

    if 0.0 not in cycle_stats:
        print(f"跳过 {cycle_name}: 没有零负载基线")
        continue

    zero_stats_cycle = cycle_stats[0.0]["stats"]
    zero_segments.append(zero_stats_cycle["data"])
    zero_mean_cycle = zero_stats_cycle["mean"]

    for force in sorted(force for force in cycle_stats if force > 0):
        stats_data = cycle_stats[force]["stats"]
        delta_s = stats_data["mean"] - zero_mean_cycle
        force_segments[force].append(stats_data["data"])
        cycle_rows.append(
            {
                "Cycle": cycle_name,
                "Force_N": force,
                "Force_mN": force * 1000,
                "Mean_LSB": stats_data["mean"],
                "Std_LSB": stats_data["std"],
                "Zero_Mean_LSB": zero_mean_cycle,
                "Delta_S_LSB": delta_s,
                "Abs_Delta_S_LSB": abs(delta_s),
            }
        )

if not cycle_rows:
    raise RuntimeError("未加载到可用的 cycle1..cycle10 检测下限数据。")

zero_data_all = np.concatenate(zero_segments)
noise_mean = float(np.mean(zero_data_all))
noise_std = float(np.std(zero_data_all, ddof=1))
noise_3sigma = 3 * noise_std
zero_data_centered = zero_data_all - noise_mean
zero_bin_width = 0.5
zero_bin_start = np.floor(zero_data_centered.min() / zero_bin_width) * zero_bin_width
zero_bin_end = np.ceil(zero_data_centered.max() / zero_bin_width) * zero_bin_width
bins_z = np.arange(zero_bin_start, zero_bin_end + zero_bin_width, zero_bin_width)
if bins_z.size < 2:
    bins_z = np.array([zero_bin_start, zero_bin_start + zero_bin_width], dtype=float)

df_cycles = pd.DataFrame(cycle_rows)

summary_rows = []
for force, group in df_cycles.groupby("Force_N", sort=True):
    force_data_all = np.concatenate(force_segments[force])
    t_stat, p_value = stats.ttest_ind(force_data_all, zero_data_all, equal_var=False)
    abs_delta_mean = float(group["Abs_Delta_S_LSB"].mean())
    abs_delta_std = float(group["Abs_Delta_S_LSB"].std(ddof=1))
    snr = abs_delta_mean / noise_std if noise_std > 0 else np.inf
    summary_rows.append(
        {
            "Force_N": force,
            "Force_mN": force * 1000,
            "Mean_LSB": float(group["Mean_LSB"].mean()),
            "Std_LSB": float(group["Mean_LSB"].std(ddof=1)),
            "Delta_S_mean_LSB": float(group["Delta_S_LSB"].mean()),
            "Abs_Delta_S_mean_LSB": abs_delta_mean,
            "Abs_Delta_S_std_LSB": abs_delta_std,
            "SNR": snr,
            "SNR_dB": 20 * np.log10(snr) if snr > 0 else -np.inf,
            "Detectable_VppNoise": abs_delta_mean > VPP_NOISE_THRESHOLD_LSB,
            "Detectable_3sigma": abs_delta_mean > noise_3sigma,
            "p_value": p_value,
            "Significant": p_value < 0.001,
            "N_cycles": int(group["Cycle"].nunique()),
        }
    )

df_results = pd.DataFrame(summary_rows)

print(f"\n共加载 {df_cycles['Cycle'].nunique()} 个cycle, {len(df_results)} 个非零力值点")
print("=" * 60)
print("【零负载噪声基准】")
print("=" * 60)
print(f"零点均值: {noise_mean:.3f} LSB")
print(f"噪声标准差 σ: {noise_std:.3f} LSB")
print(f"噪声 3σ: {noise_3sigma:.3f} LSB")
print(f"噪声峰峰阈值 Vpp,noise: {VPP_NOISE_THRESHOLD_LSB:.3f} LSB")

print("微小力值检测结果:")
print("-" * 100)
print(
    df_results[
        [
            "Force_mN",
            "Abs_Delta_S_mean_LSB",
            "Abs_Delta_S_std_LSB",
            "SNR",
            "Detectable_VppNoise",
            "Significant",
        ]
    ].to_string(index=False)
)

detectable_forces = df_results[df_results["Detectable_VppNoise"]]["Force_N"].values
LOD_vpp = detectable_forces.min() if len(detectable_forces) > 0 else np.nan

significant_forces = df_results[df_results["Significant"]]["Force_N"].values
LOD_stat = significant_forces.min() if len(significant_forces) > 0 else np.nan

max_force = float(df_results["Force_N"].max())
max_force_delta = float(df_results.loc[df_results["Force_N"] == max_force, "Abs_Delta_S_mean_LSB"].iloc[0])
sensitivity = max_force_delta / max_force if max_force > 0 else np.nan
LOD_theoretical = VPP_NOISE_THRESHOLD_LSB / sensitivity if sensitivity > 0 else np.nan

print("=" * 60)
print("【检测下限（LOD）确定】")
print("=" * 60)
if not np.isnan(LOD_vpp):
    print(f"基于Vpp噪声阈值的检测下限: {LOD_vpp*1000:.2f} mN ({LOD_vpp:.4f} N)")
else:
    print("基于Vpp噪声阈值: 所有测试力值均未达到检测阈值")

if not np.isnan(LOD_stat):
    print(f"基于统计显著性的检测下限: {LOD_stat*1000:.2f} mN ({LOD_stat:.4f} N)")
else:
    print("基于统计显著性: 所有测试力值均未达到显著性阈值")

if not np.isnan(LOD_theoretical):
    print(f"理论检测下限 (Vpp/灵敏度): {LOD_theoretical*1000:.2f} mN ({LOD_theoretical:.4f} N)")
else:
    print("理论检测下限: 无法计算")

print(f"灵敏度估计: {sensitivity:.2f} LSB/N")

# ── Fig 1: 2-panel detection limit ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (a) Zero-load noise histogram
ax = axes[0]
zero_data = zero_data_centered
ax.hist(
    zero_data,
    bins=bins_z,
    density=True,
    color=COLORS["zero"],
    alpha=0.65,
    edgecolor="none",
    label="Zero-load noise",
)
x_g = np.linspace(zero_data.min(), zero_data.max(), 200)
ax.plot(x_g, norm.pdf(x_g, 0, noise_std), color=COLORS["red"], linewidth=1.5, label="Gaussian fit")
ax.set_xlabel("ADC - Mean (LSB)")
ax.set_ylabel("Probability Density")
ax.text(
    0.97,
    0.95,
    f"σ = {noise_std:.3f} LSB",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
)
ax.legend(frameon=False, fontsize=8)
add_subplot_label(ax, "(a)")

# (b) Signal change vs force, 10-cycle mean +/- 1 sigma
ax = axes[1]
forces_mN = df_results["Force_mN"].to_numpy()
delta_mean = df_results["Abs_Delta_S_mean_LSB"].to_numpy()
delta_std = df_results["Abs_Delta_S_std_LSB"].to_numpy()
bar_labels = [f"{f:.0f}" for f in forces_mN]
ax.bar(
    bar_labels,
    delta_mean,
    yerr=delta_std,
    capsize=4,
    color=COLORS["blue"],
    edgecolor="none",
    alpha=0.86,
    error_kw=dict(ecolor=COLORS["gray"], elinewidth=1.0, capthick=1.0),
    label=r"10-cycle mean $\pm$ 1$\sigma$",
)
ax.axhline(
    VPP_NOISE_THRESHOLD_LSB,
    color=COLORS["red"],
    linewidth=1.2,
    linestyle="--",
    label=r"$V_{\mathrm{pp,noise}} = 7$ LSB",
)
ax.set_xlabel("Force (mN)")
ax.set_ylabel(r"$|\Delta S|$ (LSB)")
ax.text(
    0.97,
    0.80,
    f"LOD = {LOD_vpp*1000:.0f} mN\nSensitivity = {abs(sensitivity):.0f} LSB/N",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
)
ax.legend(frameon=False, fontsize=8)
add_subplot_label(ax, "(b)")

plt.tight_layout()
plt.close(fig)

# ── Individual sub-figures ────────────────────────────────────────────────────
for i, stem in enumerate(["05a_noise_hist", "05b_signal_change"]):
    fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
    if i == 0:
        ax_s.hist(
            zero_data,
            bins=bins_z,
            density=True,
            color=COLORS["zero"],
            alpha=0.65,
            edgecolor="none",
            label="Zero-load noise",
        )
        ax_s.plot(x_g, norm.pdf(x_g, 0, noise_std), color=COLORS["red"], linewidth=1.5,
                  label="Gaussian fit")
        ax_s.set_xlabel("ADC - Mean (LSB)")
        ax_s.set_ylabel("Probability Density")
        ax_s.text(
            0.97,
            0.95,
            f"σ = {noise_std:.3f} LSB",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )
        ax_s.legend(frameon=False, fontsize=8)
    else:
        ax_s.bar(
            bar_labels,
            delta_mean,
            yerr=delta_std,
            capsize=4,
            color=COLORS["blue"],
            edgecolor="none",
            alpha=0.86,
            error_kw=dict(ecolor=COLORS["gray"], elinewidth=1.0, capthick=1.0),
            label=r"10-cycle mean $\pm$ 1$\sigma$",
        )
        ax_s.axhline(
            VPP_NOISE_THRESHOLD_LSB,
            color=COLORS["red"],
            linewidth=1.2,
            linestyle="--",
            label=r"$V_{\mathrm{pp,noise}} = 7$ LSB",
        )
        ax_s.set_xlabel("Force (mN)")
        ax_s.set_ylabel(r"$|\Delta S|$ (LSB)")
        ax_s.text(
            0.97,
            0.80,
            f"LOD = {LOD_vpp*1000:.0f} mN\nSensitivity = {abs(sensitivity):.0f} LSB/N",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )
        ax_s.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    p = save_figure(fig_s, SAVE_DIR, stem)
    print(f"Saved: {p}")
    plt.close(fig_s)

force_resolution_1sigma = noise_std / sensitivity if sensitivity > 0 else np.nan
force_resolution_vpp = VPP_NOISE_THRESHOLD_LSB / sensitivity if sensitivity > 0 else np.nan

print("=" * 60)
print("【力分辨率】")
print("=" * 60)
if not np.isnan(force_resolution_1sigma):
    print(f"1σ 力分辨率: {force_resolution_1sigma*1000:.3f} mN")
    print(f"Vpp 力分辨率: {force_resolution_vpp*1000:.3f} mN")
else:
    print("无法计算力分辨率（灵敏度为零）")

df_cycles.to_csv(os.path.join(CSV_DIR, "05_detection_limit_cycle_data.csv"), index=False)
df_results.to_csv(os.path.join(CSV_DIR, "05_detection_limit_data.csv"), index=False)

lod_params = {
    "Noise_Std_LSB": noise_std,
    "Noise_3sigma_LSB": noise_3sigma,
    "Noise_Vpp_threshold_LSB": VPP_NOISE_THRESHOLD_LSB,
    "Sensitivity_LSB_per_N": sensitivity,
    "LOD_Vpp_N": LOD_vpp if not np.isnan(LOD_vpp) else None,
    "LOD_Vpp_mN": LOD_vpp * 1000 if not np.isnan(LOD_vpp) else None,
    "LOD_theoretical_N": LOD_theoretical if not np.isnan(LOD_theoretical) else None,
    "Resolution_1sigma_mN": force_resolution_1sigma * 1000 if not np.isnan(force_resolution_1sigma) else None,
    "Resolution_Vpp_mN": force_resolution_vpp * 1000 if not np.isnan(force_resolution_vpp) else None,
    "N_cycles": int(df_cycles["Cycle"].nunique()),
}
pd.DataFrame([lod_params]).to_csv(os.path.join(CSV_DIR, "05_detection_limit_params.csv"), index=False)

lod_vpp_mN = LOD_vpp * 1000 if not np.isnan(LOD_vpp) else 0
lod_theo_mN = LOD_theoretical * 1000 if not np.isnan(LOD_theoretical) else 0
res_1sigma_mN = force_resolution_1sigma * 1000 if not np.isnan(force_resolution_1sigma) else 0
res_vpp_mN = force_resolution_vpp * 1000 if not np.isnan(force_resolution_vpp) else 0

print("=" * 60)
print("【实验05 结果汇总】")
print("=" * 60)
print(f"""
┌─────────────────────────────────────────────────────────┐
│  检测下限测试结果                                       │
├─────────────────────────────────────────────────────────┤
│  循环次数:           {df_cycles['Cycle'].nunique():>10}                        │
│  测试力值点数:       {len(df_results):>10}                        │
│  噪声标准差 σ:       {noise_std:>10.3f} LSB                    │
│  Vpp噪声阈值:        {VPP_NOISE_THRESHOLD_LSB:>10.3f} LSB                    │
│  灵敏度:             {sensitivity:>10.2f} LSB/N                  │
├─────────────────────────────────────────────────────────┤
│  检测下限 (Vpp):     {lod_vpp_mN:>10.2f} mN                     │
│  理论检测下限:       {lod_theo_mN:>10.2f} mN                     │
│  1σ 力分辨率:        {res_1sigma_mN:>10.3f} mN                    │
│  Vpp 力分辨率:       {res_vpp_mN:>10.3f} mN                    │
└─────────────────────────────────────────────────────────┘
""")

print("检测下限汇总数据已保存至: 05_detection_limit_data.csv")
print("检测下限逐cycle数据已保存至: 05_detection_limit_cycle_data.csv")
print("检测下限参数已保存至: 05_detection_limit_params.csv")
