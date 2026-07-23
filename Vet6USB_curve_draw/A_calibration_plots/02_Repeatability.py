#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import os, glob, re, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from plot_style import apply_measurement_style, COLORS, add_subplot_label, save_figure, SAVE_DIRS, clear_experiment_outputs

apply_measurement_style()
SAVE_DIR = SAVE_DIRS[2]
CSV_DIR = os.path.join(SAVE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
clear_experiment_outputs((SCRIPT_DIR, CSV_DIR), SAVE_DIR, ("02_", "02a_", "02b_", "02c_"))

FS = 200
V_REF = 5.0
LSB = V_REF / (2**15)
NOISE_STD = 1.16
REPEATABILITY_LEGEND_NOTE = "RSD < 1.4 %\nFS error < 0.3 % FS"


def add_repeatability_legend(ax, handles=None, labels=None, loc="upper right", fontsize=10):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    note_handle = Line2D([], [], linestyle="none", marker="", color="none")
    return ax.legend(
        list(handles) + [note_handle],
        list(labels) + [REPEATABILITY_LEGEND_NOTE],
        loc=loc,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.6,
    )

def load_sensor_data(file_path):
    """加载传感器CSV数据"""
    df = pd.read_csv(file_path)
    df.columns = ['Time_ms', 'RawValue', 'RawGround']
    df['Time_s'] = df['Time_ms'] / 1000
    return df

def extract_stable_mean(df, start_ratio=0.2, end_ratio=0.8):
    """提取稳定区间的均值"""
    n = len(df)
    start_idx = int(n * start_ratio)
    end_idx = int(n * end_ratio)
    stable_data = df['RawValue'].iloc[start_idx:end_idx]
    return np.mean(stable_data), np.std(stable_data)

def get_csv_files_in_folder(folder_path):
    """获取文件夹内所有CSV文件，按时间戳排序"""
    pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(pattern)
    # 按文件名排序（文件名包含时间戳）
    files.sort()
    return files

def parse_force_from_folder_name(folder_name):
    """从文件夹名称解析力值，如 '0N' -> 0.0, '7N' -> 7.0"""
    match = re.match(r'(\d+(?:\.\d+)?)N?', folder_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "Workspace",
    "A_calibration",
    "A02_repeatability",
)
BASE_PATH = _BASE

ENABLED_FORCES = ["0N","1N","2N","3N","4N","5N","6N","7N","8N","9N","10N"]

print(f"主路径: {BASE_PATH}")
print(f"选择的力值文件夹: {ENABLED_FORCES}")

# 自动扫描文件夹并加载数据
repeatability_config = {}

for force_folder in ENABLED_FORCES:
    folder_path = os.path.join(BASE_PATH, force_folder)
    
    if not os.path.isdir(folder_path):
        print(f"警告: 文件夹不存在 - {folder_path}")
        continue
    
    # 解析力值
    force_value = parse_force_from_folder_name(force_folder)
    if force_value is None:
        print(f"警告: 无法解析文件夹名称中的力值 - {force_folder}")
        continue
    
    # 获取该文件夹内的所有CSV文件
    csv_files = get_csv_files_in_folder(folder_path)
    
    if len(csv_files) == 0:
        print(f"警告: 文件夹内没有CSV文件 - {folder_path}")
        continue
    
    repeatability_config[force_value] = csv_files
    print(f"✓ {force_folder}: 找到 {len(csv_files)} 个CSV文件")

print(f"\n共加载 {len(repeatability_config)} 个力值点的数据")

# 显示详细的文件列表
print("\n详细文件列表:")
print("="*80)
for force, files in sorted(repeatability_config.items()):
    print(f"\n【{force} N】共 {len(files)} 个文件:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {os.path.basename(f)}")

results = []

for force, files in sorted(repeatability_config.items()):
    measurements = []
    stds = []
    
    for f in files:
        df = load_sensor_data(f)
        mean_val, std_val = extract_stable_mean(df)
        measurements.append(mean_val)
        stds.append(std_val)
    
    measurements = np.array(measurements)
    
    results.append({
        'Force_N': force,
        'Mean_LSB': np.mean(measurements),
        'Std_LSB': np.std(measurements),
        'Max_LSB': np.max(measurements),
        'Min_LSB': np.min(measurements),
        'Range_LSB': np.max(measurements) - np.min(measurements),
        'RSD_percent': (np.std(measurements) / np.mean(measurements) * 100) if np.mean(measurements) != 0 else 0,
        'N_trials': len(measurements),
        'Avg_noise_LSB': np.mean(stds)
    })

df_results = pd.DataFrame(results)
print("重复性测试结果:")
print(df_results.to_string(index=False))

full_scale_output = df_results[df_results['Force_N'] == df_results['Force_N'].max()]['Mean_LSB'].values[0]
zero_output = df_results[df_results['Force_N'] == 0]['Mean_LSB'].values[0] if 0 in df_results['Force_N'].values else df_results['Mean_LSB'].min()
span = full_scale_output - zero_output

# Compute repeatability %FS for all rows (0N gets NaN naturally)
df_results['Repeatability_FS_percent'] = df_results['Std_LSB'] / abs(span) * 100
df_results_nonzero = df_results[df_results['Force_N'] > 0].copy()

print("="*60)
print("【重复性误差分析】")
print("="*60)
print(f"零点输出:     {zero_output:.2f} LSB")
print(f"满量程输出:   {full_scale_output:.2f} LSB")
print(f"量程跨度:     {span:.2f} LSB")
print()

print("各测试点重复性:")
print("-"*60)
for _, row in df_results.iterrows():
    rep = row['Repeatability_FS_percent']
    rep_str = f"{rep:.4f}" if not pd.isna(rep) else "N/A"
    print(f"  {row['Force_N']:>5.1f} N: σ = {row['Std_LSB']:.3f} LSB, "
          f"重复性 = {rep_str} %FS, 测量次数 = {row['N_trials']:.0f}")

max_repeatability = df_results_nonzero['Repeatability_FS_percent'].max()
print()
print(f"整体重复性: {max_repeatability:.4f} %FS")

# ── Fig 1: 2-panel repeatability overview ────────────────────────────────────
# (a) mean response with dispersion bands  (b) repeatability %FS bar
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

forces_arr = df_results['Force_N'].values
means_arr  = df_results['Mean_LSB'].values
stds_arr   = df_results['Std_LSB'].values
mins_arr = df_results["Min_LSB"].values
maxs_arr = df_results["Max_LSB"].values

# (a) Mean response with error bars
ax = axes[0]
ax.errorbar(forces_arr, means_arr, yerr=stds_arr,
            fmt="o", color=COLORS["red"], ecolor=COLORS["red"],
            capsize=3, elinewidth=1.2, markersize=3.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label="Mean ± σ")
ax.plot(forces_arr, means_arr, "-", color=COLORS["red"], linewidth=1.5,
        alpha=0.7, label="Mean trend")
ax.set_xlabel("Force (N)"); ax.set_ylabel("ADC Output (LSB)")
handles, labels = ax.get_legend_handles_labels()
order = [labels.index("Mean trend"), labels.index("Mean ± σ")]
add_repeatability_legend(
    ax,
    [handles[idx] for idx in order],
    [labels[idx] for idx in order],
    loc="upper right",
    fontsize=10,
)
add_subplot_label(ax, "(a)")

# (b) Repeatability %FS bar chart (non-zero points only)
ax = axes[1]
df_nz = df_results[df_results['Force_N'] > 0].copy()
df_nz['Repeatability_FS_percent'] = df_nz['Std_LSB'] / abs(span) * 100
bar_colors = [COLORS["blue"] if v < 0.5 else COLORS["orange"]
              for v in df_nz['Repeatability_FS_percent']]
ax.bar([f"{int(f)}N" for f in df_nz['Force_N']], df_nz['Repeatability_FS_percent'],
       color=bar_colors, edgecolor="none", alpha=0.85)
ax.axhline(max_repeatability, color=COLORS["red"], linewidth=0.8, linestyle="--",
           label=f"Max = {max_repeatability:.3f} %FS")
ax.set_xlabel("Force (N)"); ax.set_ylabel("Repeatability (%FS)")
add_repeatability_legend(ax, loc="upper right", fontsize=8)
ax.tick_params(axis="x", labelsize=7)
add_subplot_label(ax, "(b)")

fig.suptitle("Repeatability Test — 10 Cycles", fontsize=11, y=1.01)
plt.tight_layout()
plt.close(fig)

# ── Individual sub-figures ────────────────────────────────────────────────────
stems = ["02a_overlay", "02b_repeatability_bar"]
for i, stem in enumerate(stems):
    fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
    if i == 0:
        ax_s.errorbar(forces_arr, means_arr, yerr=stds_arr,
                      fmt="o", color=COLORS["red"], ecolor=COLORS["red"],
                      capsize=3, elinewidth=1.2, markersize=3.5,
                      markeredgecolor="white", markeredgewidth=0.5,
                      label="Mean ± σ")
        ax_s.plot(forces_arr, means_arr, "-", color=COLORS["red"],
                  linewidth=1.5, alpha=0.7, label="Mean trend")
        ax_s.set_xlabel("Force (N)"); ax_s.set_ylabel("ADC Output (LSB)")
        handles, labels = ax_s.get_legend_handles_labels()
        order = [labels.index("Mean trend"), labels.index("Mean ± σ")]
        add_repeatability_legend(
            ax_s,
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper right",
            fontsize=10,
        )
    else:
        ax_s.bar([f"{int(f)}N" for f in df_nz['Force_N']], df_nz['Repeatability_FS_percent'],
                 color=bar_colors, edgecolor="none", alpha=0.85, label="Repeatability")
        ax_s.axhline(max_repeatability, color=COLORS["red"], linewidth=0.8, linestyle="--",
                     label=f"Max = {max_repeatability:.3f} %FS")
        ax_s.set_xlabel("Force (N)"); ax_s.set_ylabel("Repeatability (%FS)")
        ax_s.tick_params(axis="x", labelsize=7)
        add_repeatability_legend(ax_s, loc="upper right", fontsize=8)
    plt.tight_layout()
    p = save_figure(fig_s, SAVE_DIR, stem)
    print(f"Saved: {p}")
    plt.close(fig_s)

# Zero-point drift across 10 trials
if 0.0 in repeatability_config:
    zero_files = repeatability_config[0.0]
    zero_means = np.array([extract_stable_mean(load_sensor_data(f))[0] for f in zero_files])
    zero_drift = np.max(zero_means) - np.min(zero_means)
    zero_drift_percent = zero_drift / abs(span) * 100

    print(f"零点漂移: {zero_drift:.3f} LSB  ({zero_drift_percent:.4f} %FS)")

    fig_z, ax_z = plt.subplots(figsize=(6, 4))
    ax_z.plot(range(1, len(zero_means)+1), zero_means, "o-",
              color=COLORS["blue"], markersize=5, linewidth=1.2)
    ax_z.axhline(np.mean(zero_means), color=COLORS["red"], linewidth=0.8, linestyle="--",
                 label=f"Mean = {np.mean(zero_means):.2f} LSB")
    ax_z.fill_between(range(1, len(zero_means)+1),
                      np.mean(zero_means) - np.std(zero_means),
                      np.mean(zero_means) + np.std(zero_means),
                      color=COLORS["red"], alpha=0.15, label="±σ")
    ax_z.set_xlabel("Trial #"); ax_z.set_ylabel("Zero Point (LSB)")
    ax_z.legend(frameon=False)
    plt.tight_layout()
    plt.close(fig_z)
else:
    zero_drift_percent = 0
    print("注意: 没有0N的数据，跳过零点漂移分析")

# Per-force scatter distribution (compact grid)
n_forces = len(repeatability_config)
n_cols = min(4, n_forces)
n_rows = (n_forces + n_cols - 1) // n_cols

fig_d, axes_d = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3*n_rows))
axes_d = np.array(axes_d).reshape(n_rows, n_cols)

for idx, (force, files) in enumerate(sorted(repeatability_config.items())):
    row, col = idx // n_cols, idx % n_cols
    ax = axes_d[row, col]
    meas = np.array([extract_stable_mean(load_sensor_data(f))[0] for f in files])
    ax.scatter(range(1, len(meas)+1), meas, s=25, color=COLORS["blue"], alpha=0.7)
    ax.axhline(np.mean(meas), color=COLORS["red"], linewidth=0.8, linestyle="--")
    ax.fill_between(range(0, len(meas)+2),
                    np.mean(meas) - np.std(meas), np.mean(meas) + np.std(meas),
                    color=COLORS["red"], alpha=0.15)
    ax.set_title(f"{int(force)} N  σ={np.std(meas):.2f}", fontsize=9)
    ax.set_xlabel("Trial #", fontsize=8); ax.set_ylabel("LSB", fontsize=8)
    ax.tick_params(labelsize=7)

for idx in range(n_forces, n_rows * n_cols):
    axes_d[idx // n_cols, idx % n_cols].set_visible(False)

plt.tight_layout()
plt.close(fig_d)

# 保存结果
df_results.to_csv(os.path.join(CSV_DIR, '02_repeatability_results.csv'), index=False)

print("="*60)
print("【实验02 结果汇总】")
print("="*60)
print(f"""
┌─────────────────────────────────────────────────────────┐
│  重复性测试结果                                         │
├─────────────────────────────────────────────────────────┤
│  测试力值点数:     {len(repeatability_config):>10}                        │
│  量程跨度:         {span:>10.2f} LSB                    │
│  整体重复性:       {max_repeatability:>10.4f} %FS                  │
│  零点漂移:         {zero_drift_percent:>10.4f} %FS                  │
├─────────────────────────────────────────────────────────┤
│  评价标准 (参考):                                       │
│    优秀: < 0.1%FS                                       │
│    良好: 0.1% ~ 0.5%FS                                  │
│    一般: 0.5% ~ 1%FS                                    │
└─────────────────────────────────────────────────────────┘
""")

print("详细结果已保存至: 02_repeatability_results.csv")
