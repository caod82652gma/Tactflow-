#!/usr/bin/env python
# coding: utf-8

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import cauchy, laplace, norm, pearsonr, spearmanr, t as student_t

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

# Default input. You can also pass a file path on the command line:
# python 01_Noise_Analysis.py Workspace/A_calibration/A01_noise/Tactile_All_AD2_20260507_120050.csv
INPUT_FILE = os.path.join(
    "Workspace",
    "A_calibration",
    "A01_noise",
    "Tactile_All_AD2_20260507_120050.csv",
)

apply_measurement_style()
SAVE_DIR = SAVE_DIRS[1]
CSV_DIR = os.path.join(SAVE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
clear_experiment_outputs((SCRIPT_DIR, CSV_DIR), SAVE_DIR, ("01_", "01a_", "01b_"))

FS = 200
DT = 1 / FS
V_REF = 5.0
ADC_BITS = 16
LSB = V_REF / (2**15)

print(f"采样频率: {FS} Hz")
print(f"采样周期: {DT*1000:.1f} ms")
print(f"LSB电压: {LSB*1000:.4f} mV")


def resolve_input_path(path):
    if os.path.isabs(path):
        return path
    candidates = [
        os.path.join(os.getcwd(), path),
        os.path.join(SCRIPT_DIR, "..", "..", path),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(candidates[0])


def parse_channel_name(name):
    row_part, col_part = name.split("_")
    return int(row_part[1:]), int(col_part[1:])


def find_single_channel_columns(df):
    time_col = None
    raw_col = None
    ground_col = None

    for col in df.columns:
        lower = col.lower()
        if time_col is None and "time" in lower:
            time_col = col
        if raw_col is None and ("tactileraw" in lower or lower in {"rawvalue", "raw"}):
            raw_col = col
        if ground_col is None and ("groundref" in lower or lower in {"rawground", "ground"}):
            ground_col = col

    if raw_col is None and len(df.columns) >= 2:
        raw_col = df.columns[1]
    if ground_col is None and len(df.columns) >= 3:
        ground_col = df.columns[2]
    return time_col, raw_col, ground_col


def load_noise_data(file_path):
    df = pd.read_csv(file_path)
    array_cols = [
        col for col in df.columns
        if col.startswith("R") and "_C" in col
    ]

    if array_cols:
        channel_cols = sorted(array_cols, key=parse_channel_name)
        sample_index = (
            df["Index"].to_numpy(dtype=float)
            if "Index" in df.columns
            else np.arange(len(df), dtype=float)
        )
        time_s = (sample_index - sample_index[0]) * DT
        channel_values = df[channel_cols].astype(float)
        channel_baselines = channel_values.mean(axis=0)
        channel_residuals = channel_values - channel_baselines

        reference_baseline = np.nan
        reference_residual = None
        reference_name = ""
        if "VGND" in df.columns:
            reference_name = "VGND"
            vgnd = df["VGND"].astype(float)
            reference_baseline = float(vgnd.mean())
            reference_residual = (vgnd - reference_baseline).to_numpy()

        return {
            "mode": "array_baseline_removed",
            "title": "Noise Characterization — 6×6 Array Baseline Removed",
            "object_label": f"{len(channel_cols)}阵点合并",
            "reference_name": reference_name,
            "time_s": time_s,
            "channel_cols": channel_cols,
            "channel_values": channel_values,
            "channel_baselines": channel_baselines,
            "channel_residuals": channel_residuals,
            "reference_baseline": reference_baseline,
            "reference_residual": reference_residual,
        }

    time_col, raw_col, ground_col = find_single_channel_columns(df)
    if raw_col is None:
        raise ValueError("未找到可分析的单通道原始信号列或 R*_C* 阵列列")

    raw = df[raw_col].astype(float)
    raw_baseline = float(raw.mean())
    time_s = (
        df[time_col].to_numpy(dtype=float) / 1000
        if time_col is not None
        else np.arange(len(df), dtype=float) * DT
    )

    channel_residuals = pd.DataFrame({"Signal": raw - raw_baseline})
    channel_values = pd.DataFrame({"Signal": raw})
    channel_baselines = pd.Series({"Signal": raw_baseline})

    reference_baseline = np.nan
    reference_residual = None
    reference_name = ""
    if ground_col is not None:
        reference_name = ground_col
        ground = df[ground_col].astype(float)
        reference_baseline = float(ground.mean())
        reference_residual = (ground - reference_baseline).to_numpy()

    return {
        "mode": "single_channel_baseline_removed",
        "title": "Noise Characterization — Single Channel Baseline Removed",
        "object_label": "单通道信号",
        "reference_name": reference_name,
        "time_s": time_s,
        "channel_cols": ["Signal"],
        "channel_values": channel_values,
        "channel_baselines": channel_baselines,
        "channel_residuals": channel_residuals,
        "reference_baseline": reference_baseline,
        "reference_residual": reference_residual,
    }


def fit_noise_distribution(data, x_values, bin_edges):
    candidates = []
    hist_density, _ = np.histogram(data, bins=bin_edges, density=True)
    bin_weights = np.histogram(data, bins=bin_edges)[0]
    min_scale = 0.5

    def add_candidate(name, pdf_x, cdf_edges):
        bin_widths = np.diff(bin_edges)
        pdf_bins = np.diff(cdf_edges) / bin_widths
        weighted_sse = np.average((hist_density - pdf_bins) ** 2, weights=bin_weights)
        candidates.append((weighted_sse, name, pdf_x))

    fit_data = data
    if len(data) > 50000:
        rng = np.random.default_rng(20260507)
        fit_data = rng.choice(data, size=50000, replace=False)

    std_gauss = max(np.std(data), min_scale)
    add_candidate(
        "Gaussian",
        norm.pdf(x_values, 0.0, std_gauss),
        norm.cdf(bin_edges, 0.0, std_gauss),
    )

    _, scale_laplace = laplace.fit(fit_data, floc=0.0)
    scale_laplace = max(scale_laplace, min_scale)
    add_candidate(
        "Laplace",
        laplace.pdf(x_values, 0.0, scale_laplace),
        laplace.cdf(bin_edges, 0.0, scale_laplace),
    )

    df_t, _, scale_t = student_t.fit(fit_data, floc=0.0)
    scale_t = max(scale_t, min_scale)
    add_candidate(
        "Student-t",
        student_t.pdf(x_values, df_t, 0.0, scale_t),
        student_t.cdf(bin_edges, df_t, 0.0, scale_t),
    )

    _, scale_cauchy = cauchy.fit(fit_data, floc=0.0)
    scale_cauchy = max(scale_cauchy, min_scale)
    add_candidate(
        "Cauchy",
        cauchy.pdf(x_values, 0.0, scale_cauchy),
        cauchy.cdf(bin_edges, 0.0, scale_cauchy),
    )

    best_score, best_name, best_pdf = min(candidates, key=lambda item: item[0])
    return best_name, best_pdf, candidates


def compute_noise_metrics(residual):
    residual = np.asarray(residual, dtype=float)
    q1, q3 = np.percentile(residual, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    valid_mask = (residual >= lower_bound) & (residual <= upper_bound)
    valid = residual[valid_mask]

    mean_value = float(np.mean(valid))
    std_value = float(np.std(valid))
    pp_value = float(np.max(valid) - np.min(valid))
    sigma_6 = 6 * std_value
    enob_value = ADC_BITS - np.log2(std_value * 6.6) if std_value > 0 else np.nan
    full_scale = 2 ** (ADC_BITS - 1)
    snr_value = 20 * np.log10(full_scale / std_value) if std_value > 0 else np.inf

    return {
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "valid_mask": valid_mask,
        "valid_data": valid,
        "mean_LSB": mean_value,
        "std_LSB": std_value,
        "pp_LSB": pp_value,
        "sigma6_LSB": float(sigma_6),
        "ENOB_bits": float(enob_value),
        "SNR_dB": float(snr_value),
        "outlier_count": int(np.size(residual) - np.size(valid)),
        "valid_count": int(np.size(valid)),
    }


input_arg = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
FILE_NOISE = resolve_input_path(input_arg)
if not os.path.exists(FILE_NOISE):
    raise FileNotFoundError(f"输入文件不存在: {FILE_NOISE}")

data = load_noise_data(FILE_NOISE)
time_s = data["time_s"]
channel_cols = data["channel_cols"]
residual_df = data["channel_residuals"]
residual_matrix = residual_df.to_numpy()
noise_all = residual_matrix.ravel()

channel_stats = []
for col in channel_cols:
    residual = residual_df[col].to_numpy(dtype=float)
    metrics = compute_noise_metrics(residual)
    channel_stats.append(
        {
            "channel": col,
            "baseline_LSB": float(data["channel_baselines"][col]),
            "mean_after_baseline_LSB": metrics["mean_LSB"],
            "std_LSB": metrics["std_LSB"],
            "pp_LSB": metrics["pp_LSB"],
            "ENOB_bits": metrics["ENOB_bits"],
            "SNR_dB": metrics["SNR_dB"],
            "valid_count": metrics["valid_count"],
            "outlier_count": metrics["outlier_count"],
        }
    )

channel_stats_df = pd.DataFrame(channel_stats)
best_channel_row = channel_stats_df.sort_values(
    by=["std_LSB", "pp_LSB", "ENOB_bits"],
    ascending=[True, True, False],
).iloc[0]
best_channel = str(best_channel_row["channel"])
best_residual_full = residual_df[best_channel].to_numpy(dtype=float)
best_metrics = compute_noise_metrics(best_residual_full)

print(f"数据文件: {FILE_NOISE}")
print(f"数据格式: {data['mode']}")
print(f"时间采样点数: {len(time_s)}")
print(f"通道数量: {len(channel_cols)}")
print(f"最佳通道: {best_channel}")
print(f"最佳通道有效样本数: {best_metrics['valid_count']}")
print(f"采样时长: {time_s[-1] - time_s[0]:.2f} s")

print(f"异常值数量: {best_metrics['outlier_count']}")
print(f"有效总体噪声样本: {best_metrics['valid_count']}")

signal_mean = best_metrics["mean_LSB"]
signal_std = best_metrics["std_LSB"]
signal_pp = best_metrics["pp_LSB"]
signal_6sigma = best_metrics["sigma6_LSB"]

reference_residual = data["reference_residual"]
if reference_residual is not None:
    reference_mean = np.mean(reference_residual)
    reference_std = np.std(reference_residual)
    reference_pp = np.max(reference_residual) - np.min(reference_residual)
else:
    reference_mean = reference_std = reference_pp = np.nan

print("=" * 70)
print("【噪声统计结果】")
print("=" * 70)
print(f"{'对象':<18} {'均值(LSB)':<15} {'标准差σ(LSB)':<15} {'峰峰值(LSB)':<15}")
print("-" * 70)
print(f"{best_channel:<18} {signal_mean:<15.3f} {signal_std:<15.3f} {signal_pp:<15.3f}")
if reference_residual is not None:
    print(f"{data['reference_name']:<18} {reference_mean:<15.3f} {reference_std:<15.3f} {reference_pp:<15.3f}")

common_signal = residual_matrix.mean(axis=1)
selected_signal = best_residual_full
if reference_residual is not None and np.std(selected_signal) > 0 and np.std(reference_residual) > 0:
    pearson_corr, pearson_p = pearsonr(selected_signal, reference_residual)
    spearman_corr, spearman_p = spearmanr(selected_signal, reference_residual)
    correlation = signal.correlate(selected_signal, reference_residual, mode="full")
    lags = signal.correlation_lags(len(selected_signal), len(reference_residual), mode="full")
    correlation_normalized = correlation / (
        np.std(selected_signal) * np.std(reference_residual) * len(selected_signal)
    )
    max_corr_idx = np.argmax(np.abs(correlation_normalized))
    max_corr_lag = lags[max_corr_idx]
    max_corr_value = correlation_normalized[max_corr_idx]
    covariance = np.cov(selected_signal, reference_residual)[0, 1]
else:
    pearson_corr = pearson_p = spearman_corr = spearman_p = np.nan
    max_corr_lag = np.nan
    max_corr_value = np.nan
    covariance = np.nan

print("=" * 70)
print("【信号与参考通道相关性分析】")
print("=" * 70)
print(f"Pearson相关系数:   {pearson_corr:>10.6f} (p = {pearson_p:.2e})")
print(f"Spearman相关系数:  {spearman_corr:>10.6f} (p = {spearman_p:.2e})")
print(f"协方差:            {covariance:>10.4f} LSB²")
print(f"最大互相关:        {max_corr_value:>10.6f} (时延 = {max_corr_lag} 采样点)")
print()
print("相关性解释:")
if np.isfinite(pearson_corr) and abs(pearson_corr) > 0.7:
    print("  → 信号公共分量与参考通道存在强相关性")
elif np.isfinite(pearson_corr) and abs(pearson_corr) > 0.3:
    print("  → 信号公共分量与参考通道存在中等相关性")
elif np.isfinite(pearson_corr):
    print("  → 信号公共分量与参考通道相关性较弱")
else:
    print("  → 当前数据无法计算参考通道相关性")

ENOB_signal = best_metrics["ENOB_bits"]
SNR_signal_dB = best_metrics["SNR_dB"]

noise_centered = best_metrics["valid_data"]
hist_bins = np.arange(
    np.floor(noise_centered.min()) - 0.5,
    np.ceil(noise_centered.max()) + 1.5,
    1.0,
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

ax = axes[0]
counts, bins, _ = ax.hist(
    noise_centered,
    bins=hist_bins,
    density=True,
    rwidth=0.7,
    color=COLORS["blue"],
    alpha=0.65,
    edgecolor="none",
    label=f"σ = {signal_std:.3f} LSB",
)
x_fit = np.sort(np.unique(np.append(np.linspace(bins[0], bins[-1], 400), 0.0)))
best_fit_name, pdf_fit, fit_candidates = fit_noise_distribution(noise_centered, x_fit, bins)
print("Noise distribution fit weighted SSE:")
for fit_score, fit_name, _ in sorted(fit_candidates, key=lambda item: item[0]):
    print(f"  {fit_name:<10}: {fit_score:.6g}")
ax.plot(x_fit, pdf_fit, color=COLORS["red"], linewidth=1.5, label=f"{best_fit_name} fit")
ax.set_xlabel("Baseline-corrected ADC (LSB)")
ax.set_ylabel("Probability Density")
ax.legend(frameon=False, fontsize=8)
ax.text(
    0.97,
    0.95,
    f"ENOB = {ENOB_signal:.2f} bit\nσ = {signal_std:.3f} LSB\nSNR = {SNR_signal_dB:.1f} dB",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
)
add_subplot_label(ax, "(a)")

ax = axes[1]
n_show = len(time_s)
t_show = time_s[:n_show] - time_s[0]
ax.plot(
    t_show,
    best_residual_full[:n_show],
    color=COLORS["blue"],
    linewidth=0.8,
    alpha=0.85,
)
if residual_matrix.shape[1] > 1:
    ax.plot(t_show, common_signal[:n_show], color=COLORS["red"], linewidth=1.0, label="Mean")
ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
ax.fill_between(t_show, -signal_std, signal_std, color=COLORS["blue"], alpha=0.12,
                label=f"±σ = {signal_std:.2f} LSB")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Baseline-corrected ADC (LSB)")
ax.legend(frameon=False, fontsize=8)
add_subplot_label(ax, "(b)")

ax = axes[2]
nperseg = min(512, len(time_s))
f_welch, psd = signal.welch(best_residual_full, fs=FS, nperseg=nperseg)
psd_dB = 10 * np.log10(psd + 1e-30)
ax.plot(f_welch, psd_dB, color=COLORS["blue"], linewidth=1.0, label=f"{best_channel} PSD")
ax.axvline(FS / 2, color=COLORS["red"], linewidth=0.8, linestyle="--",
           label=f"Nyquist = {FS//2} Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD (dB/Hz)")
ax.legend(frameon=False, fontsize=8)
add_subplot_label(ax, "(c)")

fig.suptitle(data["title"], fontsize=11, y=1.01)
plt.tight_layout()
plt.close(fig)

for panel_idx, stem in enumerate(["01a_hist", "01b_time", "01b_time_no_mean"]):
    fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
    if panel_idx == 0:
        ax_s.hist(noise_centered, bins=hist_bins, density=True, rwidth=0.7,
                  color=COLORS["blue"], alpha=0.65, edgecolor="none")
        ax_s.plot(x_fit, pdf_fit, color=COLORS["red"], linewidth=1.5,
                  label=f"{best_fit_name} fit")
        ax_s.set_xlabel("Baseline-corrected ADC (LSB)")
        ax_s.set_ylabel("Probability Density")
        ax_s.text(
            0.97, 0.95,
            f"ENOB = {ENOB_signal:.2f} bit\nσ = {signal_std:.3f} LSB\nSNR = {SNR_signal_dB:.1f} dB",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
        )
        ax_s.legend(frameon=False, fontsize=8)
    else:
        ax_s.plot(
            t_show,
            best_residual_full[:n_show],
            color=COLORS["blue"],
            linewidth=0.8,
            alpha=0.85,
        )
        if panel_idx == 1 and residual_matrix.shape[1] > 1:
            ax_s.plot(t_show, common_signal[:n_show], color=COLORS["red"],
                      linewidth=1.0, label="Mean")
        ax_s.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
        ax_s.fill_between(t_show, -signal_std, signal_std, color=COLORS["blue"], alpha=0.12)
        ax_s.set_xlabel("Time (s)")
        ax_s.set_ylabel("Baseline-corrected ADC (LSB)")
        if panel_idx == 1 and residual_matrix.shape[1] > 1:
            ax_s.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    p = save_figure(fig_s, SAVE_DIR, stem)
    print(f"Saved: {p}")
    plt.close(fig_s)

channel_stats_df["is_best_channel"] = channel_stats_df["channel"] == best_channel
channel_stats_df.to_csv(os.path.join(CSV_DIR, "01_channel_noise_stats.csv"), index=False)

noise_params = {
    "analysis_mode": data["mode"],
    "data_file": os.path.basename(FILE_NOISE),
    "input_file": FILE_NOISE,
    "n_time_samples": len(time_s),
    "n_channels": len(channel_cols),
    "total_noise_samples": len(noise_all),
    "selected_channel": best_channel,
    "valid_noise_samples": best_metrics["valid_count"],
    "outlier_count": best_metrics["outlier_count"],
    "signal_mean_LSB": signal_mean,
    "signal_std_LSB": signal_std,
    "signal_pp_LSB": signal_pp,
    "signal_6sigma_LSB": signal_6sigma,
    "signal_ENOB_bits": ENOB_signal,
    "signal_SNR_dB": SNR_signal_dB,
    "reference_name": data["reference_name"],
    "reference_baseline_LSB": data["reference_baseline"],
    "reference_mean_LSB": reference_mean,
    "reference_std_LSB": reference_std,
    "reference_pp_LSB": reference_pp,
    "signal_reference_pearson_correlation": pearson_corr,
    "signal_reference_spearman_correlation": spearman_corr,
    "signal_reference_covariance_LSB2": covariance,
    "signal_reference_max_crosscorr": max_corr_value,
    "signal_reference_max_crosscorr_lag_samples": max_corr_lag,
    "best_fit_distribution": best_fit_name,
}

pd.DataFrame([noise_params]).to_csv(os.path.join(CSV_DIR, "01_noise_params.csv"), index=False)

print("=" * 70)
print("【ADC性能指标】")
print("=" * 70)
print(f"总体噪声标准差:        {signal_std:.4f} LSB")
print(f"总体噪声6σ:            {signal_6sigma:.4f} LSB")
print(f"有效位数ENOB:          {ENOB_signal:.2f} bits")
print(f"满量程SNR:             {SNR_signal_dB:.1f} dB")
print(f"最佳分布拟合:          {best_fit_name}")

print("=" * 70)
print("【实验01 结果汇总】")
print("=" * 70)
print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  噪声特性参数                                                    │
├──────────────────────────────────────────────────────────────────┤
│  数据文件: {os.path.basename(FILE_NOISE):<50} │
│  数据格式: {data["mode"]:<50} │
│  通道数量:         {len(channel_cols):>10}                                  │
│  最佳通道:         {best_channel:<10}                                  │
│  时间采样点数:     {len(time_s):>10}                                  │
│  有效噪声样本:     {best_metrics["valid_count"]:>10}                                  │
├──────────────────────────────────────────────────────────────────┤
│  最佳通道去基线后噪声:                                           │
│    均值:           {signal_mean:>10.3f} LSB                             │
│    标准差 σ:       {signal_std:>10.3f} LSB                             │
│    峰峰值:         {signal_pp:>10.3f} LSB                             │
│    6σ:             {signal_6sigma:>10.3f} LSB                             │
│    有效位数 ENOB:  {ENOB_signal:>10.2f} bits                            │
├──────────────────────────────────────────────────────────────────┤
│  参考通道: {data["reference_name"] or "None":<48} │
│    基线:           {data["reference_baseline"]:>10.3f} LSB                             │
│    标准差 σ:       {reference_std:>10.3f} LSB                             │
│    Pearson相关:    {pearson_corr:>10.4f}                                  │
└──────────────────────────────────────────────────────────────────┘
""")

print("噪声参数已保存至: 01_noise_params.csv")
print("各通道噪声统计已保存至: 01_channel_noise_stats.csv")
