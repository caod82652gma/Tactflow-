"""
Batch tactile slip analysis for VET6USB 6x6 AD1 CSV exports.

conda activate vet6usb_pyqt
python Vet6USB_curve_draw/B_experiment_plots/08_tactile_slip_pipeline.py

Pipeline:
  1. load each V*/Tactile_All_AD1_*.csv as T x 6 x 6
  2. baseline-correct, optionally spatially Gaussian-blur each frame
  3. detect the slip window from the whole-array envelope
  4. extract COP, activation timing, velocity estimate, and STFT features
  5. process all trials, select the lowest-variance subset per speed
  6. save summary CSV files and publication-style figures
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import stft


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
sys.path.insert(0, str(PLOT_ROOT))

from plot_style import COLORS, add_subplot_label, apply_measurement_style, save_figure  # noqa: E402


# ============================================================
# CONFIG
# ============================================================
FS = 30.0
PITCH_MM = 10.0
N_ROWS, N_COLS = 6, 6
BASELINE_N = 40
FIXED_ACTIVITY_THRESHOLD_LSB = 35.0
MIN_EVENT_LEN = 5
TAIL_GUARD = 3

STFT_WIN_FRAC = 0.25
STFT_HOP_FRAC = 0.10
STFT_NFFT_MIN = 64
SPECTRAL_MIN_FREQ_HZ = 1.0

NOMINAL_SPEEDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
SUPPLEMENTAL_DISPLAY_SPEEDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
DATA_ROOT = REPO_ROOT / "Workspace" / "B_experiments" / "B08_tactile_slip"
RESULT_DISPLAY_ROOT = PLOT_ROOT / "result_display" / "B_experiment"
OUTPUT_DIR = RESULT_DISPLAY_ROOT / "test8_tactile_slip"

APPLY_GAUSSIAN_BLUR = False
GAUSSIAN_SIGMA_CELLS = 0.65

SELECT_N_PER_SPEED = 5
SELECTION_FEATURE = "f_centroid"
REPRESENTATIVE_SPEED = 0.10
REPRESENTATIVE_STFT_SPEEDS = (0.01, 0.10, 1.00)
SUPP_S2_PLANE_DIRECTION_WEIGHT = 0.5
EPS = 1e-12


@dataclass
class TrialResult:
    path: Path
    speed_cmps: float
    v_nominal: float
    label: str
    X_bc: np.ndarray
    X_analysis: np.ndarray
    X_slip: np.ndarray
    t_start: int
    t_end: int
    slip_len: int
    envelope: np.ndarray
    frame_diff: np.ndarray
    cop_x: np.ndarray
    cop_y: np.ndarray
    t_peak_map: np.ndarray
    plane_coeff: tuple[float, float, float]
    direction_deg: float
    v_measured: float
    stft_f: np.ndarray
    stft_t: np.ndarray
    stft_power: np.ndarray
    f_peak: float
    f_centroid: float
    f_bandwidth: float
    selected: bool = False
    error_message: str = ""


def load_trial(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    drop_cols = [name for name in ("Index", "VGND") if name in df.columns]
    df = df.drop(columns=drop_cols)

    ordered_cols = [f"R{row}_C{col}" for row in range(N_ROWS) for col in range(N_COLS)]
    if all(col in df.columns for col in ordered_cols):
        values = df[ordered_cols].to_numpy(dtype=float)
    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < N_ROWS * N_COLS:
            raise ValueError(f"{csv_path} has only {numeric.shape[1]} numeric tactile columns")
        values = numeric.iloc[:, : N_ROWS * N_COLS].to_numpy(dtype=float)

    return values.reshape(values.shape[0], N_ROWS, N_COLS)


def baseline_correct(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline_count = min(BASELINE_N, X.shape[0])
    baseline = np.mean(X[:baseline_count], axis=0)
    baseline_std = np.std(X[:baseline_count], axis=0)
    return X - baseline, baseline_std


def maybe_blur_frames(X_bc: np.ndarray, apply_blur: bool, sigma_cells: float) -> np.ndarray:
    if not apply_blur or sigma_cells <= 0:
        return X_bc.copy()
    return gaussian_filter(X_bc, sigma=(0.0, sigma_cells, sigma_cells), mode="nearest")


def apply_activity_threshold(X: np.ndarray, threshold_lsb: float) -> np.ndarray:
    if threshold_lsb <= 0:
        return X.copy()
    return np.where(np.abs(X) >= threshold_lsb, X, 0.0)


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = best_end = current_start = 0
    in_run = False

    for idx, value in enumerate(mask):
        if value and not in_run:
            current_start = idx
            in_run = True
        elif not value and in_run:
            if idx - current_start > best_end - best_start:
                best_start, best_end = current_start, idx
            in_run = False

    if in_run and len(mask) - current_start > best_end - best_start:
        best_start, best_end = current_start, len(mask)

    return best_start, best_end


def detect_slip_window(X_analysis: np.ndarray) -> tuple[int, int, np.ndarray, float]:
    envelope = np.sum(np.abs(X_analysis), axis=(1, 2))
    threshold = float(FIXED_ACTIVITY_THRESHOLD_LSB)
    above = envelope > threshold
    t_start, t_end = longest_true_run(above)

    if (t_end - t_start) < MIN_EVENT_LEN:
        raise ValueError(
            f"no valid slip event detected: longest run={t_end - t_start}, threshold={threshold:.3f}"
        )

    t_start = max(0, t_start - TAIL_GUARD)
    t_end = min(X_analysis.shape[0], t_end + TAIL_GUARD)
    return t_start, t_end, envelope, threshold


def compute_frame_diff(X_analysis: np.ndarray) -> np.ndarray:
    dX = np.diff(X_analysis, axis=0)
    return np.sqrt(np.sum(dX * dX, axis=(1, 2)))


def compute_cop(X_slip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.abs(X_slip)
    total = np.sum(W, axis=(1, 2))
    j_grid, i_grid = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS))
    active = total > EPS
    x_c = np.full(X_slip.shape[0], np.nan, dtype=float)
    y_c = np.full(X_slip.shape[0], np.nan, dtype=float)
    x_c[active] = np.sum(W[active] * j_grid, axis=(1, 2)) / total[active]
    y_c[active] = np.sum(W[active] * i_grid, axis=(1, 2)) / total[active]
    return x_c, y_c


def compute_activation_map(X_analysis: np.ndarray, t_start: int, t_end: int) -> np.ndarray:
    window = np.abs(X_analysis[t_start:t_end])
    active = np.max(window, axis=0) > EPS
    t_peak_map = np.full((N_ROWS, N_COLS), np.nan, dtype=float)
    t_peak_map[active] = np.argmax(window[:, active], axis=0).astype(float)
    return t_peak_map


def fit_activation_plane(t_peak_map: np.ndarray) -> tuple[float, float, float]:
    j_grid, i_grid = np.meshgrid(np.arange(N_COLS, dtype=float), np.arange(N_ROWS, dtype=float))
    valid = np.isfinite(t_peak_map)
    if np.count_nonzero(valid) < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([j_grid[valid], i_grid[valid], np.ones(np.count_nonzero(valid))])
    coeff, *_ = np.linalg.lstsq(A, t_peak_map[valid], rcond=None)
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def estimate_slip_direction(t_peak_map: np.ndarray) -> tuple[float, float, tuple[float, float, float]]:
    a, b, c = fit_activation_plane(t_peak_map)
    grad_norm = math.hypot(a, b)
    if grad_norm < EPS:
        return float("nan"), float("nan"), (a, b, c)

    direction_deg = math.degrees(math.atan2(b, a))
    speed_cells_per_sample = 1.0 / grad_norm
    speed_mps = speed_cells_per_sample * PITCH_MM * 1e-3 * FS
    return direction_deg, speed_mps, (a, b, c)


def next_pow2(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def compute_stft(D_slip: np.ndarray, slip_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if D_slip.size < 2:
        raise ValueError("slip window is too short for frame-difference STFT")

    win_len = max(8, int(slip_len * STFT_WIN_FRAC))
    win_len = min(win_len, D_slip.size)
    hop_len = max(2, int(slip_len * STFT_HOP_FRAC))
    hop_len = min(hop_len, win_len)
    noverlap = max(0, win_len - hop_len)
    nfft = max(STFT_NFFT_MIN, next_pow2(win_len))

    f, t, Zxx = stft(
        D_slip,
        fs=FS,
        nperseg=win_len,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None,
        padded=False,
    )
    return f, t, np.abs(Zxx) ** 2


def extract_spectral_features(f: np.ndarray, power: np.ndarray) -> tuple[float, float, float]:
    freq_mask = f >= SPECTRAL_MIN_FREQ_HZ
    if not np.any(freq_mask):
        freq_mask = np.ones_like(f, dtype=bool)

    f_use = f[freq_mask]
    P_avg = np.mean(power[freq_mask], axis=1)
    total = float(np.sum(P_avg))
    if total <= EPS:
        return float("nan"), float("nan"), float("nan")

    f_peak = float(f_use[int(np.argmax(P_avg))])
    f_centroid = float(np.sum(f_use * P_avg) / total)
    f_bandwidth = float(np.sqrt(np.sum((f_use - f_centroid) ** 2 * P_avg) / total))
    return f_peak, f_centroid, f_bandwidth


def process_single_trial(
    csv_path: Path,
    nominal_speed: float,
    apply_blur: bool,
    gaussian_sigma: float,
) -> TrialResult:
    X = load_trial(csv_path)
    X_bc, _baseline_std = baseline_correct(X)
    X_filtered = maybe_blur_frames(X_bc, apply_blur=apply_blur, sigma_cells=gaussian_sigma)
    X_analysis = apply_activity_threshold(X_filtered, FIXED_ACTIVITY_THRESHOLD_LSB)

    t_start, t_end, envelope, _threshold = detect_slip_window(X_analysis)
    slip_len = t_end - t_start
    X_slip = X_analysis[t_start:t_end]

    D_full = compute_frame_diff(X_analysis)
    D_slip = D_full[t_start : max(t_start, t_end - 1)]
    cop_x, cop_y = compute_cop(X_slip)
    t_peak_map = compute_activation_map(X_analysis, t_start, t_end)
    direction_deg, v_measured, plane_coeff = estimate_slip_direction(t_peak_map)

    f, t_stft, power = compute_stft(D_slip, slip_len)
    f_peak, f_centroid, f_bandwidth = extract_spectral_features(f, power)

    speed_cmps = nominal_speed * 100.0
    return TrialResult(
        path=csv_path,
        speed_cmps=speed_cmps,
        v_nominal=nominal_speed,
        label=make_label(csv_path),
        X_bc=X_bc,
        X_analysis=X_analysis,
        X_slip=X_slip,
        t_start=t_start,
        t_end=t_end,
        slip_len=slip_len,
        envelope=envelope,
        frame_diff=D_full,
        cop_x=cop_x,
        cop_y=cop_y,
        t_peak_map=t_peak_map,
        plane_coeff=plane_coeff,
        direction_deg=direction_deg,
        v_measured=v_measured,
        stft_f=f,
        stft_t=t_stft,
        stft_power=power,
        f_peak=f_peak,
        f_centroid=f_centroid,
        f_bandwidth=f_bandwidth,
    )


def make_label(path: Path) -> str:
    suffix = path.stem.split("_")[-1]
    if len(suffix) == 6 and suffix.isdigit():
        return f"{suffix[:2]}:{suffix[2:4]}:{suffix[4:]}"
    return path.stem


def discover_trials(data_root: Path) -> dict[float, list[Path]]:
    trials: dict[float, list[Path]] = {}
    for folder in sorted(data_root.glob("V*"), key=lambda p: speed_from_folder(p.name)):
        speed_cmps = speed_from_folder(folder.name)
        if speed_cmps <= 0:
            continue
        paths = sorted(folder.glob("*.csv"))
        if paths:
            trials[speed_cmps / 100.0] = paths
    return trials


def speed_from_folder(name: str) -> float:
    if not name.startswith("V"):
        return -1.0
    try:
        return float(name[1:])
    except ValueError:
        return -1.0


def batch_process(
    trial_paths_dict: dict[float, list[Path]],
    apply_blur: bool,
    gaussian_sigma: float,
) -> tuple[dict[float, list[TrialResult]], list[dict[str, object]]]:
    results: dict[float, list[TrialResult]] = {}
    failures: list[dict[str, object]] = []

    for speed, paths in trial_paths_dict.items():
        speed_results: list[TrialResult] = []
        for path in paths:
            try:
                speed_results.append(
                    process_single_trial(
                        csv_path=path,
                        nominal_speed=speed,
                        apply_blur=apply_blur,
                        gaussian_sigma=gaussian_sigma,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"v_nominal": speed, "file": path.name, "error": str(exc)})
        results[speed] = speed_results

    return results, failures


def select_low_variance_trials(
    results: dict[float, list[TrialResult]],
    select_n: int,
    feature: str,
) -> dict[float, list[TrialResult]]:
    selected: dict[float, list[TrialResult]] = {}
    for speed, trials in results.items():
        for trial in trials:
            trial.selected = False

        valid = [trial for trial in trials if np.isfinite(getattr(trial, feature, float("nan")))]
        if not valid:
            selected[speed] = []
            continue

        k = min(max(1, select_n), len(valid))
        if k == len(valid):
            chosen = valid
        else:
            chosen = min(
                itertools.combinations(valid, k),
                key=lambda combo: float(np.var([getattr(trial, feature) for trial in combo], ddof=0)),
            )

        for trial in chosen:
            trial.selected = True
        selected[speed] = list(chosen)

    return selected


def representative_trial(trials: list[TrialResult], feature: str = SELECTION_FEATURE) -> TrialResult:
    if not trials:
        raise ValueError("no trials available for representative selection")
    values = np.asarray([getattr(trial, feature, np.nan) for trial in trials], dtype=float)
    if np.all(~np.isfinite(values)):
        return trials[0]
    center = float(np.nanmedian(values))
    idx = int(np.nanargmin(np.abs(values - center)))
    return trials[idx]


def cop_forward_score(result: TrialResult) -> tuple[float, float]:
    valid = cop_valid_mask(result)
    if np.count_nonzero(valid) < 2:
        return float("inf"), float("inf")

    x = result.cop_x[valid]
    y = result.cop_y[valid]
    edge_n = min(3, x.size)
    start_x = float(np.median(x[:edge_n]))
    start_y = float(np.median(y[:edge_n]))
    end_x = float(np.median(x[-edge_n:]))
    end_y = float(np.median(y[-edge_n:]))
    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.hypot(dx, dy)
    if distance < EPS:
        return float("inf"), float("inf")

    angle_abs_deg = abs(math.degrees(math.atan2(dy, dx)))
    backward_penalty = 180.0 if dx < 0 else 0.0
    plane_angle_abs_deg = abs(result.direction_deg) if np.isfinite(result.direction_deg) else 180.0
    direction_score = angle_abs_deg + backward_penalty + SUPP_S2_PLANE_DIRECTION_WEIGHT * plane_angle_abs_deg
    return direction_score, -distance


def representative_forward_cop_trial(trials: list[TrialResult]) -> TrialResult:
    if not trials:
        raise ValueError("no trials available for representative selection")
    return min(trials, key=cop_forward_score)


def nearest_speed(speed_values: list[float], target: float) -> float:
    if not speed_values:
        raise ValueError("no speed groups available")
    return min(speed_values, key=lambda value: abs(value - target))


def supplemental_display_speeds(selected: dict[float, list[TrialResult]]) -> list[float]:
    available = sorted(selected)
    speeds: list[float] = []
    for target in SUPPLEMENTAL_DISPLAY_SPEEDS:
        matches = [speed for speed in available if np.isclose(speed, target)]
        if matches:
            speeds.append(matches[0])
    return speeds


def format_speed_cmps(speed_mps: float) -> str:
    speed_cmps = speed_mps * 100.0
    if np.isclose(speed_cmps, round(speed_cmps)):
        return f"{int(round(speed_cmps))} cm/s"
    return f"{speed_cmps:g} cm/s"


def speed_folder_name(speed_mps: float) -> str:
    speed_cmps = speed_mps * 100.0
    if np.isclose(speed_cmps, round(speed_cmps)):
        return f"V{int(round(speed_cmps))}"
    return f"V{speed_cmps:g}"


def active_slip_frame_indices(result: TrialResult, n_frames: int = 5) -> np.ndarray:
    activity = np.sum(np.abs(result.X_slip), axis=(1, 2))
    active = np.flatnonzero(activity > EPS)
    source = active if active.size else np.arange(result.X_slip.shape[0])
    if source.size == 0:
        return np.array([], dtype=int)
    pick = np.linspace(0, source.size - 1, min(n_frames, source.size)).round().astype(int)
    return source[pick]


def cop_valid_mask(result: TrialResult) -> np.ndarray:
    return np.isfinite(result.cop_x) & np.isfinite(result.cop_y)


def build_summary_table(results: dict[float, list[TrialResult]]) -> pd.DataFrame:
    rows = []
    for speed in sorted(results):
        for trial in results[speed]:
            err = (trial.v_measured - trial.v_nominal) / trial.v_nominal * 100.0
            rows.append(
                {
                    "v_nominal_mps": trial.v_nominal,
                    "v_nominal_cmps": trial.speed_cmps,
                    "file": trial.path.name,
                    "selected": trial.selected,
                    "activity_threshold_lsb": FIXED_ACTIVITY_THRESHOLD_LSB,
                    "t_start": trial.t_start,
                    "t_end": trial.t_end,
                    "slip_len_samples": trial.slip_len,
                    "slip_len_ms": trial.slip_len / FS * 1000.0,
                    "v_measured_mps": trial.v_measured,
                    "velocity_error_pct": err,
                    "f_peak_hz": trial.f_peak,
                    "f_centroid_hz": trial.f_centroid,
                    "f_bandwidth_hz": trial.f_bandwidth,
                    "direction_deg": trial.direction_deg,
                    "plane_a_samples_per_col": trial.plane_coeff[0],
                    "plane_b_samples_per_row": trial.plane_coeff[1],
                    "plane_c_samples": trial.plane_coeff[2],
                }
            )
    return pd.DataFrame(rows)


def build_selected_stats(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()

    grouped = selected.groupby(["v_nominal_mps", "v_nominal_cmps"], as_index=False)
    stats = grouped.agg(
        n_selected=("file", "count"),
        v_measured_mean_mps=("v_measured_mps", "mean"),
        v_measured_std_mps=("v_measured_mps", "std"),
        velocity_error_mean_pct=("velocity_error_pct", "mean"),
        f_peak_mean_hz=("f_peak_hz", "mean"),
        f_centroid_mean_hz=("f_centroid_hz", "mean"),
        f_centroid_std_hz=("f_centroid_hz", "std"),
        f_bandwidth_mean_hz=("f_bandwidth_hz", "mean"),
        direction_mean_deg=("direction_deg", "mean"),
        slip_len_mean_ms=("slip_len_ms", "mean"),
    )
    return stats.sort_values("v_nominal_mps")


def sanity_check_velocity(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[summary["selected"]].copy()
    if selected.empty:
        return pd.DataFrame()

    rows = []
    for speed, group in selected.groupby("v_nominal_mps"):
        measured = float(group["v_measured_mps"].mean())
        err = (measured - speed) / speed * 100.0
        rows.append(
            {
                "v_nominal_mps": speed,
                "v_measured_mean_mps": measured,
                "error_pct": err,
                "flag_error_gt_30pct": abs(err) > 30.0,
            }
        )
    return pd.DataFrame(rows)


def plot_fig_x1_spatial(result: TrialResult, output_dir: Path) -> str:
    fig = plt.figure(figsize=(9.4, 7.2), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.0])

    framespec = outer[0].subgridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.06], wspace=0.08)
    frame_indices = active_slip_frame_indices(result, n_frames=5)
    phase_labels = np.linspace(0.0, 1.0, len(frame_indices)) if len(frame_indices) else []
    vmax = float(np.nanmax(np.abs(result.X_slip)))
    vmax = vmax if vmax > 0 else 1.0
    im = None
    for idx, (frame_idx, phase) in enumerate(zip(frame_indices, phase_labels)):
        ax = fig.add_subplot(framespec[0, idx])
        frame = result.X_slip[frame_idx]
        im = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        active_mask = np.abs(frame) >= FIXED_ACTIVITY_THRESHOLD_LSB
        if np.any(active_mask):
            ax.contour(active_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(f"{phase:.2f} T")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            add_subplot_label(ax, "(a)")

    cax = fig.add_subplot(framespec[0, 5])
    fig.colorbar(im, cax=cax, label="Delta signal (LSB)")

    midspec = outer[1].subgridspec(1, 2, width_ratios=[1, 1.25], wspace=0.24)
    ax_cop = fig.add_subplot(midspec[0, 0])
    ax_cop.set_title("COP trajectory")
    ax_cop.set_xlim(-0.5, N_COLS - 0.5)
    ax_cop.set_ylim(N_ROWS - 0.5, -0.5)
    ax_cop.set_aspect("equal")
    ax_cop.set_xticks(range(N_COLS))
    ax_cop.set_yticks(range(N_ROWS))
    ax_cop.grid(True, linewidth=0.7)
    valid_cop = cop_valid_mask(result)
    if np.any(valid_cop):
        time_color = np.linspace(0, 1, len(result.cop_x))[valid_cop]
        scatter = ax_cop.scatter(
            result.cop_x[valid_cop],
            result.cop_y[valid_cop],
            c=time_color,
            cmap="viridis",
            s=18,
            edgecolor="none",
        )
        ax_cop.plot(
            result.cop_x[valid_cop],
            result.cop_y[valid_cop],
            color=COLORS["gray"],
            linewidth=0.8,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax_cop, label="Normalized time", fraction=0.046, pad=0.04)
    else:
        ax_cop.text(0.5, 0.5, "No active COP", transform=ax_cop.transAxes, ha="center", va="center")
    add_subplot_label(ax_cop, "(b)")

    ax_series = fig.add_subplot(midspec[0, 1])
    t_full = np.arange(result.envelope.size) / FS
    d_t = np.arange(result.frame_diff.size) / FS
    env_norm = result.envelope / (np.max(result.envelope) + EPS)
    diff_norm = result.frame_diff / (np.max(result.frame_diff) + EPS)
    ax_series.plot(t_full, env_norm, color=COLORS["blue"], label="Envelope")
    ax_series.plot(d_t, diff_norm, color=COLORS["red"], label="Frame diff")
    ax_series.axvline(result.t_start / FS, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax_series.axvline(result.t_end / FS, color=COLORS["gray"], linestyle="--", linewidth=0.9)
    ax_series.set_title("Detected slip window")
    ax_series.set_xlabel("Time (s)")
    ax_series.set_ylabel("Normalized amplitude")
    ax_series.legend(frameon=False)
    add_subplot_label(ax_series, "(c)")

    ax_meta = fig.add_subplot(outer[2])
    ax_meta.axis("off")
    text = (
        f"Representative trial: {result.path.parent.name}/{result.path.name}\n"
        f"v nominal = {result.v_nominal:.3f} m/s, v measured = {result.v_measured:.3f} m/s, "
        f"direction = {result.direction_deg:.1f} deg, slip window = {result.slip_len / FS * 1000:.1f} ms\n"
        f"fixed activity threshold = {FIXED_ACTIVITY_THRESHOLD_LSB:.1f} LSB, "
        f"f_peak = {result.f_peak:.1f} Hz, f_centroid = {result.f_centroid:.1f} Hz, "
        f"f_bandwidth = {result.f_bandwidth:.1f} Hz"
    )
    ax_meta.text(0.0, 0.88, text, ha="left", va="top", transform=ax_meta.transAxes)

    path = save_figure(fig, os.fspath(output_dir), "08_fig_X1_spatial")
    plt.close(fig)
    return path


def plot_trial_all_trajectory(result: TrialResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = active_slip_frame_indices(result, n_frames=5)
    phase_labels = np.linspace(0.0, 1.0, len(frame_indices)) if len(frame_indices) else []
    vmax = float(np.nanmax(np.abs(result.X_slip)))
    vmax = vmax if vmax > 0 else 1.0

    fig, axes = plt.subplots(1, 6, figsize=(9.4, 1.9), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.06]})
    im = None
    for ax in axes[:5]:
        ax.axis("off")

    for ax, frame_idx, phase in zip(axes[:5], frame_indices, phase_labels):
        frame = result.X_slip[frame_idx]
        im = ax.imshow(frame, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper")
        active_mask = np.abs(frame) >= FIXED_ACTIVITY_THRESHOLD_LSB
        if np.any(active_mask):
            ax.contour(active_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        ax.set_title(f"{phase:.2f} T")
        ax.set_xticks([])
        ax.set_yticks([])

    if im is None:
        im = axes[0].imshow(np.zeros((N_ROWS, N_COLS)), cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    fig.colorbar(im, cax=axes[5], label="Delta signal (LSB)")
    fig.suptitle(f"{result.path.parent.name}/{result.path.name} | threshold >= {FIXED_ACTIVITY_THRESHOLD_LSB:g} LSB")
    fig.tight_layout()

    path = output_dir / f"{result.path.stem}_ALL_trajectory.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trial_cop_trajectory(result: TrialResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.1, 3.8))
    ax.set_title(f"COP trajectory | {result.path.parent.name}/{result.label}")
    ax.set_xlim(-0.5, N_COLS - 0.5)
    ax.set_ylim(N_ROWS - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.grid(True, linewidth=0.7)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    valid = cop_valid_mask(result)
    if np.any(valid):
        t_norm = np.linspace(0, 1, len(result.cop_x))[valid]
        scatter = ax.scatter(
            result.cop_x[valid],
            result.cop_y[valid],
            c=t_norm,
            cmap="viridis",
            s=26,
            edgecolor="none",
        )
        ax.plot(result.cop_x[valid], result.cop_y[valid], color=COLORS["gray"], linewidth=0.9, alpha=0.75)
        ax.plot(result.cop_x[valid][0], result.cop_y[valid][0], marker="o", color=COLORS["green"], markersize=6)
        ax.plot(result.cop_x[valid][-1], result.cop_y[valid][-1], marker="s", color=COLORS["red"], markersize=5)
        fig.colorbar(scatter, ax=ax, label="Normalized time", fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No active COP", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    path = output_dir / f"{result.path.stem}_COP_trajectoryy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_trial_sliding_trajectory_figures(results: dict[float, list[TrialResult]], output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    root = output_dir / "png" / "Sliding trajectory"
    for speed in sorted(results):
        speed_dir = root / speed_folder_name(speed)
        for trial in results[speed]:
            saved_paths.append(plot_trial_all_trajectory(trial, speed_dir))
            saved_paths.append(plot_trial_cop_trajectory(trial, speed_dir))
    return saved_paths


def plot_fig_x2_frequency(selected: dict[float, list[TrialResult]], stats: pd.DataFrame, output_dir: Path) -> str:
    fig = plt.figure(figsize=(8.2, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.15])

    available = sorted(speed for speed, trials in selected.items() if trials)
    for row, target_speed in enumerate(REPRESENTATIVE_STFT_SPEEDS):
        speed = nearest_speed(available, target_speed)
        trial = representative_trial(selected[speed])
        ax = fig.add_subplot(grid[row, 0])
        t_norm = trial.stft_t / max(EPS, trial.slip_len / FS)
        power_db = 10.0 * np.log10(trial.stft_power + EPS)
        mesh = ax.pcolormesh(t_norm, trial.stft_f, power_db, shading="auto", cmap="magma")
        ax.set_ylim(0, min(500.0, float(np.max(trial.stft_f))))
        ax.set_ylabel("f (Hz)")
        ax.set_title(f"STFT, {speed:.2f} m/s ({trial.label})")
        if row == 0:
            add_subplot_label(ax, "(a)")
        if row == 2:
            ax.set_xlabel("t / T_slip")
        fig.colorbar(mesh, ax=ax, label="Power (dB)", fraction=0.025, pad=0.02)

    ax_fit = fig.add_subplot(grid[3, 0])
    x = stats["v_nominal_mps"].to_numpy(dtype=float)
    y = stats["f_centroid_mean_hz"].to_numpy(dtype=float)
    yerr = stats["f_centroid_std_hz"].fillna(0.0).to_numpy(dtype=float)
    ax_fit.errorbar(x, y, yerr=yerr, fmt="o", color=COLORS["data"], capsize=3, label="Selected trials")

    r2_text = "R^2 = n/a"
    if len(x) >= 2 and np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
        coeff = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        yy = np.polyval(coeff, xx)
        ax_fit.plot(xx, yy, color=COLORS["fit"], label=f"Linear fit: {coeff[0]:.1f}x + {coeff[1]:.1f}")
        y_hat = np.polyval(coeff, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot > EPS:
            r2_text = f"R^2 = {1.0 - ss_res / ss_tot:.3f}"

    ax_fit.text(0.02, 0.92, r2_text, transform=ax_fit.transAxes, ha="left", va="top")
    ax_fit.set_xlabel("Nominal speed (m/s)")
    ax_fit.set_ylabel("Frequency centroid (Hz)")
    ax_fit.set_title("Frequency centroid vs nominal speed")
    ax_fit.legend(frameon=False)
    add_subplot_label(ax_fit, "(b)")

    path = save_figure(fig, os.fspath(output_dir), "08_fig_X2_frequency")
    plt.close(fig)
    return path


def plot_supp_s1_waveforms(selected: dict[float, list[TrialResult]], output_dir: Path) -> str:
    speeds = supplemental_display_speeds(selected)
    fig, axes = plt.subplots(1, len(speeds), figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    if len(speeds) == 1:
        axes = [axes]

    for ax, speed in zip(axes, speeds):
        trials = selected[speed]
        if not trials:
            ax.set_visible(False)
            continue
        rep = representative_trial(trials)
        for trial in trials:
            start = trial.t_start
            stop = max(start + 1, trial.t_end - 1)
            y = trial.frame_diff[start:stop]
            x = np.arange(y.size) / FS
            color = COLORS["red"] if trial is rep else COLORS["gray"]
            alpha = 0.95 if trial is rep else 0.35
            lw = 1.5 if trial is rep else 0.8
            ax.plot(x, y, color=color, alpha=alpha, linewidth=lw)
        ax.set_title(format_speed_cmps(speed))
        ax.set_xlabel("Time (s)")
        ax.margins(x=0.02, y=0.08)
    axes[0].set_ylabel("Frame diff (LSB)")

    path = save_figure(fig, os.fspath(output_dir), "08_supp_S1_waveforms")
    plt.close(fig)
    return path


def plot_supp_s2_activation_maps(selected: dict[float, list[TrialResult]], output_dir: Path) -> str:
    speeds = supplemental_display_speeds(selected)
    fig, axes = plt.subplots(1, len(speeds), figsize=(10.8, 3.55), constrained_layout=True)
    if len(speeds) == 1:
        axes = [axes]

    all_maps = [representative_forward_cop_trial(selected[speed]).t_peak_map for speed in speeds if selected[speed]]
    finite_maps = [m for m in all_maps if np.any(np.isfinite(m))]
    vmax = max(float(np.nanmax(m)) for m in finite_maps) if finite_maps else 1.0

    for ax, speed in zip(axes, speeds):
        trials = selected[speed]
        if not trials:
            ax.set_visible(False)
            continue
        trial = representative_forward_cop_trial(trials)
        im = ax.imshow(trial.t_peak_map, cmap="viridis", vmin=0.0, vmax=vmax, origin="upper")
        if np.count_nonzero(np.isfinite(trial.t_peak_map)) >= 3:
            ax.contour(trial.t_peak_map, colors="white", linewidths=0.5, alpha=0.75)
        theta = math.radians(trial.direction_deg) if np.isfinite(trial.direction_deg) else 0.0
        ax.arrow(
            (N_COLS - 1) / 2,
            (N_ROWS - 1) / 2,
            math.cos(theta) * 1.3,
            math.sin(theta) * 1.3,
            color="white",
            width=0.035,
            head_width=0.25,
            length_includes_head=True,
        )
        ax.set_title(format_speed_cmps(speed))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    path = save_figure(fig, os.fspath(output_dir), "08_supp_S2_activation_maps")
    plt.close(fig)
    return path


def plot_supp_s3_velocity_table(stats: pd.DataFrame, output_dir: Path) -> str:
    table_df = stats[
        [
            "v_nominal_mps",
            "n_selected",
            "v_measured_mean_mps",
            "v_measured_std_mps",
            "velocity_error_mean_pct",
            "f_centroid_mean_hz",
            "f_centroid_std_hz",
        ]
    ].copy()

    display_df = table_df.round(
        {
            "v_nominal_mps": 3,
            "v_measured_mean_mps": 3,
            "v_measured_std_mps": 3,
            "velocity_error_mean_pct": 1,
            "f_centroid_mean_hz": 1,
            "f_centroid_std_hz": 1,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 0.55 * max(3, len(display_df)) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.fillna("").astype(str).values,
        colLabels=[
            "v nom. (m/s)",
            "n",
            "v meas. mean",
            "v meas. std",
            "err. (%)",
            "f centroid mean",
            "f centroid std",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    ax.set_title("Selected-trial velocity and spectral summary", pad=12)

    path = save_figure(fig, os.fspath(output_dir), "08_supp_S3_velocity_table")
    plt.close(fig)
    return path


def write_failures(failures: list[dict[str, object]], output_dir: Path) -> Path | None:
    path = output_dir / "processing_failures.csv"
    if not failures:
        if path.exists():
            path.unlink()
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def reset_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    result_root = RESULT_DISPLAY_ROOT.resolve()

    try:
        output_dir.relative_to(result_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to clean output directory outside {result_root}: {output_dir}") from exc

    protected_paths = {result_root, PLOT_ROOT.resolve(), REPO_ROOT.resolve(), Path(output_dir.anchor).resolve()}
    if output_dir in protected_paths:
        raise ValueError(f"Refusing to clean protected directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process VET6USB tactile slip CSV trials.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--pitch-mm", type=float, default=PITCH_MM)
    parser.add_argument("--gaussian-blur", action="store_true", default=APPLY_GAUSSIAN_BLUR)
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA_CELLS)
    parser.add_argument("--select-n", type=int, default=SELECT_N_PER_SPEED)
    parser.add_argument("--selection-feature", default=SELECTION_FEATURE)
    parser.add_argument("--spectral-min-freq", type=float, default=SPECTRAL_MIN_FREQ_HZ)
    parser.add_argument("--activity-threshold", type=float, default=FIXED_ACTIVITY_THRESHOLD_LSB)
    return parser.parse_args()


def update_runtime_config(args: argparse.Namespace) -> None:
    global FS, PITCH_MM, SPECTRAL_MIN_FREQ_HZ, FIXED_ACTIVITY_THRESHOLD_LSB
    FS = float(args.fs)
    PITCH_MM = float(args.pitch_mm)
    SPECTRAL_MIN_FREQ_HZ = float(args.spectral_min_freq)
    FIXED_ACTIVITY_THRESHOLD_LSB = float(args.activity_threshold)


def print_velocity_check(check: pd.DataFrame) -> None:
    if check.empty:
        print("No selected trials for velocity sanity check.")
        return
    print("Velocity sanity check (selected trials)")
    print("nominal_mps, measured_mean_mps, error_pct, flag_error_gt_30pct")
    for row in check.itertuples(index=False):
        print(
            f"{row.v_nominal_mps:.3f}, "
            f"{row.v_measured_mean_mps:.3f}, "
            f"{row.error_pct:.1f}, "
            f"{bool(row.flag_error_gt_30pct)}"
        )


def main() -> None:
    args = parse_args()
    update_runtime_config(args)
    apply_measurement_style()
    reset_output_dir(args.output_dir)

    trial_paths = discover_trials(args.data_root)
    if not trial_paths:
        raise FileNotFoundError(f"No V*/CSV tactile trials found under {args.data_root}")

    results, failures = batch_process(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
    )
    selected = select_low_variance_trials(
        results=results,
        select_n=args.select_n,
        feature=args.selection_feature,
    )

    summary = build_summary_table(results)
    stats = build_selected_stats(summary)
    velocity_check = sanity_check_velocity(summary)

    summary_path = args.output_dir / "summary_all_trials.csv"
    stats_path = args.output_dir / "summary_selected_stats.csv"
    velocity_path = args.output_dir / "velocity_sanity_check.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    velocity_check.to_csv(velocity_path, index=False, encoding="utf-8-sig")
    failure_path = write_failures(failures, args.output_dir)

    selected_speeds = sorted(speed for speed, trials in selected.items() if trials)
    rep_speed = nearest_speed(selected_speeds, REPRESENTATIVE_SPEED)
    rep_trial = representative_trial(selected[rep_speed])

    figure_paths = [
        plot_fig_x1_spatial(rep_trial, args.output_dir),
        plot_fig_x2_frequency(selected, stats, args.output_dir),
        plot_supp_s1_waveforms(selected, args.output_dir),
        plot_supp_s2_activation_maps(selected, args.output_dir),
        plot_supp_s3_velocity_table(stats, args.output_dir),
    ]
    sliding_paths = save_trial_sliding_trajectory_figures(results, args.output_dir)

    print_velocity_check(velocity_check)
    print(f"Processed trials: {len(summary)}")
    print(f"Selected per speed: requested {args.select_n}, feature={args.selection_feature}")
    print(f"Gaussian blur: {args.gaussian_blur}, sigma={args.gaussian_sigma:g} cells")
    print(f"Fixed activity threshold: {FIXED_ACTIVITY_THRESHOLD_LSB:g} LSB")
    print(f"Saved summary: {summary_path}")
    print(f"Saved selected stats: {stats_path}")
    print(f"Saved velocity check: {velocity_path}")
    if failure_path:
        print(f"Saved failures: {failure_path}")
    for path in figure_paths:
        print(f"Saved figure: {path}")
    print(f"Saved sliding trajectory figures: {len(sliding_paths)}")
    print(f"Sliding trajectory root: {args.output_dir / 'png' / 'Sliding trajectory'}")


if __name__ == "__main__":
    main()
