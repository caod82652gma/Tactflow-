"""
Batch tactile force-rise analysis for VET6USB 6x6 CSV exports.

conda activate vet6usb_pyqt
python Vet6USB_curve_draw/B_experiment_plots/09a_Tactile_force_broken_curve.py --mode force-rise

Pipeline:
  1. load Workspace/B_experiments/B09_tactile_force/1cm through 6cm CSV trials as T x 6 x 6
  2. baseline-correct each taxel, optionally spatially Gaussian-blur each frame
  3. build the whole-array tactile envelope and extract only the short rising segment
  4. select up to 10 low-variance rising trials per distance folder
  5. plot one common-time mean curve per force level with a standard-deviation band
  6. find the longest trial in each distance group and draw the final 150 samples
     as the 600 s long-hold segment
  7. save response/creep statistics and publication-style figures
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
PITCH_MM = 5.0
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
DATA_ROOT = REPO_ROOT / "Workspace" / "B_experiments" / "B08_tactile_slip"
FORCE_DATA_ROOT = REPO_ROOT / "Workspace" / "B_experiments" / "B09_tactile_force"
RESULT_DISPLAY_ROOT = PLOT_ROOT / "result_display" / "B_experiment"
OUTPUT_DIR = RESULT_DISPLAY_ROOT / "test8_tactile_slip"
FORCE_OUTPUT_DIR = RESULT_DISPLAY_ROOT / "test9a_tactile_force"

APPLY_GAUSSIAN_BLUR = False
GAUSSIAN_SIGMA_CELLS = 0.65

SELECT_N_PER_SPEED = 5
SELECT_N_PER_DISTANCE = 10
SELECTION_FEATURE = "f_centroid"
REPRESENTATIVE_SPEED = 0.10
REPRESENTATIVE_STFT_SPEEDS = (0.01, 0.10, 1.00)
EPS = 1e-12

FORCE_DISTANCE_GROUPS = ("1cm", "2cm", "3cm", "4cm", "5cm", "6cm")
FORCE_SELECTION_GRID_POINTS = 200
FORCE_MAX_EXACT_SELECTION_COMBOS = 100000
FORCE_RISE_SMOOTH_WINDOW = 31
FORCE_ONSET_FRACTION_OF_PEAK = 0.05
FORCE_RISE_PLOT_DURATION_S = 6.0
FORCE_PLATEAU_FRACTION_OF_PEAK = 0.95
FORCE_PLATEAU_SLOPE_WINDOW_S = 2.0
FORCE_LONG_TAIL_POINTS = 150
FORCE_LONG_TAIL_START_S = 600.0
FORCE_GRAMS_TO_NEWTON = 9.80665e-3
FORCE_GROUP_COLORS = {
    "1cm": COLORS["blue"],
    "2cm": COLORS["green"],
    "3cm": COLORS["orange"],
    "4cm": COLORS["red"],
    "5cm": COLORS["purple"],
    "6cm": COLORS["teal"],
}
FORCE_RISE_DIAGNOSTIC_GROUPS = ("1cm", "3cm", "5cm")
FORCE_RISE_MANUAL_EXCLUDE_STEMS = {
    "1cm": {
        "Tactile_All_AD2_20260503_232205",
        "Tactile_All_AD2_20260503_232338",
        "Tactile_All_AD2_20260503_232654",
    },
    "3cm": {
        "Tactile_All_AD2_20260503_234347",
    },
}


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


@dataclass
class ForceRiseTrial:
    path: Path
    distance_label: str
    label: str
    envelope: np.ndarray
    analysis_envelope: np.ndarray
    t_start: int
    peak_idx: int
    rise_x: np.ndarray
    rise_y: np.ndarray
    peak_envelope: float
    baseline_level: float
    rise_10_90_ms: float
    plateau_entry_ms: float
    plateau_slope_lsb_s: float
    plateau_std_lsb: float
    plateau_value_lsb: float
    quality_score: float = float("nan")
    selected: bool = False
    manual_excluded: bool = False


@dataclass
class ForceLongTailTrial:
    path: Path
    distance_label: str
    label: str
    envelope: np.ndarray
    tail_x_s: np.ndarray
    tail_y: np.ndarray
    tail_start_idx: int
    peak_idx: int


@dataclass(frozen=True)
class ForceLevel:
    mean_n: float
    std_n: float
    n: int


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


def load_force_levels(force_sensor_path: Path) -> dict[str, ForceLevel]:
    if not force_sensor_path.exists():
        return {}

    levels: dict[str, ForceLevel] = {}
    current_label: str | None = None
    for raw_line in force_sensor_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith("cm"):
            current_label = line
            continue
        if current_label is None:
            continue
        values_g = np.fromstring(line, sep=" ", dtype=float)
        if values_g.size == 0:
            continue
        values_n = values_g * FORCE_GRAMS_TO_NEWTON
        levels[current_label] = ForceLevel(
            mean_n=float(np.mean(values_n)),
            std_n=float(np.std(values_n, ddof=1)) if values_n.size > 1 else 0.0,
            n=int(values_n.size),
        )
        current_label = None
    return levels


def force_legend_label(distance_label: str, force_levels: dict[str, ForceLevel]) -> str:
    level = force_levels.get(distance_label)
    cm = distance_label.replace("cm", " cm")
    if level is None:
        return cm
    return f"{level.mean_n:.2f} N +/- {level.std_n:.2f} N ({cm})"


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


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 2:
        return values.copy()
    window = min(int(window), values.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def force_level_crossing_time_ms(
    values: np.ndarray,
    start_idx: int,
    stop_idx: int,
    target: float,
) -> float:
    segment = values[start_idx : stop_idx + 1]
    if segment.size < 2:
        return float("nan")
    hits = np.flatnonzero(segment >= target)
    if hits.size == 0:
        return float("nan")
    hit = int(hits[0])
    if hit == 0:
        return 0.0
    y0 = float(segment[hit - 1])
    y1 = float(segment[hit])
    frac = 0.0 if abs(y1 - y0) < EPS else (target - y0) / (y1 - y0)
    return float((hit - 1 + np.clip(frac, 0.0, 1.0)) / FS * 1000.0)


def force_level_crossing_times_ms(
    smooth_envelope: np.ndarray,
    t_start: int,
    peak_idx: int,
    baseline_level: float,
    amplitude: float,
) -> tuple[float, float]:
    if amplitude <= EPS:
        return float("nan"), float("nan")
    t10 = force_level_crossing_time_ms(
        smooth_envelope,
        start_idx=t_start,
        stop_idx=peak_idx,
        target=baseline_level + 0.10 * amplitude,
    )
    t90 = force_level_crossing_time_ms(
        smooth_envelope,
        start_idx=t_start,
        stop_idx=peak_idx,
        target=baseline_level + 0.90 * amplitude,
    )
    return t10, t90


def force_plateau_entry_ms(
    smooth_envelope: np.ndarray,
    t_start: int,
    peak_idx: int,
    baseline_level: float,
    amplitude: float,
) -> float:
    if amplitude <= EPS:
        return float("nan")
    target = baseline_level + FORCE_PLATEAU_FRACTION_OF_PEAK * amplitude
    return force_level_crossing_time_ms(smooth_envelope, t_start, peak_idx, target)


def force_plateau_metrics(
    smooth_envelope: np.ndarray,
    start_idx: int,
    duration_s: float,
) -> tuple[float, float, float]:
    window_n = max(2, int(round(duration_s * FS)))
    stop_idx = min(len(smooth_envelope), start_idx + window_n)
    segment = smooth_envelope[start_idx:stop_idx]
    if segment.size < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.arange(segment.size, dtype=float) / FS
    slope = float(np.polyfit(x, segment, 1)[0])
    std = float(np.std(segment, ddof=0))
    mean_value = float(np.mean(segment))
    return slope, std, mean_value


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


def discover_force_rise_trials(
    data_root: Path,
    groups: tuple[str, ...] | list[str],
) -> dict[str, list[Path]]:
    trials: dict[str, list[Path]] = {}
    for group in groups:
        folder = data_root / group
        if folder.exists():
            trials[group] = sorted(path for path in folder.glob("*.csv") if path.is_file())
        else:
            trials[group] = []
    return trials


def iter_grouped_force_csvs(
    trial_paths_dict: dict[str, list[Path]],
) -> list[tuple[str, Path]]:
    return [(distance_label, path) for distance_label, paths in trial_paths_dict.items() for path in paths]


def process_force_rise_trial(
    csv_path: Path,
    distance_label: str,
    apply_blur: bool,
    gaussian_sigma: float,
) -> ForceRiseTrial:
    X = load_trial(csv_path)
    X_bc, _baseline_std = baseline_correct(X)
    X_filtered = maybe_blur_frames(X_bc, apply_blur=apply_blur, sigma_cells=gaussian_sigma)
    X_analysis = apply_activity_threshold(X_filtered, FIXED_ACTIVITY_THRESHOLD_LSB)

    raw_envelope = np.sum(np.abs(X_filtered), axis=(1, 2))
    smooth_envelope = smooth_1d(raw_envelope, FORCE_RISE_SMOOTH_WINDOW)
    analysis_envelope = np.sum(np.abs(X_analysis), axis=(1, 2))

    baseline_count = min(BASELINE_N, smooth_envelope.size)
    baseline_level = float(np.mean(smooth_envelope[:baseline_count]))
    baseline_noise = float(np.std(smooth_envelope[:baseline_count]))
    peak_idx = int(np.argmax(smooth_envelope))
    peak_level = float(smooth_envelope[peak_idx])
    if peak_level <= baseline_level:
        raise ValueError("no rising peak above baseline")

    onset_threshold = baseline_level + max(
        5.0 * baseline_noise,
        FORCE_ONSET_FRACTION_OF_PEAK * (peak_level - baseline_level),
        FIXED_ACTIVITY_THRESHOLD_LSB,
    )
    before_peak = smooth_envelope[: peak_idx + 1]
    hits = np.flatnonzero(before_peak >= onset_threshold)
    if hits.size == 0:
        raise ValueError("no rising onset detected before peak")

    t_start = int(hits[0])
    stop = min(len(smooth_envelope), peak_idx + 1)
    if stop - t_start < 2:
        raise ValueError("rising segment is too short")

    rise_window = max(stop, min(len(smooth_envelope), t_start + int(round(FORCE_RISE_PLOT_DURATION_S * FS)) + 1))
    rise_y = smooth_envelope[t_start:rise_window]
    rise_x = np.arange(rise_y.size, dtype=float) / FS

    amplitude = peak_level - baseline_level
    rise_10_ms, rise_90_ms = force_level_crossing_times_ms(
        smooth_envelope=smooth_envelope,
        t_start=t_start,
        peak_idx=peak_idx,
        baseline_level=baseline_level,
        amplitude=amplitude,
    )
    plateau_entry_ms = force_plateau_entry_ms(
        smooth_envelope=smooth_envelope,
        t_start=t_start,
        peak_idx=peak_idx,
        baseline_level=baseline_level,
        amplitude=amplitude,
    )
    plateau_slope, plateau_std, plateau_value = force_plateau_metrics(
        smooth_envelope=smooth_envelope,
        start_idx=max(t_start, peak_idx),
        duration_s=FORCE_PLATEAU_SLOPE_WINDOW_S,
    )
    return ForceRiseTrial(
        path=csv_path,
        distance_label=distance_label,
        label=make_label(csv_path),
        envelope=smooth_envelope,
        analysis_envelope=analysis_envelope,
        t_start=t_start,
        peak_idx=peak_idx,
        rise_x=rise_x,
        rise_y=rise_y,
        peak_envelope=float(raw_envelope[peak_idx]),
        baseline_level=baseline_level,
        rise_10_90_ms=rise_90_ms - rise_10_ms,
        plateau_entry_ms=plateau_entry_ms,
        plateau_slope_lsb_s=plateau_slope,
        plateau_std_lsb=plateau_std,
        plateau_value_lsb=plateau_value,
    )


def batch_process_force_rise(
    trial_paths_dict: dict[str, list[Path]],
    apply_blur: bool,
    gaussian_sigma: float,
) -> tuple[dict[str, list[ForceRiseTrial]], list[dict[str, object]]]:
    results: dict[str, list[ForceRiseTrial]] = {}
    failures: list[dict[str, object]] = []
    for distance_label, paths in trial_paths_dict.items():
        group_results: list[ForceRiseTrial] = []
        for path in paths:
            try:
                group_results.append(
                    process_force_rise_trial(
                        csv_path=path,
                        distance_label=distance_label,
                        apply_blur=apply_blur,
                        gaussian_sigma=gaussian_sigma,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failures.append({"distance": distance_label, "file": path.name, "error": str(exc)})
        results[distance_label] = group_results
    return results, failures


def interpolate_rise_to_phase_grid(
    trials: list[ForceRiseTrial],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    phase_grid = np.linspace(0.0, 1.0, n_points)
    curves = []
    for trial in trials:
        if trial.rise_y.size < 2:
            continue
        trial_phase = np.linspace(0.0, 1.0, trial.rise_y.size)
        curves.append(np.interp(phase_grid, trial_phase, trial.rise_y))
    if not curves:
        return phase_grid, np.empty((0, n_points), dtype=float)
    return phase_grid, np.vstack(curves)


def subset_variance_score(curves: np.ndarray, indices: tuple[int, ...] | list[int]) -> float:
    subset = curves[np.asarray(indices, dtype=int)]
    return float(np.mean(np.var(subset, axis=0, ddof=0)))


def select_low_variance_force_trials(
    results: dict[str, list[ForceRiseTrial]],
    select_n: int,
    n_grid_points: int,
) -> dict[str, list[ForceRiseTrial]]:
    selected: dict[str, list[ForceRiseTrial]] = {}
    for distance_label, trials in results.items():
        for trial in trials:
            trial.selected = False
            trial.manual_excluded = trial.path.stem in FORCE_RISE_MANUAL_EXCLUDE_STEMS.get(distance_label, set())
            trial.quality_score = float("nan")

        valid = [trial for trial in trials if trial.rise_y.size >= 2 and not trial.manual_excluded]
        if not valid:
            selected[distance_label] = []
            continue

        k = min(max(1, select_n), len(valid))
        _phase_grid, curves = interpolate_rise_to_phase_grid(valid, n_grid_points)
        if curves.shape[0] == 0:
            selected[distance_label] = []
            continue

        median_curve = np.median(curves, axis=0)
        for trial, curve in zip(valid, curves):
            trial.quality_score = float(np.mean((curve - median_curve) ** 2))

        if k == len(valid):
            chosen_indices = tuple(range(len(valid)))
        else:
            combo_count = math.comb(len(valid), k)
            if combo_count <= FORCE_MAX_EXACT_SELECTION_COMBOS:
                chosen_indices = min(
                    itertools.combinations(range(len(valid)), k),
                    key=lambda combo: subset_variance_score(curves, combo),
                )
            else:
                ranked = np.argsort([trial.quality_score for trial in valid])
                chosen_indices = tuple(int(idx) for idx in ranked[:k])

        chosen = [valid[idx] for idx in chosen_indices]
        for trial in chosen:
            trial.selected = True
        selected[distance_label] = chosen

    return selected


def align_force_rise_curves_by_time(
    trials: list[ForceRiseTrial],
    duration_s: float = FORCE_RISE_PLOT_DURATION_S,
) -> tuple[np.ndarray, np.ndarray]:
    valid = [trial for trial in trials if trial.rise_x.size >= 2 and trial.rise_y.size >= 2]
    if not valid:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    x_max = min(float(duration_s), min(float(trial.rise_x[-1]) for trial in valid))
    if x_max <= 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)

    grid = np.linspace(0.0, x_max, max(FORCE_SELECTION_GRID_POINTS, int(round(x_max * FS)) + 1))
    curves = np.vstack([np.interp(grid, trial.rise_x, trial.rise_y) for trial in valid])
    return grid, curves


def compute_force_rise_shared_ylim(
    selected: dict[str, list[ForceRiseTrial]],
    long_tails: list[ForceLongTailTrial],
) -> tuple[float, float] | None:
    y_values: list[np.ndarray] = []

    for distance_label in FORCE_DISTANCE_GROUPS:
        trials = selected.get(distance_label, [])
        if not trials:
            continue
        _grid, curves = align_force_rise_curves_by_time(trials)
        if curves.size == 0:
            continue
        mean_y = np.mean(curves, axis=0)
        std_y = np.std(curves, axis=0, ddof=0)
        y_values.append(mean_y - std_y)
        y_values.append(mean_y + std_y)

    for tail in long_tails:
        if tail.tail_y.size:
            y_values.append(tail.tail_y)

    if not y_values:
        return None

    y_all = np.concatenate([values[np.isfinite(values)] for values in y_values if np.any(np.isfinite(values))])
    if y_all.size == 0:
        return None

    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if y_max <= y_min:
        pad = max(1.0, abs(y_max) * 0.05)
        return y_min - pad, y_max + pad

    pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def plot_force_rise_summary(
    selected: dict[str, list[ForceRiseTrial]],
    output_dir: Path,
    force_levels: dict[str, ForceLevel] | None = None,
    shared_ylim: tuple[float, float] | None = None,
) -> str:
    force_levels = force_levels or {}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    for distance_label in FORCE_DISTANCE_GROUPS:
        trials = selected.get(distance_label, [])
        if not trials:
            continue
        grid, curves = align_force_rise_curves_by_time(trials)
        if curves.size == 0:
            continue
        mean_y = np.mean(curves, axis=0)
        std_y = np.std(curves, axis=0, ddof=0)
        color = FORCE_GROUP_COLORS.get(distance_label, None)
        label = f"{force_legend_label(distance_label, force_levels)}"
        ax.plot(grid, mean_y, color=color, linewidth=1.8, label=label)
        ax.fill_between(grid, mean_y - std_y, mean_y + std_y, color=color, alpha=0.18, linewidth=0)

    ax.set_title("Contact establishment and short-hold response")
    ax.set_xlabel("Time from rise onset (s)")
    ax.set_ylabel("Array envelope (LSB)")
    ax.set_xlim(0.0, FORCE_RISE_PLOT_DURATION_S)
    if shared_ylim is not None:
        ax.set_ylim(*shared_ylim)
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0.02, y=0.08)
    path = save_figure(fig, os.fspath(output_dir), "09_force_rise_1cm_to_6cm_mean_std")
    plt.close(fig)
    return path


def build_force_rise_diagnostic_table(
    results: dict[str, list[ForceRiseTrial]],
    groups: tuple[str, ...] = FORCE_RISE_DIAGNOSTIC_GROUPS,
) -> pd.DataFrame:
    rows = []
    for distance_label in groups:
        trials = sorted(results.get(distance_label, []), key=lambda trial: trial.path.name)
        for order, trial in enumerate(trials, start=1):
            rows.append(
                {
                    "distance": distance_label,
                    "trial_order": order,
                    "file": trial.path.name,
                    "selected": trial.selected,
                    "manual_excluded": trial.manual_excluded,
                    "quality_score": trial.quality_score,
                    "rise_10_90_ms": trial.rise_10_90_ms,
                    "plateau_entry_ms": trial.plateau_entry_ms,
                    "rise_duration_s": float(trial.rise_x[-1]) if trial.rise_x.size else 0.0,
                    "rise_len_samples": int(trial.rise_y.size),
                    "t_start": trial.t_start,
                    "peak_idx": trial.peak_idx,
                    "peak_envelope_lsb": trial.peak_envelope,
                    "plateau_value_lsb": trial.plateau_value_lsb,
                    "plateau_std_lsb": trial.plateau_std_lsb,
                }
            )
    return pd.DataFrame(rows)


def plot_force_rise_diagnostic_curves(
    results: dict[str, list[ForceRiseTrial]],
    output_dir: Path,
    groups: tuple[str, ...] = FORCE_RISE_DIAGNOSTIC_GROUPS,
) -> str:
    available_groups = [group for group in groups if results.get(group)]
    if not available_groups:
        return ""

    fig, axes = plt.subplots(
        len(available_groups),
        1,
        figsize=(7.8, 3.6 * len(available_groups)),
        sharex=True,
        constrained_layout=True,
    )
    if len(available_groups) == 1:
        axes = [axes]

    for ax, distance_label in zip(axes, available_groups):
        trials = sorted(results.get(distance_label, []), key=lambda trial: trial.path.name)
        color = FORCE_GROUP_COLORS.get(distance_label, COLORS["data"])
        for order, trial in enumerate(trials, start=1):
            if trial.rise_x.size < 2 or trial.rise_y.size < 2:
                continue
            if trial.manual_excluded:
                line_color = COLORS["gray"]
                alpha = 0.85
                linewidth = 1.15
                linestyle = ":"
                suffix = " excluded"
            else:
                line_color = color
                alpha = 0.9 if trial.selected else 0.38
                linewidth = 1.45 if trial.selected else 0.85
                linestyle = "-" if trial.selected else "--"
                suffix = ""
            label = f"{order:02d} {trial.path.stem} ({trial.rise_10_90_ms:.1f} ms){suffix}"
            ax.plot(
                trial.rise_x,
                trial.rise_y,
                color=line_color,
                alpha=alpha,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
            )

        ax.set_title(f"{distance_label}: all rising curves")
        ax.set_ylabel("Array envelope (LSB)")
        ax.set_xlim(0.0, FORCE_RISE_PLOT_DURATION_S)
        ax.margins(x=0.02, y=0.08)
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    axes[-1].set_xlabel("Time from rise onset (s)")
    path = save_figure(fig, os.fspath(output_dir), "09_force_rise_diagnostic_1cm_3cm_5cm_all_trials")
    plt.close(fig)
    return path


def build_force_rise_summary_table(results: dict[str, list[ForceRiseTrial]]) -> pd.DataFrame:
    rows = []
    for distance_label in FORCE_DISTANCE_GROUPS:
        for trial in results.get(distance_label, []):
            rows.append(
                {
                    "distance": distance_label,
                    "file": trial.path.name,
                    "selected": trial.selected,
                    "manual_excluded": trial.manual_excluded,
                    "quality_score": trial.quality_score,
                    "t_start": trial.t_start,
                    "peak_idx": trial.peak_idx,
                    "rise_len_samples": int(trial.rise_y.size),
                    "rise_duration_s": float(trial.rise_x[-1]) if trial.rise_x.size else 0.0,
                    "peak_envelope_lsb": trial.peak_envelope,
                    "baseline_level_lsb": trial.baseline_level,
                    "rise_10_90_ms": trial.rise_10_90_ms,
                    "plateau_entry_ms": trial.plateau_entry_ms,
                    "plateau_slope_lsb_s": trial.plateau_slope_lsb_s,
                    "plateau_std_lsb": trial.plateau_std_lsb,
                    "plateau_value_lsb": trial.plateau_value_lsb,
                }
            )
    return pd.DataFrame(rows)


def build_force_response_metrics_table(
    selected: dict[str, list[ForceRiseTrial]],
    force_levels: dict[str, ForceLevel],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for distance_label in FORCE_DISTANCE_GROUPS:
        trials = selected.get(distance_label, [])
        if not trials:
            continue
        peak_values = np.asarray([trial.peak_envelope for trial in trials], dtype=float)
        plateau_values = np.asarray([trial.plateau_value_lsb for trial in trials], dtype=float)
        force_level = force_levels.get(distance_label)
        rows.append(
            {
                "distance": distance_label,
                "force_label": force_legend_label(distance_label, force_levels),
                "force_mean_n": force_level.mean_n if force_level else float("nan"),
                "force_std_n": force_level.std_n if force_level else float("nan"),
                "n_selected": len(trials),
                "rise_10_90_mean_ms": float(np.nanmean([trial.rise_10_90_ms for trial in trials])),
                "rise_10_90_std_ms": float(np.nanstd([trial.rise_10_90_ms for trial in trials], ddof=0)),
                "plateau_entry_mean_ms": float(np.nanmean([trial.plateau_entry_ms for trial in trials])),
                "plateau_slope_abs_max_lsb_s": float(
                    np.nanmax(np.abs([trial.plateau_slope_lsb_s for trial in trials]))
                ),
                "plateau_slope_abs_mean_lsb_s": float(
                    np.nanmean(np.abs([trial.plateau_slope_lsb_s for trial in trials]))
                ),
                "peak_cv_pct": float(np.nanstd(peak_values, ddof=0) / max(EPS, np.nanmean(peak_values)) * 100.0),
                "plateau_value_sigma_lsb": float(np.nanstd(plateau_values, ddof=0)),
                "plateau_within_trial_std_mean_lsb": float(
                    np.nanmean([trial.plateau_std_lsb for trial in trials])
                ),
            }
        )
    return pd.DataFrame(rows)


def find_longest_force_csv(
    trial_paths_dict: dict[str, list[Path]],
) -> tuple[str, Path, int]:
    longest: tuple[str, Path, int] | None = None
    for distance_label, path in iter_grouped_force_csvs(trial_paths_dict):
        try:
            sample_count = int(load_trial(path).shape[0])
        except Exception:
            continue
        if longest is None or sample_count > longest[2]:
            longest = (distance_label, path, sample_count)

    if longest is None:
        raise FileNotFoundError("No readable force-test CSV files found for long-tail plot")
    return longest


def find_longest_force_csvs_by_group(
    trial_paths_dict: dict[str, list[Path]],
) -> dict[str, tuple[Path, int]]:
    longest_by_group: dict[str, tuple[Path, int]] = {}
    for distance_label, paths in trial_paths_dict.items():
        for path in paths:
            try:
                sample_count = int(load_trial(path).shape[0])
            except Exception:
                continue
            current = longest_by_group.get(distance_label)
            if current is None or sample_count > current[1]:
                longest_by_group[distance_label] = (path, sample_count)
    return longest_by_group


def process_force_long_tail_trial(
    csv_path: Path,
    distance_label: str,
    apply_blur: bool,
    gaussian_sigma: float,
    tail_points: int,
) -> ForceLongTailTrial:
    X = load_trial(csv_path)
    X_bc, _baseline_std = baseline_correct(X)
    X_filtered = maybe_blur_frames(X_bc, apply_blur=apply_blur, sigma_cells=gaussian_sigma)
    raw_envelope = np.sum(np.abs(X_filtered), axis=(1, 2))
    smooth_envelope = smooth_1d(raw_envelope, FORCE_RISE_SMOOTH_WINDOW)

    baseline_count = min(BASELINE_N, smooth_envelope.size)
    baseline_level = float(np.mean(smooth_envelope[:baseline_count]))
    baseline_noise = float(np.std(smooth_envelope[:baseline_count]))
    peak_idx = int(np.argmax(smooth_envelope))
    peak_level = float(smooth_envelope[peak_idx])
    onset_threshold = baseline_level + max(
        5.0 * baseline_noise,
        FORCE_ONSET_FRACTION_OF_PEAK * max(0.0, peak_level - baseline_level),
        FIXED_ACTIVITY_THRESHOLD_LSB,
    )

    before_peak = smooth_envelope[: peak_idx + 1]
    hits = np.flatnonzero(before_peak >= onset_threshold)
    t_start = int(hits[0]) if hits.size else 0
    rise_stop = min(len(smooth_envelope), peak_idx + 1)
    if rise_stop - t_start < 2:
        t_start = max(0, peak_idx - 1)
        rise_stop = min(len(smooth_envelope), peak_idx + 1)

    tail_count = min(max(1, tail_points), len(smooth_envelope))
    tail_start_idx = len(smooth_envelope) - tail_count
    tail_y = smooth_envelope[tail_start_idx:]
    tail_x_s = FORCE_LONG_TAIL_START_S + np.arange(tail_count, dtype=float) / FS

    return ForceLongTailTrial(
        path=csv_path,
        distance_label=distance_label,
        label=make_label(csv_path),
        envelope=smooth_envelope,
        tail_x_s=tail_x_s,
        tail_y=tail_y,
        tail_start_idx=tail_start_idx,
        peak_idx=peak_idx,
    )


def process_force_long_tail_trials_by_group(
    trial_paths_dict: dict[str, list[Path]],
    apply_blur: bool,
    gaussian_sigma: float,
    tail_points: int,
) -> list[ForceLongTailTrial]:
    longest_by_group = find_longest_force_csvs_by_group(trial_paths_dict)
    tails: list[ForceLongTailTrial] = []
    for distance_label in FORCE_DISTANCE_GROUPS:
        item = longest_by_group.get(distance_label)
        if item is None:
            continue
        path, _sample_count = item
        tails.append(
            process_force_long_tail_trial(
                csv_path=path,
                distance_label=distance_label,
                apply_blur=apply_blur,
                gaussian_sigma=gaussian_sigma,
                tail_points=tail_points,
            )
        )
    return tails


def _add_break_marks(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    kwargs = dict(
        marker=[(-1, -0.8), (1, 0.8)],
        markersize=8,
        linestyle="none",
        color="black",
        mec="black",
        mew=0.8,
        clip_on=False,
    )
    ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kwargs)


def plot_force_long_tail_broken(
    results: list[ForceLongTailTrial],
    output_dir: Path,
    force_levels: dict[str, ForceLevel] | None = None,
    shared_ylim: tuple[float, float] | None = None,
) -> str:
    force_levels = force_levels or {}
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    for result in results:
        color = FORCE_GROUP_COLORS.get(result.distance_label, None)
        label = force_legend_label(result.distance_label, force_levels)
        ax.plot(result.tail_x_s, result.tail_y, color=color, linewidth=1.6, label=label)

    ax.set_title("Long-hold tail response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Array envelope (LSB)")
    ax.set_xlim(FORCE_LONG_TAIL_START_S, FORCE_LONG_TAIL_START_S + max(1, FORCE_LONG_TAIL_POINTS - 1) / FS)
    if shared_ylim is not None:
        ax.set_ylim(*shared_ylim)
    ax.legend(frameon=False, ncol=2)
    ax.margins(x=0.02, y=0.08)

    path = save_figure(fig, os.fspath(output_dir), "09_force_longest_rise_tail_broken")
    plt.close(fig)
    return path


def plot_force_rise_tail_combined(
    selected: dict[str, list[ForceRiseTrial]],
    long_tails: list[ForceLongTailTrial],
    output_dir: Path,
    force_levels: dict[str, ForceLevel] | None = None,
) -> str:
    force_levels = force_levels or {}
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2), sharey=False, constrained_layout=True)
    ax_rise, ax_tail = axes

    for distance_label in FORCE_DISTANCE_GROUPS:
        trials = selected.get(distance_label, [])
        if not trials:
            continue
        grid, curves = align_force_rise_curves_by_time(trials)
        if curves.size == 0:
            continue
        mean_y = np.mean(curves, axis=0)
        std_y = np.std(curves, axis=0, ddof=0)
        color = FORCE_GROUP_COLORS.get(distance_label, None)
        label = force_legend_label(distance_label, force_levels)
        ax_rise.plot(grid, mean_y, color=color, linewidth=1.8, label=label)
        ax_rise.fill_between(grid, mean_y - std_y, mean_y + std_y, color=color, alpha=0.16, linewidth=0)

    for tail in long_tails:
        color = FORCE_GROUP_COLORS.get(tail.distance_label, None)
        label = force_legend_label(tail.distance_label, force_levels)
        ax_tail.plot(tail.tail_x_s, tail.tail_y, color=color, linewidth=1.55, label=label)

    add_subplot_label(ax_rise, "(a)")
    add_subplot_label(ax_tail, "(b)")
    ax_rise.set_title("Contact establishment and short-hold response")
    ax_tail.set_title("Long-hold response after 10 min")
    ax_rise.set_xlabel("Time from rise onset (s)")
    ax_tail.set_xlabel("Time (s)")
    ax_rise.set_ylabel("Array envelope (LSB)")
    ax_tail.set_ylabel("Array envelope (LSB)")
    ax_rise.set_xlim(0.0, FORCE_RISE_PLOT_DURATION_S)
    ax_tail.set_xlim(FORCE_LONG_TAIL_START_S, FORCE_LONG_TAIL_START_S + max(1, FORCE_LONG_TAIL_POINTS - 1) / FS)
    ax_rise.legend(frameon=False, ncol=2)
    ax_tail.legend(frameon=False, ncol=2)
    ax_rise.margins(x=0.02, y=0.08)
    ax_tail.margins(x=0.02, y=0.08)

    path = save_figure(fig, os.fspath(output_dir), "09_force_rise_tail_combined")
    plt.close(fig)
    return path


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


def nearest_speed(speed_values: list[float], target: float) -> float:
    if not speed_values:
        raise ValueError("no speed groups available")
    return min(speed_values, key=lambda value: abs(value - target))


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
    speeds = sorted(selected)
    fig, axes = plt.subplots(1, len(speeds), figsize=(12.2, 3.2), sharey=True, constrained_layout=True)
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
        ax.set_title(f"{speed:.2f} m/s")
        ax.set_xlabel("Time (s)")
        ax.margins(x=0.02, y=0.08)
    axes[0].set_ylabel("Frame diff (LSB)")

    path = save_figure(fig, os.fspath(output_dir), "08_supp_S1_waveforms")
    plt.close(fig)
    return path


def plot_supp_s2_activation_maps(selected: dict[float, list[TrialResult]], output_dir: Path) -> str:
    speeds = sorted(selected)
    fig, axes = plt.subplots(1, len(speeds), figsize=(12.2, 3.1), constrained_layout=True)
    if len(speeds) == 1:
        axes = [axes]

    all_maps = [representative_trial(selected[speed]).t_peak_map for speed in speeds if selected[speed]]
    finite_maps = [m for m in all_maps if np.any(np.isfinite(m))]
    vmax = max(float(np.nanmax(m)) for m in finite_maps) if finite_maps else 1.0

    for ax, speed in zip(axes, speeds):
        trials = selected[speed]
        if not trials:
            ax.set_visible(False)
            continue
        trial = representative_trial(trials)
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
        ax.set_title(f"{speed:.2f} m/s")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=list(axes), label="Peak time in slip window (samples)", fraction=0.02, pad=0.02)
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
    parser = argparse.ArgumentParser(description="Process VET6USB tactile force-rise CSV trials.")
    parser.add_argument("--mode", choices=("auto", "slip", "force-rise"), default="auto")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--force-data-root", type=Path, default=FORCE_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force-output-dir", type=Path, default=FORCE_OUTPUT_DIR)
    parser.add_argument("--force-groups", nargs="+", default=list(FORCE_DISTANCE_GROUPS))
    parser.add_argument("--force-tail-points", type=int, default=FORCE_LONG_TAIL_POINTS)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--pitch-mm", type=float, default=PITCH_MM)
    parser.add_argument("--gaussian-blur", action="store_true", default=APPLY_GAUSSIAN_BLUR)
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA_CELLS)
    parser.add_argument("--select-n", type=int, default=SELECT_N_PER_SPEED)
    parser.add_argument("--force-select-n", type=int, default=SELECT_N_PER_DISTANCE)
    parser.add_argument("--selection-feature", default=SELECTION_FEATURE)
    parser.add_argument("--spectral-min-freq", type=float, default=SPECTRAL_MIN_FREQ_HZ)
    parser.add_argument("--activity-threshold", type=float, default=FIXED_ACTIVITY_THRESHOLD_LSB)
    return parser.parse_args()


def update_runtime_config(args: argparse.Namespace) -> None:
    global FS, PITCH_MM, SPECTRAL_MIN_FREQ_HZ, FIXED_ACTIVITY_THRESHOLD_LSB, FORCE_DISTANCE_GROUPS
    FS = float(args.fs)
    PITCH_MM = float(args.pitch_mm)
    SPECTRAL_MIN_FREQ_HZ = float(args.spectral_min_freq)
    FIXED_ACTIVITY_THRESHOLD_LSB = float(args.activity_threshold)
    FORCE_DISTANCE_GROUPS = tuple(args.force_groups)


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


def force_rise_data_available(data_root: Path, groups: tuple[str, ...] | list[str]) -> bool:
    return any((data_root / group).exists() and any((data_root / group).glob("*.csv")) for group in groups)


def run_force_rise_pipeline(args: argparse.Namespace) -> None:
    apply_measurement_style()
    reset_output_dir(args.force_output_dir)

    trial_paths = discover_force_rise_trials(args.force_data_root, FORCE_DISTANCE_GROUPS)
    if not any(paths for paths in trial_paths.values()):
        searched = "\n".join(str(args.force_data_root / group) for group in FORCE_DISTANCE_GROUPS)
        raise FileNotFoundError(f"No short force-test CSV files found under:\n{searched}")

    results, failures = batch_process_force_rise(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
    )
    selected = select_low_variance_force_trials(
        results=results,
        select_n=args.force_select_n,
        n_grid_points=FORCE_SELECTION_GRID_POINTS,
    )

    summary = build_force_rise_summary_table(results)
    summary_path = args.force_output_dir / "force_rise_summary_all_trials.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    failure_path = write_failures(failures, args.force_output_dir)
    force_levels = load_force_levels(args.force_data_root / "force_sensor.txt")
    metrics = build_force_response_metrics_table(selected, force_levels)
    metrics_path = args.force_output_dir / "force_response_metrics_by_group.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    diagnostic = build_force_rise_diagnostic_table(results)
    diagnostic_path = args.force_output_dir / "force_rise_1cm_3cm_5cm_all_rise_times.csv"
    diagnostic.to_csv(diagnostic_path, index=False, encoding="utf-8-sig")

    long_tails = process_force_long_tail_trials_by_group(
        trial_paths_dict=trial_paths,
        apply_blur=args.gaussian_blur,
        gaussian_sigma=args.gaussian_sigma,
        tail_points=args.force_tail_points,
    )
    shared_ylim = compute_force_rise_shared_ylim(selected, long_tails)

    figure_path = plot_force_rise_summary(selected, args.force_output_dir, force_levels, shared_ylim=shared_ylim)
    diagnostic_figure_path = plot_force_rise_diagnostic_curves(results, args.force_output_dir)
    long_tail_path = plot_force_long_tail_broken(long_tails, args.force_output_dir, force_levels, shared_ylim=shared_ylim)
    combined_path = plot_force_rise_tail_combined(selected, long_tails, args.force_output_dir, force_levels)

    print("Short force-test rising summary")
    print("distance, csv_count, selected_count")
    for distance_label in FORCE_DISTANCE_GROUPS:
        print(
            f"{distance_label}, "
            f"{len(trial_paths.get(distance_label, []))}, "
            f"{len(selected.get(distance_label, []))}"
        )
    print(f"Selected per distance: requested {args.force_select_n}, metric=minimum rising-curve variance")
    print(f"Fixed activity threshold: {FIXED_ACTIVITY_THRESHOLD_LSB:g} LSB")
    print(f"Gaussian blur: {args.gaussian_blur}, sigma={args.gaussian_sigma:g} cells")
    manual_excluded_count = int(summary["manual_excluded"].sum()) if "manual_excluded" in summary else 0
    print(f"Manual force-rise exclusions: {manual_excluded_count}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved 1cm/3cm/5cm rise-time diagnostic table: {diagnostic_path}")
    if failure_path:
        print(f"Saved failures: {failure_path}")
    print(f"Saved force rising figure: {figure_path}")
    if diagnostic_figure_path:
        print(f"Saved 1cm/3cm/5cm all-trial diagnostic figure: {diagnostic_figure_path}")
    if not diagnostic.empty:
        print("1cm/3cm/5cm all-sample 10-90 rise times")
        print("distance, file, selected, manual_excluded, rise_10_90_ms")
        for row in diagnostic.itertuples(index=False):
            print(
                f"{row.distance}, {row.file}, {bool(row.selected)}, "
                f"{bool(row.manual_excluded)}, {row.rise_10_90_ms:.3f}"
            )
    print("Longest trials for long-tail figure")
    for tail in long_tails:
        print(f"{tail.distance_label}/{tail.path.name}, tail_start_idx={tail.tail_start_idx}")
    print(f"Saved force long-tail figure: {long_tail_path}")
    print(f"Saved combined force figure: {combined_path}")


def main() -> None:
    args = parse_args()
    update_runtime_config(args)
    if args.mode == "slip":
        raise ValueError(
            "This is the force-response script. Use 08_tactile_slip_pipeline.py for section 8/test8 outputs."
        )
    run_force_rise_pipeline(args)


if __name__ == "__main__":
    main()
