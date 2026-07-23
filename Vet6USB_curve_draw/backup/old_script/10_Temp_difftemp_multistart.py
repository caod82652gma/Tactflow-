"""
Time-domain temperature curves for the P0/P1/P2/P3 different-temperature experiment.

Each CSV is converted with the same thermocouple path used by
01_6_PartsA_test/06_Temperature_Calibration.py:
raw signal -> subtract offset -> thermocouple mV -> cold-junction compensation
-> T-type table interpolation -> temperature in degC.

--- FIXES applied (2026-05-09) ---
FIX 1 (load_and_process_trial): T_hat now normalised against T_c[0] (the
       sensor's actual pre-contact baseline ≈ cold_junction_c) instead of the
       filename-metadata t_jaw0_c.  The filename value is kept in t_jaw0_c for
       β / consistency diagnostics only.  Using the filename value was causing
       T_hat[0] ≈ -1.7 … -3.3 instead of 0, which made the double-exp model
       (which assumes T_hat[0]=0) completely invalid.

FIX 2 (fit_normalized_response): Replaced the fragile "peak > 1.05" heuristic
       for choosing the fit start with a fixed skip of FIT_START_SKIP_S (2 s).
       The old heuristic was picking spurious noise spikes near t ≈ 18.9 s and
       starting the fit on <50 near-steady-state samples, yielding norm_R² < 0.

FIX 3 (fit_normalized_response): When the grid search returns no result
       (best is None) or when the data is too short, the fallback NormalizedFit
       now preserves the beta / consistency fields computed from trial metadata
       instead of silently returning NaN for everything (which was hiding valid
       physical information for the 58 °C trial).

--- Improvements (2026-05-09) ---
- CSV output: removed process-only columns (offset details, raw ADC values,
  contact detection internals) to keep only experiment-relevant results.
- Plot labels: use clean "T_liquid = XX°C" instead of raw CSV filenames.
- Normalized fit plot: display A = α and B = (1-α) model coefficients
  alongside τ_fast, τ_slow and R² in the legend.
- Temperature curve plot: show RC fit parameters (τ, T_inf, R²) in legend.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PLOT_ROOT.parent
sys.path.insert(0, str(PLOT_ROOT))

from plot_style import COLORS, apply_measurement_style, save_figure  # noqa: E402


CALIBRATION_SCRIPT = PLOT_ROOT / "01_6_PartsA_test" / "06_Temperature_Calibration.py"
DATA_ROOT = REPO_ROOT / "workspace" / "temp_test_difftemp"
DEFAULT_POINTS = ("P0", "P1", "P2", "P3")
OUTPUT_DIR = PLOT_ROOT / "result_display" / "test10_temperature_curves" / "difftemp_multistart"

CSV_PATTERN = "*.csv"
SAMPLING_RATE_HZ = 8000.0
BASELINE_WINDOW_S = 0.5
NOISE_TAIL_S = 5.0
COLD_JUNCTION_C = 25.0
OFFSET_MODE = "per-file-baseline"
SMOOTH_WINDOW = 31
FIT_START_SKIP_S = 2.0
CONTACT_THRESHOLD_RAW16 = 35.0
CONTACT_RISE_FRAMES = 2
CONTACT_FIT_FRAMES = 8
CONTACT_SMOOTH_WINDOW = 5
NORMALIZED_FIT_POINTS = 60
MIN_R2_THRESHOLD = 0.93

# ===========================================================================
# Double-exponential fit parameters (tune here, all in one place)
# ===========================================================================
# tau_fast: 0.05 s often hits the lower bound and degenerates to single-exp.
# Raising it above 0.1 s keeps the fast component physically separable.
TAU_FAST_MIN_S = 0.1
TAU_FAST_INIT_S = 1.0
TAU_FAST_MAX_S = 8.0

# tau_slow: P0 data suggests the dominant time constant is around 5-6 s.
TAU_SLOW_MIN_S = 1.0
TAU_SLOW_INIT_S = 6.0
TAU_SLOW_MAX_S = 60.0

# alpha (= A, fast exponential weight).
ALPHA_MIN = 0.02
ALPHA_MAX = 0.98
ALPHA_INIT = 0.30

# Multistart strategy to avoid local minima.
N_MULTISTART = 12

# Soft penalty enforcing tau_slow >= factor * tau_fast. This prevents the
# classic non-identifiability where the two exponentials collapse together.
TAU_SEPARATION_FACTOR = 1.5
TAU_SEPARATION_WEIGHT = 5.0

# AIC comparison: delta_AIC = AIC_double - AIC_single.
ENABLE_SINGLE_EXP_COMPARISON = True


# ---------------------------------------------------------------------------
# Calibration module loader
# ---------------------------------------------------------------------------

def _load_temperature_calibration_module():
    spec = importlib.util.spec_from_file_location("temperature_calibration_06", CALIBRATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load calibration script: {CALIBRATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CALIBRATION = _load_temperature_calibration_module()
thermocouple_mv = _CALIBRATION.thermocouple_mv
raw16_to_thermocouple_mv = _CALIBRATION.raw16_to_thermocouple_mv
temperature_from_mv = _CALIBRATION.temperature_from_mv


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RCFit:
    tau_s: float = float("nan")
    T_inf_c: float = float("nan")
    T_0_c: float = float("nan")
    R2: float = float("nan")
    delta_T_steady_c: float = float("nan")
    snr: float = float("nan")
    popt: tuple[float, float, float] | None = None


@dataclass
class TrialMetadata:
    target_temperature_c: float
    t_liquid_c: float | None
    t_jaw0_c: float | None
    t_env_c: float | None
    source: str


@dataclass
class NormalizedFit:
    alpha: float = float("nan")
    tau_fast_s: float = float("nan")
    tau_slow_s: float = float("nan")
    R2: float = float("nan")
    t_fit_start_s: float = float("nan")
    beta: float = float("nan")
    consistency: str = "not_available"
    popt: tuple[float, float, float] | None = None
    tau_fast_at_bound: bool = False
    single_exp_tau_s: float = float("nan")
    single_exp_R2: float = float("nan")
    delta_AIC: float = float("nan")


@dataclass
class TemperatureTrial:
    path: Path
    target_temperature_c: float
    t_liquid_c: float | None
    t_jaw0_c: float
    t_env_c: float | None
    metadata_source: str
    label: str
    t_s: np.ndarray
    T_c: np.ndarray
    T_hat: np.ndarray | None
    raw_temp_adc: np.ndarray
    raw_vgnd_adc: np.ndarray
    raw_signal_adc: np.ndarray
    offset_raw16: float
    contact_index: int
    contact_time_s: float
    contact_threshold_raw16: float
    contact_rise_frames: int
    corrected_raw16: np.ndarray
    thermocouple_mv: np.ndarray
    total_mv: np.ndarray
    baseline_c: float
    steady_c: float
    noise_std_c: float
    fit: RCFit | None = None
    normalized_fit: NormalizedFit | None = None


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------

def _pick_column(
    fieldnames: Iterable[str],
    preferred: tuple[str, ...],
    contains: tuple[str, ...],
) -> str:
    names = list(fieldnames)
    for name in preferred:
        if name in names:
            return name
    for name in names:
        lowered = name.lower()
        if any(token.lower() in lowered for token in contains):
            return name
    raise ValueError(f"Could not find required column in {names}")


def parse_target_temperature(path: Path) -> float:
    try:
        return float(path.stem)
    except ValueError as exc:
        raise ValueError(f"{path.name} does not use a numeric temperature filename") from exc


def parse_trial_metadata(path: Path) -> TrialMetadata:
    name = path.name
    t_env = _parse_named_temperature(name, ("Tenv", "T_env", "Tenvironment"))
    t_liquid = _parse_named_temperature(name, ("Tliquid", "T_liquid", "Tliq"))
    t_jaw = _parse_named_temperature(name, ("Tgripper", "T_gripper", "Tjaw", "T_jaw"))
    if t_liquid is not None and t_jaw is not None:
        return TrialMetadata(
            target_temperature_c=t_liquid,
            t_liquid_c=t_liquid,
            t_jaw0_c=t_jaw,
            t_env_c=t_env,
            source="filename_metadata",
        )
    return TrialMetadata(
        target_temperature_c=parse_target_temperature(path),
        t_liquid_c=None,
        t_jaw0_c=None,
        t_env_c=None,
        source="numeric_filename",
    )


def _parse_named_temperature(name: str, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        match = re.search(rf"{re.escape(label)}(-?\d+(?:[\._p]\d+)?)C", name, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace("p", ".").replace("_", "."))
    return None


def discover_temperature_files(data_dir: Path, pattern: str) -> list[Path]:
    if not data_dir.exists():
        return []
    files = []
    for path in data_dir.glob(pattern):
        if not path.is_file():
            continue
        try:
            parse_trial_metadata(path)
        except ValueError:
            continue
        files.append(path)
    return sorted(files, key=lambda item: (parse_trial_metadata(item).target_temperature_c, item.name))


def load_temperature_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[float] = []
    temp_values: list[float] = []
    vgnd_values: list[float] = []

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")

        temp_col = _pick_column(
            reader.fieldnames,
            preferred=("TemperatureRaw16", "TemperatureRaw(16bit)", "TemperatureRaw", "Raw16", "RawValue"),
            contains=("Temperature", "Temp"),
        )
        vgnd_col = _pick_column(
            reader.fieldnames,
            preferred=("VGNDRaw16", "VGNDRaw(16bit)", "VGNDRaw", "VGND"),
            contains=("VGND",),
        )
        index_col = "Index" if "Index" in reader.fieldnames else None

        for row_idx, row in enumerate(reader):
            if not row:
                continue
            try:
                temp = float(row[temp_col])
                vgnd = float(row[vgnd_col])
                index = float(row[index_col]) if index_col else float(row_idx)
            except (KeyError, TypeError, ValueError):
                continue
            indices.append(index)
            temp_values.append(temp)
            vgnd_values.append(vgnd)

    if not temp_values:
        raise ValueError(f"{path} contains no readable temperature samples")

    return (
        np.asarray(indices, dtype=float),
        np.asarray(temp_values, dtype=float),
        np.asarray(vgnd_values, dtype=float),
    )

# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
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


def thermocouple_temperature_from_raw16(
    raw_signal_adc: np.ndarray,
    offset_raw16: float,
    cold_junction_c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected_raw16 = raw_signal_adc - offset_raw16
    tc_mv = np.asarray([raw16_to_thermocouple_mv(value) for value in corrected_raw16], dtype=float)
    cold_mv = thermocouple_mv(int(round(cold_junction_c)))
    total_mv = tc_mv + cold_mv
    T_c = np.asarray([temperature_from_mv(value) for value in total_mv], dtype=float)
    return T_c, corrected_raw16, total_mv


def normalized_response(T_c: np.ndarray, t_jaw0_c: float, t_ss_c: float) -> np.ndarray:
    denom = t_ss_c - t_jaw0_c
    if abs(denom) < 1e-9:
        return np.full_like(T_c, np.nan, dtype=float)
    return (T_c - t_jaw0_c) / denom


def consistency_status(t_jaw0_c: float, t_ss_c: float, t_liquid_c: float | None) -> str:
    if t_liquid_c is None:
        return "not_available"
    if t_liquid_c > t_jaw0_c:
        return "ok_heat" if t_jaw0_c < t_ss_c < t_liquid_c else "fail_heat"
    if t_liquid_c < t_jaw0_c:
        return "ok_cold" if t_liquid_c < t_ss_c < t_jaw0_c else "fail_cold"
    return "fail_equal_liquid_jaw"


def beta_from_temperatures(t_jaw0_c: float, t_ss_c: float, t_liquid_c: float | None) -> float:
    if t_liquid_c is None:
        return float("nan")
    denom = t_liquid_c - t_jaw0_c
    if abs(denom) < 1e-9:
        return float("nan")
    return float((t_ss_c - t_jaw0_c) / denom)


def response_direction(raw_signal_adc: np.ndarray, tail_n: int) -> float:
    if raw_signal_adc.size == 0:
        return 1.0
    head_n = max(1, min(raw_signal_adc.size, CONTACT_SMOOTH_WINDOW))
    baseline = float(np.median(raw_signal_adc[:head_n]))
    tail = float(np.median(raw_signal_adc[-tail_n:]))
    return 1.0 if tail >= baseline else -1.0


# ---------------------------------------------------------------------------
# Contact detection
# ---------------------------------------------------------------------------

def detect_contact_start(
    raw_signal_adc: np.ndarray,
    threshold_raw16: float,
    rise_frames: int,
    fit_frames: int,
    smooth_window: int,
    sampling_rate_hz: float,
    tail_n: int,
) -> tuple[int, float]:
    if raw_signal_adc.size == 0 or threshold_raw16 <= 0:
        return 0, 0.0

    direction = response_direction(raw_signal_adc, tail_n)
    head_n = max(1, min(raw_signal_adc.size, smooth_window))
    baseline = float(np.median(raw_signal_adc[:head_n]))
    response = direction * (raw_signal_adc - baseline)
    smooth = moving_average(response, smooth_window)
    diffs = np.diff(smooth)

    rise_frames = max(1, int(rise_frames))
    crossing_candidates = np.flatnonzero(smooth >= threshold_raw16)
    if not crossing_candidates.size:
        return 0, 0.0

    crossing = int(crossing_candidates[0])
    rise_end = crossing
    for idx in crossing_candidates:
        idx = int(idx)
        left = max(0, idx - rise_frames)
        if idx > left and np.all(diffs[left:idx] > 0):
            rise_end = idx
            break

    rise_start = rise_end
    while rise_start > 0 and smooth[rise_start - 1] < smooth[rise_start] and smooth[rise_start - 1] > 0:
        rise_start -= 1

    fit_end = min(raw_signal_adc.size, max(rise_end + 1, rise_start + rise_frames + 1))
    fit_end = min(raw_signal_adc.size, fit_end + max(0, fit_frames - (fit_end - rise_start)))
    fit_start = max(0, rise_start)
    fit_x = np.arange(fit_start, fit_end, dtype=float)
    fit_y = smooth[fit_start:fit_end]
    valid = np.isfinite(fit_y)

    if np.count_nonzero(valid) >= 2:
        slope, intercept = np.polyfit(fit_x[valid], fit_y[valid], 1)
        if slope > 0:
            zero_sample = float(-intercept / slope)
        else:
            zero_sample = float(rise_start)
    else:
        zero_sample = float(rise_start)

    zero_sample = max(0.0, min(float(crossing), zero_sample))
    contact_index = max(0, min(raw_signal_adc.size - 1, int(np.floor(zero_sample))))
    contact_time_s = zero_sample / sampling_rate_hz
    return contact_index, contact_time_s


def resolve_offset(
    raw_signal_adc: np.ndarray,
    mode: str,
    fixed_offset: float | None,
    baseline_n: int,
    contact_index: int,
) -> float:
    if mode == "fixed":
        if fixed_offset is None:
            raise ValueError("--offset-mode fixed requires --offset-raw16")
        return float(fixed_offset)
    if mode == "per-file-baseline":
        if contact_index > 0:
            return float(np.mean(raw_signal_adc[:contact_index]))
        return float(np.mean(raw_signal_adc[:baseline_n]))
    raise ValueError(f"Unsupported offset mode: {mode}")


def prepend_contact_zero_sample(
    t_s: np.ndarray,
    raw_temp: np.ndarray,
    raw_vgnd: np.ndarray,
    raw_signal: np.ndarray,
    offset_raw16: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if t_s.size == 0:
        return t_s, raw_temp, raw_vgnd, raw_signal
    if t_s[0] <= 0:
        t_s = t_s.copy()
        t_s[0] = 0.0
        return t_s, raw_temp, raw_vgnd, raw_signal

    first_vgnd = raw_vgnd[0] if raw_vgnd.size else 0.0
    return (
        np.concatenate(([0.0], t_s)),
        np.concatenate(([offset_raw16 + first_vgnd], raw_temp)),
        np.concatenate(([first_vgnd], raw_vgnd)),
        np.concatenate(([offset_raw16], raw_signal)),
    )


# ---------------------------------------------------------------------------
# Trial loading & processing
# ---------------------------------------------------------------------------

def load_and_process_trial(
    csv_path: Path,
    sampling_rate_hz: float,
    cold_junction_c: float,
    offset_mode: str,
    fixed_offset_raw16: float | None,
    baseline_window_s: float,
    noise_tail_s: float,
    contact_threshold_raw16: float,
    contact_rise_frames: int,
    contact_fit_frames: int,
    contact_smooth_window: int,
) -> TemperatureTrial:
    metadata = parse_trial_metadata(csv_path)
    index, raw_temp, raw_vgnd = load_temperature_csv(csv_path)
    baseline_n = max(1, min(raw_temp.size, int(round(baseline_window_s * sampling_rate_hz))))
    full_tail_n = max(1, min(raw_temp.size, int(round(noise_tail_s * sampling_rate_hz))))

    raw_signal = raw_temp - raw_vgnd
    contact_index, contact_time_s = detect_contact_start(
        raw_signal,
        threshold_raw16=contact_threshold_raw16,
        rise_frames=contact_rise_frames,
        fit_frames=contact_fit_frames,
        smooth_window=contact_smooth_window,
        sampling_rate_hz=sampling_rate_hz,
        tail_n=full_tail_n,
    )
    offset = resolve_offset(raw_signal, offset_mode, fixed_offset_raw16, baseline_n, contact_index)

    index = index[contact_index:]
    raw_temp = raw_temp[contact_index:]
    raw_vgnd = raw_vgnd[contact_index:]
    raw_signal = raw_signal[contact_index:]
    t_s = index / sampling_rate_hz - contact_time_s
    keep = t_s >= 0
    if np.any(keep):
        t_s = t_s[keep]
        raw_temp = raw_temp[keep]
        raw_vgnd = raw_vgnd[keep]
        raw_signal = raw_signal[keep]

    if t_s.size == 0:
        raise ValueError(f"{csv_path} has no samples after contact alignment")

    t_s, raw_temp, raw_vgnd, raw_signal = prepend_contact_zero_sample(
        t_s,
        raw_temp,
        raw_vgnd,
        raw_signal,
        offset_raw16=offset,
    )
    tail_n = max(1, min(raw_temp.size, int(round(noise_tail_s * sampling_rate_hz))))

    T_c, corrected_raw16, total_mv = thermocouple_temperature_from_raw16(
        raw_signal,
        offset_raw16=offset,
        cold_junction_c=cold_junction_c,
    )
    tc_mv = np.asarray([raw16_to_thermocouple_mv(value) for value in corrected_raw16], dtype=float)
    tail = T_c[-tail_n:]
    smooth_tail = moving_average(tail, min(SMOOTH_WINDOW, tail_n))
    steady_c = float(np.mean(tail))

    # Keep filename-based t_jaw0_c for β / consistency diagnostics only.
    t_jaw0_c = metadata.t_jaw0_c if metadata.t_jaw0_c is not None else cold_junction_c

    # FIX 1: normalise T_hat against the sensor's actual pre-contact baseline
    t_jaw0_for_norm = float(T_c[0])
    T_hat = normalized_response(T_c, t_jaw0_for_norm, steady_c)

    # --- Improved label: clean temperature-based, no raw filename ---
    if metadata.t_liquid_c is not None and metadata.t_jaw0_c is not None:
        label = (
            f"T_liquid={metadata.t_liquid_c:g}°C, "
            f"T_jaw0={metadata.t_jaw0_c:g}°C"
        )
    else:
        label = f"T_target={metadata.target_temperature_c:g}°C"

    return TemperatureTrial(
        path=csv_path,
        target_temperature_c=metadata.target_temperature_c,
        t_liquid_c=metadata.t_liquid_c,
        t_jaw0_c=t_jaw0_c,
        t_env_c=metadata.t_env_c,
        metadata_source=metadata.source,
        label=label,
        t_s=t_s,
        T_c=T_c,
        T_hat=T_hat,
        raw_temp_adc=raw_temp,
        raw_vgnd_adc=raw_vgnd,
        raw_signal_adc=raw_signal,
        offset_raw16=offset,
        contact_index=contact_index,
        contact_time_s=contact_time_s,
        contact_threshold_raw16=contact_threshold_raw16,
        contact_rise_frames=contact_rise_frames,
        corrected_raw16=corrected_raw16,
        thermocouple_mv=tc_mv,
        total_mv=total_mv,
        baseline_c=float(np.mean(T_c[:baseline_n])),
        steady_c=steady_c,
        noise_std_c=float(np.std(tail - smooth_tail)),
    )


# ---------------------------------------------------------------------------
# RC (single exponential) fit
# ---------------------------------------------------------------------------

def rc_model(t_s: np.ndarray, T_inf_c: float, delta_T0_c: float, tau_s: float) -> np.ndarray:
    return T_inf_c + delta_T0_c * np.exp(-t_s / tau_s)


def _fit_for_tau_grid(t_s: np.ndarray, T_c: np.ndarray) -> tuple[np.ndarray, float]:
    max_time = max(float(t_s[-1] - t_s[0]), 1.0)
    tau_grid = np.geomspace(0.05, max(100.0, max_time * 2.0), 700)
    best_popt: np.ndarray | None = None
    best_sse = float("inf")

    for tau_s in tau_grid:
        basis = np.exp(-t_s / tau_s)
        design = np.column_stack([np.ones_like(basis), basis])
        coeff, *_ = np.linalg.lstsq(design, T_c, rcond=None)
        pred = design @ coeff
        sse = float(np.sum((T_c - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_popt = np.asarray([coeff[0], coeff[1], tau_s], dtype=float)

    if best_popt is None:
        raise RuntimeError("RC fit grid search failed")
    return best_popt, best_sse


def fit_first_order_rc(
    t_s: np.ndarray,
    T_c: np.ndarray,
    t_start_skip_s: float,
    baseline_window_s: float,
    noise_tail_s: float,
) -> RCFit:
    mask = t_s >= t_start_skip_s
    if np.count_nonzero(mask) < 4:
        return RCFit()

    t_fit = t_s[mask] - t_start_skip_s
    T_fit = T_c[mask]
    finite = np.isfinite(t_fit) & np.isfinite(T_fit)
    t_fit = t_fit[finite]
    T_fit = T_fit[finite]
    if t_fit.size < 4 or float(np.ptp(t_fit)) <= 0:
        return RCFit()

    try:
        popt_arr, _ = _fit_for_tau_grid(t_fit, T_fit)
    except (RuntimeError, np.linalg.LinAlgError, FloatingPointError, ValueError):
        return RCFit()

    T_inf_c, delta_T0_c, tau_s = (float(popt_arr[0]), float(popt_arr[1]), float(popt_arr[2]))
    T_pred = rc_model(t_fit, T_inf_c, delta_T0_c, tau_s)
    ss_res = float(np.sum((T_fit - T_pred) ** 2))
    ss_tot = float(np.sum((T_fit - np.mean(T_fit)) ** 2))
    R2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    baseline_mask = t_s < baseline_window_s
    T_baseline = float(np.mean(T_c[baseline_mask])) if np.any(baseline_mask) else float(T_c[0])
    tail_mask = t_s > (t_s[-1] - noise_tail_s)
    tail = T_c[tail_mask] if np.any(tail_mask) else T_c[-min(T_c.size, 10) :]
    sigma_noise = float(np.std(tail))
    delta_T_steady = T_inf_c - T_baseline
    snr = abs(delta_T_steady) / sigma_noise if sigma_noise > 0 else float("nan")

    return RCFit(
        tau_s=tau_s,
        T_inf_c=T_inf_c,
        T_0_c=T_inf_c + delta_T0_c,
        R2=R2,
        delta_T_steady_c=delta_T_steady,
        snr=snr,
        popt=(T_inf_c, delta_T0_c, tau_s),
    )


def fit_trials(trials: list[TemperatureTrial], fit_start_skip_s: float, baseline_window_s: float, noise_tail_s: float) -> None:
    for trial in trials:
        trial.fit = fit_first_order_rc(
            trial.t_s,
            trial.T_c,
            t_start_skip_s=fit_start_skip_s,
            baseline_window_s=baseline_window_s,
            noise_tail_s=noise_tail_s,
        )


# ---------------------------------------------------------------------------
# Double-exponential (normalized) fit
# ---------------------------------------------------------------------------

def double_exp_normalized(t_s: np.ndarray, alpha: float, tau_fast_s: float, tau_slow_s: float) -> np.ndarray:
    return 1.0 - alpha * np.exp(-t_s / tau_fast_s) - (1.0 - alpha) * np.exp(-t_s / tau_slow_s)


def _double_exp_residuals(params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residuals with soft penalty enforcing tau_slow > k * tau_fast."""
    alpha, tau_fast, tau_slow = params
    pred = 1.0 - alpha * np.exp(-t / tau_fast) - (1.0 - alpha) * np.exp(-t / tau_slow)
    res = pred - y
    gap = tau_slow - TAU_SEPARATION_FACTOR * tau_fast
    if gap < 0:
        penalty = TAU_SEPARATION_WEIGHT * (-gap) * np.ones(3)
        return np.concatenate([res, penalty])
    return res


def _single_exp_residuals(params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    (tau,) = params
    pred = 1.0 - np.exp(-t / tau)
    return pred - y


def _aic(sse: float, n: int, k: int) -> float:
    """Akaike Information Criterion for Gaussian residuals."""
    if n <= 0 or sse <= 0:
        return float("inf")
    return n * np.log(sse / n) + 2 * k


def _generate_multistart_seeds(rng: np.random.Generator) -> list[tuple[float, float, float]]:
    """Deterministic coarse grid plus random perturbations."""
    alpha_seeds = [0.15, ALPHA_INIT, 0.50, 0.70]
    tau_fast_seeds = [TAU_FAST_INIT_S * 0.5, TAU_FAST_INIT_S, TAU_FAST_INIT_S * 2.0]
    tau_slow_seeds = [TAU_SLOW_INIT_S * 0.6, TAU_SLOW_INIT_S, TAU_SLOW_INIT_S * 1.8]

    seeds = []
    for alpha in alpha_seeds:
        for tau_fast in tau_fast_seeds:
            for tau_slow in tau_slow_seeds:
                if tau_slow > TAU_SEPARATION_FACTOR * tau_fast:
                    seeds.append((alpha, tau_fast, tau_slow))

    while len(seeds) < N_MULTISTART:
        seeds.append(
            (
                rng.uniform(ALPHA_MIN + 0.05, ALPHA_MAX - 0.05),
                rng.uniform(TAU_FAST_MIN_S * 1.5, TAU_FAST_MAX_S * 0.5),
                rng.uniform(TAU_SLOW_MIN_S * 1.5, TAU_SLOW_MAX_S * 0.3),
            )
        )
    return seeds[:N_MULTISTART]


def fit_normalized_response(trial: TemperatureTrial, max_points: int = NORMALIZED_FIT_POINTS) -> NormalizedFit:
    def _fallback(**extra) -> NormalizedFit:
        return NormalizedFit(
            beta=beta_from_temperatures(trial.t_jaw0_c, trial.steady_c, trial.t_liquid_c),
            consistency=consistency_status(trial.t_jaw0_c, trial.steady_c, trial.t_liquid_c),
            **extra,
        )

    if trial.T_hat is None or trial.T_hat.size < 6:
        return _fallback()

    T_hat = np.asarray(trial.T_hat, dtype=float)
    t_s = np.asarray(trial.t_s, dtype=float)
    finite = np.isfinite(t_s) & np.isfinite(T_hat)
    t_s, T_hat = t_s[finite], T_hat[finite]
    if t_s.size < 6 or float(np.ptp(t_s)) <= 0:
        return _fallback()

    # FIX 2 (revised): use absolute time, skip only the prepended t=0 sample.
    fit_start_index = 1 if t_s.size > 1 else 0

    t_fit = t_s[fit_start_index:]
    y_fit = T_hat[fit_start_index:]
    m = np.isfinite(t_fit) & np.isfinite(y_fit)
    t_fit, y_fit = t_fit[m], y_fit[m]
    if t_fit.size < 6:
        return _fallback()

    if t_fit.size > max_points:
        idx = np.unique(np.linspace(0, t_fit.size - 1, max_points).round().astype(int))
        t_fit, y_fit = t_fit[idx], y_fit[idx]

    rng = np.random.default_rng(seed=42)
    lower = [ALPHA_MIN, TAU_FAST_MIN_S, TAU_SLOW_MIN_S]
    upper = [ALPHA_MAX, TAU_FAST_MAX_S, TAU_SLOW_MAX_S]

    best_params: tuple[float, float, float] | None = None
    best_sse = float("inf")

    for x0 in _generate_multistart_seeds(rng):
        x0 = tuple(min(max(v, lo + 1e-6), up - 1e-6) for v, lo, up in zip(x0, lower, upper))
        try:
            result = least_squares(
                _double_exp_residuals,
                x0=x0,
                args=(t_fit, y_fit),
                bounds=(lower, upper),
                method="trf",
                max_nfev=200,
                xtol=1e-9,
                ftol=1e-9,
            )
        except (ValueError, RuntimeError):
            continue
        if not result.success:
            continue

        alpha, tau_fast, tau_slow = result.x
        pred = 1.0 - alpha * np.exp(-t_fit / tau_fast) - (1.0 - alpha) * np.exp(-t_fit / tau_slow)
        sse = float(np.sum((y_fit - pred) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_params = (float(alpha), float(tau_fast), float(tau_slow))

    if best_params is None:
        return _fallback()

    alpha, tau_fast, tau_slow = best_params
    if tau_fast > tau_slow:
        tau_fast, tau_slow = tau_slow, tau_fast
        alpha = 1.0 - alpha

    pred = double_exp_normalized(t_fit, alpha, tau_fast, tau_slow)
    ss_res = float(np.sum((y_fit - pred) ** 2))
    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    R2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    n = t_fit.size
    aic_double = _aic(ss_res, n, k=3)

    at_bound = tau_fast < TAU_FAST_MIN_S * 1.05

    single_tau = float("nan")
    single_R2 = float("nan")
    aic_single = float("inf")
    if ENABLE_SINGLE_EXP_COMPARISON:
        try:
            res_single = least_squares(
                _single_exp_residuals,
                x0=[TAU_SLOW_INIT_S],
                args=(t_fit, y_fit),
                bounds=([TAU_SLOW_MIN_S], [TAU_SLOW_MAX_S]),
                method="trf",
            )
            if res_single.success:
                single_tau = float(res_single.x[0])
                pred_s = 1.0 - np.exp(-t_fit / single_tau)
                sse_s = float(np.sum((y_fit - pred_s) ** 2))
                single_R2 = float("nan") if ss_tot <= 0 else 1.0 - sse_s / ss_tot
                aic_single = _aic(sse_s, n, k=1)
        except (ValueError, RuntimeError):
            pass

    delta_aic = aic_double - aic_single

    return NormalizedFit(
        alpha=alpha,
        tau_fast_s=tau_fast,
        tau_slow_s=tau_slow,
        R2=R2,
        t_fit_start_s=float(t_s[fit_start_index]),
        beta=beta_from_temperatures(trial.t_jaw0_c, trial.steady_c, trial.t_liquid_c),
        consistency=consistency_status(trial.t_jaw0_c, trial.steady_c, trial.t_liquid_c),
        popt=(alpha, tau_fast, tau_slow),
        tau_fast_at_bound=at_bound,
        single_exp_tau_s=single_tau,
        single_exp_R2=single_R2,
        delta_AIC=delta_aic,
    )


def fit_normalized_trials(trials: list[TemperatureTrial]) -> None:
    for trial in trials:
        trial.normalized_fit = fit_normalized_response(trial)


def filter_trials_by_r2(
    trials: list[TemperatureTrial],
    min_r2: float | None,
) -> tuple[list[TemperatureTrial], list[dict[str, str]]]:
    if min_r2 is None or not np.isfinite(min_r2):
        return trials, []

    kept: list[TemperatureTrial] = []
    filtered: list[dict[str, str]] = []
    for trial in trials:
        fit = trial.fit or RCFit()
        if np.isfinite(fit.R2) and fit.R2 >= min_r2:
            kept.append(trial)
            continue
        r2_text = "nan" if not np.isfinite(fit.R2) else f"{fit.R2:.6g}"
        filtered.append(
            {
                "point": trial.path.parent.name,
                "file": trial.path.name,
                "reason": f"R2 {r2_text} < {min_r2:g}",
            }
        )
    return kept, filtered


# ---------------------------------------------------------------------------
# Plotting — temperature time curves (improved)
# ---------------------------------------------------------------------------

def plot_temperature_curves(trials: list[TemperatureTrial], cold_junction_c: float, point_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    colors = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["purple"]]
    for idx, trial in enumerate(trials):
        color = colors[idx % len(colors)]
        smooth_window = min(SMOOTH_WINDOW, trial.T_c.size)
        T_smooth = moving_average(trial.T_c, smooth_window)

        # Build legend: temperature + RC fit params
        fit = trial.fit or RCFit()
        if fit.popt is not None:
            legend_text = (
                f"{trial.label}\n"
                f"  τ={fit.tau_s:.2f}s, T∞={fit.T_inf_c:.1f}°C, R²={fit.R2:.3f}"
            )
        else:
            legend_text = trial.label

        ax.plot(trial.t_s, trial.T_c, color=color, linewidth=0.55, alpha=0.18)
        ax.plot(trial.t_s, T_smooth, color=color, linewidth=1.35, alpha=0.92, label=legend_text)

    ax.axhline(cold_junction_c, color=COLORS["gray"], linewidth=0.9, linestyle="--", label=f"Cold junction ({cold_junction_c:g}°C)")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{point_name} Temperature Time Curves (Different Liquid Temperatures)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting — steady-state summary (improved)
# ---------------------------------------------------------------------------

def plot_steady_temperature(trials: list[TemperatureTrial], point_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    target = np.asarray([trial.target_temperature_c for trial in trials], dtype=float)
    steady = np.asarray([trial.steady_c for trial in trials], dtype=float)
    noise = np.asarray([trial.noise_std_c for trial in trials], dtype=float)
    ax.errorbar(
        target,
        steady,
        yerr=noise,
        fmt="o-",
        capsize=3,
        color=COLORS["data"],
        ecolor=COLORS["gray"],
        elinewidth=0.8,
        alpha=0.92,
        label="Measured T_ss",
    )

    # Annotate each point with β value
    for trial in trials:
        nf = trial.normalized_fit or NormalizedFit()
        if np.isfinite(nf.beta):
            ax.annotate(
                f"β={nf.beta:.2f}",
                xy=(trial.target_temperature_c, trial.steady_c),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=6.5,
                color=COLORS.get("gray", "#666666"),
            )

    lo = float(np.nanmin([np.nanmin(target), np.nanmin(steady)]))
    hi = float(np.nanmax([np.nanmax(target), np.nanmax(steady)]))
    ax.plot([lo, hi], [lo, hi], "--", color=COLORS["gray"], linewidth=1.0, label="y = x (ideal)")
    ax.set_xlabel("Target liquid temperature [°C]")
    ax.set_ylabel("Measured steady-state temperature [°C]")
    ax.set_title(f"{point_name} Steady-State Temperature Summary")
    ax.legend(frameon=False)
    ax.margins(x=0.06, y=0.10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting — normalized double-exponential fit (improved with A, B params)
# ---------------------------------------------------------------------------

def plot_normalized_fits(trials: list[TemperatureTrial], point_name: str) -> plt.Figure:
    """
    Improved normalized-fit plot.  The legend now shows:
        T_liquid = XX°C : A=α, B=(1-α), τ_f=…s, τ_s=…s, R²=…

    where A and B are the two amplitude coefficients of the double-exponential
    model  T_hat(t) = 1 - A·exp(-t/τ_fast) - B·exp(-t/τ_slow).
    """
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    colors = [COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["orange"], COLORS["red"], COLORS["purple"]]

    for idx, trial in enumerate(trials):
        if trial.T_hat is None:
            continue
        fit = trial.normalized_fit or NormalizedFit()
        color = colors[idx % len(colors)]
        t0 = fit.t_fit_start_s if np.isfinite(fit.t_fit_start_s) else 0.0
        mask = trial.t_s >= t0
        t_plot = trial.t_s[mask] - t0
        y_plot = trial.T_hat[mask]
        if t_plot.size == 0:
            continue
        step = max(1, int(np.ceil(t_plot.size / 120)))

        # --- Build legend with A, B, τ_fast, τ_slow, R² ---
        if fit.popt is not None:
            A = fit.alpha
            B = 1.0 - fit.alpha
            warn = " tau_f@bound" if fit.tau_fast_at_bound else ""
            label = (
                f"{trial.label}\n"
                f"  A={A:.2f}, B={B:.2f}, "
                f"tau_f={fit.tau_fast_s:.2f}s{warn}, tau_s={fit.tau_slow_s:.2f}s\n"
                f"  R2={fit.R2:.3f}, delta_AIC={fit.delta_AIC:+.1f}"
            )
        else:
            label = f"{trial.label} (no fit)"

        ax.scatter(t_plot[::step], y_plot[::step], s=8, alpha=0.28, color=color)
        if fit.popt is not None:
            dense_t = np.linspace(0.0, float(t_plot[-1]), 240)
            ax.plot(dense_t, double_exp_normalized(dense_t, *fit.popt), color=color, linewidth=1.5, label=label)
            if np.isfinite(fit.single_exp_tau_s):
                ax.plot(
                    dense_t,
                    1.0 - np.exp(-dense_t / fit.single_exp_tau_s),
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.6,
                )
        else:
            ax.plot(t_plot, moving_average(y_plot, min(SMOOTH_WINDOW, y_plot.size)), color=color, linewidth=1.2, label=label)

    ax.axhline(1.0, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    ax.axhline(0.0, color=COLORS["gray"], linewidth=0.7, linestyle=":")
    ax.set_xlabel("Time since contact [s]")
    ax.set_ylabel("Normalized response T̂")
    ax.set_title(f"{point_name} Normalized Response — Double-Exponential Fit\n"
                 r"$\hat{T}(t) = 1 - A \cdot e^{-t/\tau_f} - B \cdot e^{-t/\tau_s}$",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.margins(x=0.02, y=0.08)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CSV output — cleaned up (removed process-only columns)
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_temperature_traces(output_dir: Path, point_name: str, trials: list[TemperatureTrial]) -> Path:
    trace_dir = output_dir / "csv" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for trial in trials:
        rows = [
            {
                "time_s": round(float(t), 6),
                "temperature_c": round(float(temp), 6),
                "temperature_normalized": "" if trial.T_hat is None else round(float(t_hat), 9),
                "raw_signal_adc": round(float(raw), 6),
                "offset_raw16": round(float(trial.offset_raw16), 6),
                "contact_index": trial.contact_index,
                "contact_time_s": round(float(trial.contact_time_s), 6),
                "contact_rise_frames": trial.contact_rise_frames,
                "corrected_raw16": round(float(corrected), 6),
                "thermocouple_mV": round(float(tc_mv), 9),
                "total_mV_after_cold_junction": round(float(total_mv), 9),
            }
            for t, temp, t_hat, raw, corrected, tc_mv, total_mv in zip(
                trial.t_s,
                trial.T_c,
                trial.T_hat if trial.T_hat is not None else np.full_like(trial.T_c, np.nan),
                trial.raw_signal_adc,
                trial.corrected_raw16,
                trial.thermocouple_mv,
                trial.total_mv,
            )
        ]
        write_csv(trace_dir / f"10_{point_name}_{trial.path.stem}_temperature_trace.csv", rows)
    return trace_dir


def trial_summary_rows(
    trials: list[TemperatureTrial],
    cold_junction_c: float,
    offset_mode: str,
    sampling_rate_hz: float,
    noise_tail_s: float,
) -> list[dict[str, object]]:
    """
    Build the summary CSV rows.

    Compared to the original version, the following process-only columns have
    been removed:  metadata_source, offset_mode, offset_raw16,
    contact_threshold_raw16, contact_rise_frames, contact_index,
    contact_time_s, raw_signal_first_adc, raw_signal_last_adc,
    corrected_raw16_tail_mean, total_mV_tail_mean, T0_fit_c,
    tail_error_vs_filename_c, norm_t_fit_start_s.

    Retained columns focus on experimental results and model parameters.
    """
    rows: list[dict[str, object]] = []
    for trial in trials:
        fit = trial.fit or RCFit()
        norm_fit = trial.normalized_fit or NormalizedFit()
        rows.append(
            {
                "point": trial.path.parent.name,
                "file": trial.path.name,
                "target_temperature_c": trial.target_temperature_c,
                "T_liquid_c": trial.t_liquid_c,
                "T_jaw0_c": trial.t_jaw0_c,
                "T_env_c": trial.t_env_c,
                "samples": trial.T_c.size,
                "cold_junction_c": cold_junction_c,
                "baseline_temperature_c": trial.baseline_c,
                "T_ss_c": trial.steady_c,
                "sigma_ss_c": trial.noise_std_c,
                # --- Physical diagnostics ---
                "beta": norm_fit.beta,
                "consistency": norm_fit.consistency,
                # --- RC (single-exp) fit ---
                "tau_s": fit.tau_s,
                "T_inf_c": fit.T_inf_c,
                "R2": fit.R2,
                "delta_T_steady_c": fit.delta_T_steady_c,
                "snr": fit.snr,
                # --- Double-exp normalized fit ---
                "norm_A_alpha": norm_fit.alpha,
                "norm_B_1_minus_alpha": round(1.0 - norm_fit.alpha, 6) if np.isfinite(norm_fit.alpha) else float("nan"),
                "norm_tau_fast_s": norm_fit.tau_fast_s,
                "norm_tau_slow_s": norm_fit.tau_slow_s,
                "norm_R2": norm_fit.R2,
                # --- Normalized-fit diagnostics ---
                "tau_fast_at_bound": norm_fit.tau_fast_at_bound,
                "single_exp_tau_s": norm_fit.single_exp_tau_s,
                "single_exp_R2": norm_fit.single_exp_R2,
                "delta_AIC": norm_fit.delta_AIC,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Legacy cleanup
# ---------------------------------------------------------------------------

def clear_legacy_outputs(output_dir: Path) -> None:
    legacy_paths = [
        output_dir / "processing_failures.csv",
        output_dir / "temperature_group_fit_summary.csv",
        output_dir / "temperature_trial_summary.csv",
    ]
    legacy_stems = (
        "10_fig_B_temperature_rc_fit",
        "10_fig_C_tau_snr_vs_deltaT",
        "10_selected_low_variance_trials",
        "10_P1_60_65_75_time_domain",
    )
    for figure_dir in (output_dir / "png", output_dir / "pdf"):
        for stem in legacy_stems:
            legacy_paths.append(figure_dir / f"{stem}{figure_dir.suffix or ''}")
        legacy_paths.extend(figure_dir.glob("10_fig_B_temperature_rc_fit.*"))
        legacy_paths.extend(figure_dir.glob("10_fig_C_tau_snr_vs_deltaT.*"))
        legacy_paths.extend(figure_dir.glob("10_selected_low_variance_trials.*"))
        legacy_paths.extend(figure_dir.glob("10_P1_60_65_75_time_domain.*"))

    for path in legacy_paths:
        if path.exists() and path.is_file():
            path.unlink()


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temperature-time curves for temp_test_difftemp.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--points", nargs="+", default=list(DEFAULT_POINTS))
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional single point directory override.")
    parser.add_argument("--pattern", default=CSV_PATTERN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sampling-rate-hz", type=float, default=SAMPLING_RATE_HZ)
    parser.add_argument("--cold-junction-c", type=float, default=COLD_JUNCTION_C)
    parser.add_argument("--offset-mode", choices=("per-file-baseline", "fixed"), default=OFFSET_MODE)
    parser.add_argument("--offset-raw16", type=float, default=None)
    parser.add_argument("--baseline-window-s", type=float, default=BASELINE_WINDOW_S)
    parser.add_argument("--noise-tail-s", type=float, default=NOISE_TAIL_S)
    parser.add_argument("--fit-start-skip-s", type=float, default=FIT_START_SKIP_S)
    parser.add_argument("--contact-threshold-raw16", type=float, default=CONTACT_THRESHOLD_RAW16)
    parser.add_argument("--contact-rise-frames", type=int, default=CONTACT_RISE_FRAMES)
    parser.add_argument("--contact-fit-frames", type=int, default=CONTACT_FIT_FRAMES)
    parser.add_argument("--contact-smooth-window", type=int, default=CONTACT_SMOOTH_WINDOW)
    parser.add_argument(
        "--min-r2",
        type=float,
        default=MIN_R2_THRESHOLD,
        help="Filter out trials whose single-exponential RC fit R2 is below this value. Use NaN to disable.",
    )
    parser.add_argument("--skip-fit", action="store_true")
    return parser.parse_args()


def process_point(args: argparse.Namespace, point_name: str, data_dir: Path) -> dict[str, object] | None:
    files = discover_temperature_files(data_dir, args.pattern)
    if not files:
        print(f"Skipping {point_name}: no matching CSV files found under {data_dir}")
        return None

    failures: list[dict[str, str]] = []
    trials: list[TemperatureTrial] = []
    for path in files:
        try:
            trials.append(
                load_and_process_trial(
                    csv_path=path,
                    sampling_rate_hz=args.sampling_rate_hz,
                    cold_junction_c=args.cold_junction_c,
                    offset_mode=args.offset_mode,
                    fixed_offset_raw16=args.offset_raw16,
                    baseline_window_s=args.baseline_window_s,
                    noise_tail_s=args.noise_tail_s,
                    contact_threshold_raw16=args.contact_threshold_raw16,
                    contact_rise_frames=args.contact_rise_frames,
                    contact_fit_frames=args.contact_fit_frames,
                    contact_smooth_window=args.contact_smooth_window,
                )
            )
        except Exception as exc:
            failures.append({"point": path.parent.name, "file": path.name, "reason": str(exc)})

    if not trials:
        failures_path = args.output_dir / "csv" / f"10_{point_name}_processing_failures.csv"
        write_csv(failures_path, failures)
        print(f"Skipping {point_name}: no readable CSV files. Failure table saved to {failures_path}")
        return None

    processed_count = len(trials)
    r2_filtered_count = 0
    if not args.skip_fit:
        fit_trials(
            trials,
            fit_start_skip_s=args.fit_start_skip_s,
            baseline_window_s=args.baseline_window_s,
            noise_tail_s=args.noise_tail_s,
        )
        fit_normalized_trials(trials)
        trials, r2_filtered = filter_trials_by_r2(trials, args.min_r2)
        r2_filtered_count = len(r2_filtered)
        failures.extend(r2_filtered)
        if not trials:
            failures_path = args.output_dir / "csv" / f"10_{point_name}_processing_failures.csv"
            write_csv(failures_path, failures)
            print(
                f"Skipping {point_name}: no CSV files passed the R2 filter "
                f"(min R2 = {args.min_r2:g}). Failure table saved to {failures_path}"
            )
            return None

    trace_dir = write_temperature_traces(args.output_dir, point_name, trials)
    summary_path = args.output_dir / "csv" / f"10_{point_name}_temperature_summary.csv"
    failures_path = args.output_dir / "csv" / f"10_{point_name}_processing_failures.csv"
    write_csv(
        summary_path,
        trial_summary_rows(
            trials,
            args.cold_junction_c,
            args.offset_mode,
            args.sampling_rate_hz,
            args.noise_tail_s,
        ),
    )
    write_csv(failures_path, failures)

    fig_curves = plot_temperature_curves(trials, args.cold_junction_c, point_name)
    curves_path = save_figure(fig_curves, os.fspath(args.output_dir), f"10_{point_name}_temperature_time_curves")
    plt.close(fig_curves)

    fig_steady = plot_steady_temperature(trials, point_name)
    steady_path = save_figure(fig_steady, os.fspath(args.output_dir), f"10_{point_name}_tail_temperature_summary")
    plt.close(fig_steady)

    fig_normalized = plot_normalized_fits(trials, point_name)
    normalized_path = save_figure(fig_normalized, os.fspath(args.output_dir), f"10_{point_name}_normalized_double_exp_fit")
    plt.close(fig_normalized)

    print(f"{point_name} different-temperature processing summary")
    print(f"Input data dir: {data_dir}")
    print(f"CSV files processed: {processed_count}")
    print(f"Cold junction: {args.cold_junction_c:g} degC")
    print(f"Offset mode: {args.offset_mode}")
    print(f"Contact threshold: {args.contact_threshold_raw16:g} raw16")
    print(f"Contact rise frames: {args.contact_rise_frames}")
    if not args.skip_fit:
        print(f"R2 filter: kept {len(trials)} trials with R2 >= {args.min_r2:g}; filtered {r2_filtered_count}")
    if args.offset_raw16 is not None:
        print(f"Fixed offset: {args.offset_raw16:g} raw16")
    print(f"Saved time-curve figure: {curves_path}")
    print(f"Saved tail summary figure: {steady_path}")
    print(f"Saved normalized fit figure: {normalized_path}")
    print(f"Saved temperature traces: {trace_dir}")
    print(f"Saved summary table: {summary_path}")
    if failures:
        print(f"Saved processing failures: {failures_path}")
    return {
        "point": point_name,
        "data_dir": data_dir,
        "trials": trials,
        "failures": failures,
        "summary_path": summary_path,
        "curves_path": curves_path,
        "steady_path": steady_path,
        "normalized_path": normalized_path,
        "trace_dir": trace_dir,
    }


def main() -> None:
    args = parse_args()
    apply_measurement_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clear_legacy_outputs(args.output_dir)

    if args.data_dir is not None:
        point_dirs = [(args.data_dir.name, args.data_dir)]
    else:
        point_dirs = [(point, args.data_root / point) for point in args.points]

    results = []
    all_summary_rows: list[dict[str, object]] = []
    all_failures: list[dict[str, str]] = []
    for point_name, data_dir in point_dirs:
        result = process_point(args, point_name, data_dir)
        if result is None:
            continue
        results.append(result)
        all_summary_rows.extend(
            trial_summary_rows(
                result["trials"],
                args.cold_junction_c,
                args.offset_mode,
                args.sampling_rate_hz,
                args.noise_tail_s,
            )
        )
        all_failures.extend(result["failures"])

    if not results:
        print("No points were processed; all requested point directories were missing or empty.")
        return

    if len(results) > 1:
        all_summary_path = args.output_dir / "csv" / "10_all_points_temperature_summary.csv"
        all_failures_path = args.output_dir / "csv" / "10_all_points_processing_failures.csv"
        write_csv(all_summary_path, all_summary_rows)
        write_csv(all_failures_path, all_failures)
        print("All points summary")
        print(f"Processed points: {', '.join(result['point'] for result in results)}")
        print(f"Saved combined summary: {all_summary_path}")


if __name__ == "__main__":
    main()
