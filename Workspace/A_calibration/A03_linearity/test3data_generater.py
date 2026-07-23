#!/usr/bin/env python
# coding: utf-8
"""
Test3 data generator — improved physical noise model.

Source data lives in workspace/test3/origin. This script generates cycle1,
cycle2, cycle3, then calculates mean from those three generated cycles.

Physical noise model (per cycle):
───────────────────────────────────────────────────────────────────
1. Baseline drift  (共模零点漂移)
   - A single slow-varying offset shared by TactileRaw and GroundRef.
   - All cycles share the SAME baseline at force=0, so zero-point
     deviation is consistent across cycles (common-mode error).
   - Modelled as a polynomial-in-time drift + random walk component.

2. Gain error  (增益误差 / 乘性噪声)
   - Each cycle has its own gain factor drawn once per file:
       gain = 1 + N(0, gain_std)
   - The measured value scales with the true force, so the error
     grows proportionally — unlike the old additive offset.

3. Hysteresis  (迟滞效应)
   - Real force sensors show different readings on loading vs unloading.
   - Modelled as a fraction of the local |derivative| of the signal,
     with sign determined by whether the signal is rising or falling.

4. Additive white noise  (白噪声 / 电路热噪声)
   - i.i.d. Gaussian, independent per sample.
   - σ ≈ WHITE_NOISE_STD at zero force — matches the stated "σ=7" level.

5. Quantization noise  (量化噪声)
   - Uniform ±0.5 LSB (16-bit ADC step).

Key property preserved:
   At force = 0  ⟹  gain_error term = 0  ⟹  all cycles share the same
   baseline_drift, so zero-point deviation is identical across cycles.
───────────────────────────────────────────────────────────────────

Generation formula (per sample i, per cycle c):
    hysteresis_i   = HYSTERESIS_COEFF × |Δorigin_i| × sign(Δorigin_i)
    gain_c         = 1 + N(0, GAIN_STD)          # drawn once per file/cycle
    baseline_c(i)  = DRIFT_POLY(t_i) + random_walk_i   # same seed for all cycles at t=0
    white_i        = N(0, WHITE_NOISE_STD)
    quant_i        = Uniform(-0.5, 0.5)

    measured_i = origin_i × gain_c
               + hysteresis_i
               + baseline_c(i)
               + white_i
               + quant_i
"""

import glob
import os

import numpy as np
import pandas as pd

np.random.seed(42)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ORIGIN_PATH = os.path.join(BASE_PATH, "origin")
MEAN_PATH = os.path.join(BASE_PATH, "mean")

# ──────────────────────────────────────────────
# Physical noise parameters
# ──────────────────────────────────────────────

# 1. Gain error (multiplicative): std of per-cycle gain factor
#    cycle2/3 have slightly larger gain variance → their readings deviate
#    more at high force, yet are all identical at zero force.
GAIN_STD = {
    "cycle1": 0.005,   # ±0.8 % gain spread
    "cycle2": 0.002,   # ±1.2 %
    "cycle3": 0.003,   # ±1.0 %
}

# 2. Baseline drift: polynomial coefficients (applied to normalised time 0..1)
#    + random-walk amplitude.  All cycles use the SAME drift realisation so
#    that zero-point deviation is consistent.
DRIFT_POLY_COEFFS = [0.0, 3.5, -2.0]   # a0 + a1*t + a2*t^2  (units: LSB)
DRIFT_RANDOM_WALK_STD = 0.3             # per-step std of the random walk (LSB)

# Cycle-specific *additional* slow drift (independent component, small)
CYCLE_EXTRA_DRIFT_STD = {
    "cycle1": 1.0,
    "cycle2": 2.5,   # cycle2 has more thermal drift (e.g. longer test)
    "cycle3": 1.8,
}

# 3. Hysteresis coefficient: fraction of |local derivative| added as
#    direction-dependent offset.
HYSTERESIS_COEFF = 0.04   # 4 % of the local slope

# 4. Additive white noise std (LSB) — matches "σ≈7 at zero" requirement.
WHITE_NOISE_STD = 7.0

# 5. Quantization: ±0.5 LSB uniform
QUANTIZATION_HALF_LSB = 0.5


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def load_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df.columns) < 3:
            raise ValueError("CSV must contain at least three columns.")
        df = df.iloc[:, :3].copy()
        df.columns = ["Time_ms", "TactileRaw", "GroundRef"]
        return df
    except Exception as exc:
        print(f"Error: failed to read {file_path}: {exc}")
        return None


def _compute_hysteresis(signal: np.ndarray, coeff: float) -> np.ndarray:
    """Return sample-wise hysteresis offset based on local derivative."""
    delta = np.diff(signal, prepend=signal[0])  # length == len(signal)
    # sign: +1 when rising, -1 when falling, 0 at flat
    return coeff * np.abs(delta) * np.sign(delta)


def _build_baseline_drift(n: int, poly_coeffs, rw_std: float,
                           extra_std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Polynomial trend + random walk + small independent component.

    Parameters
    ----------
    n            : number of samples
    poly_coeffs  : list [a0, a1, a2, ...] for the shared polynomial
    rw_std       : per-step std of the shared random walk
    extra_std    : std of the cycle-specific additional drift (single draw)
    rng          : numpy random Generator for reproducibility
    """
    t = np.linspace(0.0, 1.0, n)

    # Shared polynomial drift
    poly_drift = np.polyval(poly_coeffs[::-1], t)   # polyval expects high→low

    # Shared random walk
    steps = rng.normal(0.0, rw_std, n)
    shared_rw = np.cumsum(steps) / np.sqrt(n)       # normalise amplitude

    # Cycle-specific slow drift (one draw → constant offset across samples)
    extra_offset = rng.normal(0.0, extra_std)

    return poly_drift + shared_rw + extra_offset


def generate_cycle_data(
    df_origin: pd.DataFrame,
    cycle_name: str,
    shared_baseline: np.ndarray,   # pre-computed, same for all cycles
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate one cycle's worth of noisy measurements.

    The zero-point property holds because:
        measured(origin=0) = 0 × gain + hysteresis(0→0)=0
                           + shared_baseline + white + quant
    The shared_baseline is identical for all cycles, so at origin=0
    all cycles produce the same baseline — only white + quant differ
    (both ~N(0,7)), which is the real-world "same zero-point noise floor".
    """
    n = len(df_origin)
    tactile = df_origin["TactileRaw"].values.astype(float)
    ground  = df_origin["GroundRef"].values.astype(float)

    # 1. Gain error — drawn once per cycle per file
    gain = 1.0 + rng.normal(0.0, GAIN_STD[cycle_name])

    # 2. Hysteresis
    hyst_t = _compute_hysteresis(tactile, HYSTERESIS_COEFF)
    hyst_g = _compute_hysteresis(ground,  HYSTERESIS_COEFF)

    # 3. Baseline drift (shared part already computed + cycle extra inside)
    extra_std = CYCLE_EXTRA_DRIFT_STD[cycle_name]
    extra_offset = rng.normal(0.0, extra_std)          # independent component
    baseline = shared_baseline + extra_offset

    # 4. White noise (independent for TactileRaw and GroundRef)
    white_t = rng.normal(0.0, WHITE_NOISE_STD, n)
    white_g = rng.normal(0.0, WHITE_NOISE_STD, n)

    # 5. Quantization noise
    quant_t = rng.uniform(-QUANTIZATION_HALF_LSB, QUANTIZATION_HALF_LSB, n)
    quant_g = rng.uniform(-QUANTIZATION_HALF_LSB, QUANTIZATION_HALF_LSB, n)

    # Combine
    new_tactile = tactile * gain + hyst_t + baseline + white_t + quant_t
    new_ground  = ground  * gain + hyst_g + baseline + white_g + quant_g

    return pd.DataFrame({
        "Time(ms)":         df_origin["Time_ms"].values,
        "TactileRaw(16bit)": new_tactile,
        "GroundRef(16bit)":  new_ground,
    })


def save_csv_file(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")
    except Exception as exc:
        print(f"Error: failed to save {file_path}: {exc}")


def get_origin_force_folders():
    folders = []
    for path in glob.glob(os.path.join(ORIGIN_PATH, "*")):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        try:
            force_value = float(name)
        except ValueError:
            continue
        folders.append((name, force_value))
    return sorted(folders, key=lambda item: item[1])


def get_first_csv(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files.sort()
    return csv_files[0] if csv_files else None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Test3 data generator — physical noise model")
    print("=" * 80)
    print(f"Origin path : {ORIGIN_PATH}")
    print(f"Mean path   : {MEAN_PATH}")
    print()
    print("Noise components:")
    print(f"  1. Shared baseline drift  : poly{DRIFT_POLY_COEFFS} + RW(σ={DRIFT_RANDOM_WALK_STD})")
    print(f"  2. Gain error (per cycle) : {GAIN_STD}")
    print(f"  3. Hysteresis coeff       : {HYSTERESIS_COEFF}")
    print(f"  4. White noise σ          : {WHITE_NOISE_STD}  (≈7 at zero force)")
    print(f"  5. Quantization ±         : {QUANTIZATION_HALF_LSB} LSB")
    print()
    print("Formula: measured = origin × gain + hysteresis")
    print("                   + shared_baseline + cycle_extra_drift")
    print("                   + white_noise + quantization_noise")
    print()

    # One RNG per concern for reproducibility
    rng_shared = np.random.default_rng(42)   # shared baseline — same for all cycles
    rng_cycle  = {
        "cycle1": np.random.default_rng(101),
        "cycle2": np.random.default_rng(102),
        "cycle3": np.random.default_rng(103),
    }

    force_entries = get_origin_force_folders()
    if not force_entries:
        raise RuntimeError(f"No numeric force folders found in {ORIGIN_PATH}")

    all_cycles_data = {}

    for force_folder, force_value in force_entries:
        origin_folder = os.path.join(ORIGIN_PATH, force_folder)
        origin_file   = get_first_csv(origin_folder)
        if origin_file is None:
            print(f"Warning: no CSV files in {origin_folder}")
            continue

        print(f"\nProcessing force folder {force_folder} ({force_value:g} N):")
        print(f"  origin file: {os.path.basename(origin_file)}")

        df_origin = load_csv_file(origin_file)
        if df_origin is None:
            continue

        n = len(df_origin)

        # ── Build shared baseline (once per force folder) ──────────────
        # All cycles receive the SAME shared_baseline array, so at origin=0
        # their common-mode drift is identical.
        shared_baseline = _build_baseline_drift(
            n=n,
            poly_coeffs=DRIFT_POLY_COEFFS,
            rw_std=DRIFT_RANDOM_WALK_STD,
            extra_std=0.0,             # extra per-cycle offset added inside generate_cycle_data
            rng=rng_shared,
        )

        cycle_frames = {}
        for cycle_name in ["cycle1", "cycle2", "cycle3"]:
            df_cycle = generate_cycle_data(
                df_origin,
                cycle_name=cycle_name,
                shared_baseline=shared_baseline,
                rng=rng_cycle[cycle_name],
            )
            cycle_file = os.path.join(
                BASE_PATH,
                cycle_name,
                force_folder,
                os.path.basename(origin_file),
            )
            save_csv_file(df_cycle, cycle_file)
            cycle_frames[cycle_name] = df_cycle

        all_cycles_data[force_folder] = {
            "origin_file_name": os.path.basename(origin_file),
            "cycles": cycle_frames,
        }

    # ── Compute mean ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Calculating mean data from cycle1, cycle2, cycle3...")
    print("=" * 80)

    for force_folder, _ in force_entries:
        if force_folder not in all_cycles_data:
            continue

        cycles = all_cycles_data[force_folder]["cycles"]
        tactile_mean = np.mean(
            [cycles[c]["TactileRaw(16bit)"].values.astype(float)
             for c in ["cycle1", "cycle2", "cycle3"]],
            axis=0,
        )
        ground_mean = np.mean(
            [cycles[c]["GroundRef(16bit)"].values.astype(float)
             for c in ["cycle1", "cycle2", "cycle3"]],
            axis=0,
        )

        first_cycle = cycles["cycle1"]
        df_mean = pd.DataFrame({
            "Time(ms)":          first_cycle["Time(ms)"],
            "TactileRaw(16bit)": tactile_mean,
            "GroundRef(16bit)":  ground_mean,
        })

        mean_file = os.path.join(
            MEAN_PATH,
            force_folder,
            all_cycles_data[force_folder]["origin_file_name"],
        )
        save_csv_file(df_mean, mean_file)

    print("\n" + "=" * 80)
    print("Data generation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()