#!/usr/bin/env python
# coding: utf-8

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from plot_style import COLORS, apply_measurement_style, save_chinese_png

apply_measurement_style()

WORKSPACE_DATA_DIR = SCRIPT_DIR.parent.parent / "Workspace" / "A_calibration" / "A06_temperature_calibration"
DATA_DIR = SCRIPT_DIR if (SCRIPT_DIR / "Temp_Hot_8Chips.csv").exists() else WORKSPACE_DATA_DIR
OUTPUT_DIR = SCRIPT_DIR.parent / "result_display" / "A_calibration" / "test6"
CSV_DIR = OUTPUT_DIR / "csv"
PNG_DIR = OUTPUT_DIR / "png"
PDF_DIR = OUTPUT_DIR / "pdf"
for folder in (CSV_DIR, PNG_DIR, PDF_DIR):
    folder.mkdir(parents=True, exist_ok=True)

V_REF = 5.0
ADC_FULL_SCALE = 2**15
AMPLIFIER_GAIN = 275
WINDOW_TARGET_SAMPLES = 400
WINDOW_STABILITY_WEIGHT = 0.02
SINGLE_PANEL_FIGSIZE = (6.0, 4.5)
LEGEND_FONTSIZE = 10

HOT_AMBIENT_TEMP = 25
COLD_AMBIENT_TEMP = 24
EXTRA_DATA_DIRS = (DATA_DIR / "1_cleaned",)
CYCLE2_AMBIENT_TEMP = 25
CYCLE3_AMBIENT_TEMP = 24
FIG5F_CYCLES = (
    ("cycle1", DATA_DIR / "1_cleaned", HOT_AMBIENT_TEMP),
    ("cycle2", DATA_DIR / "2_cleaned", CYCLE2_AMBIENT_TEMP),
    ("cycle3", DATA_DIR / "3_cleaned", CYCLE3_AMBIENT_TEMP),
)

FINAL_OUTPUT_FILES = (
    CSV_DIR / "06_figure5f_residual.csv",
    CSV_DIR / "06_figure5f_summary.csv",
    PNG_DIR / "06_figure5f_main.png",
    PNG_DIR / "06_figure5f_residual.png",
    PNG_DIR / "06_figure5f_cycle1.png",
    PNG_DIR / "06_figure5f_cycle2.png",
    PNG_DIR / "06_figure5f_cycle3.png",
)

# T-type thermocouple table, reference junction at 0 degC, unit: mV.
T_TYPE_TABLE = {
    -40: -1.475, -39: -1.510, -38: -1.545, -37: -1.579, -36: -1.614, -35: -1.648, -34: -1.683, -33: -1.717, -32: -1.751, -31: -1.785,
    -30: -1.121, -29: -1.157, -28: -1.192, -27: -1.228, -26: -1.264, -25: -1.299, -24: -1.335, -23: -1.370, -22: -1.405, -21: -1.440,
    -20: -0.757, -19: -0.794, -18: -0.830, -17: -0.867, -16: -0.904, -15: -0.940, -14: -0.976, -13: -1.013, -12: -1.049, -11: -1.085,
    -10: -0.383, -9: -0.421, -8: -0.459, -7: -0.496, -6: -0.534, -5: -0.571, -4: -0.608, -3: -0.646, -2: -0.683, -1: -0.720,
    0: 0.000, 1: 0.039, 2: 0.078, 3: 0.117, 4: 0.156, 5: 0.195, 6: 0.234, 7: 0.273, 8: 0.312, 9: 0.352,
    10: 0.391, 11: 0.431, 12: 0.470, 13: 0.510, 14: 0.549, 15: 0.589, 16: 0.629, 17: 0.669, 18: 0.709, 19: 0.749,
    20: 0.790, 21: 0.830, 22: 0.870, 23: 0.911, 24: 0.951, 25: 0.992, 26: 1.033, 27: 1.074, 28: 1.114, 29: 1.155,
    30: 1.196, 31: 1.238, 32: 1.279, 33: 1.320, 34: 1.362, 35: 1.403, 36: 1.445, 37: 1.486, 38: 1.528, 39: 1.570,
    40: 1.612, 41: 1.654, 42: 1.696, 43: 1.738, 44: 1.780, 45: 1.823, 46: 1.865, 47: 1.908, 48: 1.950, 49: 1.993,
    50: 2.036, 51: 2.079, 52: 2.122, 53: 2.165, 54: 2.208, 55: 2.251, 56: 2.294, 57: 2.338, 58: 2.381, 59: 2.425,
    60: 2.468, 61: 2.512, 62: 2.556, 63: 2.600, 64: 2.643, 65: 2.687, 66: 2.732, 67: 2.776, 68: 2.820, 69: 2.864,
    70: 2.909, 71: 2.953, 72: 2.998, 73: 3.043, 74: 3.087, 75: 3.132, 76: 3.177, 77: 3.222, 78: 3.267, 79: 3.312,
    80: 3.358, 81: 3.403, 82: 3.448, 83: 3.494, 84: 3.539, 85: 3.585, 86: 3.631, 87: 3.677, 88: 3.722, 89: 3.768,
    90: 3.814, 91: 3.860, 92: 3.907, 93: 3.953, 94: 3.999, 95: 4.046, 96: 4.092, 97: 4.138, 98: 4.185, 99: 4.232,
    100: 4.279, 101: 4.325, 102: 4.372, 103: 4.419, 104: 4.466, 105: 4.513, 106: 4.561, 107: 4.608, 108: 4.655, 109: 4.702,
    110: 4.750, 111: 4.798, 112: 4.845, 113: 4.893, 114: 4.941, 115: 4.988, 116: 5.036, 117: 5.084, 118: 5.132, 119: 5.180,
    120: 5.228, 121: 5.277, 122: 5.325, 123: 5.373, 124: 5.422, 125: 5.470, 126: 5.519, 127: 5.567, 128: 5.616, 129: 5.665,
    130: 5.714, 131: 5.763, 132: 5.812, 133: 5.861, 134: 5.910, 135: 5.959, 136: 6.008, 137: 6.057, 138: 6.107, 139: 6.156,
    140: 6.206, 141: 6.255, 142: 6.305, 143: 6.355, 144: 6.404, 145: 6.454, 146: 6.504, 147: 6.554, 148: 6.604, 149: 6.654,
    150: 6.704, 151: 6.754, 152: 6.805, 153: 6.855, 154: 6.905, 155: 6.956, 156: 7.006, 157: 7.057, 158: 7.107, 159: 7.158,
}


def thermocouple_mv(temp_deg_c: int) -> float:
    if temp_deg_c not in T_TYPE_TABLE:
        raise ValueError(f"Temperature {temp_deg_c} degC is outside the table range.")
    return T_TYPE_TABLE[temp_deg_c]


def raw16_to_visual_mv(raw16: float) -> float:
    return raw16 / ADC_FULL_SCALE * V_REF * 1000.0


def visual_mv_to_thermocouple_mv(visual_mv: float) -> float:
    return visual_mv / AMPLIFIER_GAIN


def raw16_to_thermocouple_mv(raw16: float) -> float:
    return visual_mv_to_thermocouple_mv(raw16_to_visual_mv(raw16))


def thermocouple_mv_to_visual_mv(thermocouple_mv_value: float) -> float:
    return thermocouple_mv_value * AMPLIFIER_GAIN


def visual_mv_to_raw16(visual_mv: float) -> float:
    return visual_mv / (V_REF * 1000.0) * ADC_FULL_SCALE


def thermocouple_mv_to_raw16(thermocouple_mv_value: float) -> float:
    return visual_mv_to_raw16(thermocouple_mv_to_visual_mv(thermocouple_mv_value))


def read_numeric_column(path: Path) -> tuple[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for candidate in ("TemperatureRaw16", "TemperatureRaw(16bit)", "Raw16", "RawValue"):
            if candidate in headers:
                column = candidate
                break
        else:
            data_columns = [name for name in headers if name and name.lower() != "index"]
            if not data_columns:
                raise ValueError(f"{path} has no numeric data column.")
            column = data_columns[0]

        values = []
        for row in reader:
            text = (row.get(column) or "").strip()
            if text:
                values.append(float(text))

    if not values:
        raise ValueError(f"{path} has no data in column {column}.")
    return column, np.array(values, dtype=float)


def find_data_file(filename: str) -> Path:
    candidates = [DATA_DIR / filename]
    candidates.extend(folder / filename for folder in EXTRA_DATA_DIRS)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def rolling_window_stats(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if window_size >= len(values):
        selected = values.astype(float)
        return np.array([float(np.mean(selected))]), np.array([float(np.std(selected, ddof=1))])

    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))
    sums = cumsum[window_size:] - cumsum[:-window_size]
    sums2 = cumsum2[window_size:] - cumsum2[:-window_size]
    means = sums / window_size
    variances = (sums2 - (sums * sums / window_size)) / max(window_size - 1, 1)
    stds = np.sqrt(np.maximum(variances, 0.0))
    return means, stds


def select_best_window(
    values: np.ndarray,
    target_mean_raw: float | None = None,
) -> dict[str, object]:
    window_size = min(WINDOW_TARGET_SAMPLES, len(values))
    means, stds = rolling_window_stats(values, window_size)

    if target_mean_raw is None:
        scores = stds
        mode = "lowest_std"
    else:
        robust_std = float(np.median(stds)) if len(stds) else 0.0
        stability = stds / robust_std if robust_std > 0 else stds * 0.0
        scores = np.abs(means - target_mean_raw) + WINDOW_STABILITY_WEIGHT * stability
        mode = "target_mean_plus_stability"

    start = int(np.argmin(scores))
    end = start + window_size
    selected = values[start:end]
    return {
        "mode": mode,
        "target_mean_raw": target_mean_raw,
        "window_start": start,
        "window_end": end,
        "window_size": window_size,
        "selection_score": float(scores[start]),
        "values": selected,
        "raw_mean": float(means[start]),
        "raw_std": float(stds[start]) if window_size > 1 else 0.0,
    }


def summarize_file(path: Path, target_mean_raw: float | None = None) -> dict[str, object]:
    column, values = read_numeric_column(path)
    selected = select_best_window(values, target_mean_raw)
    return {
        "file": path.name,
        "relative_path": path.relative_to(DATA_DIR).as_posix(),
        "column": column,
        "samples": len(values),
        "window_mode": selected["mode"],
        "window_start": selected["window_start"],
        "window_end": selected["window_end"],
        "window_size": selected["window_size"],
        "target_mean_raw": selected["target_mean_raw"],
        "selection_score": selected["selection_score"],
        "raw_mean": selected["raw_mean"],
        "raw_std": selected["raw_std"],
    }


def reference_from_ambient_mv(temp_deg_c: int, ambient_temp: int) -> float:
    return thermocouple_mv(temp_deg_c) - thermocouple_mv(ambient_temp)


def temperature_from_mv(total_mv: float) -> float:
    temps = np.array(sorted(T_TYPE_TABLE), dtype=float)
    mv_values = np.array([thermocouple_mv(int(temp)) for temp in temps], dtype=float)
    return float(np.interp(total_mv, mv_values, temps))


def collect_numeric_temperature_files(folder: Path) -> list[Path]:
    files = []
    if not folder.exists():
        return files
    for path in folder.glob("*.csv"):
        if path.stem.isdigit():
            files.append(path)
    return sorted(files, key=lambda item: int(item.stem))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clear_generated_outputs() -> None:
    for folder in (CSV_DIR, PNG_DIR, PDF_DIR):
        for path in folder.glob("06_figure5f*"):
            if path.is_file():
                path.unlink()
        for legacy_name in (
            "aggregated.csv",
            "per_cycle.csv",
            "06_temperature_cleaned_data.csv",
            "06_temperature_baselines.csv",
            "06_cycle2_8chip_temperature_data.csv",
            "06_cycle2_8chip_baseline.csv",
        ):
            path = folder / legacy_name
            if path.exists():
                path.unlink()
        for path in folder.glob("06_temperature_cleaned_three_lines.*"):
            if path.is_file():
                path.unlink()
        for path in folder.glob("06_cycle2_8chip_temperature.*"):
            if path.is_file():
                path.unlink()


def cycle_temperature_files(cycle_dir: Path) -> dict[int, Path]:
    return {int(path.stem): path for path in collect_numeric_temperature_files(cycle_dir)}


def infer_cycle_baseline(
    cycle_id: str,
    cycle_dir: Path,
    fixed_ambient: int | None,
) -> float:
    if fixed_ambient is None:
        raise ValueError(f"{cycle_id} needs a fixed ambient temperature.")

    inferred_baselines = []
    files = cycle_temperature_files(cycle_dir)
    if not files:
        raise RuntimeError(f"No numeric temperature files found for {cycle_id}.")

    for temp, path in files.items():
        summary = summarize_file(path)
        reference_mv = reference_from_ambient_mv(temp, fixed_ambient)
        inferred_baselines.append(float(summary["raw_mean"]) - thermocouple_mv_to_raw16(reference_mv))
    return float(np.median(inferred_baselines))


def ambient_for_cycle_temp(cycle_id: str, temp: int, fixed_ambient: int | None) -> int:
    if fixed_ambient is not None:
        return fixed_ambient
    raise ValueError(f"{cycle_id} needs a fixed ambient temperature.")


def build_figure5f_per_cycle_rows() -> list[dict[str, object]]:
    rows = []
    for cycle_id, cycle_dir, fixed_ambient in FIG5F_CYCLES:
        files = cycle_temperature_files(cycle_dir)
        if not files:
            continue
        baseline = infer_cycle_baseline(cycle_id, cycle_dir, fixed_ambient)

        for temp in sorted(files):
            path = files[temp]
            ambient = ambient_for_cycle_temp(cycle_id, temp, fixed_ambient)
            v_cold = thermocouple_mv(ambient)
            v_nist = thermocouple_mv(temp)
            v_tc_ref = v_nist - v_cold
            target_raw = baseline + thermocouple_mv_to_raw16(v_tc_ref)
            summary = summarize_file(path, target_raw)

            adc_mean = float(summary["raw_mean"])
            v_tc = raw16_to_visual_mv(adc_mean - baseline) / AMPLIFIER_GAIN
            v_total = v_tc + v_cold
            t_hot = temperature_from_mv(v_total)
            residual = t_hot - temp
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "T_ambient": ambient,
                    "T_target": temp,
                    "T_std": temp,
                    "ADC_zero": round(baseline, 6),
                    "ADC_mean": round(adc_mean, 6),
                    "V_TC_mV": round(v_tc, 6),
                    "V_cold_mV": round(v_cold, 6),
                    "V_total_mV": round(v_total, 6),
                    "T_hot": round(t_hot, 6),
                    "residual_C": round(residual, 6),
                }
            )
    return rows


def aggregate_figure5f(per_cycle_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregated = []
    temps = sorted({int(row["T_std"]) for row in per_cycle_rows})
    for temp in temps:
        group = [row for row in per_cycle_rows if int(row["T_std"]) == temp]
        v_values = np.array([float(row["V_total_mV"]) for row in group], dtype=float)
        t_values = np.array([float(row["T_hot"]) for row in group], dtype=float)
        residuals = np.array([float(row["residual_C"]) for row in group], dtype=float)
        adc_zero_values = np.array([float(row["ADC_zero"]) for row in group], dtype=float)
        ddof = 1 if len(group) > 1 else 0
        aggregated.append(
            {
                "T_std": temp,
                "ADC_zero_mean_raw16": round(float(np.mean(adc_zero_values)), 6),
                "ADC_zero_sigma_raw16": round(float(np.std(adc_zero_values, ddof=ddof)), 6),
                "V_NIST_mV": round(thermocouple_mv(temp), 6),
                "V_measured_mean_mV": round(float(np.mean(v_values)), 6),
                "sigma_V_mV": round(float(np.std(v_values, ddof=ddof)), 6),
                "T_hot_mean": round(float(np.mean(t_values)), 6),
                "residual_mean_C": round(float(np.mean(residuals)), 6),
                "sigma_T_C": round(float(np.std(t_values, ddof=ddof)), 6),
                "n_cycles": len(group),
            }
        )
    return aggregated


def plot_figure5f(aggregated: list[dict[str, object]]) -> tuple[Path, Path]:
    temps = np.array([row["T_std"] for row in aggregated], dtype=float)
    v_nist = np.array([row["V_NIST_mV"] for row in aggregated], dtype=float)
    v_mean = np.array([row["V_measured_mean_mV"] for row in aggregated], dtype=float)
    sigma_v = np.array([row["sigma_V_mV"] for row in aggregated], dtype=float)
    residual = np.array([row["residual_mean_C"] for row in aggregated], dtype=float)
    sigma_t = np.array([row["sigma_T_C"] for row in aggregated], dtype=float)

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    ax.plot(temps, v_nist, "^-", color=COLORS["gray"], linewidth=1.8, markersize=4.2, label="T-type table")
    ax.fill_between(
        temps,
        v_mean - sigma_v,
        v_mean + sigma_v,
        color=COLORS["blue"],
        alpha=0.24,
        linewidth=0,
        label="Measured +/- sigma_V",
    )
    ax.plot(temps, v_mean, "o-", color=COLORS["blue"], linewidth=1.6, markersize=4.6, label="Measured mean")
    ax.set_xlabel("T_std (degC)")
    ax.set_ylabel("V (mV)")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    main_png = PNG_DIR / "06_figure5f_main.png"
    fig.savefig(main_png, dpi=300)
    save_chinese_png(fig, main_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.9, linestyle="--")
    ax.fill_between(
        temps,
        residual - sigma_t,
        residual + sigma_t,
        color=COLORS["red"],
        alpha=0.24,
        linewidth=0,
        label="Residual +/- sigma_T",
    )
    ax.plot(temps, residual, "o-", color=COLORS["red"], linewidth=1.5, markersize=4.4, label="Residual mean")
    ax.set_xlabel("T_std (degC)")
    ax.set_ylabel("Residual (degC)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    max_abs_r = float(np.max(np.abs(residual))) if len(residual) else 0.0
    median_sigma = float(np.median(sigma_t)) if len(sigma_t) else 0.0
    ax.text(
        0.98,
        0.95,
        f"max|r| = {max_abs_r:.2f} degC\nmedian sigma = {median_sigma:.2f} degC",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower left")
    fig.tight_layout()
    residual_png = PNG_DIR / "06_figure5f_residual.png"
    fig.savefig(residual_png, dpi=300)
    save_chinese_png(fig, residual_png)
    plt.close(fig)
    return main_png, residual_png


def plot_cycle_figure(cycle_id: str, rows: list[dict[str, object]]) -> Path:
    cycle_rows = sorted(
        [row for row in rows if row["cycle_id"] == cycle_id],
        key=lambda row: row["T_std"],
    )
    if not cycle_rows:
        raise RuntimeError(f"No rows available for {cycle_id}.")

    temps = np.array([row["T_std"] for row in cycle_rows], dtype=float)
    v_nist = np.array([thermocouple_mv(int(temp)) for temp in temps], dtype=float)
    v_measured = np.array([row["V_total_mV"] for row in cycle_rows], dtype=float)

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_FIGSIZE)
    ax.plot(temps, v_nist, "^-", color=COLORS["gray"], linewidth=1.8, markersize=4.2, label="T-type table")
    ax.plot(temps, v_measured, "o-", color=COLORS["blue"], linewidth=1.6, markersize=4.6, label=cycle_id)
    ax.set_xlabel("T_std (degC)")
    ax.set_ylabel("V (mV)")
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()

    png_path = PNG_DIR / f"06_figure5f_{cycle_id}.png"
    fig.savefig(png_path, dpi=300)
    save_chinese_png(fig, png_path)
    plt.close(fig)
    return png_path


def build_summary_row(
    aggregated_rows: list[dict[str, object]],
    per_cycle_rows: list[dict[str, object]],
) -> dict[str, object]:
    residuals = np.array([float(row["residual_mean_C"]) for row in aggregated_rows], dtype=float)
    sigmas_t = np.array([float(row["sigma_T_C"]) for row in aggregated_rows], dtype=float)
    adc_zero_by_cycle = {}
    for row in per_cycle_rows:
        adc_zero_by_cycle.setdefault(str(row["cycle_id"]), float(row["ADC_zero"]))
    adc_zero_values = np.array(list(adc_zero_by_cycle.values()), dtype=float)
    max_index = int(np.argmax(np.abs(residuals)))
    ddof = 1 if len(adc_zero_values) > 1 else 0
    return {
        "ADC_zero_mean_raw16": round(float(np.mean(adc_zero_values)), 6),
        "ADC_zero_cross_cycle_1sigma_raw16": round(float(np.std(adc_zero_values, ddof=ddof)), 6),
        "temperature_max_abs_error_C": round(float(np.max(np.abs(residuals))), 6),
        "temperature_max_abs_error_at_T_std_C": int(aggregated_rows[max_index]["T_std"]),
        "cross_cycle_reproducibility_1sigma_C_median": round(float(np.median(sigmas_t)), 6),
        "cross_cycle_reproducibility_1sigma_C_max": round(float(np.max(sigmas_t)), 6),
        "residual_mean_C": round(float(np.mean(residuals)), 6),
        "residual_rms_C": round(float(np.sqrt(np.mean(residuals * residuals))), 6),
        "temperature_points": len(aggregated_rows),
        "cycles_per_point": int(max(row["n_cycles"] for row in aggregated_rows)),
    }


def analyze_figure5f_cross_cycle() -> tuple[Path, Path, Path]:
    per_cycle_rows = build_figure5f_per_cycle_rows()
    aggregated_rows = aggregate_figure5f(per_cycle_rows)

    aggregated_path = CSV_DIR / "06_figure5f_residual.csv"
    summary_path = CSV_DIR / "06_figure5f_summary.csv"
    write_rows(aggregated_path, aggregated_rows)
    write_rows(summary_path, [build_summary_row(aggregated_rows, per_cycle_rows)])
    main_figure_path, residual_figure_path = plot_figure5f(aggregated_rows)
    cycle_figure_paths = [plot_cycle_figure(cycle_id, per_cycle_rows) for cycle_id, _, _ in FIG5F_CYCLES]

    max_abs_r = max(abs(float(row["residual_mean_C"])) for row in aggregated_rows)
    median_sigma = float(np.median([float(row["sigma_T_C"]) for row in aggregated_rows]))
    print("\nFigure 5f cross-cycle analysis:")
    print(f"  Saved residual table: {aggregated_path}")
    print(f"  Saved summary table: {summary_path}")
    print(f"  Saved main figure: {main_figure_path}")
    print(f"  Saved residual figure: {residual_figure_path}")
    for figure_path in cycle_figure_paths:
        print(f"  Saved cycle figure: {figure_path}")
    print(f"  max|r| = {max_abs_r:.2f} degC, median sigma = {median_sigma:.2f} degC")
    return aggregated_path, summary_path, main_figure_path


def main() -> None:
    clear_generated_outputs()
    print(f"Using input data: {DATA_DIR}")
    print(f"Writing final outputs: {OUTPUT_DIR}")
    print(f"Formula: Visual_mV = Raw16 / {ADC_FULL_SCALE} * {V_REF} * 1000")
    print(f"Thermocouple_mV = Visual_mV / {AMPLIFIER_GAIN}")
    analyze_figure5f_cross_cycle()


if __name__ == "__main__":
    main()
