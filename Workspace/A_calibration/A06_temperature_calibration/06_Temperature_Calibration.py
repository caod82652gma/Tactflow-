#!/usr/bin/env python
# coding: utf-8

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_DIR = SCRIPT_DIR / "csv"
PNG_DIR = SCRIPT_DIR / "png"
PDF_DIR = SCRIPT_DIR / "pdf"
for folder in (CSV_DIR, PNG_DIR, PDF_DIR):
    folder.mkdir(parents=True, exist_ok=True)

V_REF = 5.0
ADC_FULL_SCALE = 2**15
AMPLIFIER_GAIN = 275
WINDOW_TARGET_SAMPLES = 400
WINDOW_STABILITY_WEIGHT = 0.02

HOT_AMBIENT_TEMP = 25
COLD_AMBIENT_TEMP = 24
EXTRA_DATA_DIRS = (SCRIPT_DIR / "1_cleaned",)
CYCLE2_DIR = SCRIPT_DIR / "cycle2"
CYCLE2_AMBIENT_TEMP = 25
CYCLE3_DIR = SCRIPT_DIR / "cycle3"
CYCLE3_AMBIENT_TEMP = 24
FIG5F_CYCLES = (
    ("cycle1", SCRIPT_DIR / "1_cleaned", HOT_AMBIENT_TEMP),
    ("cycle2", SCRIPT_DIR / "2_cleaned", CYCLE2_AMBIENT_TEMP),
    ("cycle3", SCRIPT_DIR / "3_cleaned", CYCLE3_AMBIENT_TEMP),
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
    candidates = [SCRIPT_DIR / filename]
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
        "relative_path": path.relative_to(SCRIPT_DIR).as_posix(),
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


def collect_temperature_files() -> list[Path]:
    files_by_temperature = {}
    for path in SCRIPT_DIR.glob("*.csv"):
        if path.stem.isdigit():
            files_by_temperature[int(path.stem)] = path

    for folder in EXTRA_DATA_DIRS:
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            if path.stem.isdigit():
                files_by_temperature.setdefault(int(path.stem), path)

    return [files_by_temperature[temp] for temp in sorted(files_by_temperature)]


def device_for_temperature(temp_deg_c: int) -> str | None:
    if temp_deg_c <= 24:
        return "cold_device_1"
    if temp_deg_c >= 25:
        return "hot_device_2"
    return None


def ambient_temperature_for_device(device: str) -> int:
    if device == "cold_device_1":
        return COLD_AMBIENT_TEMP
    if device == "hot_device_2":
        return HOT_AMBIENT_TEMP
    raise ValueError(f"Unknown device: {device}")


def ambient_zero_reference_mv(temp_deg_c: int, device: str) -> float:
    ambient_temp = ambient_temperature_for_device(device)
    return thermocouple_mv(temp_deg_c) - thermocouple_mv(ambient_temp)


def target_raw_for_temperature(temp_deg_c: int, device: str, baseline_raw: float) -> float:
    return baseline_raw + thermocouple_mv_to_raw16(ambient_zero_reference_mv(temp_deg_c, device))


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


def plot_curves(rows: list[dict[str, object]]) -> Path:
    cold_rows = [row for row in rows if row["Device"] == "cold_device_1"]
    hot_rows = [row for row in rows if row["Device"] == "hot_device_2"]
    reference_rows = sorted(rows, key=lambda row: row["Temperature_degC"])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(
        [row["Temperature_degC"] for row in reference_rows],
        [row["Reference_Thermocouple_mV"] for row in reference_rows],
        "^-",
        color="#2f4858",
        linewidth=1.8,
        markersize=4.5,
        label="T-type table, ambient-zero",
    )

    if cold_rows:
        ax.plot(
            [row["Temperature_degC"] for row in cold_rows],
            [row["Corrected_Thermocouple_mV"] for row in cold_rows],
            "o-",
            color="#3366cc",
            linewidth=1.5,
            markersize=4.5,
            label="Low-temp device 1",
        )

    if hot_rows:
        ax.plot(
            [row["Temperature_degC"] for row in hot_rows],
            [row["Corrected_Thermocouple_mV"] for row in hot_rows],
            "s-",
            color="#cc5533",
            linewidth=1.5,
            markersize=4.5,
            label="High-temp device 2",
        )

    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("Thermocouple EMF after ambient zeroing (mV)")
    ax.set_title("Temperature Test After Ambient Offset Removal")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = PNG_DIR / "06_temperature_cleaned_three_lines.png"
    pdf_path = PDF_DIR / "06_temperature_cleaned_three_lines.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def plot_cycle2(rows: list[dict[str, object]]) -> Path:
    reference_rows = sorted(rows, key=lambda row: row["Temperature_degC"])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(
        [row["Temperature_degC"] for row in reference_rows],
        [row["Reference_Thermocouple_mV"] for row in reference_rows],
        "^-",
        color="#2f4858",
        linewidth=1.8,
        markersize=4.5,
        label="T-type table, ambient-zero",
    )
    ax.plot(
        [row["Temperature_degC"] for row in reference_rows],
        [row["Corrected_Thermocouple_mV"] for row in reference_rows],
        "o-",
        color="#6a51a3",
        linewidth=1.5,
        markersize=4.2,
        label="Cycle2 8chip",
    )
    ax.set_xlabel("Temperature (degC)")
    ax.set_ylabel("Thermocouple EMF after 25 degC zeroing (mV)")
    ax.set_title("Cycle2 Temperature Calibration - 8chip")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = PNG_DIR / "06_cycle2_8chip_temperature.png"
    pdf_path = PDF_DIR / "06_cycle2_8chip_temperature.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def analyze_cycle2_8chip() -> tuple[Path | None, Path | None]:
    files = collect_numeric_temperature_files(CYCLE2_DIR)
    if not files:
        return None, None

    preliminary = []
    for path in files:
        temp = int(path.stem)
        summary = summarize_file(path)
        reference_mv = reference_from_ambient_mv(temp, CYCLE2_AMBIENT_TEMP)
        inferred_baseline = float(summary["raw_mean"]) - thermocouple_mv_to_raw16(reference_mv)
        preliminary.append((path, temp, summary, reference_mv, inferred_baseline))

    estimated_baseline = float(np.median([item[4] for item in preliminary]))
    rows = []
    for path, temp, _, reference_mv, _ in preliminary:
        target_raw = estimated_baseline + thermocouple_mv_to_raw16(reference_mv)
        summary = summarize_file(path, target_raw)
        corrected_raw = float(summary["raw_mean"]) - estimated_baseline
        corrected_visual_mv = raw16_to_visual_mv(corrected_raw)
        corrected_tc_mv = visual_mv_to_thermocouple_mv(corrected_visual_mv)
        error_mv = corrected_tc_mv - reference_mv

        rows.append(
            {
                "Temperature_degC": temp,
                "Device": "cycle2_8chip",
                "Ambient_Temperature_degC": CYCLE2_AMBIENT_TEMP,
                "Source_File": summary["relative_path"],
                "Raw_Column": summary["column"],
                "Samples": summary["samples"],
                "Window_Mode": summary["window_mode"],
                "Window_Start": summary["window_start"],
                "Window_End": summary["window_end"],
                "Window_Size": summary["window_size"],
                "Target_Mean_Raw16": round(float(summary["target_mean_raw"]), 6),
                "Selection_Score": round(float(summary["selection_score"]), 6),
                "Estimated_Baseline_Raw16": round(estimated_baseline, 6),
                "Measured_Mean_Raw16": round(float(summary["raw_mean"]), 6),
                "Measured_Std_Raw16": round(float(summary["raw_std"]), 6),
                "Corrected_Raw16": round(corrected_raw, 6),
                "Corrected_Visual_mV": round(corrected_visual_mv, 6),
                "Corrected_Thermocouple_mV": round(corrected_tc_mv, 6),
                "T_Table_Absolute_mV": round(thermocouple_mv(temp), 6),
                "Reference_Thermocouple_mV": round(reference_mv, 6),
                "Error_mV": round(error_mv, 6),
                "Error_uV": round(error_mv * 1000.0, 3),
            }
        )

    rows = sorted(rows, key=lambda row: row["Temperature_degC"])
    data_path = CSV_DIR / "06_cycle2_8chip_temperature_data.csv"
    write_rows(data_path, rows)
    baseline_path = CSV_DIR / "06_cycle2_8chip_baseline.csv"
    write_rows(
        baseline_path,
        [
            {
                "Device": "cycle2_8chip",
                "Ambient_Temperature_degC": CYCLE2_AMBIENT_TEMP,
                "Estimated_Baseline_Raw16": round(estimated_baseline, 6),
                "Baseline_Method": "median inferred from all temperature points",
                "Temperature_Min_degC": rows[0]["Temperature_degC"],
                "Temperature_Max_degC": rows[-1]["Temperature_degC"],
                "Num_Files": len(rows),
            }
        ],
    )
    print("\nCycle2 8chip analysis:")
    print(f"  Ambient temperature: {CYCLE2_AMBIENT_TEMP} degC")
    print(f"  Estimated baseline: {estimated_baseline:.3f} raw16")
    print(f"  Files: {len(rows)}")
    print(f"  Saved data: {data_path}")
    print(f"  Saved baseline: {baseline_path}")
    return data_path, None


def cycle_temperature_files(cycle_dir: Path) -> dict[int, Path]:
    return {int(path.stem): path for path in collect_numeric_temperature_files(cycle_dir)}


def ambient_baseline_file(cycle_dir: Path, ambient_temp: int) -> Path | None:
    if ambient_temp == HOT_AMBIENT_TEMP:
        candidates = ("Temp_Hot_8Chips.csv", "Temp_Hot_4Chips.csv")
    elif ambient_temp == COLD_AMBIENT_TEMP:
        candidates = ("Temp_Cold_8Chips.csv",)
    else:
        candidates = ()
    for filename in candidates:
        path = cycle_dir / filename
        if path.exists():
            return path
    return None


def infer_cycle_baselines(
    cycle_id: str,
    cycle_dir: Path,
    fixed_ambient: int | None,
) -> dict[str, float]:
    inferred = []
    if fixed_ambient is not None:
        baseline_path = ambient_baseline_file(cycle_dir, fixed_ambient)
        if baseline_path is not None:
            return {"all": float(summarize_file(baseline_path)["raw_mean"])}

    for temp, path in cycle_temperature_files(cycle_dir).items():
        if fixed_ambient is None:
            raise ValueError(f"{cycle_id} needs a fixed ambient temperature.")
        ambient = fixed_ambient
        summary = summarize_file(path)
        reference_mv = reference_from_ambient_mv(temp, ambient)
        inferred.append(float(summary["raw_mean"]) - thermocouple_mv_to_raw16(reference_mv))
    if not inferred:
        raise RuntimeError(f"No numeric temperature files found for {cycle_id}.")
    return {"all": float(np.median(inferred))}


def baseline_for_cycle_temp(
    cycle_id: str,
    temp: int,
    baselines: dict[str, float],
) -> float:
    return baselines["all"]


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
        baselines = infer_cycle_baselines(cycle_id, cycle_dir, fixed_ambient)

        for temp in sorted(files):
            path = files[temp]
            ambient = ambient_for_cycle_temp(cycle_id, temp, fixed_ambient)
            baseline = baseline_for_cycle_temp(cycle_id, temp, baselines)
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


def adc_zero_for_standard_temp(temp: int) -> float:
    if temp <= 24:
        return float(summarize_file(find_data_file("Temp_Cold_8Chips.csv"))["raw_mean"])
    return float(summarize_file(find_data_file("Temp_Hot_4Chips.csv"))["raw_mean"])


def aggregate_figure5f(per_cycle_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregated = []
    temps = sorted({int(row["T_std"]) for row in per_cycle_rows})
    for temp in temps:
        group = [row for row in per_cycle_rows if int(row["T_std"]) == temp]
        v_values = np.array([float(row["V_total_mV"]) for row in group], dtype=float)
        t_values = np.array([float(row["T_hot"]) for row in group], dtype=float)
        residuals = np.array([float(row["residual_C"]) for row in group], dtype=float)
        ddof = 1 if len(group) > 1 else 0
        aggregated.append(
            {
                "ADC_zero": round(adc_zero_for_standard_temp(temp), 6),
                "T_std": temp,
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

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(temps, v_nist, "^-", color="#2f4858", linewidth=1.8, markersize=3.6, label="T-type table")
    ax.fill_between(
        temps,
        v_mean - sigma_v,
        v_mean + sigma_v,
        color="#3366cc",
        alpha=0.30,
        linewidth=0,
        label="Measured +/- sigma_V",
    )
    ax.plot(temps, v_mean, "o-", color="#3366cc", linewidth=1.5, markersize=4.0, label="Measured mean")
    ax.set_xlabel("T_std (degC)")
    ax.set_ylabel("V (mV)")
    ax.set_title("Fig. 5f Main: Cross-Cycle Mean EMF")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    main_png = PNG_DIR / "06_figure5f_main.png"
    main_pdf = PDF_DIR / "06_figure5f_main.pdf"
    fig.savefig(main_png, dpi=300)
    fig.savefig(main_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.fill_between(
        temps,
        residual - sigma_t,
        residual + sigma_t,
        color="#cc5533",
        alpha=0.30,
        linewidth=0,
        label="Residual +/- sigma_T",
    )
    ax.plot(temps, residual, "o-", color="#cc5533", linewidth=1.3, markersize=3.8, label="Residual mean")
    ax.set_xlabel("T_std (degC)")
    ax.set_ylabel("Residual (degC)")
    ax.set_title("Fig. 5f Residual: T_hot_mean - T_std")
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
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    residual_png = PNG_DIR / "06_figure5f_residual.png"
    residual_pdf = PDF_DIR / "06_figure5f_residual.pdf"
    fig.savefig(residual_png, dpi=300)
    fig.savefig(residual_pdf)
    plt.close(fig)
    return main_png, residual_png


def analyze_figure5f_cross_cycle() -> tuple[Path, Path, Path]:
    per_cycle_rows = build_figure5f_per_cycle_rows()
    aggregated_rows = aggregate_figure5f(per_cycle_rows)
    per_cycle_output_rows = [
        {
            "cycle_id": row["cycle_id"],
            "T_ambient": row["T_ambient"],
            "T_target": row["T_target"],
            "T_std": row["T_std"],
            "ADC_mean": row["ADC_mean"],
            "V_TC_mV": row["V_TC_mV"],
            "V_cold_mV": row["V_cold_mV"],
            "V_total_mV": row["V_total_mV"],
            "T_hot": row["T_hot"],
            "residual_C": row["residual_C"],
        }
        for row in per_cycle_rows
    ]

    per_cycle_path = CSV_DIR / "per_cycle.csv"
    aggregated_path = CSV_DIR / "aggregated.csv"
    write_rows(per_cycle_path, per_cycle_output_rows)
    write_rows(aggregated_path, aggregated_rows)
    main_figure_path, residual_figure_path = plot_figure5f(aggregated_rows)

    max_abs_r = max(abs(float(row["residual_mean_C"])) for row in aggregated_rows)
    median_sigma = float(np.median([float(row["sigma_T_C"]) for row in aggregated_rows]))
    print("\nFigure 5f cross-cycle analysis:")
    print(f"  Saved per-cycle table: {per_cycle_path}")
    print(f"  Saved aggregated table: {aggregated_path}")
    print(f"  Saved main figure: {main_figure_path}")
    print(f"  Saved residual figure: {residual_figure_path}")
    print(f"  max|r| = {max_abs_r:.2f} degC, median sigma = {median_sigma:.2f} degC")
    return per_cycle_path, aggregated_path, main_figure_path


def main() -> None:
    hot_baseline_csv = find_data_file("Temp_Hot_4Chips.csv")
    cold_baseline_csv = find_data_file("Temp_Cold_8Chips.csv")

    hot_baseline = summarize_file(hot_baseline_csv)
    cold_baseline = summarize_file(cold_baseline_csv)
    baselines = {
        "hot_device_2": hot_baseline["raw_mean"],
        "cold_device_1": cold_baseline["raw_mean"],
    }

    print("Ambient zero-offset baselines:")
    print(f"  hot device 2 @ {HOT_AMBIENT_TEMP} degC: {baselines['hot_device_2']:.3f} raw16")
    print(f"  cold device 1 @ {COLD_AMBIENT_TEMP} degC: {baselines['cold_device_1']:.3f} raw16")

    rows = []
    for path in collect_temperature_files():
        temp = int(path.stem)
        device = device_for_temperature(temp)
        if device is None:
            continue

        target_raw = target_raw_for_temperature(temp, device, float(baselines[device]))
        summary = summarize_file(path, target_raw)
        corrected_raw = float(summary["raw_mean"]) - float(baselines[device])
        corrected_visual_mv = raw16_to_visual_mv(corrected_raw)
        corrected_tc_mv = visual_mv_to_thermocouple_mv(corrected_visual_mv)
        table_mv = thermocouple_mv(temp)
        reference_mv = ambient_zero_reference_mv(temp, device)
        error_mv = corrected_tc_mv - reference_mv

        rows.append(
            {
                "Temperature_degC": temp,
                "Device": device,
                "Source_File": summary["relative_path"],
                "Raw_Column": summary["column"],
                "Samples": summary["samples"],
                "Window_Mode": summary["window_mode"],
                "Window_Start": summary["window_start"],
                "Window_End": summary["window_end"],
                "Window_Size": summary["window_size"],
                "Target_Mean_Raw16": round(float(summary["target_mean_raw"]), 6),
                "Selection_Score": round(float(summary["selection_score"]), 6),
                "Ambient_Baseline_Raw16": round(float(baselines[device]), 6),
                "Measured_Mean_Raw16": round(float(summary["raw_mean"]), 6),
                "Measured_Std_Raw16": round(float(summary["raw_std"]), 6),
                "Corrected_Raw16": round(corrected_raw, 6),
                "Corrected_Visual_mV": round(corrected_visual_mv, 6),
                "Corrected_Thermocouple_mV": round(corrected_tc_mv, 6),
                "T_Table_Absolute_mV": round(table_mv, 6),
                "Reference_Thermocouple_mV": round(reference_mv, 6),
                "Error_mV": round(error_mv, 6),
                "Error_uV": round(error_mv * 1000.0, 3),
            }
        )

    if not rows:
        raise RuntimeError("No temperature CSV files matched the configured ranges.")

    rows = sorted(rows, key=lambda row: row["Temperature_degC"])
    data_path = CSV_DIR / "06_temperature_cleaned_data.csv"
    write_rows(data_path, rows)

    baseline_path = CSV_DIR / "06_temperature_baselines.csv"
    write_rows(
        baseline_path,
        [
            {
                "Device": "hot_device_2",
                "Ambient_Temperature_degC": HOT_AMBIENT_TEMP,
                "Baseline_File": hot_baseline["relative_path"],
                "Baseline_Column": hot_baseline["column"],
                "Window_Mode": hot_baseline["window_mode"],
                "Window_Start": hot_baseline["window_start"],
                "Window_End": hot_baseline["window_end"],
                "Window_Size": hot_baseline["window_size"],
                "Selection_Score": round(float(hot_baseline["selection_score"]), 6),
                "Baseline_Raw16": round(float(hot_baseline["raw_mean"]), 6),
                "Baseline_Visual_mV": round(raw16_to_visual_mv(float(hot_baseline["raw_mean"])), 6),
                "Baseline_Thermocouple_mV": round(raw16_to_thermocouple_mv(float(hot_baseline["raw_mean"])), 6),
            },
            {
                "Device": "cold_device_1",
                "Ambient_Temperature_degC": COLD_AMBIENT_TEMP,
                "Baseline_File": cold_baseline["relative_path"],
                "Baseline_Column": cold_baseline["column"],
                "Window_Mode": cold_baseline["window_mode"],
                "Window_Start": cold_baseline["window_start"],
                "Window_End": cold_baseline["window_end"],
                "Window_Size": cold_baseline["window_size"],
                "Selection_Score": round(float(cold_baseline["selection_score"]), 6),
                "Baseline_Raw16": round(float(cold_baseline["raw_mean"]), 6),
                "Baseline_Visual_mV": round(raw16_to_visual_mv(float(cold_baseline["raw_mean"])), 6),
                "Baseline_Thermocouple_mV": round(raw16_to_thermocouple_mv(float(cold_baseline["raw_mean"])), 6),
            },
        ],
    )

    print("\nCleaned temperature data:")
    for row in rows:
        print(
            f"  {row['Temperature_degC']:>3} degC {row['Device']}: "
            f"{row['Corrected_Thermocouple_mV']:.4f} mV "
            f"(reference {row['Reference_Thermocouple_mV']:.4f}, error {row['Error_uV']:.1f} uV, "
            f"window {row['Window_Start']}:{row['Window_End']})"
        )

    print(f"\nSaved data: {data_path}")
    print(f"Saved baselines: {baseline_path}")
    print(f"Formula: Visual_mV = Raw16 / {ADC_FULL_SCALE} * {V_REF} * 1000")
    print(f"Thermocouple_mV = Visual_mV / {AMPLIFIER_GAIN}")
    analyze_cycle2_8chip()
    analyze_figure5f_cross_cycle()


if __name__ == "__main__":
    main()
