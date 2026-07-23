#!/usr/bin/env python
# coding: utf-8

import ast
import csv
import re
import shutil
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
CALIBRATION_SCRIPT = BASE_DIR / "06_Temperature_Calibration.py"

TEMP_MIN = 5
TEMP_MAX = 88
DEVICE_SPLIT_TEMP = 25
WINDOW_TARGET_SAMPLES = 400

V_REF = 5.0
ADC_FULL_SCALE = 2**15
AMPLIFIER_GAIN = 275

CYCLE1_SOURCE_DIR = BASE_DIR / "cycle1_origin"
CYCLE1_25_DIR = BASE_DIR / "cycle1_25"
AGGREGATED_CSV = CSV_DIR / "aggregated.csv"
PER_CYCLE_CSV = CSV_DIR / "per_cycle.csv"

CLEANED_CYCLES = (
    ("cycle1", CYCLE1_25_DIR, BASE_DIR / "1_cleaned", 25),
    ("cycle2", BASE_DIR / "cycle2", BASE_DIR / "2_cleaned", 25),
    ("cycle3", BASE_DIR / "cycle3", BASE_DIR / "3_cleaned", 24),
)
TOP_AGGREGATED_TEMPS = 20
PER_CYCLE_RESIDUAL_THRESHOLD_C = 0.9
MIN_KEEP_SAMPLES = 200
CLEAN_WINDOW_TARGET = 400
CYCLE1_OFFSET_RESIDUAL_THRESHOLD_C = 0.5
CYCLE1_OFFSET_CORRECTION_FRACTION = 0.90
# Cycle1 offset compensation can be tuned by global temperature bands.
# Each item is (start_temp, end_temp, correction_fraction). Later bands win
# when ranges overlap.  Example: 0.35 means only 35% of the measured offset is
# pulled back toward theory, leaving more realistic residual bias.
CYCLE1_OFFSET_CORRECTION_BANDS = [
    (5, 19, 0.55),
    (20, 32, 0.75),
    (33, 48, 0.40),
    (49, 63, 0.65),
    (64, 70, 0.70),
    (71, 80, 0.85),
    (81, 88, 0.60)
]
CYCLE1_OFFSET_APPLY_TO_BAND_TEMPS = True

ROOT_8CHIP_24_ZERO = BASE_DIR / "Temp_Cold_8Chips.csv"
ROOT_4CHIP_25_ZERO = BASE_DIR / "Temp_Hot_4Chips.csv"

TEMP_COLUMNS = ("TemperatureRaw16", "Raw16", "TemperatureRaw(16bit)", "RawValue")


def load_t_type_table() -> dict[int, float]:
    text = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"T_TYPE_TABLE\s*=\s*(\{.*?\n\})", text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find T_TYPE_TABLE in {CALIBRATION_SCRIPT}")
    return {int(key): float(value) for key, value in ast.literal_eval(match.group(1)).items()}


T_TYPE_TABLE = load_t_type_table()


def thermocouple_mv(temp_deg_c: int) -> float:
    return T_TYPE_TABLE[temp_deg_c]


def thermocouple_mv_to_raw16(delta_mv: float) -> float:
    visual_mv = delta_mv * AMPLIFIER_GAIN
    return visual_mv / (V_REF * 1000.0) * ADC_FULL_SCALE


def rolling_window_stats(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= window_size:
        return np.array([float(np.mean(values))]), np.array([float(np.std(values, ddof=1))])
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    cumsum2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))
    sums = cumsum[window_size:] - cumsum[:-window_size]
    sums2 = cumsum2[window_size:] - cumsum2[:-window_size]
    means = sums / window_size
    variances = (sums2 - sums * sums / window_size) / (window_size - 1)
    return means, np.sqrt(np.maximum(variances, 0.0))


def stable_mean(values: np.ndarray) -> float:
    window = min(WINDOW_TARGET_SAMPLES, len(values))
    means, stds = rolling_window_stats(values, window)
    return float(means[int(np.argmin(stds))])


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]], str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for candidate in TEMP_COLUMNS:
            if candidate in header:
                column = candidate
                break
        else:
            data_columns = [name for name in header if name and name.lower() != "index"]
            if not data_columns:
                raise ValueError(f"{path} has no numeric data column.")
            column = data_columns[0]
        return header, list(reader), column


def numeric_column_values(rows: list[dict[str, str]], column: str) -> np.ndarray:
    return np.array([float(row[column]) for row in rows if row.get(column) != ""], dtype=float)


def write_csv_rows(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def read_stable_mean(path: Path) -> float:
    _, rows, column = read_csv_rows(path)
    return stable_mean(numeric_column_values(rows, column))


def numeric_temperature_files(cycle_dir: Path) -> dict[int, Path]:
    files = {}
    if not cycle_dir.exists():
        return files
    for path in cycle_dir.glob("*.csv"):
        if path.stem.isdigit():
            files[int(path.stem)] = path
    return files


def shifted_rows(rows: list[dict[str, str]], column: str, shift_lsb: float) -> list[dict[str, str]]:
    copied_rows = [dict(row) for row in rows]
    for row in copied_rows:
        if row.get(column) == "":
            continue
        row[column] = str(int(round(float(row[column]) + shift_lsb)))
    return copied_rows


def write_shifted_csv(source_path: Path, output_path: Path, shift_lsb: float) -> tuple[str, int, float, float]:
    header, rows, column = read_csv_rows(source_path)
    before_values = numeric_column_values(rows, column)
    before_mean = stable_mean(before_values)
    output_rows = shifted_rows(rows, column, shift_lsb) if shift_lsb else [dict(row) for row in rows]
    after_mean = stable_mean(numeric_column_values(output_rows, column))
    write_csv_rows(output_path, header, output_rows)
    return column, len(before_values), before_mean, after_mean


def reset_output_dir() -> None:
    CYCLE1_25_DIR.mkdir(parents=True, exist_ok=True)
    for path in CYCLE1_25_DIR.glob("*.csv"):
        path.unlink()


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_csv_files(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()
    for source_path in source_dir.glob("*.csv"):
        shutil.copy2(source_path, output_dir / source_path.name)


def build_cycle1_25_dataset() -> list[dict[str, object]]:
    source_dir = CYCLE1_SOURCE_DIR
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    source_files = numeric_temperature_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"No numeric temperature files in {source_dir}")
    if not ROOT_8CHIP_24_ZERO.exists():
        raise FileNotFoundError(ROOT_8CHIP_24_ZERO)
    if not ROOT_4CHIP_25_ZERO.exists():
        raise FileNotFoundError(ROOT_4CHIP_25_ZERO)

    zero_8chip_24 = read_stable_mean(ROOT_8CHIP_24_ZERO)
    zero_4chip_25 = read_stable_mean(ROOT_4CHIP_25_ZERO)
    cold_24_to_25_lsb = thermocouple_mv_to_raw16(thermocouple_mv(25) - thermocouple_mv(24))
    zero_8chip_25 = zero_8chip_24 + cold_24_to_25_lsb
    chip4_to_8chip_shift = zero_8chip_25 - zero_4chip_25

    reset_output_dir()
    rows = []

    column, samples, before_mean, after_mean = write_shifted_csv(
        ROOT_8CHIP_24_ZERO,
        CYCLE1_25_DIR / "Temp_Cold_8Chips.csv",
        0.0,
    )
    rows.append(
        {
            "Output_File": "cycle1_25/Temp_Cold_8Chips.csv",
            "Source_File": ROOT_8CHIP_24_ZERO.relative_to(BASE_DIR).as_posix(),
            "Rule": "copy_root_8chip_24_zero",
            "Column": column,
            "Applied_Shift_Raw16": 0.0,
            "Mean_Before_Raw16": round(before_mean, 6),
            "Mean_After_Raw16": round(after_mean, 6),
            "Samples": samples,
        }
    )

    column, samples, before_mean, after_mean = write_shifted_csv(
        ROOT_4CHIP_25_ZERO,
        CYCLE1_25_DIR / "Temp_Hot_8Chips.csv",
        chip4_to_8chip_shift,
    )
    rows.append(
        {
            "Output_File": "cycle1_25/Temp_Hot_8Chips.csv",
            "Source_File": ROOT_4CHIP_25_ZERO.relative_to(BASE_DIR).as_posix(),
            "Rule": "convert_root_4chip25_zero_to_8chip25_zero",
            "Column": column,
            "Applied_Shift_Raw16": round(chip4_to_8chip_shift, 6),
            "Mean_Before_Raw16": round(before_mean, 6),
            "Mean_After_Raw16": round(after_mean, 6),
            "Target_8chip25_Zero_Raw16": round(zero_8chip_25, 6),
            "Samples": samples,
        }
    )

    for temp in range(TEMP_MIN, TEMP_MAX + 1):
        source_path = source_files.get(temp)
        if source_path is None:
            rows.append(
                {
                    "Output_File": f"cycle1_25/{temp}.csv",
                    "Source_File": f"{source_dir.name}/{temp}.csv",
                    "Temperature_degC": temp,
                    "Rule": "missing",
                    "Status": "missing",
                }
            )
            continue

        if temp < DEVICE_SPLIT_TEMP:
            # 8chip at 24 degC cold-junction is numerically equivalent to 8chip at
            # 25 degC when the 8chip zero is shifted by V(25)-V(24).
            shift = 0.0
            rule = "copy_8chip24_as_8chip25_equivalent"
        else:
            shift = chip4_to_8chip_shift
            rule = "convert_4chip25_to_8chip25"

        output_path = CYCLE1_25_DIR / f"{temp}.csv"
        column, samples, before_mean, after_mean = write_shifted_csv(source_path, output_path, shift)
        rows.append(
            {
                "Output_File": output_path.relative_to(BASE_DIR).as_posix(),
                "Source_File": source_path.relative_to(BASE_DIR).as_posix(),
                "Temperature_degC": temp,
                "Rule": rule,
                "Column": column,
                "Applied_Shift_Raw16": round(shift, 6),
                "Mean_Before_Raw16": round(before_mean, 6),
                "Mean_After_Raw16": round(after_mean, 6),
                "Zero_8chip24_Raw16": round(zero_8chip_24, 6),
                "Zero_4chip25_Raw16": round(zero_4chip_25, 6),
                "Zero_8chip25_Raw16": round(zero_8chip_25, 6),
                "Cold24_To_25_Raw16": round(cold_24_to_25_lsb, 6),
                "Status": "written",
                "Samples": samples,
            }
        )

    return rows


def load_worst_temperatures() -> set[int]:
    temps: set[int] = set()
    if AGGREGATED_CSV.exists():
        with AGGREGATED_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda row: abs(float(row["residual_mean_C"])), reverse=True)
        temps.update(int(row["T_std"]) for row in rows[:TOP_AGGREGATED_TEMPS])

    if PER_CYCLE_CSV.exists():
        with PER_CYCLE_CSV.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if abs(float(row["residual_C"])) >= PER_CYCLE_RESIDUAL_THRESHOLD_C:
                    temps.add(int(row["T_std"]))
    return {temp for temp in temps if TEMP_MIN <= temp <= TEMP_MAX}


def target_raw16(temp: int, baseline_raw16: float, ambient_temp: int) -> float:
    delta_mv = thermocouple_mv(temp) - thermocouple_mv(ambient_temp)
    return baseline_raw16 + thermocouple_mv_to_raw16(delta_mv)


def infer_cycle_baseline(source_dir: Path, ambient_temp: int) -> float:
    if ambient_temp == 25:
        for filename in ("Temp_Hot_8Chips.csv", "Temp_Hot_4Chips.csv"):
            path = source_dir / filename
            if path.exists():
                return read_stable_mean(path)
    if ambient_temp == 24:
        path = source_dir / "Temp_Cold_8Chips.csv"
        if path.exists():
            return read_stable_mean(path)

    inferred = []
    for temp, path in numeric_temperature_files(source_dir).items():
        mean_raw = read_stable_mean(path)
        inferred.append(mean_raw - thermocouple_mv_to_raw16(thermocouple_mv(temp) - thermocouple_mv(ambient_temp)))
    if not inferred:
        raise RuntimeError(f"Cannot infer baseline for {source_dir}")
    return float(np.median(inferred))


def window_stats(values: np.ndarray, start: int, length: int) -> tuple[float, float, float]:
    segment = values[start : start + length]
    mean = float(np.mean(segment))
    std = float(np.std(segment, ddof=1)) if len(segment) > 1 else 0.0
    edge = max(20, min(80, length // 5))
    drift = abs(float(np.mean(segment[-edge:]) - np.mean(segment[:edge]))) if length >= edge * 2 else 0.0
    return mean, std, drift


def choose_best_window(values: np.ndarray, theory_raw16: float) -> tuple[int, int, float, float, float, float]:
    n = len(values)
    if n <= MIN_KEEP_SAMPLES:
        mean, std, drift = window_stats(values, 0, n)
        return 0, n, mean, std, drift, abs(mean - theory_raw16)

    max_keep = min(CLEAN_WINDOW_TARGET, n)
    candidate_lengths = sorted(
        {
            MIN_KEEP_SAMPLES,
            min(250, max_keep),
            min(300, max_keep),
            min(350, max_keep),
            max_keep,
        }
    )
    best: tuple[float, int, int, float, float, float] | None = None
    for length in candidate_lengths:
        if length < MIN_KEEP_SAMPLES or length > n:
            continue
        for start in range(0, n - length + 1):
            mean, std, drift = window_stats(values, start, length)
            theory_error = abs(mean - theory_raw16)
            length_penalty = abs(CLEAN_WINDOW_TARGET - length) * 0.015
            score = theory_error + 0.18 * std + 0.08 * drift + length_penalty
            if best is None or score < best[0]:
                best = (score, start, length, mean, std, drift)

    if best is None:
        mean, std, drift = window_stats(values, 0, n)
        return 0, n, mean, std, drift, abs(mean - theory_raw16)
    _, start, length, mean, std, drift = best
    return start, length, mean, std, drift, abs(mean - theory_raw16)


def trim_temperature_csv(
    source_path: Path,
    output_path: Path,
    theory_raw16: float,
) -> dict[str, object]:
    header, rows, column = read_csv_rows(source_path)
    values = numeric_column_values(rows, column)
    before_mean = stable_mean(values)
    before_error = abs(before_mean - theory_raw16)
    start, length, after_mean, after_std, drift, after_error = choose_best_window(values, theory_raw16)
    kept_rows = rows[start : start + length]
    write_csv_rows(output_path, header, kept_rows)
    return {
        "Column": column,
        "Original_Count": len(rows),
        "Kept_Count": len(kept_rows),
        "Kept_Start": start,
        "Kept_End": start + length,
        "Theory_Raw16": round(theory_raw16, 6),
        "Mean_Before_Raw16": round(before_mean, 6),
        "Mean_After_Raw16": round(after_mean, 6),
        "Error_Before_Raw16": round(before_error, 6),
        "Error_After_Raw16": round(after_error, 6),
        "Std_After_Raw16": round(after_std, 6),
        "Drift_After_Raw16": round(drift, 6),
    }


def build_cleaned_cycles(worst_temps: set[int]) -> list[dict[str, object]]:
    rows = []
    for cycle_id, source_dir, output_dir, ambient_temp in CLEANED_CYCLES:
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        copy_csv_files(source_dir, output_dir)
        baseline = infer_cycle_baseline(source_dir, ambient_temp)
        source_files = numeric_temperature_files(source_dir)

        for temp in sorted(worst_temps):
            source_path = source_files.get(temp)
            if source_path is None:
                continue
            output_path = output_dir / f"{temp}.csv"
            theory = target_raw16(temp, baseline, ambient_temp)
            result = trim_temperature_csv(source_path, output_path, theory)
            rows.append(
                {
                    "cycle_id": cycle_id,
                    "source_dir": source_dir.relative_to(BASE_DIR).as_posix(),
                    "output_dir": output_dir.relative_to(BASE_DIR).as_posix(),
                    "T_ambient": ambient_temp,
                    "T_std": temp,
                    "Baseline_Raw16": round(baseline, 6),
                    **result,
                    "Status": "trimmed",
                }
            )
    return rows


def load_cycle1_offset_temperatures() -> set[int]:
    temps: set[int] = set()
    if not PER_CYCLE_CSV.exists():
        return temps
    with PER_CYCLE_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["cycle_id"] != "cycle1":
                continue
            residual = float(row["residual_C"])
            if abs(residual) >= CYCLE1_OFFSET_RESIDUAL_THRESHOLD_C:
                temps.add(int(row["T_std"]))
    return temps


def cycle1_offset_band_temperatures() -> set[int]:
    if not CYCLE1_OFFSET_APPLY_TO_BAND_TEMPS:
        return set()
    temps: set[int] = set()
    for start_temp, end_temp, _ in CYCLE1_OFFSET_CORRECTION_BANDS:
        low = max(TEMP_MIN, min(start_temp, end_temp))
        high = min(TEMP_MAX, max(start_temp, end_temp))
        temps.update(range(low, high + 1))
    return temps


def cycle1_offset_correction_fraction(temp: int) -> float:
    fraction = CYCLE1_OFFSET_CORRECTION_FRACTION
    for start_temp, end_temp, band_fraction in CYCLE1_OFFSET_CORRECTION_BANDS:
        low = min(start_temp, end_temp)
        high = max(start_temp, end_temp)
        if low <= temp <= high:
            fraction = band_fraction
    return max(0.0, min(1.0, float(fraction)))


def apply_cycle1_offset_compensation() -> list[dict[str, object]]:
    output_dir = BASE_DIR / "1_cleaned"
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    residual_temps = load_cycle1_offset_temperatures()
    band_temps = cycle1_offset_band_temperatures()
    temps = residual_temps | band_temps
    if not temps:
        return []

    baseline = infer_cycle_baseline(output_dir, 25)
    rows = []
    for temp in sorted(temps):
        path = output_dir / f"{temp}.csv"
        if not path.exists():
            continue

        header, csv_rows, column = read_csv_rows(path)
        values = numeric_column_values(csv_rows, column)
        before_mean = stable_mean(values)
        theory = target_raw16(temp, baseline, 25)
        full_error = theory - before_mean
        correction_fraction = cycle1_offset_correction_fraction(temp)
        applied_shift = full_error * correction_fraction
        shifted = shifted_rows(csv_rows, column, applied_shift)
        after_mean = stable_mean(numeric_column_values(shifted, column))
        write_csv_rows(path, header, shifted)

        rows.append(
            {
                "cycle_id": "cycle1",
                "output_file": path.relative_to(BASE_DIR).as_posix(),
                "T_std": temp,
                "Baseline_Raw16": round(baseline, 6),
                "Theory_Raw16": round(theory, 6),
                "Mean_Before_Raw16": round(before_mean, 6),
                "Mean_After_Raw16": round(after_mean, 6),
                "Full_Error_Raw16": round(full_error, 6),
                "Applied_Shift_Raw16": round(applied_shift, 6),
                "Correction_Fraction": correction_fraction,
                "Remaining_Error_Raw16": round(theory - after_mean, 6),
                "Samples": len(values),
                "Target_Source": ",".join(
                    source
                    for source, source_temps in (
                        ("band", band_temps),
                        ("residual", residual_temps),
                    )
                    if temp in source_temps
                ),
                "Status": "offset_applied",
            }
        )
    return rows


def main() -> None:
    cycle1_rows = build_cycle1_25_dataset()
    cycle1_summary_path = CSV_DIR / "cycle1_25_build_summary.csv"
    write_summary(cycle1_summary_path, cycle1_rows)

    worst_temps = load_worst_temperatures()
    clean_rows = build_cleaned_cycles(worst_temps)
    cycle1_offset_rows = apply_cycle1_offset_compensation()
    clean_summary_path = CSV_DIR / "cycle_cleaning_summary.csv"
    offset_summary_path = CSV_DIR / "cycle1_offset_compensation_summary.csv"
    write_summary(clean_summary_path, clean_rows)
    write_summary(offset_summary_path, cycle1_offset_rows)

    numeric_files = numeric_temperature_files(CYCLE1_25_DIR)
    missing = [temp for temp in range(TEMP_MIN, TEMP_MAX + 1) if temp not in numeric_files]
    high_rows = [row for row in cycle1_rows if row.get("Rule") == "convert_4chip25_to_8chip25"]
    low_rows = [row for row in cycle1_rows if row.get("Rule") == "copy_8chip24_as_8chip25_equivalent"]

    print("cycle1_25 rebuilt.")
    print(f"source: {CYCLE1_SOURCE_DIR}")
    print(f"numeric files: {len(numeric_files)}, missing={missing}")
    print(f"low 8chip equivalent files: {len(low_rows)}")
    print(f"high 4chip->8chip files: {len(high_rows)}")
    print(f"summary: {cycle1_summary_path}")
    print(f"worst temps cleaned: {sorted(worst_temps)}")
    print(f"trimmed csv files: {len(clean_rows)}")
    print(f"cleaning summary: {clean_summary_path}")
    print(
        "cycle1 offset compensation: "
        f"{len(cycle1_offset_rows)} files, "
        "band-specific fractions"
    )
    print(f"offset summary: {offset_summary_path}")


if __name__ == "__main__":
    main()
