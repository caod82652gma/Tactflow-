#!/usr/bin/env python
# coding: utf-8

import csv
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# =========================
# Adjustable parameters
# =========================
# python workspace\test4\test4data_generater.py

BASE_DIR = Path(__file__).resolve().parent
SOURCE_CYCLE = 1
GENERATE_CYCLES = range(2, 9)
FORCE_FOLDERS = [str(i) for i in range(11)]

RANDOM_SEED = 20260506
STABLE_START_RATIO = 0.20
STABLE_END_RATIO = 0.80

# Break-in: first-cycle hmax is usually 1.5-2.0x steady hmax.
INITIAL_TO_STEADY_RATIO = 1.48
BREAK_IN_TAU_CYCLES = 2.0

# Cycle-to-cycle random fluctuation for hmax(k): epsilon ~ N(0, sigma).
HMAX_RANDOM_SIGMA = 0.045
HMAX_RANDOM_CLIP = 0.12

# Measurement noise and quantization. Percent values are relative to full scale.
MEASUREMENT_NOISE_PERCENT_FS = 0.035
ADC_BITS = 16

# Slow drift, per cycle, relative to full scale.
DRIFT_PERCENT_FS_PER_CYCLE = 0.010
DRIFT_RANDOM_WALK_PERCENT_FS = 0.004
DRIFT_DIRECTION = 1.0

# Extra tiny point-level variation so every force point does not shrink identically.
POINT_HYSTERESIS_SIGMA = 0.020
POINT_COMMON_MODE_SIGMA_PERCENT_FS = 0.006

OUTPUT_TIMESTAMP_START = "20260506_010000"
WRITE_CYCLE1_SUMMARY = True


RAW_COLUMN = "TactileRaw(16bit)"
GROUND_COLUMN = "GroundRef(16bit)"
TIME_COLUMN = "Time(ms)"
SUMMARY_DATA_NAME = "04_hysteresis_data.csv"
SUMMARY_PARAMS_NAME = "04_hysteresis_params.csv"
ALL_CYCLES_SUMMARY_NAME = "all_cycles_hysteresis_summary.csv"


def timestamp_key(path: Path) -> str:
    match = re.search(r"(\d{8})_(\d{6})", path.name)
    if match:
        return match.group(1) + match.group(2)
    return path.name


def sorted_csv_files(folder: Path) -> list[Path]:
    data_files = [
        path
        for path in folder.glob("*.csv")
        if path.name not in {SUMMARY_DATA_NAME, SUMMARY_PARAMS_NAME}
    ]
    return sorted(data_files, key=timestamp_key)


def read_sensor_csv(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        times, raw_values, ground_values = [], [], []
        for row in reader:
            times.append(float(row[TIME_COLUMN]))
            raw_values.append(float(row[RAW_COLUMN]))
            ground_values.append(float(row[GROUND_COLUMN]))
    return header, np.array(times), np.array(raw_values), np.array(ground_values)


def write_sensor_csv(
    path: Path,
    header: list[str],
    times: np.ndarray,
    raw_values: np.ndarray,
    ground_values: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for t, raw, ground in zip(times, raw_values, ground_values):
            writer.writerow(
                {
                    TIME_COLUMN: f"{t:.2f}",
                    RAW_COLUMN: f"{int(raw)}",
                    GROUND_COLUMN: f"{int(ground)}",
                }
            )


def stable_slice(values: np.ndarray) -> np.ndarray:
    start = int(len(values) * STABLE_START_RATIO)
    end = int(len(values) * STABLE_END_RATIO)
    return values[start:end]


def stable_mean(values: np.ndarray) -> float:
    return float(np.mean(stable_slice(values)))


def load_cycle_templates(cycle_dir: Path) -> dict[str, dict[str, object]]:
    templates = {}
    for force_folder in FORCE_FOLDERS:
        folder = cycle_dir / force_folder
        files = sorted_csv_files(folder)
        if len(files) < 2:
            raise FileNotFoundError(f"{folder} needs two csv files for loading/unloading.")

        header_l, time_l, raw_l, ground_l = read_sensor_csv(files[0])
        header_u, time_u, raw_u, ground_u = read_sensor_csv(files[1])
        if header_l != header_u:
            raise ValueError(f"CSV headers differ in {folder}.")

        templates[force_folder] = {
            "force": float(force_folder),
            "loading_file": files[0],
            "unloading_file": files[1],
            "header": header_l,
            "loading_time": time_l,
            "unloading_time": time_u,
            "loading_raw": raw_l,
            "unloading_raw": raw_u,
            "loading_ground": ground_l,
            "unloading_ground": ground_u,
            "loading_mean": stable_mean(raw_l),
            "unloading_mean": stable_mean(raw_u),
        }
    return templates


def cycle_summary_from_means(means: dict[str, tuple[float, float]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    forces = np.array([float(k) for k in means.keys()])
    loading = np.array([means[k][0] for k in means.keys()])
    unloading = np.array([means[k][1] for k in means.keys()])
    full_scale = float(np.max(loading) - np.min(loading))

    rows = []
    for force, load_value, unload_value in zip(forces, loading, unloading):
        error_lsb = float(unload_value - load_value)
        error_percent = abs(error_lsb) / full_scale * 100.0
        rows.append(
            {
                "Force_N": float(force),
                "Loading_LSB": float(load_value),
                "Unloading_LSB": float(unload_value),
                "Error_LSB": error_lsb,
                "Error_percentFS": float(error_percent),
            }
        )

    errors = [row["Error_percentFS"] for row in rows]
    params = {
        "Full_Scale_LSB": full_scale,
        "Max_Hysteresis_percentFS": float(max(errors)),
        "Avg_Hysteresis_percentFS": float(np.mean(errors)),
        "Num_Test_Points": len(rows),
    }
    return rows, params


def write_dict_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_output_name(source_name: str, timestamp: datetime) -> str:
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    if re.search(r"\d{8}_\d{6}", source_name):
        return re.sub(r"\d{8}_\d{6}", stamp, source_name, count=1)
    return source_name.replace(".csv", f"_{stamp}.csv")


def synthesize_trace(
    source_raw: np.ndarray,
    source_mean: float,
    target_mean: float,
    full_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    quant_lsb = full_scale / (2**ADC_BITS)
    noise_sigma = full_scale * MEASUREMENT_NOISE_PERCENT_FS / 100.0
    centered_source = source_raw - source_mean
    white_noise = rng.normal(0.0, noise_sigma, size=source_raw.shape)
    synthesized = target_mean + centered_source + white_noise
    synthesized = np.round(synthesized / quant_lsb) * quant_lsb
    return np.rint(synthesized).astype(int)


def synthesize_ground(source_ground: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ground_noise = rng.normal(0.0, 0.35, size=source_ground.shape)
    return np.rint(source_ground + ground_noise).astype(int)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    source_cycle_dir = BASE_DIR / f"cycle{SOURCE_CYCLE}"
    templates = load_cycle_templates(source_cycle_dir)

    source_means = {
        folder: (data["loading_mean"], data["unloading_mean"])
        for folder, data in templates.items()
    }
    cycle1_rows, cycle1_params = cycle_summary_from_means(source_means)
    full_scale = cycle1_params["Full_Scale_LSB"]
    initial_hmax = cycle1_params["Max_Hysteresis_percentFS"]
    steady_hmax = initial_hmax / INITIAL_TO_STEADY_RATIO

    source_errors = {
        folder: data["unloading_mean"] - data["loading_mean"]
        for folder, data in templates.items()
    }
    max_source_error = max(abs(value) for value in source_errors.values())

    if WRITE_CYCLE1_SUMMARY:
        write_dict_csv(source_cycle_dir / SUMMARY_DATA_NAME, cycle1_rows)
        write_dict_csv(source_cycle_dir / SUMMARY_PARAMS_NAME, [cycle1_params])

    all_cycle_rows = [
        {
            "Cycle": SOURCE_CYCLE,
            "Target_Hmax_percentFS": initial_hmax,
            **cycle1_params,
        }
    ]

    timestamp_base = datetime.strptime(OUTPUT_TIMESTAMP_START, "%Y%m%d_%H%M%S")
    drift_random_walk = 0.0

    for cycle in GENERATE_CYCLES:
        cycle_index = cycle - SOURCE_CYCLE
        deterministic_hmax = steady_hmax + (initial_hmax - steady_hmax) * math.exp(
            -cycle_index / BREAK_IN_TAU_CYCLES
        )
        epsilon = float(np.clip(rng.normal(0.0, HMAX_RANDOM_SIGMA), -HMAX_RANDOM_CLIP, HMAX_RANDOM_CLIP))
        target_hmax = deterministic_hmax * (1.0 + epsilon)
        target_max_error_lsb = target_hmax / 100.0 * full_scale

        drift_random_walk += rng.normal(0.0, DRIFT_RANDOM_WALK_PERCENT_FS / 100.0 * full_scale)
        drift_lsb = (
            DRIFT_DIRECTION * cycle_index * DRIFT_PERCENT_FS_PER_CYCLE / 100.0 * full_scale
            + drift_random_walk
        )

        cycle_dir = BASE_DIR / f"cycle{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        generated_means = {}

        for force_offset, folder in enumerate(FORCE_FOLDERS):
            data = templates[folder]
            force_dir = cycle_dir / folder
            force_dir.mkdir(parents=True, exist_ok=True)

            point_factor = 1.0 + rng.normal(0.0, POINT_HYSTERESIS_SIGMA)
            target_error = source_errors[folder] / max_source_error * target_max_error_lsb * point_factor

            common_mode = rng.normal(0.0, POINT_COMMON_MODE_SIGMA_PERCENT_FS / 100.0 * full_scale)
            source_mid = 0.5 * (data["loading_mean"] + data["unloading_mean"])
            target_mid = source_mid + drift_lsb + common_mode
            target_loading_mean = target_mid - 0.5 * target_error
            target_unloading_mean = target_mid + 0.5 * target_error

            loading_raw = synthesize_trace(
                data["loading_raw"],
                data["loading_mean"],
                target_loading_mean,
                full_scale,
                rng,
            )
            unloading_raw = synthesize_trace(
                data["unloading_raw"],
                data["unloading_mean"],
                target_unloading_mean,
                full_scale,
                rng,
            )
            loading_ground = synthesize_ground(data["loading_ground"], rng)
            unloading_ground = synthesize_ground(data["unloading_ground"], rng)

            timestamp = timestamp_base + timedelta(minutes=(cycle - 2) * 30 + force_offset * 2)
            loading_name = make_output_name(data["loading_file"].name, timestamp)
            unloading_name = make_output_name(data["unloading_file"].name, timestamp + timedelta(seconds=40))

            write_sensor_csv(
                force_dir / loading_name,
                data["header"],
                data["loading_time"],
                loading_raw,
                loading_ground,
            )
            write_sensor_csv(
                force_dir / unloading_name,
                data["header"],
                data["unloading_time"],
                unloading_raw,
                unloading_ground,
            )

            generated_means[folder] = (stable_mean(loading_raw), stable_mean(unloading_raw))

        cycle_rows, cycle_params = cycle_summary_from_means(generated_means)
        write_dict_csv(cycle_dir / SUMMARY_DATA_NAME, cycle_rows)
        write_dict_csv(cycle_dir / SUMMARY_PARAMS_NAME, [cycle_params])
        all_cycle_rows.append(
            {
                "Cycle": cycle,
                "Target_Hmax_percentFS": target_hmax,
                **cycle_params,
            }
        )

    write_dict_csv(BASE_DIR / ALL_CYCLES_SUMMARY_NAME, all_cycle_rows)
    print(f"Generated cycle2-cycle8 under: {BASE_DIR}")
    print(f"Cycle1 hmax: {initial_hmax:.4f}%FS, steady hmax: {steady_hmax:.4f}%FS")
    for row in all_cycle_rows:
        print(
            f"cycle{int(row['Cycle'])}: max={row['Max_Hysteresis_percentFS']:.4f}%FS, "
            f"avg={row['Avg_Hysteresis_percentFS']:.4f}%FS"
        )


if __name__ == "__main__":
    main()
