#!/usr/bin/env python
# coding: utf-8

import csv
import re
from pathlib import Path

import numpy as np


# =========================
# Adjustable parameters
# =========================
# python workspace\test5\test5data_generater.py

BASE_DIR = Path(__file__).resolve().parent
SOURCE_CYCLE = 1
GENERATE_CYCLES = list(range(2, 11))

RANDOM_SEED = 20260506
STABLE_START_RATIO = 0.20
STABLE_END_RATIO = 0.80

# Detection-limit force points used by 05_Detection_Limit.py.
FORCE_CONFIG = {
    "0": 0.0,
    "1": 0.05,
    "2": 0.10,
    "3": 0.20,
    "4": 0.50,
}

# Cycle-to-cycle response variation.
GAIN_RANDOM_SIGMA = 0.012
GAIN_RANDOM_CLIP = 0.035
POINT_GAIN_SIGMA = 0.006

# Slow zero-load drift, relative to the cycle1 signal span.
DRIFT_PERCENT_FS_PER_CYCLE = 0.010
DRIFT_RANDOM_WALK_PERCENT_FS = 0.004
DRIFT_DIRECTION = 1.0

# Measurement noise. Percent values are relative to the cycle1 signal span.
MEASUREMENT_NOISE_PERCENT_FS = 0.030
POINT_COMMON_MODE_SIGMA_PERCENT_FS = 0.006
GROUND_NOISE_SIGMA_LSB = 0.35

RAW_COLUMN = "TactileRaw(16bit)"
GROUND_COLUMN = "GroundRef(16bit)"
TIME_COLUMN = "Time(ms)"

SUMMARY_CYCLE_DATA_NAME = "05_detection_limit_generated_cycle_data.csv"
SUMMARY_PARAMS_NAME = "05_detection_limit_generated_params.csv"


def numeric_folder_key(path: Path) -> tuple[int, str]:
    return (0, f"{int(path.name):08d}") if path.name.isdigit() else (1, path.name)


def timestamp_key(path: Path) -> str:
    match = re.search(r"(\d{8})_(\d{6})", path.name)
    if match:
        return match.group(1) + match.group(2)
    return path.name


def sorted_csv_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.csv"), key=timestamp_key)


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


def stable_std(values: np.ndarray) -> float:
    return float(np.std(stable_slice(values), ddof=1))


def load_cycle_templates(cycle_dir: Path) -> dict[str, dict[str, object]]:
    templates = {}
    force_dirs = sorted(
        [path for path in cycle_dir.iterdir() if path.is_dir()],
        key=numeric_folder_key,
    )
    if not force_dirs:
        raise FileNotFoundError(f"{cycle_dir} has no force-point folders.")

    for force_dir in force_dirs:
        files = sorted_csv_files(force_dir)
        if len(files) < 1:
            raise FileNotFoundError(f"{force_dir} needs one csv file.")
        if len(files) > 1:
            print(f"Warning: {force_dir} has {len(files)} csv files; using {files[0].name}.")

        header, times, raw, ground = read_sensor_csv(files[0])
        templates[force_dir.name] = {
            "file": files[0],
            "header": header,
            "time": times,
            "raw": raw,
            "ground": ground,
            "mean": stable_mean(raw),
            "std": stable_std(raw),
        }
    return templates


def signal_span(templates: dict[str, dict[str, object]]) -> float:
    means = np.array([data["mean"] for data in templates.values()], dtype=float)
    span = float(np.max(means) - np.min(means))
    if span <= 0:
        raise ValueError("Cycle1 signal span must be greater than zero.")
    return span


def synthesize_trace(
    source_raw: np.ndarray,
    source_mean: float,
    target_mean: float,
    full_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise_sigma = full_scale * MEASUREMENT_NOISE_PERCENT_FS / 100.0
    centered_source = source_raw - source_mean
    white_noise = rng.normal(0.0, noise_sigma, size=source_raw.shape)
    synthesized = target_mean + centered_source + white_noise
    return np.rint(synthesized).astype(int)


def synthesize_ground(source_ground: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ground_noise = rng.normal(0.0, GROUND_NOISE_SIGMA_LSB, size=source_ground.shape)
    return np.rint(source_ground + ground_noise).astype(int)


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_cycle_rows(
    cycle_name: str,
    means: dict[str, float],
    stds: dict[str, float],
    zero_mean: float,
) -> list[dict[str, object]]:
    rows = []
    for folder in sorted(means.keys(), key=lambda name: numeric_folder_key(Path(name))):
        force_n = FORCE_CONFIG.get(folder, float(folder) if folder.isdigit() else np.nan)
        rows.append(
            {
                "Cycle": cycle_name,
                "Folder": folder,
                "Force_N": force_n,
                "Force_mN": force_n * 1000 if not np.isnan(force_n) else np.nan,
                "Mean_LSB": means[folder],
                "Std_LSB": stds[folder],
                "Zero_Mean_LSB": zero_mean,
                "Delta_S_LSB": means[folder] - zero_mean,
                "Abs_Delta_S_LSB": abs(means[folder] - zero_mean),
            }
        )
    return rows


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    source_cycle_dir = BASE_DIR / f"cycle{SOURCE_CYCLE}"
    templates = load_cycle_templates(source_cycle_dir)

    if "0" not in templates:
        raise FileNotFoundError(f"{source_cycle_dir} must contain folder 0 as zero-load baseline.")

    full_scale = signal_span(templates)
    source_zero_mean = float(templates["0"]["mean"])
    source_means = {folder: float(data["mean"]) for folder, data in templates.items()}
    source_stds = {folder: float(data["std"]) for folder, data in templates.items()}

    all_cycle_rows = make_cycle_rows(
        f"cycle{SOURCE_CYCLE}",
        source_means,
        source_stds,
        source_zero_mean,
    )
    drift_random_walk = 0.0

    for cycle in GENERATE_CYCLES:
        cycle_index = cycle - SOURCE_CYCLE
        gain = 1.0 + float(
            np.clip(rng.normal(0.0, GAIN_RANDOM_SIGMA), -GAIN_RANDOM_CLIP, GAIN_RANDOM_CLIP)
        )

        drift_random_walk += rng.normal(0.0, DRIFT_RANDOM_WALK_PERCENT_FS / 100.0 * full_scale)
        drift_lsb = (
            DRIFT_DIRECTION * cycle_index * DRIFT_PERCENT_FS_PER_CYCLE / 100.0 * full_scale
            + drift_random_walk
        )

        cycle_dir = BASE_DIR / f"cycle{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        generated_means = {}
        generated_stds = {}

        for folder, data in templates.items():
            force_dir = cycle_dir / folder
            force_dir.mkdir(parents=True, exist_ok=True)

            point_gain = 1.0 + rng.normal(0.0, POINT_GAIN_SIGMA)
            common_mode = rng.normal(0.0, POINT_COMMON_MODE_SIGMA_PERCENT_FS / 100.0 * full_scale)
            source_signal = float(data["mean"]) - source_zero_mean
            target_mean = source_zero_mean + drift_lsb + common_mode + source_signal * gain * point_gain

            raw = synthesize_trace(
                data["raw"],
                float(data["mean"]),
                target_mean,
                full_scale,
                rng,
            )
            ground = synthesize_ground(data["ground"], rng)

            write_sensor_csv(
                force_dir / data["file"].name,
                data["header"],
                data["time"],
                raw,
                ground,
            )

            generated_means[folder] = stable_mean(raw)
            generated_stds[folder] = stable_std(raw)

        generated_zero_mean = generated_means["0"]
        all_cycle_rows.extend(
            make_cycle_rows(
                f"cycle{cycle}",
                generated_means,
                generated_stds,
                generated_zero_mean,
            )
        )

    write_dict_csv(BASE_DIR / SUMMARY_CYCLE_DATA_NAME, all_cycle_rows)

    nonzero_rows = [row for row in all_cycle_rows if row["Folder"] != "0"]
    zero_rows = [row for row in all_cycle_rows if row["Folder"] == "0"]
    params = {
        "Source_Cycle": f"cycle{SOURCE_CYCLE}",
        "Generated_Cycles": ", ".join(f"cycle{cycle}" for cycle in GENERATE_CYCLES),
        "Num_Force_Points": len(templates),
        "Signal_Span_LSB": full_scale,
        "Zero_Mean_Avg_LSB": float(np.mean([row["Mean_LSB"] for row in zero_rows])),
        "Zero_Std_Avg_LSB": float(np.mean([row["Std_LSB"] for row in zero_rows])),
        "Nonzero_Abs_Delta_Avg_LSB": float(np.mean([row["Abs_Delta_S_LSB"] for row in nonzero_rows])),
    }
    write_dict_csv(BASE_DIR / SUMMARY_PARAMS_NAME, [params])

    generated_cycle_names = ", ".join(f"cycle{cycle}" for cycle in GENERATE_CYCLES)
    print(f"Generated {generated_cycle_names} from cycle{SOURCE_CYCLE} under: {BASE_DIR}")
    print(f"Force folders: {', '.join(templates.keys())}")
    print(f"Signal span: {full_scale:.3f} LSB")


if __name__ == "__main__":
    main()
