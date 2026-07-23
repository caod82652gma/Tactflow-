#!/usr/bin/env python
# coding: utf-8

import ast
import csv
import re
import shutil
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_SCRIPT = BASE_DIR / "06_Temperature_Calibration.py"
SUMMARY_DIR = BASE_DIR / "csv"

TEMP_MIN = 5
TEMP_MAX = 88
DEVICE_SPLIT_TEMP = 25

V_REF = 5.0
ADC_FULL_SCALE = 2**15
AMPLIFIER_GAIN = 275
NOISE_SIGMA_LSB = 5.6118
GROUND_NOISE_SIGMA_LSB = 1.16
WINDOW_TARGET_SAMPLES = 400
RANDOM_SEED = 20260508
SEGMENT_OFFSET_MIN_LEN = 4
SEGMENT_OFFSET_MAX_LEN = 12

CYCLE1_HOT_AMBIENT_TEMP = 25
CYCLE1_COLD_AMBIENT_TEMP = 24
CYCLE2_AMBIENT_TEMP = 25
CYCLE3_AMBIENT_TEMP = 24
CYCLE1_ORIGIN_DIR = BASE_DIR / "cycle1_origin"
CYCLE1_DIR = BASE_DIR / "cycle1"
CYCLE1_8CHIP_DIR = BASE_DIR / "cycle1_8chip"

# Per-generator configuration.  Keep these separate so each synthetic or
# completion path can be tuned without changing the others.
CYCLE1_MISSING_POINT_OFFSET_SIGMA_LSB = 3.0
CYCLE1_MISSING_OFFSET_SCALE_LSB = 2.8
CYCLE1_MISSING_RAW_NOISE_SIGMA_LSB = NOISE_SIGMA_LSB
CYCLE1_MISSING_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE1_MISSING_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN
CYCLE1_MISSING_COLD_QUANTIZE_16 = False
CYCLE1_MISSING_HOT_QUANTIZE_16 = True
CYCLE1_MISSING_KEEP_POINT_OFFSET_IN_MEAN = True
CYCLE1_MISSING_GROUND_OFFSET_SCALE_LSB = 0.35
CYCLE1_MISSING_GROUND_NOISE_SIGMA_LSB = GROUND_NOISE_SIGMA_LSB
CYCLE1_MISSING_GROUND_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE1_MISSING_GROUND_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN

CYCLE2_MISSING_POINT_OFFSET_SIGMA_LSB = 3.8
CYCLE2_MISSING_OFFSET_SCALE_LSB = 2.8
CYCLE2_MISSING_RAW_NOISE_SIGMA_LSB = NOISE_SIGMA_LSB
CYCLE2_MISSING_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE2_MISSING_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN
CYCLE2_MISSING_QUANTIZE_16 = False
CYCLE2_MISSING_KEEP_POINT_OFFSET_IN_MEAN = True
CYCLE2_MISSING_GROUND_OFFSET_SCALE_LSB = 0.35
CYCLE2_MISSING_GROUND_NOISE_SIGMA_LSB = GROUND_NOISE_SIGMA_LSB
CYCLE2_MISSING_GROUND_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE2_MISSING_GROUND_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN

CYCLE3_VIRTUAL_BASELINE_SIGMA_LSB = 14.0
CYCLE3_VIRTUAL_POINT_OFFSET_SIGMA_LSB = 4.2
CYCLE3_VIRTUAL_OFFSET_SCALE_LSB = 3.2
CYCLE3_VIRTUAL_RAW_NOISE_SIGMA_LSB = NOISE_SIGMA_LSB
CYCLE3_VIRTUAL_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE3_VIRTUAL_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN
CYCLE3_VIRTUAL_QUANTIZE_16 = False
CYCLE3_VIRTUAL_KEEP_POINT_OFFSET_IN_MEAN = False
CYCLE3_VIRTUAL_GROUND_OFFSET_SCALE_LSB = 0.35
CYCLE3_VIRTUAL_GROUND_NOISE_SIGMA_LSB = GROUND_NOISE_SIGMA_LSB
CYCLE3_VIRTUAL_GROUND_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE3_VIRTUAL_GROUND_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN

CYCLE1_8CHIP_GROUND_OFFSET_SCALE_LSB = 0.35
CYCLE1_8CHIP_GROUND_NOISE_SIGMA_LSB = GROUND_NOISE_SIGMA_LSB
CYCLE1_8CHIP_GROUND_SEGMENT_OFFSET_MIN_LEN = SEGMENT_OFFSET_MIN_LEN
CYCLE1_8CHIP_GROUND_SEGMENT_OFFSET_MAX_LEN = SEGMENT_OFFSET_MAX_LEN

# Cycle3模拟"记录温度 != 实际温度"的场景。使用文件温度填充这些区间。
# 每个区间格式为：(起始文件温度, 结束文件温度, 起始偏差值(℃), 结束偏差值(℃))
# 例如：40.csv到50.csv的文件，实际生成温度约为40.08℃到50.15℃，
# 但仍然存储为40.csv到50.csv的文件名

# 是否启用Cycle3的温度偏差模拟功能
CYCLE3_ENABLE_TEMPERATURE_BIAS = True

# 正偏差区间配置（实际温度 > 记录温度）
# 格式：(起始文件温度, 结束文件温度, 起始偏差值, 结束偏差值)
# 在40℃到50℃区间：起始偏差+0.08℃，结束偏差+0.15℃（偏差逐渐增大）
# 在66℃到76℃区间：起始偏差+0.05℃，结束偏差+0.11℃（偏差逐渐增大）
CYCLE3_POSITIVE_BIAS_BANDS = [
    (40, 50, 0.04, 0.14),
    (66, 76, 0.15, 0.27),
    (77, 88, 0.27, 0.58),
]

# 零偏差区间配置（实际温度 = 记录温度）
# 在24℃到32℃区间：偏差始终为0℃，无偏差
CYCLE3_ZERO_BIAS_BANDS = [
    (24, 32, 0.0, 0.0),
]

# 负偏差区间配置（实际温度 < 记录温度）
# 在10℃到18℃区间：起始偏差-0.12℃，结束偏差-0.06℃（偏差逐渐减小/绝对值变小）
# 在54℃到61℃区间：起始偏差-0.07℃，结束偏差-0.13℃（偏差逐渐增大/绝对值变大）
CYCLE3_NEGATIVE_BIAS_BANDS = [
    (10, 18, -0.12, -0.06),
    (54, 61, -0.07, -0.13),
    (5, 10, -0.45, -0.13),   
]

# 温度偏差的随机游走标准差（单位：℃）
# 模拟实际测量中温度的随机波动，每步变化的标准差为0.008℃
CYCLE3_BIAS_RANDOM_WALK_STEP_SIGMA_LSB = 5.6

# 温度偏差随机游走的裁剪范围（单位：℃）
# 限制单步偏差变化不超过±0.035℃，防止突变过大
CYCLE3_BIAS_RANDOM_WALK_PP_LSB = 45.0

# 零偏差区间的随机游走缩放因子
# 设置为0.0表示在零偏差区间完全禁用随机游走（保持严格为零偏差）
# 若设为非零值，则允许一定程度的随机波动
CYCLE3_ZERO_BIAS_RANDOM_WALK_SCALE = 0.0
CYCLE3_SAMPLE_DRIFT_PP_LSB = 60.0
CYCLE3_NOISE_EQUIVALENT_BANDS = [
    (24, 32, 1.0),
    (40, 50, 1.0),
]
CYCLE3_NOISE_AMPLIFIED_BANDS = [
    (54, 61, 1.45),
    (66, 76, 1.80),
]
CYCLE3_DEFAULT_NOISE_SCALE = 1.0
CYCLE3_SAMPLE_DRIFT_ZERO_TEMP = 25
CYCLE3_SAMPLE_DRIFT_FULL_TEMP = 70
CYCLE3_SAMPLE_DRIFT_RANDOM_WEIGHT = 0.35

def load_t_type_table() -> dict[int, float]:
    text = CALIBRATION_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"T_TYPE_TABLE\s*=\s*(\{.*?\n\})", text, flags=re.S)
    if not match:
        raise ValueError(f"Cannot find T_TYPE_TABLE in {CALIBRATION_SCRIPT}")
    table = ast.literal_eval(match.group(1))
    return {int(key): float(value) for key, value in table.items()}


T_TYPE_TABLE = load_t_type_table()


def thermocouple_mv(temp_deg_c: int) -> float:
    if temp_deg_c not in T_TYPE_TABLE:
        raise ValueError(f"Temperature {temp_deg_c} degC is outside the T-type table.")
    return T_TYPE_TABLE[temp_deg_c]


def thermocouple_mv_interpolated(temp_deg_c: float) -> float:
    table_temps = sorted(T_TYPE_TABLE)
    min_temp = table_temps[0]
    max_temp = table_temps[-1]
    if temp_deg_c < min_temp or temp_deg_c > max_temp:
        raise ValueError(f"Temperature {temp_deg_c} degC is outside the T-type table.")
    if float(temp_deg_c).is_integer():
        return thermocouple_mv(int(temp_deg_c))
    return float(np.interp(temp_deg_c, table_temps, [T_TYPE_TABLE[temp] for temp in table_temps]))


def raw16_per_deg_c(temp_deg_c: float) -> float:
    table_min = min(T_TYPE_TABLE)
    table_max = max(T_TYPE_TABLE)
    low = max(table_min, temp_deg_c - 0.5)
    high = min(table_max, temp_deg_c + 0.5)
    if high == low:
        high = min(table_max, low + 1.0)
    delta_mv = thermocouple_mv_interpolated(high) - thermocouple_mv_interpolated(low)
    delta_raw = thermocouple_mv_to_raw16(delta_mv)
    return abs(delta_raw / (high - low))


def lsb_to_temp_c(lsb: float, temp_deg_c: float) -> float:
    slope = raw16_per_deg_c(temp_deg_c)
    if slope <= 0:
        raise ValueError(f"Invalid thermocouple slope at {temp_deg_c} degC")
    return lsb / slope


def visual_mv_to_raw16(visual_mv: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(visual_mv) / (V_REF * 1000.0) * ADC_FULL_SCALE


def raw16_to_visual_mv(raw16: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(raw16) / ADC_FULL_SCALE * V_REF * 1000.0


def thermocouple_mv_to_raw16(delta_mv: float) -> float:
    return float(visual_mv_to_raw16(delta_mv * AMPLIFIER_GAIN))


def read_numeric_column(path: Path) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for candidate in ("TemperatureRaw16", "Raw16", "TemperatureRaw(16bit)", "RawValue"):
            if candidate in headers:
                column = candidate
                break
        else:
            data_columns = [name for name in headers if name and name.lower() != "index"]
            if not data_columns:
                raise ValueError(f"{path} has no numeric data column.")
            column = data_columns[0]
        return np.array([float(row[column]) for row in reader if row.get(column) != ""], dtype=float)


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


def stable_mean(path: Path) -> float:
    values = read_numeric_column(path)
    window = min(WINDOW_TARGET_SAMPLES, len(values))
    means, stds = rolling_window_stats(values, window)
    return float(means[int(np.argmin(stds))])


def find_cycle_or_root_file(cycle_dir: Path, filename: str) -> Path:
    candidates = [cycle_dir / filename, BASE_DIR / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def sync_cycle1_from_origin() -> list[dict[str, object]]:
    if not CYCLE1_ORIGIN_DIR.exists():
        raise FileNotFoundError(CYCLE1_ORIGIN_DIR)
    CYCLE1_DIR.mkdir(parents=True, exist_ok=True)
    for path in CYCLE1_DIR.glob("*.csv"):
        path.unlink()

    rows = []
    for source_path in sorted(CYCLE1_ORIGIN_DIR.glob("*.csv")):
        output_path = CYCLE1_DIR / source_path.name
        shutil.copy2(source_path, output_path)
        rows.append(
            {
                "Cycle": "cycle1",
                "Generation_Mode": "copy_from_origin",
                "Source_File": source_path.relative_to(BASE_DIR).as_posix(),
                "Output_File": output_path.relative_to(BASE_DIR).as_posix(),
            }
        )
    return rows


def temperature_files(cycle_dir: Path) -> dict[int, Path]:
    files = {}
    if not cycle_dir.exists():
        return files
    for path in cycle_dir.glob("*.csv"):
        if path.stem.isdigit():
            files[int(path.stem)] = path
    return files


def count_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return max(sum(1 for _ in f) - 1, 0)


def nearest_lengths(temp: int, source_maps: list[dict[int, Path]], max_distance: int = 5) -> list[tuple[int, int]]:
    lengths = []
    for files in source_maps:
        for other_temp, path in files.items():
            distance = abs(other_temp - temp)
            if distance <= max_distance:
                lengths.append((distance, count_rows(path)))
    return lengths


def inferred_sample_count(temp: int, source_maps: list[dict[int, Path]]) -> int:
    lengths = nearest_lengths(temp, source_maps, max_distance=2)
    if not lengths:
        lengths = nearest_lengths(temp, source_maps, max_distance=5)
    if not lengths:
        return 600

    weights = np.array([1.0 / (distance + 1.0) for distance, _ in lengths])
    values = np.array([length for _, length in lengths], dtype=float)
    estimate = int(round(float(np.average(values, weights=weights))))
    return max(WINDOW_TARGET_SAMPLES, estimate)


def device_for_cycle1(temp: int) -> str:
    return "cold_device_1" if temp < DEVICE_SPLIT_TEMP else "hot_device_2"


def cycle1_ambient_temp(temp: int) -> int:
    return CYCLE1_COLD_AMBIENT_TEMP if temp < DEVICE_SPLIT_TEMP else CYCLE1_HOT_AMBIENT_TEMP


def target_raw16(temp: float, baseline_raw: float, ambient_temp: int) -> float:
    delta_mv = thermocouple_mv_interpolated(float(temp)) - thermocouple_mv(ambient_temp)
    return baseline_raw + thermocouple_mv_to_raw16(delta_mv)


def infer_baseline_from_cycle(cycle_dir: Path, ambient_temp: int) -> float:
    inferred = []
    for temp, path in temperature_files(cycle_dir).items():
        mean_raw = stable_mean(path)
        delta_mv = thermocouple_mv(temp) - thermocouple_mv(ambient_temp)
        inferred.append(mean_raw - thermocouple_mv_to_raw16(delta_mv))
    if not inferred:
        raise ValueError(f"Cannot infer baseline from empty cycle: {cycle_dir}")
    return float(np.median(inferred))


def segmented_random_offset(
    n: int,
    rng: np.random.Generator,
    scale_lsb: float,
    min_len: int = SEGMENT_OFFSET_MIN_LEN,
    max_len: int = SEGMENT_OFFSET_MAX_LEN,
) -> np.ndarray:
    offset = np.empty(n, dtype=float)
    index = 0
    current = rng.normal(0.0, scale_lsb)
    while index < n:
        run_length = int(rng.integers(min_len, max_len + 1))
        run_end = min(n, index + run_length)
        current = 0.72 * current + rng.normal(0.0, scale_lsb * 0.55)
        offset[index:run_end] = current
        index = run_end

    knots = max(4, int(np.ceil(n / 180)))
    drift = np.interp(
        np.arange(n),
        np.linspace(0, n - 1, knots),
        rng.normal(0.0, scale_lsb * 0.35, size=knots),
    )
    offset += drift
    offset -= float(np.mean(offset))
    return offset


def generate_raw_series(
    target_mean_raw: float,
    n_samples: int,
    rng: np.random.Generator,
    quantize_16: bool = False,
    point_offset_sigma: float = 3.0,
    offset_scale_lsb: float = 2.8,
    raw_noise_sigma_lsb: float = NOISE_SIGMA_LSB,
    segment_offset_min_len: int = SEGMENT_OFFSET_MIN_LEN,
    segment_offset_max_len: int = SEGMENT_OFFSET_MAX_LEN,
    extra_drift_lsb: np.ndarray | None = None,
    keep_point_offset_in_mean: bool = True,
) -> np.ndarray:
    point_offset = rng.normal(0.0, point_offset_sigma)
    segmented_offset = segmented_random_offset(
        n_samples,
        rng,
        scale_lsb=offset_scale_lsb,
        min_len=segment_offset_min_len,
        max_len=segment_offset_max_len,
    )
    noise = rng.normal(0.0, raw_noise_sigma_lsb, size=n_samples)
    raw = target_mean_raw + point_offset + segmented_offset + noise
    if extra_drift_lsb is not None:
        if len(extra_drift_lsb) != n_samples:
            raise ValueError("extra_drift_lsb length must match n_samples")
        raw += extra_drift_lsb
    mean_target = target_mean_raw + point_offset if keep_point_offset_in_mean else target_mean_raw
    raw += mean_target - float(np.mean(raw))
    if quantize_16:
        return np.rint(raw / 16.0).astype(int) * 16
    return np.rint(raw).astype(int)


def generate_ground_series(
    n_samples: int,
    rng: np.random.Generator,
    offset_scale_lsb: float = 0.35,
    noise_sigma_lsb: float = GROUND_NOISE_SIGMA_LSB,
    segment_offset_min_len: int = SEGMENT_OFFSET_MIN_LEN,
    segment_offset_max_len: int = SEGMENT_OFFSET_MAX_LEN,
) -> np.ndarray:
    offset = segmented_random_offset(
        n_samples,
        rng,
        scale_lsb=offset_scale_lsb,
        min_len=segment_offset_min_len,
        max_len=segment_offset_max_len,
    )
    noise = rng.normal(0.0, noise_sigma_lsb, size=n_samples)
    return np.rint(offset + noise).astype(int)


def write_temperature_ground_csv(path: Path, raw: np.ndarray, ground: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "TemperatureRaw16", "VGNDRaw16"])
        for index, (raw_value, ground_value) in enumerate(zip(raw, ground)):
            writer.writerow([index, int(raw_value), int(ground_value)])


def write_raw_visual_csv(path: Path, raw: np.ndarray) -> None:
    visual_mv = raw16_to_visual_mv(raw)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Raw16", "Visual_mV"])
        for index, (raw_value, mv_value) in enumerate(zip(raw, visual_mv)):
            writer.writerow([index, int(raw_value), f"{float(mv_value):.8f}".rstrip("0").rstrip(".")])


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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


def read_numeric_column_with_name(path: Path) -> tuple[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for candidate in ("TemperatureRaw16", "Raw16", "TemperatureRaw(16bit)", "RawValue"):
            if candidate in headers:
                column = candidate
                break
        else:
            data_columns = [name for name in headers if name and name.lower() != "index"]
            if not data_columns:
                raise ValueError(f"{path} has no numeric data column.")
            column = data_columns[0]
        values = np.array([float(row[column]) for row in reader if row.get(column) != ""], dtype=float)
    return column, values


def read_ground_column(path: Path) -> np.ndarray | None:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for candidate in ("VGNDRaw16", "VGNDRaw(16bit)", "RawGround", "Ground"):
            if candidate in headers:
                return np.array([float(row[candidate]) for row in reader if row.get(candidate) != ""], dtype=float)
    return None


def stable_mean_from_values(values: np.ndarray) -> float:
    window = min(WINDOW_TARGET_SAMPLES, len(values))
    means, stds = rolling_window_stats(values, window)
    return float(means[int(np.argmin(stds))])


def shifted_raw_like_source(source_raw: np.ndarray, target_mean_raw: float) -> np.ndarray:
    source_center = stable_mean_from_values(source_raw)
    shifted = source_raw.astype(float) + (target_mean_raw - source_center)
    return np.rint(shifted).astype(int)


def apply_constant_adc_zero_offset(source_raw: np.ndarray, source_zero: float, target_zero: float) -> np.ndarray:
    """Convert device ADC zero while preserving raw - zero exactly up to rounding."""
    shifted = source_raw.astype(float) - source_zero + target_zero
    return np.rint(shifted).astype(int)


def write_standard_8chip_csv(path: Path, raw: np.ndarray, ground: np.ndarray | None, rng: np.random.Generator) -> None:
    if ground is None or len(ground) != len(raw):
        ground = generate_ground_series(
            len(raw),
            rng,
            offset_scale_lsb=CYCLE1_8CHIP_GROUND_OFFSET_SCALE_LSB,
            noise_sigma_lsb=CYCLE1_8CHIP_GROUND_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE1_8CHIP_GROUND_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE1_8CHIP_GROUND_SEGMENT_OFFSET_MAX_LEN,
        )
    else:
        ground = np.rint(ground).astype(int)
    write_temperature_ground_csv(path, raw, ground)


def generate_cycle1_8chip_from_offsets(rng: np.random.Generator) -> list[dict[str, object]]:
    source_dir = BASE_DIR / "cycle1"
    output_dir = CYCLE1_8CHIP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.csv"):
        path.unlink()

    source_files = temperature_files(source_dir)
    cold_env_path = find_cycle_or_root_file(source_dir, "Temp_Cold_8Chips.csv")
    hot_env_path = find_cycle_or_root_file(source_dir, "Temp_Hot_4Chips.csv")

    cold_env_raw = read_numeric_column(cold_env_path)
    cold_zero = stable_mean_from_values(cold_env_raw)
    # Construct the 25 degC 8chip zero from the 24 degC 8chip zero and the T-table delta.
    hot_zero_8chip = target_raw16(CYCLE1_HOT_AMBIENT_TEMP, cold_zero, CYCLE1_COLD_AMBIENT_TEMP)
    _, hot_env_raw = read_numeric_column_with_name(hot_env_path)
    hot_zero_4chip = stable_mean_from_values(hot_env_raw)
    chip4_to_chip8_offset = hot_zero_8chip - hot_zero_4chip

    cold_env_ground = read_ground_column(cold_env_path)
    write_standard_8chip_csv(output_dir / "Temp_Cold_8Chips.csv", np.rint(cold_env_raw).astype(int), cold_env_ground, rng)

    hot_env_8chip = apply_constant_adc_zero_offset(hot_env_raw, hot_zero_4chip, hot_zero_8chip)
    write_standard_8chip_csv(output_dir / "Temp_Hot_8Chips.csv", hot_env_8chip, read_ground_column(hot_env_path), rng)

    rows = [
        {
            "Source_File": cold_env_path.relative_to(BASE_DIR).as_posix(),
            "Output_File": (output_dir / "Temp_Cold_8Chips.csv").relative_to(BASE_DIR).as_posix(),
            "Temperature_degC": CYCLE1_COLD_AMBIENT_TEMP,
            "Source_Column": "TemperatureRaw16",
            "Source_Stable_Mean_Raw16": round(stable_mean_from_values(cold_env_raw), 6),
            "Target_8chip_Mean_Raw16": round(cold_zero, 6),
            "Applied_Offset_Raw16": 0.0,
            "Samples": len(cold_env_raw),
        },
        {
            "Source_File": hot_env_path.relative_to(BASE_DIR).as_posix(),
            "Output_File": (output_dir / "Temp_Hot_8Chips.csv").relative_to(BASE_DIR).as_posix(),
            "Temperature_degC": CYCLE1_HOT_AMBIENT_TEMP,
            "Source_Column": "Raw16",
            "Source_Stable_Mean_Raw16": round(hot_zero_4chip, 6),
            "Target_8chip_Mean_Raw16": round(hot_zero_8chip, 6),
            "Applied_Offset_Raw16": round(chip4_to_chip8_offset, 6),
            "Samples": len(hot_env_raw),
        },
    ]

    for temp in range(TEMP_MIN, TEMP_MAX + 1):
        source_path = source_files[temp]
        source_column, source_raw = read_numeric_column_with_name(source_path)
        source_ground = read_ground_column(source_path)
        if temp <= 24:
            shifted_raw = np.rint(source_raw).astype(int)
            target = stable_mean_from_values(source_raw)
            applied_offset = 0.0
        else:
            shifted_raw = apply_constant_adc_zero_offset(source_raw, hot_zero_4chip, hot_zero_8chip)
            target = stable_mean_from_values(source_raw) + chip4_to_chip8_offset
            applied_offset = chip4_to_chip8_offset
        output_path = output_dir / f"{temp}.csv"
        write_standard_8chip_csv(output_path, shifted_raw, source_ground, rng)
        source_stable = stable_mean_from_values(source_raw)
        rows.append(
            {
                "Source_File": source_path.relative_to(BASE_DIR).as_posix(),
                "Output_File": output_path.relative_to(BASE_DIR).as_posix(),
                "Temperature_degC": temp,
                "Source_Column": source_column,
                "Source_Stable_Mean_Raw16": round(source_stable, 6),
                "Target_8chip_Mean_Raw16": round(target, 6),
                "Applied_Offset_Raw16": round(applied_offset, 6),
                "Samples": len(source_raw),
            }
        )

    write_rows(SUMMARY_DIR / "cycle1_8chip_offset_summary.csv", rows)
    return rows


def generate_cycle1_missing(rng: np.random.Generator) -> list[dict[str, object]]:
    cycle_dir = CYCLE1_DIR
    cycle_dir.mkdir(parents=True, exist_ok=True)
    existing = temperature_files(cycle_dir)

    cold_baseline = stable_mean(find_cycle_or_root_file(cycle_dir, "Temp_Cold_8Chips.csv"))
    hot_baseline = stable_mean(find_cycle_or_root_file(cycle_dir, "Temp_Hot_4Chips.csv"))
    rows = []
    for temp in range(TEMP_MIN, TEMP_MAX + 1):
        if temp in existing:
            continue
        device = device_for_cycle1(temp)
        baseline = cold_baseline if device == "cold_device_1" else hot_baseline
        target = target_raw16(temp, baseline, cycle1_ambient_temp(temp))
        n_samples = inferred_sample_count(temp, [existing])
        quantize_16 = CYCLE1_MISSING_HOT_QUANTIZE_16 if device == "hot_device_2" else CYCLE1_MISSING_COLD_QUANTIZE_16
        raw = generate_raw_series(
            target,
            n_samples,
            rng,
            quantize_16=quantize_16,
            point_offset_sigma=CYCLE1_MISSING_POINT_OFFSET_SIGMA_LSB,
            offset_scale_lsb=CYCLE1_MISSING_OFFSET_SCALE_LSB,
            raw_noise_sigma_lsb=CYCLE1_MISSING_RAW_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE1_MISSING_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE1_MISSING_SEGMENT_OFFSET_MAX_LEN,
            keep_point_offset_in_mean=CYCLE1_MISSING_KEEP_POINT_OFFSET_IN_MEAN,
        )
        path = cycle_dir / f"{temp}.csv"
        if device == "cold_device_1":
            ground = generate_ground_series(
                n_samples,
                rng,
                offset_scale_lsb=CYCLE1_MISSING_GROUND_OFFSET_SCALE_LSB,
                noise_sigma_lsb=CYCLE1_MISSING_GROUND_NOISE_SIGMA_LSB,
                segment_offset_min_len=CYCLE1_MISSING_GROUND_SEGMENT_OFFSET_MIN_LEN,
                segment_offset_max_len=CYCLE1_MISSING_GROUND_SEGMENT_OFFSET_MAX_LEN,
            )
            write_temperature_ground_csv(path, raw, ground)
        else:
            write_raw_visual_csv(path, raw)
        existing[temp] = path
        rows.append(
            summary_row(
                "cycle1",
                temp,
                device,
                n_samples,
                target,
                raw,
                path,
                {
                    "Generation_Mode": "missing_completion",
                    "Point_Offset_Sigma_LSB": CYCLE1_MISSING_POINT_OFFSET_SIGMA_LSB,
                    "Offset_Scale_LSB": CYCLE1_MISSING_OFFSET_SCALE_LSB,
                    "Raw_Noise_Sigma_LSB": CYCLE1_MISSING_RAW_NOISE_SIGMA_LSB,
                    "Quantize_16": quantize_16,
                },
            )
        )
    return rows


def summary_row(
    cycle: str,
    temp: int,
    device: str,
    n_samples: int,
    target: float,
    raw: np.ndarray,
    path: Path,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "Cycle": cycle,
        "Temperature_degC": temp,
        "Device": device,
        "Samples": n_samples,
        "Target_Raw16": round(target, 6),
        "Generated_Mean_Raw16": round(float(np.mean(raw)), 6),
        "Generated_Std_Raw16": round(float(np.std(raw, ddof=1)), 6),
        "Output_File": path.relative_to(BASE_DIR).as_posix(),
    }
    if extra:
        row.update(extra)
    return row


def generate_cycle2_missing(rng: np.random.Generator) -> list[dict[str, object]]:
    cycle_dir = BASE_DIR / "cycle2"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    existing = temperature_files(cycle_dir)
    cycle1_files = temperature_files(BASE_DIR / "cycle1")
    baseline = infer_baseline_from_cycle(cycle_dir, CYCLE2_AMBIENT_TEMP)

    rows = []
    for temp in range(TEMP_MIN, TEMP_MAX + 1):
        if temp in existing:
            continue
        target = target_raw16(temp, baseline, CYCLE2_AMBIENT_TEMP)
        n_samples = inferred_sample_count(temp, [existing, cycle1_files])
        raw = generate_raw_series(
            target,
            n_samples,
            rng,
            quantize_16=CYCLE2_MISSING_QUANTIZE_16,
            point_offset_sigma=CYCLE2_MISSING_POINT_OFFSET_SIGMA_LSB,
            offset_scale_lsb=CYCLE2_MISSING_OFFSET_SCALE_LSB,
            raw_noise_sigma_lsb=CYCLE2_MISSING_RAW_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE2_MISSING_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE2_MISSING_SEGMENT_OFFSET_MAX_LEN,
            keep_point_offset_in_mean=CYCLE2_MISSING_KEEP_POINT_OFFSET_IN_MEAN,
        )
        path = cycle_dir / f"{temp}.csv"
        ground = generate_ground_series(
            n_samples,
            rng,
            offset_scale_lsb=CYCLE2_MISSING_GROUND_OFFSET_SCALE_LSB,
            noise_sigma_lsb=CYCLE2_MISSING_GROUND_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE2_MISSING_GROUND_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE2_MISSING_GROUND_SEGMENT_OFFSET_MAX_LEN,
        )
        write_temperature_ground_csv(path, raw, ground)
        existing[temp] = path
        rows.append(
            summary_row(
                "cycle2",
                temp,
                "cycle2_8chip",
                n_samples,
                target,
                raw,
                path,
                {
                    "Generation_Mode": "missing_completion",
                    "Point_Offset_Sigma_LSB": CYCLE2_MISSING_POINT_OFFSET_SIGMA_LSB,
                    "Offset_Scale_LSB": CYCLE2_MISSING_OFFSET_SCALE_LSB,
                    "Raw_Noise_Sigma_LSB": CYCLE2_MISSING_RAW_NOISE_SIGMA_LSB,
                    "Quantize_16": CYCLE2_MISSING_QUANTIZE_16,
                },
            )
        )
    return rows


def cycle3_noise_scale_for_temp(temp: int) -> float:
    scale = CYCLE3_DEFAULT_NOISE_SCALE
    for start_temp, end_temp, multiplier in CYCLE3_NOISE_EQUIVALENT_BANDS:
        if min(start_temp, end_temp) <= temp <= max(start_temp, end_temp):
            scale = multiplier
    for start_temp, end_temp, multiplier in CYCLE3_NOISE_AMPLIFIED_BANDS:
        if min(start_temp, end_temp) <= temp <= max(start_temp, end_temp):
            scale = multiplier
    return max(0.0, float(scale))


def cycle3_distance_drift_scale(temp: float) -> float:
    full_distance = abs(CYCLE3_SAMPLE_DRIFT_FULL_TEMP - CYCLE3_SAMPLE_DRIFT_ZERO_TEMP)
    if full_distance <= 0:
        return 1.0
    return min(abs(temp - CYCLE3_SAMPLE_DRIFT_ZERO_TEMP) / full_distance, 1.0)


def cycle3_sample_drift_lsb(n: int, temp: int, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)

    temp_scale = cycle3_distance_drift_scale(temp)
    pp_lsb = CYCLE3_SAMPLE_DRIFT_PP_LSB * temp_scale * noise_scale
    if pp_lsb <= 0:
        return np.zeros(n, dtype=float)

    x = np.linspace(0.0, 1.0, n)
    direction = -1.0 if rng.random() < 0.5 else 1.0
    linear = direction * (x - 0.5) * pp_lsb

    knots = max(4, int(np.ceil(n / 160)))
    random_shape = np.interp(
        np.arange(n),
        np.linspace(0, n - 1, knots),
        rng.normal(0.0, pp_lsb * CYCLE3_SAMPLE_DRIFT_RANDOM_WEIGHT, size=knots),
    )
    random_shape -= float(np.mean(random_shape))
    drift = linear + random_shape
    drift -= float(np.mean(drift))

    peak_to_peak = float(np.max(drift) - np.min(drift))
    max_peak_to_peak = pp_lsb * (1.0 + CYCLE3_SAMPLE_DRIFT_RANDOM_WEIGHT)
    if peak_to_peak > max_peak_to_peak > 0:
        drift *= max_peak_to_peak / peak_to_peak
    return drift


def cycle3_random_walk_bias(count: int, temp: float, rng: np.random.Generator, scale: float) -> np.ndarray:
    if count <= 0 or scale <= 0:
        return np.zeros(max(count, 0), dtype=float)
    step_sigma_c = lsb_to_temp_c(CYCLE3_BIAS_RANDOM_WALK_STEP_SIGMA_LSB, temp) * scale
    clip_c = lsb_to_temp_c(CYCLE3_BIAS_RANDOM_WALK_PP_LSB / 2.0, temp) * scale
    walk = np.cumsum(rng.normal(0.0, step_sigma_c, size=count))
    walk -= walk[0]
    return np.clip(walk, -clip_c, clip_c)


def add_cycle3_bias_band(
    biases: dict[int, float],
    sources: dict[int, str],
    band_name: str,
    band: tuple[int, int, float, float],
    rng: np.random.Generator,
) -> None:
    start_temp, end_temp, start_bias, end_bias = band
    if start_temp > end_temp:
        start_temp, end_temp = end_temp, start_temp
        start_bias, end_bias = end_bias, start_bias

    temps = [temp for temp in range(start_temp, end_temp + 1) if TEMP_MIN <= temp <= TEMP_MAX]
    if not temps:
        return

    linear = np.interp(temps, [temps[0], temps[-1]], [start_bias, end_bias])
    mean_temp = float(np.mean(temps))
    walk_scale = (
        CYCLE3_ZERO_BIAS_RANDOM_WALK_SCALE
        if band_name == "zero"
        else cycle3_noise_scale_for_temp(int(round(mean_temp)))
    )
    random_walk = cycle3_random_walk_bias(len(temps), mean_temp, rng, walk_scale)

    for temp, bias_c in zip(temps, linear + random_walk):
        if band_name == "positive":
            bias_c = max(float(bias_c), 0.0)
        elif band_name == "negative":
            bias_c = min(float(bias_c), 0.0)
        else:
            bias_c = float(bias_c)
        biases[temp] = bias_c
        sources[temp] = f"{band_name}:{start_temp}-{end_temp}"


def cycle3_temperature_bias_profile(rng: np.random.Generator) -> tuple[dict[int, float], dict[int, str]]:
    biases = {temp: 0.0 for temp in range(TEMP_MIN, TEMP_MAX + 1)}
    sources = {temp: "default_zero" for temp in range(TEMP_MIN, TEMP_MAX + 1)}
    if not CYCLE3_ENABLE_TEMPERATURE_BIAS:
        return biases, sources

    for band in CYCLE3_POSITIVE_BIAS_BANDS:
        add_cycle3_bias_band(biases, sources, "positive", band, rng)
    for band in CYCLE3_ZERO_BIAS_BANDS:
        add_cycle3_bias_band(biases, sources, "zero", band, rng)
    for band in CYCLE3_NEGATIVE_BIAS_BANDS:
        add_cycle3_bias_band(biases, sources, "negative", band, rng)
    return biases, sources


def generate_cycle3(rng: np.random.Generator) -> list[dict[str, object]]:
    cycle_dir = BASE_DIR / "cycle3"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle1_files = temperature_files(CYCLE1_DIR)
    cycle2_files = temperature_files(BASE_DIR / "cycle2")
    for path in cycle_dir.glob("*.csv"):
        if path.stem.isdigit():
            path.unlink()
    existing = {}

    cycle2_baseline = infer_baseline_from_cycle(BASE_DIR / "cycle2", CYCLE2_AMBIENT_TEMP)
    cycle1_cold_baseline = stable_mean(find_cycle_or_root_file(CYCLE1_DIR, "Temp_Cold_8Chips.csv"))
    baseline = float(np.mean([cycle2_baseline, cycle1_cold_baseline]) + rng.normal(0.0, CYCLE3_VIRTUAL_BASELINE_SIGMA_LSB))
    temp_biases, bias_sources = cycle3_temperature_bias_profile(rng)
    table_min = min(T_TYPE_TABLE)
    table_max = max(T_TYPE_TABLE)

    rows = []
    for temp in range(TEMP_MIN, TEMP_MAX + 1):
        if temp in existing:
            continue
        temp_bias_c = temp_biases.get(temp, 0.0)
        actual_temp = min(max(temp + temp_bias_c, table_min), table_max)
        target = target_raw16(actual_temp, baseline, CYCLE3_AMBIENT_TEMP)
        n_samples = inferred_sample_count(temp, [cycle2_files, cycle1_files])
        noise_scale = cycle3_noise_scale_for_temp(temp)
        sample_drift = cycle3_sample_drift_lsb(n_samples, temp, rng, noise_scale)
        raw = generate_raw_series(
            target,
            n_samples,
            rng,
            quantize_16=CYCLE3_VIRTUAL_QUANTIZE_16,
            point_offset_sigma=CYCLE3_VIRTUAL_POINT_OFFSET_SIGMA_LSB,
            offset_scale_lsb=CYCLE3_VIRTUAL_OFFSET_SCALE_LSB,
            raw_noise_sigma_lsb=CYCLE3_VIRTUAL_RAW_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE3_VIRTUAL_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE3_VIRTUAL_SEGMENT_OFFSET_MAX_LEN,
            extra_drift_lsb=sample_drift,
            keep_point_offset_in_mean=CYCLE3_VIRTUAL_KEEP_POINT_OFFSET_IN_MEAN,
        )
        path = cycle_dir / f"{temp}.csv"
        ground = generate_ground_series(
            n_samples,
            rng,
            offset_scale_lsb=CYCLE3_VIRTUAL_GROUND_OFFSET_SCALE_LSB,
            noise_sigma_lsb=CYCLE3_VIRTUAL_GROUND_NOISE_SIGMA_LSB,
            segment_offset_min_len=CYCLE3_VIRTUAL_GROUND_SEGMENT_OFFSET_MIN_LEN,
            segment_offset_max_len=CYCLE3_VIRTUAL_GROUND_SEGMENT_OFFSET_MAX_LEN,
        )
        write_temperature_ground_csv(path, raw, ground)
        existing[temp] = path
        rows.append(
            summary_row(
                "cycle3",
                temp,
                "cycle3_8chip",
                n_samples,
                target,
                raw,
                path,
                {
                    "Recorded_Temperature_degC": temp,
                    "Actual_Temperature_degC": round(actual_temp, 6),
                    "Temperature_Bias_C": round(actual_temp - temp, 6),
                    "Bias_Source": bias_sources.get(temp, "default_zero"),
                    "Generation_Mode": "full_virtual",
                    "Point_Offset_Sigma_LSB": CYCLE3_VIRTUAL_POINT_OFFSET_SIGMA_LSB,
                    "Offset_Scale_LSB": CYCLE3_VIRTUAL_OFFSET_SCALE_LSB,
                    "Raw_Noise_Sigma_LSB": CYCLE3_VIRTUAL_RAW_NOISE_SIGMA_LSB,
                    "Quantize_16": CYCLE3_VIRTUAL_QUANTIZE_16,
                    "Noise_Scale": round(noise_scale, 6),
                    "Sample_Drift_Scale": round(cycle3_distance_drift_scale(temp), 6),
                    "Sample_Drift_PP_Raw16": round(float(np.max(sample_drift) - np.min(sample_drift)), 6),
                },
            )
        )

    baseline_path = SUMMARY_DIR / "test6data_gen_cycle3_baseline.csv"
    write_rows(
        baseline_path,
        [
            {
                "Cycle": "cycle3",
                "Ambient_Temperature_degC": CYCLE3_AMBIENT_TEMP,
                "Generated_Baseline_Raw16": round(baseline, 6),
                "Reference_Cycle2_Baseline_Raw16": round(cycle2_baseline, 6),
                "Reference_Cycle1_Cold_Baseline_Raw16": round(cycle1_cold_baseline, 6),
                "Baseline_Sigma_LSB": CYCLE3_VIRTUAL_BASELINE_SIGMA_LSB,
                "Bias_Enabled": CYCLE3_ENABLE_TEMPERATURE_BIAS,
                "Bias_Random_Walk_Step_Sigma_LSB": CYCLE3_BIAS_RANDOM_WALK_STEP_SIGMA_LSB,
                "Bias_Random_Walk_PP_LSB": CYCLE3_BIAS_RANDOM_WALK_PP_LSB,
                "Sample_Drift_PP_LSB": CYCLE3_SAMPLE_DRIFT_PP_LSB,
                "Sample_Drift_Zero_Temp": CYCLE3_SAMPLE_DRIFT_ZERO_TEMP,
                "Sample_Drift_Full_Temp": CYCLE3_SAMPLE_DRIFT_FULL_TEMP,
            }
        ],
    )
    return rows


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    all_rows.extend(sync_cycle1_from_origin())
    all_rows.extend(generate_cycle1_missing(rng))
    all_rows.extend(generate_cycle2_missing(rng))
    all_rows.extend(generate_cycle3(rng))
    cycle1_8chip_rows = generate_cycle1_8chip_from_offsets(rng)

    summary_path = SUMMARY_DIR / "test6data_gen_summary.csv"
    write_rows(summary_path, all_rows)

    for cycle in ("cycle1", "cycle2", "cycle3"):
        files = temperature_files(BASE_DIR / cycle)
        missing = [temp for temp in range(TEMP_MIN, TEMP_MAX + 1) if temp not in files]
        print(f"{cycle}: {len(files)} numeric temperature files, missing={missing}")
    print(f"Generated {len(all_rows)} files.")
    print(f"Summary: {summary_path}")
    print(f"cycle1_8chip: generated {len(cycle1_8chip_rows)} standardized 8chip CSV files.")
    print(f"cycle1_8chip summary: {SUMMARY_DIR / 'cycle1_8chip_offset_summary.csv'}")


if __name__ == "__main__":
    main()
