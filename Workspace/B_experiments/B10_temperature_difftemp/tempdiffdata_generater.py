#!/usr/bin/env python
# coding: utf-8
"""
Generate synthetic CSV data for temp_test_difftemp.

The generated CSV follows the real acquisition format:

    Index,TemperatureRaw16,VGNDRaw16

The forward analysis in 10_Temp_difftemp.py does:

    raw_signal = TemperatureRaw16 - VGNDRaw16
    corrected_raw16 = raw_signal - per-file-baseline
    thermocouple_mV = raw16_to_thermocouple_mv(corrected_raw16)
    total_mV = thermocouple_mV + thermocouple_mv(COLD_JUNCTION_C)
    T_c = temperature_from_mv(total_mV)
    T_hat = (T_c - T_c[0]) / (T_ss - T_c[0])

This generator inverts that path:

    T_hat(t) = 1 - A*exp(-t/tau_fast) - B*exp(-t/tau_slow)
    residual(t) = x*(1-x)*(c1 + c2*x + c3*x*x)
    T_c(t) = T0 + (T_ss - T0) * (T_hat(t) + residual(t))
    corrected_raw16 = thermocouple_mv_to_raw16(mv(T_c) - mv(COLD_JUNCTION_C))

The leading baseline samples are included so the contact detector in
10_Temp_difftemp.py has a realistic pre-contact region.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# User parameters
# =============================================================================

# Output
OUTPUT_SUBDIR = "1_cold"
OUTPUT_FILENAME = None
# Example fixed name:
# OUTPUT_FILENAME = "TempFit_Tenv22C_Tliquid1p5C_Tgripper21C_Temp_AD2_20260510_224022.csv"

# Metadata written into the filename and parsed by 10_Temp_difftemp.py
T_ENV_C = 22.0
T_GRIPPER_C = 21.0
T_LIQUID_C = 1.5

# The processing script uses this for cold-junction compensation. Its default is
# 25 C. With per-file-baseline offset mode, the processed curve starts at this
# temperature even if the filename contains another gripper temperature.
COLD_JUNCTION_C = 25.0

# Double-exponential normalized response:
#     T_hat = 1 - A*exp(-t/tau_fast) - B*exp(-t/tau_slow)
MODEL_A = 0.62
MODEL_B = 0.38
TAU_FAST_S = 0.45
TAU_SLOW_S = 8.5

# Steady-state temperature used to scale the normalized curve.
# If STEADY_STATE_TEMP_C is None, the final physical temperature is estimated as:
#     T_gripper + STEADY_STATE_BETA * (T_liquid - T_gripper)
# so beta/consistency in 10_Temp_difftemp.py stays in the same temperature
# reference frame as the filename metadata.
STEADY_STATE_TEMP_C = None
STEADY_STATE_BETA = 0.66


# =============================================================================
# Fixed generation parameters
# =============================================================================

# Synthetic acquisition shape
SAMPLING_RATE_HZ = 400.0
TOTAL_DURATION_S = 60.0
PRE_CONTACT_S = 0.50
NORMALIZE_A_B_TO_SUM_ONE = True

_RAW_GAUSSIAN_NOISE_SIGMA_RAW16 = 5.0
_RAW_BASELINE_OFFSET_RAW16 = 125.0
_ROUND_TO_INTEGER_RAW16 = True

# Reproducibility
RANDOM_SEED = 20260510


# =============================================================================
# Paths and calibration loading
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "tempdiffdata_config.json"
OUTPUT_DIR = SCRIPT_DIR / OUTPUT_SUBDIR
GENERATED_COLD_SUBDIRS = ("0_cold", "1_cold", "3_cold")
RESIDUAL_SCALE_BY_SUBDIR_CLASS = {
    ("0_cold", "hot"): 0.75,
    ("0_cold", "cold"): 0.75,
    ("1_cold", "hot"): 0.60,
    ("3_cold", "hot"): 0.60,
}


def _load_temperature_calibration_module():
    repo_root = SCRIPT_DIR.parents[1]
    candidates = [
        repo_root / "Vet6USB_curve_draw" / "01_6_PartsA_test" / "06_Temperature_Calibration.py",
        repo_root / "Vet6USB_curve_draw" / "07_PartsB_test" / "10_Temp_difftemp.py",
    ]
    for path in candidates:
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location("temperature_calibration_for_generator", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find temperature calibration module. Searched:\n{searched}")


_CAL = _load_temperature_calibration_module()
T_TYPE_TABLE = _CAL.T_TYPE_TABLE
thermocouple_mv_to_raw16 = _CAL.thermocouple_mv_to_raw16


# =============================================================================
# Model helpers
# =============================================================================

@dataclass(frozen=True)
class GenerationSummary:
    output_path: Path
    point_name: str
    residual_source_file: str
    t_env_c: float
    t_liquid_c: float
    t_gripper_c: float
    samples: int
    pre_contact_samples: int
    t0_c: float
    t_ss_c: float
    model_a: float
    model_b: float
    tau_fast_s: float
    tau_slow_s: float
    residual_c1: float
    residual_c2: float
    residual_c3: float
    noise_sigma_raw16: float


@dataclass(frozen=True)
class ExperimentConfig:
    output_subdir: str
    output_filename: str | None
    t_env_c: float
    t_gripper_c: float
    t_liquid_c: float
    cold_junction_c: float
    model_a: float
    model_b: float
    tau_fast_s: float
    tau_slow_s: float
    steady_state_temp_c: float | None
    steady_state_beta: float
    random_seed: int
    residual_c1: float | None
    residual_c2: float | None
    residual_c3: float | None


def _temperature_class(config: ExperimentConfig | None = None) -> str:
    if config is None:
        t_liquid_c = T_LIQUID_C
        t_gripper_c = T_GRIPPER_C
    else:
        t_liquid_c = config.t_liquid_c
        t_gripper_c = config.t_gripper_c
    return "hot" if float(t_liquid_c) >= float(t_gripper_c) else "cold"


def _residual_scale(config: ExperimentConfig | None = None) -> float:
    output_subdir = OUTPUT_SUBDIR if config is None else config.output_subdir
    return RESIDUAL_SCALE_BY_SUBDIR_CLASS.get((str(output_subdir), _temperature_class(config)), 1.0)


def _clear_generated_cold_dirs() -> None:
    for subdir in GENERATED_COLD_SUBDIRS:
        target = SCRIPT_DIR / subdir
        target.mkdir(parents=True, exist_ok=True)
        for csv_path in target.glob("*.csv"):
            csv_path.unlink()


def _format_temperature_for_filename(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _default_output_filename(config: ExperimentConfig | None = None, run_id: str | None = None, index: int | None = None) -> str:
    t_env = T_ENV_C if config is None else config.t_env_c
    t_liquid = T_LIQUID_C if config is None else config.t_liquid_c
    t_gripper = T_GRIPPER_C if config is None else config.t_gripper_c
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    elif index is None:
        timestamp = run_id
    else:
        timestamp = f"{run_id}_{index:02d}"
    return (
        f"TempFit_Tenv{_format_temperature_for_filename(t_env)}C_"
        f"Tliquid{_format_temperature_for_filename(t_liquid)}C_"
        f"Tgripper{_format_temperature_for_filename(t_gripper)}C_"
        f"Temp_AD2_{timestamp}.csv"
    )


def _thermocouple_mv_at_temperature(temp_c: np.ndarray | float) -> np.ndarray:
    temps = np.asarray(sorted(T_TYPE_TABLE), dtype=float)
    mv_values = np.asarray([T_TYPE_TABLE[int(temp)] for temp in temps], dtype=float)
    values = np.asarray(temp_c, dtype=float)
    min_temp = float(temps[0])
    max_temp = float(temps[-1])
    if np.any(values < min_temp) or np.any(values > max_temp):
        raise ValueError(f"Temperature is outside T-type table range [{min_temp:g}, {max_temp:g}] C")
    return np.interp(values, temps, mv_values)


def _resolve_model_coefficients(config: ExperimentConfig | None = None) -> tuple[float, float]:
    a = float(MODEL_A if config is None else config.model_a)
    b = float(MODEL_B if config is None else config.model_b)
    if NORMALIZE_A_B_TO_SUM_ONE:
        total = a + b
        if abs(total) < 1e-12:
            raise ValueError("MODEL_A + MODEL_B must be non-zero.")
        a /= total
        b /= total
    return a, b


def _resolve_steady_temperature(config: ExperimentConfig | None = None) -> float:
    steady_state_temp_c = STEADY_STATE_TEMP_C if config is None else config.steady_state_temp_c
    t_liquid_c = T_LIQUID_C if config is None else config.t_liquid_c
    t_gripper_c = T_GRIPPER_C if config is None else config.t_gripper_c
    steady_state_beta = STEADY_STATE_BETA if config is None else config.steady_state_beta
    if steady_state_temp_c is not None:
        return float(steady_state_temp_c)

    physical_delta_c = float(steady_state_beta) * (float(t_liquid_c) - float(t_gripper_c))
    return float(t_gripper_c) + physical_delta_c


def _double_exp_normalized(t_s: np.ndarray, a: float, b: float, config: ExperimentConfig | None = None) -> np.ndarray:
    tau_fast_s = TAU_FAST_S if config is None else config.tau_fast_s
    tau_slow_s = TAU_SLOW_S if config is None else config.tau_slow_s
    return 1.0 - a * np.exp(-t_s / float(tau_fast_s)) - b * np.exp(-t_s / float(tau_slow_s))


def _point_name_from_output_subdir(output_subdir: str | None = None) -> str:
    text = str(OUTPUT_SUBDIR if output_subdir is None else output_subdir)
    head = text.split("/")[0].split("\\")[0]
    if head.lower().endswith("_cold") and head[0].isdigit():
        return f"P{head[0]}"
    if head.upper().startswith("P"):
        return head.upper()
    raise ValueError(f"Cannot infer point name from OUTPUT_SUBDIR={OUTPUT_SUBDIR!r}")


def _load_residual_parameters(config: ExperimentConfig | None = None) -> tuple[float, float, float, str]:
    if (
        config is not None
        and config.residual_c1 is not None
        and config.residual_c2 is not None
        and config.residual_c3 is not None
    ):
        return (
            float(config.residual_c1),
            float(config.residual_c2),
            float(config.residual_c3),
            "tempdiffdata_config.json",
        )

    output_subdir = OUTPUT_SUBDIR if config is None else config.output_subdir
    t_env_c = T_ENV_C if config is None else config.t_env_c
    t_liquid_c = T_LIQUID_C if config is None else config.t_liquid_c
    t_gripper_c = T_GRIPPER_C if config is None else config.t_gripper_c
    point_name = _point_name_from_output_subdir(output_subdir)
    parameter_path = SCRIPT_DIR / "difference_traces" / f"10_{point_name}_difference_parameters.csv"
    if not parameter_path.exists():
        raise FileNotFoundError(f"Residual parameter CSV not found: {parameter_path}")

    best_row: dict[str, str] | None = None
    best_error = float("inf")
    with parameter_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                error = (
                    abs(float(row["T_env_c"]) - float(t_env_c))
                    + abs(float(row["T_liquid_c"]) - float(t_liquid_c))
                    + abs(float(row["T_jaw0_c"]) - float(t_gripper_c))
                )
            except (KeyError, TypeError, ValueError):
                continue
            if error < best_error:
                best_error = error
                best_row = row

    if best_row is None:
        raise ValueError(f"No usable residual parameters in {parameter_path}")
    c1_key = "c1" if "c1" in best_row else "diff_c1"
    c2_key = "c2" if "c2" in best_row else "diff_c2"
    c3_key = "c3" if "c3" in best_row else "diff_c3"
    return (float(best_row[c1_key]), float(best_row[c2_key]), float(best_row[c3_key]), best_row.get("file", ""))


def _residual_model_from_normalized_time(x: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * (1.0 - x) * (c1 + c2 * x + c3 * x * x)


def _normalized_contact_time(t_s: np.ndarray) -> np.ndarray:
    duration = max(float(TOTAL_DURATION_S) - float(PRE_CONTACT_S), 1.0 / float(SAMPLING_RATE_HZ))
    return np.clip(np.asarray(t_s, dtype=float), 0.0, duration) / duration


def _temperature_to_corrected_raw16(temp_c: np.ndarray, cold_junction_c: float) -> np.ndarray:
    target_mv = _thermocouple_mv_at_temperature(temp_c)
    cold_mv = float(_thermocouple_mv_at_temperature(float(cold_junction_c)))
    thermocouple_delta_mv = target_mv - cold_mv
    return np.asarray([thermocouple_mv_to_raw16(float(value)) for value in thermocouple_delta_mv], dtype=float)


def build_curve(
    config: ExperimentConfig | None = None,
    run_id: str | None = None,
    batch_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, GenerationSummary]:
    if SAMPLING_RATE_HZ <= 0:
        raise ValueError("SAMPLING_RATE_HZ must be positive.")
    if TOTAL_DURATION_S <= PRE_CONTACT_S:
        raise ValueError("TOTAL_DURATION_S must be greater than PRE_CONTACT_S.")
    tau_fast_s = TAU_FAST_S if config is None else config.tau_fast_s
    tau_slow_s = TAU_SLOW_S if config is None else config.tau_slow_s
    if tau_fast_s <= 0 or tau_slow_s <= 0:
        raise ValueError("TAU_FAST_S and TAU_SLOW_S must be positive.")

    random_seed = RANDOM_SEED if config is None else config.random_seed
    rng = np.random.default_rng(int(random_seed))
    total_samples = int(round(float(TOTAL_DURATION_S) * float(SAMPLING_RATE_HZ)))
    pre_contact_samples = int(round(float(PRE_CONTACT_S) * float(SAMPLING_RATE_HZ)))
    total_samples = max(total_samples, pre_contact_samples + 2)

    index = np.arange(total_samples, dtype=int)
    t_since_contact = (index.astype(float) - float(pre_contact_samples)) / float(SAMPLING_RATE_HZ)
    contact_t = np.clip(t_since_contact, 0.0, None)

    a, b = _resolve_model_coefficients(config)
    c1, c2, c3, residual_source_file = _load_residual_parameters(config)
    cold_junction_c = COLD_JUNCTION_C if config is None else config.cold_junction_c
    t0_c = float(cold_junction_c)
    t_ss_c = _resolve_steady_temperature(config)

    normalized = _double_exp_normalized(contact_t, a, b, config)
    residual = _residual_model_from_normalized_time(_normalized_contact_time(contact_t), c1, c2, c3)
    normalized += residual * _residual_scale(config)
    normalized[t_since_contact < 0.0] = 0.0

    temp_c = t0_c + (t_ss_c - t0_c) * normalized
    corrected_raw16 = _temperature_to_corrected_raw16(temp_c, cold_junction_c)

    raw_signal = float(_RAW_BASELINE_OFFSET_RAW16) + corrected_raw16
    raw_signal += rng.normal(0.0, _RAW_GAUSSIAN_NOISE_SIGMA_RAW16, total_samples)

    vgnd = np.zeros(total_samples, dtype=float)
    temperature_raw = raw_signal + vgnd

    if _ROUND_TO_INTEGER_RAW16:
        temperature_raw = np.rint(temperature_raw).astype(int)
        vgnd = np.rint(vgnd).astype(int)

    rows = np.column_stack([index, temperature_raw, vgnd])
    output_subdir = OUTPUT_SUBDIR if config is None else config.output_subdir
    output_filename = OUTPUT_FILENAME if config is None else config.output_filename
    output_path = SCRIPT_DIR / output_subdir / (
        output_filename or _default_output_filename(config, run_id=run_id, index=batch_index)
    )
    summary = GenerationSummary(
        output_path=output_path,
        point_name=_point_name_from_output_subdir(output_subdir),
        residual_source_file=residual_source_file,
        t_env_c=T_ENV_C if config is None else config.t_env_c,
        t_liquid_c=T_LIQUID_C if config is None else config.t_liquid_c,
        t_gripper_c=T_GRIPPER_C if config is None else config.t_gripper_c,
        samples=total_samples,
        pre_contact_samples=pre_contact_samples,
        t0_c=t0_c,
        t_ss_c=t_ss_c,
        model_a=a,
        model_b=b,
        tau_fast_s=float(tau_fast_s),
        tau_slow_s=float(tau_slow_s),
        residual_c1=c1,
        residual_c2=c2,
        residual_c3=c3,
        noise_sigma_raw16=_RAW_GAUSSIAN_NOISE_SIGMA_RAW16,
    )
    return rows, temp_c, summary


def save_csv(rows: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Index", "TemperatureRaw16", "VGNDRaw16"])
        writer.writerows(rows.tolist())


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "trial"


def _experiment_from_dict(item: dict[str, Any], index: int) -> ExperimentConfig:
    output_subdir = str(item["output_subdir"])
    output_filename = item.get("output_filename")
    return ExperimentConfig(
        output_subdir=output_subdir,
        output_filename=None if output_filename is None else str(output_filename),
        t_env_c=float(item.get("t_env_c", T_ENV_C)),
        t_gripper_c=float(item["t_gripper_c"]),
        t_liquid_c=float(item["t_liquid_c"]),
        cold_junction_c=float(item.get("cold_junction_c", COLD_JUNCTION_C)),
        model_a=float(item.get("model_a", MODEL_A)),
        model_b=float(item.get("model_b", MODEL_B)),
        tau_fast_s=float(item.get("tau_fast_s", TAU_FAST_S)),
        tau_slow_s=float(item.get("tau_slow_s", TAU_SLOW_S)),
        steady_state_temp_c=_optional_float(item.get("steady_state_temp_c", STEADY_STATE_TEMP_C)),
        steady_state_beta=float(item.get("steady_state_beta", STEADY_STATE_BETA)),
        random_seed=int(item.get("random_seed", int(RANDOM_SEED) + index)),
        residual_c1=_optional_float(item.get("residual_c1")),
        residual_c2=_optional_float(item.get("residual_c2")),
        residual_c3=_optional_float(item.get("residual_c3")),
    )


def _load_batch_config(path: Path) -> list[ExperimentConfig]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("experiments", payload if isinstance(payload, list) else None)
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain an experiments list")
    return [_experiment_from_dict(item, index + 1) for index, item in enumerate(items)]


def _batch_timestamps(count: int) -> list[str]:
    rng = np.random.default_rng(int(RANDOM_SEED))
    current = datetime.now().replace(microsecond=0)
    stamps: list[str] = []
    for index in range(count):
        if index > 0:
            current += timedelta(seconds=int(rng.integers(5 * 60, 10 * 60 + 1)))
        stamps.append(current.strftime("%Y%m%d_%H%M%S"))
    return stamps


def _print_summary(rows: np.ndarray, temp_c: np.ndarray, summary: GenerationSummary) -> None:
    raw_signal = rows[:, 1].astype(float) - rows[:, 2].astype(float)
    print("Generated temperature-difference CSV")
    print(f"  Output file        : {summary.output_path}")
    print(f"  Point              : {summary.point_name}")
    print(f"  Residual source    : {summary.residual_source_file}")
    print(f"  Samples            : {summary.samples}")
    print(f"  Pre-contact samples: {summary.pre_contact_samples}")
    print(f"  Metadata Tenv      : {summary.t_env_c:g} C")
    print(f"  Metadata Tliquid   : {summary.t_liquid_c:g} C")
    print(f"  Metadata Tgripper  : {summary.t_gripper_c:g} C")
    print(f"  Curve T0 -> Tss    : {summary.t0_c:.3f} C -> {summary.t_ss_c:.3f} C")
    print(
        "  Double-exp model   : "
        f"A={summary.model_a:.6g}, B={summary.model_b:.6g}, "
        f"tau_fast={summary.tau_fast_s:.6g}s, tau_slow={summary.tau_slow_s:.6g}s"
    )
    print(
        "  Residual model     : "
        f"x(1-x)(c1+c2*x+c3*x^2), "
        f"c1={summary.residual_c1:.9g}, "
        f"c2={summary.residual_c2:.9g}, "
        f"c3={summary.residual_c3:.9g}"
    )
    print(f"  Noise sigma        : {summary.noise_sigma_raw16:.6g} raw16")
    print(f"  Raw signal range   : {raw_signal.min():.1f} .. {raw_signal.max():.1f} raw16")
    print(f"  Ideal temp range   : {float(np.min(temp_c)):.3f} .. {float(np.max(temp_c)):.3f} C")


def main() -> None:
    if CONFIG_PATH.exists():
        configs = _load_batch_config(CONFIG_PATH)
        timestamps = _batch_timestamps(len(configs))
        _clear_generated_cold_dirs()
        print(f"Loaded batch config: {CONFIG_PATH}")
        print(f"Experiments: {len(configs)}")
        print(f"Cleared output dirs : {', '.join(GENERATED_COLD_SUBDIRS)}")
        for index, config in enumerate(configs, start=1):
            rows, temp_c, summary = build_curve(config, run_id=timestamps[index - 1])
            save_csv(rows, summary.output_path)
            _print_summary(rows, temp_c, summary)
        return

    rows, temp_c, summary = build_curve()
    save_csv(rows, summary.output_path)
    _print_summary(rows, temp_c, summary)


if __name__ == "__main__":
    main()
