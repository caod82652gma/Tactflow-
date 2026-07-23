#!/usr/bin/env python
# coding: utf-8
"""Randomize selected curve fields in tempdiffdata_config.json.

Only these fields are updated:
    steady_state_beta, model_a, model_b, tau_fast_s, tau_slow_s

Beta is sampled from group/temperature ranges, biased upward when
abs(T_liquid - T_gripper) is larger within the same group and hot/cold class.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "tempdiffdata_config.json"

MODEL_A_CHOICES = (0.60, 0.65, 0.70)
TAU_FAST_RANGES_BY_SUBDIR = {
    "0_cold": (0.07, 0.14),
    "1_cold": (0.30, 0.60),
    "3_cold": (0.95, 1.60),
}
TAU_SLOW_RANGE_S = (15.9, 20.9)

BETA_RANGES = {
    "0_cold": {
        "hot": (0.38, 0.55),
        "cold": (0.07, 0.421),
    },
    "1_cold": {
        "hot": (0.15, 0.20),
        "cold": (0.05, 0.15),
    },
    "3_cold": {
        "hot": (0.05, 0.15),
        "cold": (0.01, 0.07),
    },
}


def _temperature_class(item: dict[str, Any]) -> str:
    return "hot" if float(item["t_liquid_c"]) >= float(item["t_gripper_c"]) else "cold"


def _delta_t(item: dict[str, Any]) -> float:
    return abs(float(item["t_liquid_c"]) - float(item["t_gripper_c"]))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _beta_for_item(
    item: dict[str, Any],
    max_delta_by_group_class: dict[tuple[str, str], float],
    rng: random.Random,
) -> float:
    output_subdir = str(item["output_subdir"])
    temp_class = _temperature_class(item)
    lo, hi = BETA_RANGES[output_subdir][temp_class]
    max_delta = max_delta_by_group_class[(output_subdir, temp_class)]
    normalized_delta = _delta_t(item) / max_delta if max_delta > 0 else 0.0

    # Bias beta upward with temperature gap, while leaving small random variation.
    center = lo + (hi - lo) * normalized_delta
    jitter = rng.uniform(-0.12, 0.12) * (hi - lo)
    return round(_clamp(center + jitter, lo, hi), 6)


def randomize_config(path: Path, seed: int | None) -> None:
    rng = random.Random(seed)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError(f"{path} must contain an experiments list")

    max_delta_by_group_class: dict[tuple[str, str], float] = {}
    for item in experiments:
        key = (str(item["output_subdir"]), _temperature_class(item))
        max_delta_by_group_class[key] = max(max_delta_by_group_class.get(key, 0.0), _delta_t(item))

    for item in experiments:
        model_a = rng.choice(MODEL_A_CHOICES)
        output_subdir = str(item["output_subdir"])
        item["model_a"] = model_a
        item["model_b"] = round(1.0 - model_a, 6)
        item["tau_fast_s"] = round(rng.uniform(*TAU_FAST_RANGES_BY_SUBDIR[output_subdir]), 6)
        item["tau_slow_s"] = round(rng.uniform(*TAU_SLOW_RANGE_S), 6)
        item["steady_state_beta"] = _beta_for_item(item, max_delta_by_group_class, rng)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomize selected tempdiffdata_config curve fields.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    randomize_config(args.config, args.seed)
    print(f"Updated config: {args.config}")
    print("Fields updated: steady_state_beta, model_a, model_b, tau_fast_s, tau_slow_s")


if __name__ == "__main__":
    main()
