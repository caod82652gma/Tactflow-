from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from features.geometry import (
    load_sensor_coordinates,
    nearest_tactile_columns_for_temperature,
    temperature_geometry,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--nearest", type=int, default=6)
    args = parser.parse_args()

    config = yaml.safe_load((project_root / args.config).read_text(encoding="utf-8"))
    thermal_columns = list(config["features"]["thermal_columns"])
    coordinates = load_sensor_coordinates(project_root, config)
    xy, height, sides = temperature_geometry(coordinates, thermal_columns)
    nearest = nearest_tactile_columns_for_temperature(coordinates, thermal_columns, args.nearest)

    for column, point, h, side in zip(thermal_columns, xy, height, sides, strict=True):
        print(
            f"{column}: side={side}, x={point[0]:.2f}mm, y={point[1]:.2f}mm, "
            f"height={h:.2f}mm, nearest={';'.join(nearest[column])}"
        )


if __name__ == "__main__":
    main()
