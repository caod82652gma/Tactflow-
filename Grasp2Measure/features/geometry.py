from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_sensor_coordinates(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    path_value = config["features"].get("sensor_coordinates_csv")
    if not path_value:
        raise ValueError("features.sensor_coordinates_csv is not configured.")
    path = resolve_project_path(project_root, path_value)
    if not path.exists():
        raise FileNotFoundError(f"Sensor coordinate file not found: {path}")
    return pd.read_csv(path)


def temperature_geometry(
    coordinates: pd.DataFrame,
    thermal_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    temp_rows = coordinates.loc[coordinates["kind"] == "temperature_patch"].copy()
    if temp_rows.empty:
        raise ValueError("No temperature_patch rows found in sensor coordinate file.")

    temp_rows["column_name"] = temp_rows["ad"].astype(str) + "_Temperature_mV"
    temp_rows = temp_rows.set_index("column_name")
    missing = [column for column in thermal_columns if column not in temp_rows.index]
    if missing:
        raise ValueError(f"Missing temperature coordinates for columns: {missing}")

    ordered = temp_rows.loc[thermal_columns]
    xy = ordered[["global_center_x_mm", "global_center_y_mm"]].to_numpy(dtype=np.float32)
    height = ordered["global_center_y_mm"].to_numpy(dtype=np.float32)
    return xy, height, ordered["side"].astype(str).tolist()


def nearest_tactile_columns_for_temperature(
    coordinates: pd.DataFrame,
    thermal_columns: list[str],
    k: int = 6,
) -> dict[str, list[str]]:
    tactile = coordinates.loc[coordinates["kind"].isin(["tactile_ad_cell", "tactile_interpolation_region"])].copy()
    temp_xy, _, _ = temperature_geometry(coordinates, thermal_columns)
    tactile_names = tactile["name"].astype(str).to_numpy()
    tactile_xy = tactile[["global_center_x_mm", "global_center_y_mm"]].to_numpy(dtype=np.float32)

    mapping: dict[str, list[str]] = {}
    for column, xy in zip(thermal_columns, temp_xy, strict=True):
        dist = np.linalg.norm(tactile_xy - xy[None, :], axis=1)
        nearest = np.argsort(dist)[:k]
        mapping[column] = tactile_names[nearest].tolist()
    return mapping
