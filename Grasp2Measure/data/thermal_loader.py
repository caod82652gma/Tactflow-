from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.loader import _pressure16_trial_features, _summarize_trial
from data.normalize import StandardScaler
from features.geometry import load_sensor_coordinates, resolve_project_path, temperature_geometry
from models.mlp_multitask import TactileContainerClassifier


LEVEL_ORDER = ["Low", "Mid", "High"]


class ThermalLevelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _normalize_level(value: object) -> str:
    text = str(value).strip().lower()
    if text == "low":
        return "Low"
    if text == "mid":
        return "Mid"
    if text == "high":
        return "High"
    raise ValueError(f"Unsupported LevelClass: {value}")


def _base_tactile_columns(coordinates: pd.DataFrame, frame: pd.DataFrame) -> list[str]:
    names = coordinates.loc[coordinates["kind"] == "tactile_ad_cell", "name"].astype(str).tolist()
    return [name for name in names if name in frame.columns]


def _temperature_columns(config: dict[str, Any]) -> list[str]:
    labels = config["thermal"]["thermocouple_order"]
    return [f"{ad}_Temperature_C" for ad in labels]


def _neighborhoods(
    coordinates: pd.DataFrame,
    thermal_columns: list[str],
    tactile_columns: list[str],
    radius_mm: float,
) -> list[list[str]]:
    temp_xy, _, _ = temperature_geometry(
        coordinates,
        [column.replace("_Temperature_C", "_Temperature_mV") for column in thermal_columns],
    )
    tactile = coordinates.set_index("name").loc[tactile_columns]
    tactile_xy = tactile[["global_center_x_mm", "global_center_y_mm"]].to_numpy(dtype=np.float32)

    neighborhoods: list[list[str]] = []
    for xy in temp_xy:
        distances = np.linalg.norm(tactile_xy - xy[None, :], axis=1)
        selected = [column for column, distance in zip(tactile_columns, distances, strict=True) if distance < radius_mm]
        if not selected:
            nearest = np.argsort(distances)[:6]
            selected = [tactile_columns[index] for index in nearest]
        neighborhoods.append(selected)
    return neighborhoods


def _read_trial(path: Path, steady_tail_frames: int) -> tuple[pd.Series, pd.DataFrame]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty thermal CSV: {path}")
    tail = frame.tail(max(1, steady_tail_frames))
    summary = tail.mean(numeric_only=True)
    first = frame.iloc[0].copy()
    for column, value in summary.items():
        first[column] = value
    first["CsvPath"] = str(path)
    return first, frame


def _split_path(config: dict[str, Any], project_root: Path, split_name: str) -> Path:
    split_root = resolve_project_path(project_root, config["thermal"]["split_root"])
    configured = str(config["thermal"].get(f"{split_name}_split", f"{split_name}.csv"))
    path = Path(configured)
    if not path.is_absolute():
        path = split_root / path
    return path


def _read_split_files(data_root: Path, split_path: Path) -> list[Path]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing thermal split file: {split_path}")
    split_frame = pd.read_csv(split_path)
    if "path" not in split_frame.columns:
        raise ValueError(f"Thermal split file must contain a path column: {split_path}")

    files: list[Path] = []
    for raw_path in split_frame["path"].astype(str):
        path = Path(raw_path)
        if not path.is_absolute():
            path = data_root / path
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No thermal CSV files listed in {split_path}")
    return files


def _robust_sigma(values: np.ndarray, floor: float) -> np.ndarray:
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0)
    sigma = (1.4826 * mad).astype(np.float32)
    sigma[sigma < floor] = floor
    return sigma


def _resolve_tactile_checkpoint(project_root: Path, config: dict[str, Any], override: str | None) -> Path:
    raw_path = override or config["thermal"].get("tactile_classifier_checkpoint")
    if not raw_path:
        raise ValueError("B2 requires thermal.tactile_classifier_checkpoint or --tactile-checkpoint.")
    return resolve_project_path(project_root, str(raw_path))


def _tactile_classifier_features(
    frame: pd.DataFrame,
    tactile_columns: list[str],
    checkpoint_config: dict[str, Any],
) -> np.ndarray:
    feature_config = checkpoint_config.get("features", {})
    summary = str(feature_config.get("tactile_summary", "stable_frame"))
    input_mode = str(feature_config.get("input_mode", "tactile_summary"))
    if input_mode == "pressure_16":
        row, _, _ = _pressure16_trial_features(
            frame,
            tactile_columns,
            summary,
            float(feature_config["contact_threshold_mv"]),
        )
        return row.astype(np.float32)
    if input_mode == "tactile_summary":
        row, _, _ = _summarize_trial(frame, tactile_columns, summary)
        return row.astype(np.float32)
    raise ValueError(f"Unsupported tactile classifier input_mode: {input_mode}")


def _predict_container_one_hot(
    frames: list[pd.DataFrame],
    tactile_columns: list[str],
    checkpoint_path: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint["config"]
    x = np.stack(
        [_tactile_classifier_features(frame, tactile_columns, checkpoint_config) for frame in frames]
    ).astype(np.float32)
    if x.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Tactile classifier expected {checkpoint['input_dim']} features, "
            f"but thermal CSV tactile features have {x.shape[1]}."
        )

    scaler = StandardScaler(checkpoint["x_mean"], checkpoint["x_std"])
    x_scaled = scaler.transform(x)
    class_order = list(checkpoint["class_order"])
    model = TactileContainerClassifier(
        input_dim=int(checkpoint["input_dim"]),
        num_classes=len(class_order),
        hidden_dim=max(32, int(checkpoint_config["model"]["hidden_dim"]) * 8),
        dropout=float(checkpoint_config["model"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_scaled))
        pred_idx = logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    one_hot = np.zeros((len(pred_idx), len(class_order)), dtype=np.float32)
    one_hot[np.arange(len(pred_idx)), pred_idx] = 1.0
    return one_hot, class_order, pred_idx


def _pressure16_features(
    frames: list[pd.DataFrame],
    tactile_columns: list[str],
    checkpoint_path: Path,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_config = checkpoint.get("config", {}).get("features", {})
    summary = str(feature_config.get("tactile_summary", "stable_frame"))
    threshold_mv = float(feature_config.get("contact_threshold_mv", 50.0))
    rows = [
        _pressure16_trial_features(frame, tactile_columns, summary, threshold_mv)[0]
        for frame in frames
    ]
    return np.stack(rows).astype(np.float32)


def load_thermal_level_table(
    config: dict[str, Any],
    split_name: str | None = None,
    tactile_checkpoint: str | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    data_root = resolve_project_path(project_root, config["thermal"]["data_root"])
    tactile_checkpoint_path = _resolve_tactile_checkpoint(project_root, config, tactile_checkpoint)
    if split_name is None:
        files = sorted(data_root.glob(str(config["thermal"].get("file_pattern", "*.csv"))))
    else:
        files = _read_split_files(data_root, _split_path(config, project_root, split_name))
    if not files:
        raise FileNotFoundError(f"No thermal CSV files found in {data_root}")
    missing_files = [path for path in files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing thermal split sample: {missing_files[0]}")

    coordinates = load_sensor_coordinates(project_root, config)
    first_frame = pd.read_csv(files[0], nrows=1)
    tactile_columns = _base_tactile_columns(coordinates, first_frame)
    if not tactile_columns:
        raise ValueError("No base tactile cell columns found in thermal CSV.")

    thermal_columns = _temperature_columns(config)
    missing_thermal = [column for column in thermal_columns if column not in first_frame.columns]
    if missing_thermal:
        raise ValueError(f"Missing thermal columns: {missing_thermal}")
    neighborhoods = _neighborhoods(
        coordinates,
        thermal_columns,
        tactile_columns,
        float(config["thermal"]["neighborhood_radius_mm"]),
    )

    rows: list[pd.Series] = []
    frames: list[pd.DataFrame] = []
    tactile_values: list[np.ndarray] = []
    steady_tail_frames = int(config["thermal"]["steady_tail_frames"])
    for path in files:
        row, frame = _read_trial(path, steady_tail_frames)
        rows.append(row)
        frames.append(frame)
        tactile_values.append(row[tactile_columns].to_numpy(dtype=np.float32))

    predicted_one_hot, container_order, predicted_container_idx = _predict_container_one_hot(
        frames,
        tactile_columns,
        tactile_checkpoint_path,
    )
    pressure16 = _pressure16_features(frames, tactile_columns, tactile_checkpoint_path)
    tactile_matrix = np.stack(tactile_values).astype(np.float32)
    sigma = _robust_sigma(tactile_matrix, float(config["thermal"]["sigma_floor"]))
    threshold = float(config["thermal"]["contact_sigma_multiplier"]) * sigma
    use_abs = bool(config["thermal"].get("contact_use_abs", True))

    class_order = list(config["thermal"]["level_order"])
    class_to_idx = {label: idx for idx, label in enumerate(class_order)}

    b1_rows: list[np.ndarray] = []
    b2_rows: list[np.ndarray] = []
    b3_rows: list[np.ndarray] = []
    b4_rows: list[np.ndarray] = []
    b5_rows: list[np.ndarray] = []
    b6_rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    ambient_c: list[float] = []
    liquid_c: list[float] = []
    delta_rows: list[np.ndarray] = []
    alpha_rows: list[np.ndarray] = []
    source_files: list[str] = []

    for row, tactile, one_hot, pressure_row in zip(rows, tactile_matrix, predicted_one_hot, pressure16, strict=True):
        level = _normalize_level(row["LevelClass"])

        ambient = float(row["Ambient_C"])
        liquid = float(row["Liquid_C"])
        temperatures = row[thermal_columns].to_numpy(dtype=np.float32)
        delta = temperatures - ambient

        mask_values = np.abs(tactile) if use_abs else tactile
        contact_mask = mask_values > threshold
        alpha = np.asarray(
            [
                float(contact_mask[[tactile_columns.index(column) for column in neighborhood]].mean())
                for neighborhood in neighborhoods
            ],
            dtype=np.float32,
        )
        active_values = mask_values[contact_mask]
        v_mean = float(active_values.mean()) if active_values.size else 0.0

        b1 = delta.astype(np.float32)
        weighted_delta = (alpha * delta).astype(np.float32)
        tau_hat = np.asarray([v_mean], dtype=np.float32)
        b2 = weighted_delta
        b3 = np.concatenate([weighted_delta, alpha, tau_hat])
        b4 = np.concatenate([weighted_delta, alpha, one_hot])
        b5 = np.concatenate([weighted_delta, alpha, one_hot, tau_hat])
        b6 = np.concatenate([pressure_row, delta.astype(np.float32)])

        b1_rows.append(b1)
        b2_rows.append(b2.astype(np.float32))
        b3_rows.append(b3.astype(np.float32))
        b4_rows.append(b4.astype(np.float32))
        b5_rows.append(b5.astype(np.float32))
        b6_rows.append(b6.astype(np.float32))
        labels.append(class_to_idx[level])
        source = str(row.get("SourceFile", row["CsvPath"]))
        groups.append(source)
        source_files.append(source)
        ambient_c.append(ambient)
        liquid_c.append(liquid)
        delta_rows.append(delta.astype(np.float32))
        alpha_rows.append(alpha)

    return {
        "b1_x": np.stack(b1_rows).astype(np.float32),
        "b2_x": np.stack(b2_rows).astype(np.float32),
        "b3_x": np.stack(b3_rows).astype(np.float32),
        "b4_x": np.stack(b4_rows).astype(np.float32),
        "b5_x": np.stack(b5_rows).astype(np.float32),
        "b6_x": np.stack(b6_rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups),
        "class_order": class_order,
        "container_order": container_order,
        "predicted_container_idx": predicted_container_idx,
        "tactile_checkpoint": str(tactile_checkpoint_path),
        "ambient_c": np.asarray(ambient_c, dtype=np.float32),
        "liquid_c": np.asarray(liquid_c, dtype=np.float32),
        "delta_t": np.stack(delta_rows).astype(np.float32),
        "alpha": np.stack(alpha_rows).astype(np.float32),
        "pressure16": pressure16,
        "source_files": source_files,
        "thermal_columns": thermal_columns,
        "tactile_columns": tactile_columns,
        "neighborhoods": neighborhoods,
    }
