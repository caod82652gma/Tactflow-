from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from features.pressure import extract_pressure_features, infer_pressure_layout, is_tactile_column
from features.thermal import extract_thermal_features
from features.geometry import load_sensor_coordinates, temperature_geometry


class GraspMeasureDataset(Dataset):
    def __init__(
        self,
        pressure: np.ndarray,
        thermal: np.ndarray,
        class_labels: np.ndarray,
        volume_ml: np.ndarray,
        level_mm: np.ndarray,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required to construct GraspMeasureDataset.")
        self.pressure = torch.from_numpy(pressure.astype(np.float32))
        self.thermal = torch.from_numpy(thermal.astype(np.float32))
        self.class_labels = torch.from_numpy(class_labels.astype(np.int64))
        self.volume_ml = torch.from_numpy(volume_ml.astype(np.float32)[:, None])
        self.level_mm = torch.from_numpy(level_mm.astype(np.float32)[:, None])

    def __len__(self) -> int:
        return len(self.class_labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "pressure": self.pressure[index],
            "thermal": self.thermal[index],
            "class": self.class_labels[index],
            "volume_ml": self.volume_ml[index],
            "level_mm": self.level_mm[index],
        }


class TactileTrialDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("torch is required to construct TactileTrialDataset.")
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def parse_volume_ml(value: Any) -> float:
    text = str(value).strip().lower().replace(" ", "")
    if text.endswith("ml"):
        text = text[:-2]
    return float(text)


def _stable_pair_index(values: np.ndarray) -> tuple[int, float]:
    if len(values) <= 1:
        return 0, 0.0

    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (values - values.mean(axis=0)) / scale
    pair_motion = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    index = int(pair_motion.argmin())
    return index, float(pair_motion[index])


def _summarize_trial(
    frame: pd.DataFrame,
    tactile_columns: list[str],
    summary: str,
) -> tuple[np.ndarray, int | None, float | None]:
    values = frame[tactile_columns].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    if summary == "mean":
        return mean.astype(np.float32), None, None
    if summary == "mean_std":
        std = values.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float32), None, None
    if summary == "stable_frame":
        stable_index, stability_score = _stable_pair_index(values)
        return values[stable_index].astype(np.float32), stable_index, stability_score
    if summary == "stable_pair_mean":
        stable_index, stability_score = _stable_pair_index(values)
        if stable_index + 1 < len(values):
            stable_values = values[stable_index : stable_index + 2].mean(axis=0)
        else:
            stable_values = values[stable_index]
        return stable_values.astype(np.float32), stable_index, stability_score
    raise ValueError(f"Unsupported tactile_summary: {summary}")


def _pressure16_trial_features(
    frame: pd.DataFrame,
    tactile_columns: list[str],
    summary: str,
    threshold_mv: float,
) -> tuple[np.ndarray, int | None, float | None]:
    summarized, selected_frame_index, stability_score = _summarize_trial(frame, tactile_columns, summary)
    layout = infer_pressure_layout(tactile_columns)
    summarized_frame = pd.DataFrame([summarized], columns=tactile_columns)
    pressure, _ = extract_pressure_features(summarized_frame, layout, threshold_mv)
    return pressure[0].astype(np.float32), selected_frame_index, stability_score


def _load_tactile_trial_frame(
    data_root: Path,
    interpolation: str,
    label: str,
    path: Path,
    max_frames: int | None,
) -> pd.DataFrame:
    if interpolation == "base":
        return pd.read_csv(path, nrows=max_frames)

    base_path = data_root / "base" / label / path.name
    if not base_path.exists():
        raise FileNotFoundError(f"Missing paired base CSV for {path}: {base_path}")

    base_frame = pd.read_csv(base_path, nrows=max_frames)
    interp_frame = pd.read_csv(path, nrows=max_frames)
    if len(base_frame) != len(interp_frame):
        raise ValueError(
            f"Paired CSV row count mismatch: {base_path} has {len(base_frame)}, "
            f"{path} has {len(interp_frame)}"
        )

    key_columns = [column for column in ["Volume", "SourceFile", "Index"] if column in base_frame.columns]
    extra_columns = [
        column
        for column in interp_frame.columns
        if is_tactile_column(column) and column not in base_frame.columns
    ]
    return pd.concat(
        [
            base_frame[key_columns],
            base_frame[[column for column in base_frame.columns if is_tactile_column(column)]],
            interp_frame[extra_columns],
        ],
        axis=1,
    )


def _resolve_tactile_root(config: dict[str, Any]) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    tactile_root = Path(config["data"]["tactile_root"])
    if not tactile_root.is_absolute():
        tactile_root = (project_root / tactile_root).resolve()
    return tactile_root


def _resolve_tactile_split_root(config: dict[str, Any], tactile_root: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    split_root = Path(config["data"].get("split_root", tactile_root / "splits"))
    if not split_root.is_absolute():
        split_root = (project_root / split_root).resolve()
    return split_root


def _read_split_paths(split_root: Path, split_name: str) -> list[tuple[str, Path]]:
    split_dir = split_root
    split_path = split_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split_frame = pd.read_csv(split_path)
    if "path" not in split_frame.columns:
        raise ValueError(f"Split file must contain a path column: {split_path}")

    rows: list[tuple[str, Path]] = []
    for raw_path in split_frame["path"].astype(str):
        path = Path(raw_path)
        if path.is_absolute() or len(path.parts) != 2:
            raise ValueError(f"Split path must look like <label>/<file>.csv: {raw_path}")
        label = path.parts[0]
        rows.append((label, path))
    return rows


def load_tactile_split_table(config: dict[str, Any], split_name: str) -> dict[str, Any]:
    tactile_root = _resolve_tactile_root(config)
    split_root = _resolve_tactile_split_root(config, tactile_root)
    data_root = tactile_root / "data"

    interpolation = str(config["data"]["interpolation"])
    class_order = list(config["labels"]["class_order"])
    class_to_idx = {label: idx for idx, label in enumerate(class_order)}
    max_frames = config["data"].get("max_frames_per_trial")
    summary = str(config["features"].get("tactile_summary", "stable_frame"))
    input_mode = str(config["features"].get("input_mode", "tactile_summary"))
    if input_mode not in {"tactile_summary", "pressure_16"}:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    rows: list[np.ndarray] = []
    labels: list[int] = []
    trial_ids: list[str] = []
    selected_frame_indices: list[int] = []
    stability_scores: list[float] = []
    tactile_columns_ref: list[str] | None = None

    for label, split_path in _read_split_paths(split_root, split_name):
        path = data_root / interpolation / split_path
        if label not in class_to_idx:
            raise ValueError(f"Label {label!r} from {path} is not in class_order")
        if not path.exists():
            raise FileNotFoundError(f"Missing split sample: {path}")

        frame = _load_tactile_trial_frame(data_root, interpolation, label, path, max_frames)
        tactile_columns = [column for column in frame.columns if is_tactile_column(column)]
        if tactile_columns_ref is None:
            tactile_columns_ref = tactile_columns
        elif tactile_columns != tactile_columns_ref:
            raise ValueError(f"Tactile columns differ in {path}")
        if input_mode == "pressure_16":
            row, selected_frame_index, stability_score = _pressure16_trial_features(
                frame,
                tactile_columns,
                summary,
                float(config["features"]["contact_threshold_mv"]),
            )
        else:
            row, selected_frame_index, stability_score = _summarize_trial(frame, tactile_columns, summary)
        rows.append(row)
        labels.append(class_to_idx[label])
        trial_ids.append(str(path))
        selected_frame_indices.append(-1 if selected_frame_index is None else selected_frame_index)
        stability_scores.append(np.nan if stability_score is None else stability_score)

    if not rows:
        raise FileNotFoundError(f"No tactile CSV files found in split: {split_name}")

    return {
        "x": np.stack(rows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.int64),
        "trial_ids": trial_ids,
        "selected_frame_indices": np.asarray(selected_frame_indices, dtype=np.int64),
        "stability_scores": np.asarray(stability_scores, dtype=np.float32),
        "class_order": class_order,
        "tactile_columns": tactile_columns_ref or [],
        "root": str(tactile_root),
        "split": split_name,
    }


def load_feature_table(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    csv_dir = Path(config["data"]["csv_dir"])
    if not csv_dir.is_absolute():
        csv_dir = (root / csv_dir).resolve()

    interpolation = config["data"]["interpolation"]
    pattern = config["data"]["file_pattern"].format(interpolation=interpolation)
    files = sorted(csv_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched {csv_dir / pattern}")

    max_rows = config["data"].get("max_rows_per_file")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, nrows=max_rows)
        frame["ContainerLabel"] = frame["Volume"].astype(str)
        frame["CsvPath"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    layout = infer_pressure_layout(list(df.columns))
    pressure, projection = extract_pressure_features(
        df,
        layout,
        float(config["features"]["contact_threshold_mv"]),
    )
    coordinates = load_sensor_coordinates(root, config)
    _, thermal_heights_mm, _ = temperature_geometry(
        coordinates,
        list(config["features"]["thermal_columns"]),
    )
    thermal = extract_thermal_features(
        df,
        list(config["features"]["thermal_columns"]),
        projection,
        list(config["features"]["thermal_contact_columns"]),
        thermal_heights_mm,
    )

    class_order = list(config["labels"]["class_order"])
    class_to_idx = {label: idx for idx, label in enumerate(class_order)}
    labels_text = df["ContainerLabel"].astype(str).to_numpy()
    missing = sorted(set(labels_text) - set(class_to_idx))
    if missing:
        raise ValueError(f"Labels not found in class_order: {missing}")

    class_labels = np.asarray([class_to_idx[label] for label in labels_text], dtype=np.int64)
    radius_map = {str(k): float(v) for k, v in config["labels"]["radius_mm"].items()}
    volume_ml = np.zeros(len(labels_text), dtype=np.float32)
    level_mm = np.zeros(len(labels_text), dtype=np.float32)
    groups = (
        (df["ContainerLabel"].astype(str) + "::" + df["SourceFile"].astype(str)).to_numpy()
        if config["data"].get("split_by_source_file", True)
        else None
    )

    return {
        "pressure": pressure,
        "thermal": thermal,
        "class_labels": class_labels,
        "volume_ml": volume_ml,
        "level_mm": level_mm,
        "groups": groups,
        "class_order": class_order,
        "radius_mm": np.asarray([radius_map[label] for label in class_order], dtype=np.float32),
        "files": [str(path) for path in files],
    }
