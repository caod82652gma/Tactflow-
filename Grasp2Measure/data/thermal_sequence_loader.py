from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features.geometry import resolve_project_path


class SequenceStandardScaler:
    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> None:
        self.mean = mean
        self.std = std

    def fit(self, values: np.ndarray) -> "SequenceStandardScaler":
        flat = values.reshape(-1, values.shape[-1])
        self.mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = flat.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        return ((values - self.mean[None, :, :]) / self.std[None, :, :]).astype(np.float32)


class ThermalSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def _split_path(config: dict[str, Any], project_root: Path, split_name: str) -> Path:
    split_root = resolve_project_path(project_root, config["thermal_sequence"]["split_root"])
    configured = str(config["thermal_sequence"].get(f"{split_name}_split", f"{split_name}.csv"))
    path = Path(configured)
    if not path.is_absolute():
        path = split_root / path
    return path


def _read_split_files(data_root: Path, split_path: Path) -> list[Path]:
    if not split_path.exists():
        raise FileNotFoundError(f"Missing thermal sequence split file: {split_path}")
    split_frame = pd.read_csv(split_path)
    if "path" not in split_frame.columns:
        raise ValueError(f"Thermal sequence split file must contain a path column: {split_path}")

    files: list[Path] = []
    for raw_path in split_frame["path"].astype(str):
        path = Path(raw_path).with_suffix(".npz")
        if not path.is_absolute():
            path = data_root / path
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No thermal sequence files listed in {split_path}")
    return files


def _fit_sequence_length(values: np.ndarray, sequence_length: int | None) -> np.ndarray:
    if sequence_length is None or values.shape[0] == sequence_length:
        return values.astype(np.float32)
    if values.shape[0] == 0:
        raise ValueError("Cannot resize an empty sequence.")
    if values.shape[0] == 1:
        return np.repeat(values, sequence_length, axis=0).astype(np.float32)
    if values.shape[0] > sequence_length:
        indices = np.linspace(0, values.shape[0] - 1, sequence_length).round().astype(np.int64)
        return values[indices].astype(np.float32)
    pad = np.repeat(values[-1:], sequence_length - values.shape[0], axis=0)
    return np.concatenate([values, pad], axis=0).astype(np.float32)


def load_thermal_sequence_table(config: dict[str, Any], split_name: str | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    data_root = resolve_project_path(project_root, config["thermal_sequence"]["data_root"])
    if split_name is None:
        files = sorted(data_root.glob(str(config["thermal_sequence"].get("file_pattern", "*.npz"))))
    else:
        files = _read_split_files(data_root, _split_path(config, project_root, split_name))
    if not files:
        raise FileNotFoundError(f"No thermal sequence files found in {data_root}")
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing thermal sequence sample: {missing[0]}")

    branch_keys = ["b1_seq", "b2_seq", "b3_seq", "b4_seq", "b5_seq", "b6_seq"]
    rows: dict[str, list[np.ndarray]] = {key: [] for key in branch_keys}
    labels: list[int] = []
    source_files: list[str] = []
    class_order: list[str] | None = None
    sequence_length = int(config["thermal_sequence"]["sequence_length"]) if config["thermal_sequence"].get("sequence_length") else None

    for path in files:
        with np.load(path, allow_pickle=True) as sample:
            for key in branch_keys:
                rows[key].append(_fit_sequence_length(sample[key].astype(np.float32), sequence_length))
            labels.append(int(sample["y"]))
            source_files.append(str(sample["source_file"]))
            if class_order is None:
                class_order = [str(value) for value in sample["class_order"]]

    return {
        **{key: np.stack(value).astype(np.float32) for key, value in rows.items()},
        "y": np.asarray(labels, dtype=np.int64),
        "source_files": source_files,
        "class_order": class_order or list(config["thermal_sequence"]["level_order"]),
    }
