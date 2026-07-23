from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


TACTILE_RE = re.compile(r"^(?P<side>[LR])_(?P<body>.+)$")
CELL_RE = re.compile(r"(?P<pad>P1|P2|P3a|P3b)_R(?P<row>\d+)C(?P<col>\d+)$")
P1_P2_RE = re.compile(r"P1_P2_C(?P<col>\d+)$")
P3_RE = re.compile(r"P3a_P3b_R(?P<row>\d+)$")
P2_P3_RE = re.compile(r"P2_P3(?P<pad>a|b)_(?P<idx>\d+)$")

PAD_Y_OFFSET = {"P1": 0.0, "P2": 7.0, "P3a": 14.0, "P3b": 14.0}
PAD_X_OFFSET = {"P1": 0.0, "P2": 0.0, "P3a": 0.0, "P3b": 7.0}


@dataclass(frozen=True)
class PressureLayout:
    left_columns: list[str]
    right_columns: list[str]
    left_xy: np.ndarray
    right_xy: np.ndarray


def is_tactile_column(column: str) -> bool:
    if column in {"Volume", "SourceFile", "Index"}:
        return False
    if column.startswith("AD") or column.endswith("Temperature_mV") or column.endswith("TemperatureRaw16"):
        return False
    return column.startswith("L_") or column.startswith("R_")


def _coordinate_from_body(body: str) -> tuple[float, float]:
    match = CELL_RE.search(body)
    if match:
        pad = match.group("pad")
        row = float(match.group("row"))
        col = float(match.group("col"))
        return PAD_X_OFFSET[pad] + col, PAD_Y_OFFSET[pad] + row

    match = P1_P2_RE.search(body)
    if match:
        return float(match.group("col")), 7.0

    match = P3_RE.search(body)
    if match:
        return 6.5, 14.0 + float(match.group("row"))

    match = P2_P3_RE.search(body)
    if match:
        x = 3.0 if match.group("pad") == "a" else 6.0
        return x, 13.0 + float(match.group("idx"))

    if body == "P2_P3_center":
        return 4.5, 13.5

    return 0.0, 0.0


def infer_pressure_layout(columns: list[str]) -> PressureLayout:
    left_columns: list[str] = []
    right_columns: list[str] = []
    left_xy: list[tuple[float, float]] = []
    right_xy: list[tuple[float, float]] = []

    for column in columns:
        match = TACTILE_RE.match(column)
        if not match or not is_tactile_column(column):
            continue
        side = match.group("side")
        xy = _coordinate_from_body(match.group("body"))
        if side == "L":
            left_columns.append(column)
            left_xy.append(xy)
        else:
            right_columns.append(column)
            right_xy.append(xy)

    return PressureLayout(
        left_columns=left_columns,
        right_columns=right_columns,
        left_xy=np.asarray(left_xy, dtype=np.float32),
        right_xy=np.asarray(right_xy, dtype=np.float32),
    )


def _side_features(values: np.ndarray, xy: np.ndarray, threshold_mv: float) -> tuple[np.ndarray, np.ndarray]:
    abs_values = np.abs(values)
    mask = abs_values >= threshold_mv
    n_act = mask.sum(axis=1).astype(np.float32)
    safe_n = np.maximum(n_act, 1.0)
    weights = mask.astype(np.float32)

    x = xy[:, 0][None, :]
    y = xy[:, 1][None, :]
    x_bar = (weights * x).sum(axis=1) / safe_n
    y_bar = (weights * y).sum(axis=1) / safe_n
    dx = x - x_bar[:, None]
    dy = y - y_bar[:, None]
    sig_x2 = (weights * dx * dx).sum(axis=1) / safe_n
    sig_y2 = (weights * dy * dy).sum(axis=1) / safe_n
    sig_xy = (weights * dx * dy).sum(axis=1) / safe_n
    v_peak = abs_values.max(axis=1).astype(np.float32)
    v_mean = (weights * abs_values).sum(axis=1) / safe_n

    feats = np.stack([n_act, x_bar, y_bar, sig_x2, sig_y2, sig_xy, v_peak, v_mean], axis=1)
    feats[n_act == 0, 1:6] = 0.0

    col_index = np.clip(np.rint(xy[:, 0]).astype(int), 1, 6) - 1
    projection = np.zeros((values.shape[0], 6), dtype=np.float32)
    for idx in range(6):
        projection[:, idx] = mask[:, col_index == idx].sum(axis=1)
    return feats.astype(np.float32), projection


def extract_pressure_features(
    df: pd.DataFrame,
    layout: PressureLayout,
    threshold_mv: float,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = df[layout.left_columns].to_numpy(dtype=np.float32)
    right_values = df[layout.right_columns].to_numpy(dtype=np.float32)
    left_feats, left_proj = _side_features(left_values, layout.left_xy, threshold_mv)
    right_feats, right_proj = _side_features(right_values, layout.right_xy, threshold_mv)
    pressure = np.concatenate([left_feats, right_feats], axis=1)
    projection = left_proj + right_proj
    return pressure, projection
