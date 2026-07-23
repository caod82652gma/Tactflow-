from __future__ import annotations

import numpy as np
import pandas as pd


def extract_thermal_features(
    df: pd.DataFrame,
    thermal_columns: list[str],
    contact_projection: np.ndarray,
    thermal_contact_columns: list[int],
    thermal_heights_mm: np.ndarray | None = None,
) -> np.ndarray:
    missing = [column for column in thermal_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing thermal columns: {missing}")

    thermal = df[thermal_columns].to_numpy(dtype=np.float32)
    baseline = np.median(thermal, axis=1, keepdims=True)
    delta = thermal - baseline

    if contact_projection.size:
        max_proj = np.maximum(contact_projection.max(axis=1, keepdims=True), 1.0)
        col_weights = contact_projection / max_proj
        channel_indices = np.asarray(thermal_contact_columns, dtype=int) - 1
        channel_indices = np.clip(channel_indices, 0, contact_projection.shape[1] - 1)
        attention = col_weights[:, channel_indices]
    else:
        attention = np.ones_like(delta, dtype=np.float32)

    weighted_delta = delta * attention
    span = np.maximum(np.ptp(weighted_delta, axis=1, keepdims=True), 1e-6)
    prob = (weighted_delta - weighted_delta.min(axis=1, keepdims=True)) / span
    if thermal_heights_mm is None:
        sensor_pos = np.linspace(0.0, 1.0, weighted_delta.shape[1], dtype=np.float32)
    else:
        sensor_pos = np.asarray(thermal_heights_mm, dtype=np.float32)
        height_span = max(float(sensor_pos.max() - sensor_pos.min()), 1e-6)
        sensor_pos = (sensor_pos - sensor_pos.min()) / height_span
    sensor_pos = sensor_pos[None, :]
    h_prior = (prob * sensor_pos).sum(axis=1, keepdims=True) / np.maximum(
        prob.sum(axis=1, keepdims=True), 1e-6
    )
    return np.concatenate([weighted_delta, h_prior.astype(np.float32)], axis=1)
