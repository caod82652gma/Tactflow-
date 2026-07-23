from __future__ import annotations

import numpy as np


def split_indices(
    n_rows: int,
    groups: np.ndarray | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if groups is None:
        indices = np.arange(n_rows)
        rng.shuffle(indices)
        n_test = int(round(n_rows * test_ratio))
        n_val = int(round(n_rows * val_ratio))
        return indices[n_test + n_val :], indices[n_test : n_test + n_val], indices[:n_test]

    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_test = max(1, int(round(len(unique_groups) * test_ratio)))
    n_val = max(1, int(round(len(unique_groups) * val_ratio)))
    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    test_mask = np.asarray([group in test_groups for group in groups])
    val_mask = np.asarray([group in val_groups for group in groups])
    train_mask = ~(test_mask | val_mask)
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)


def split_train_val_indices(
    n_rows: int,
    groups: np.ndarray | None,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if groups is None:
        indices = np.arange(n_rows)
        rng.shuffle(indices)
        n_val = int(round(n_rows * val_ratio))
        return indices[n_val:], indices[:n_val]

    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_ratio)))
    val_groups = set(unique_groups[:n_val])
    val_mask = np.asarray([group in val_groups for group in groups])
    train_mask = ~val_mask
    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)
