from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StandardScaler:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean = values.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = values.std(axis=0, keepdims=True).astype(np.float32)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        return ((values - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
