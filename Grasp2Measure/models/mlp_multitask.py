from __future__ import annotations

import torch
from torch import nn


class TactileContainerClassifier(nn.Module):
    """MLP classifier for one grasp trial summarized from tactile frames."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Grasp2MeasureNet(nn.Module):
    """Three-branch asymmetric cascade for container class, level, and volume."""

    def __init__(
        self,
        num_classes: int,
        radius_mm: list[float],
        hidden_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.container_head = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.level_head = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.register_buffer("radius_mm", torch.tensor(radius_mm, dtype=torch.float32))

    def forward(self, pressure: torch.Tensor, thermal: torch.Tensor) -> dict[str, torch.Tensor]:
        class_logits = self.container_head(pressure)
        class_prob = torch.softmax(class_logits, dim=1)
        radius = class_prob @ self.radius_mm
        level_mm = self.level_head(thermal)
        volume_ml = torch.pi * radius[:, None].pow(2) * level_mm / 1000.0
        return {
            "class_logits": class_logits,
            "class_prob": class_prob,
            "radius_mm": radius[:, None],
            "level_mm": level_mm,
            "volume_ml": volume_ml,
        }
