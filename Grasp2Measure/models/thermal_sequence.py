from __future__ import annotations

import math

import torch
from torch import nn


def _attention_heads(hidden_dim: int) -> int:
    for heads in (4, 3, 2):
        if hidden_dim % heads == 0:
            return heads
    return 1


class ThermalSequenceClassifier(nn.Module):
    """Sequence classifier used by the thermal B1-B6 sequence ablations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        encoder: str = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder
        if encoder == "gru":
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            encoded_dim = hidden_dim
        elif encoder == "lstm":
            self.encoder = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            encoded_dim = hidden_dim
        elif encoder == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            encoded_dim = hidden_dim
        elif encoder == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=_attention_heads(hidden_dim),
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=min(2, max(1, num_layers)))
            encoded_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported sequence encoder: {encoder}")

        self.head = nn.Sequential(
            nn.Linear(encoded_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_name == "gru":
            _, hidden = self.encoder(x)
            encoded = hidden[-1]
        elif self.encoder_name == "lstm":
            _, (hidden, _) = self.encoder(x)
            encoded = hidden[-1]
        elif self.encoder_name == "cnn":
            encoded = self.encoder(x.transpose(1, 2)).squeeze(-1)
        else:
            x = self.input_proj(x) * math.sqrt(float(self.input_proj.out_features))
            encoded = self.encoder(x).mean(dim=1)
        return self.head(encoded)
