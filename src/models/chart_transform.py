"""Small neural chart-to-chart transform models."""

from __future__ import annotations

import torch
from torch import nn


class EventTransformMLP(nn.Module):
    """Predict target lane activations for an event from its source context."""

    def __init__(
        self, lane_count: int = 5, hidden_dim: int = 32, audio_feature_dim: int = 0
    ) -> None:
        super().__init__()
        if audio_feature_dim < 0:
            raise ValueError("audio_feature_dim must be non-negative")
        self.lane_count = lane_count
        self.audio_feature_dim = audio_feature_dim
        self.network = nn.Sequential(
            nn.Linear(lane_count + 2 + audio_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lane_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
