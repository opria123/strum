"""Small neural chart-to-chart transform models.

This module deliberately has no audio dependency.  It is the first training
stage for tasks such as Expert → lower-difficulty authoring, where paired
human charts are the supervision signal and song-level evaluation matters.
"""
from __future__ import annotations

import torch
from torch import nn


class EventTransformMLP(nn.Module):
    """Predict target lane activations for an event from its source context."""

    def __init__(self, lane_count: int = 5, hidden_dim: int = 32) -> None:
        super().__init__()
        self.lane_count = lane_count
        self.network = nn.Sequential(
            nn.Linear(lane_count + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lane_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
