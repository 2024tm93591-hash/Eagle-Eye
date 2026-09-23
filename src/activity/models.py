"""Deep sequence models for skeleton-based activity recognition (PyTorch)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_DIM


class ActivityLSTM(nn.Module):
    """Bidirectional 2-layer LSTM with temporal attention pooling."""

    def __init__(self, n_classes: int, in_dim: int = FEATURE_DIM, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_classes))

    def forward(self, x):                      # x: (B, T, F)
        h, _ = self.lstm(self.inp(x))          # (B, T, 2H)
        w = torch.softmax(self.attn(h), dim=1)  # (B, T, 1)
        return self.head((w * h).sum(1))


class _TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation), nn.BatchNorm1d(cout), nn.ReLU(),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class ActivityTCN(nn.Module):
    """Temporal Convolutional Network: dilated residual 1-D convolutions (dilation 1-2-4-8)."""

    def __init__(self, n_classes: int, in_dim: int = FEATURE_DIM, channels: int = 128, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        blocks, c = [], in_dim
        for d in (1, 2, 4, 8):
            blocks.append(_TemporalBlock(c, channels, 3, d, dropout))
            c = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, n_classes))

    def forward(self, x):                      # x: (B, T, F)
        h = self.tcn(self.norm(x).transpose(1, 2))   # (B, C, T)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


def build_model(kind: str, n_classes: int) -> nn.Module:
    return {"lstm": ActivityLSTM, "tcn": ActivityTCN}[kind](n_classes)
