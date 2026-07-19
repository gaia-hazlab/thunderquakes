"""Spectrogram CNN — seismic-only (Model A) and dual-branch (Model B) (WS4).

Requires the 'ml' pixi env (pytorch). Not imported by ``models/__init__`` so the
default env can still import the package; import this module explicitly under -e ml.

Design notes:
- AdaptiveAvgPool makes the head independent of the exact (freq×time) input size.
- Dropout is kept between conv blocks and before the head so the same network
  supports MC-dropout uncertainty at inference (WS4 uncertainty work).
"""

from __future__ import annotations

import torch
from torch import nn


def _block(cin, cout, dropout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class SeismicCNN(nn.Module):
    """Small 2D CNN over a log-spectrogram (Model A, seismic-only)."""

    def __init__(self, n_classes: int, dropout: float = 0.3, in_ch: int = 1, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, width, dropout),
            _block(width, width * 2, dropout),
            _block(width * 2, width * 4, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_seismic_cnn(n_classes: int, dropout: float = 0.3, in_ch: int = 1) -> SeismicCNN:
    """Model A factory: single-branch spectrogram CNN over 3-C or 1-C seismic input."""
    return SeismicCNN(n_classes=n_classes, dropout=dropout, in_ch=in_ch)
