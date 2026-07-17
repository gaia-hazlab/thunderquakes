"""Per-class signal characterization (WS1 deliverable).

Quantify, for each PNWML class (thunder / earthquake / explosion / surface /
noise), the discriminating properties: dominant frequency band, duration,
spectral flatness (thunder is broadband/noise-like vs an impulsive P-wave),
kurtosis, and — where infrasound is co-located — the seismo-acoustic delay.

TODO(WS1): implement feature extraction over waveform windows and emit the
per-class distribution plots + summary table for the report.
"""
from __future__ import annotations

import numpy as np


def spectral_flatness(power: np.ndarray, eps: float = 1e-20) -> float:
    """Geometric-mean / arithmetic-mean of the power spectrum (Wiener entropy).

    ~1 => broadband/noise-like (thunder); ~0 => peaky/impulsive (earthquake).
    """
    power = np.asarray(power, dtype=float) + eps
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))
