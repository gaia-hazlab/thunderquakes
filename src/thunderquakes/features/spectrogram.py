"""Spectrogram feature extraction — CNN input representation (WS1/WS4)."""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram


def log_spectrogram(
    data, fs, nperseg: int = 256, noverlap_frac: float = 0.85, eps: float = 1e-20
):
    """Return ``(f, t, S_db)`` log-power spectrogram, as used for CNN input.

    Consolidated from the ad-hoc spectrogram calls scattered through cells
    30/34/36 of the prototype.
    """
    noverlap = int(nperseg * noverlap_frac)
    f, t, sxx = spectrogram(
        np.asarray(data, dtype=float), fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    return f, t, 10.0 * np.log10(sxx + eps)
