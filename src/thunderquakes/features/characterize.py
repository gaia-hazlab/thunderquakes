"""Per-class signal characterization (WS1 deliverable).

Quantify, for each PNWML class (thunder / earthquake / explosion / surface /
noise / sonic boom), the discriminating properties: dominant frequency band,
spectral shape, envelope duration, spectral flatness (thunder is broadband /
noise-like vs an impulsive earthquake P-wave), and kurtosis.

Feature functions here are pure (array in → scalar/array out) so they are unit-
tested without any network. ``scripts/characterize_pnwml.py`` reconstructs the
labelled waveforms from FDSN using the PNWML metadata and calls ``trace_features``.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert
from scipy.stats import kurtosis


def power_spectrum(data: np.ndarray, fs: float):
    """One-sided power spectrum ``(freqs, power)`` of a real signal."""
    data = np.asarray(data, dtype=float)
    data = data - data.mean()
    freqs = rfftfreq(len(data), d=1.0 / fs)
    power = np.abs(rfft(data)) ** 2
    return freqs, power


def spectral_flatness(power: np.ndarray, eps: float = 1e-20) -> float:
    """Geometric-mean / arithmetic-mean of the power spectrum (Wiener entropy).

    ~1 => broadband/noise-like (thunder); ~0 => peaky/impulsive (earthquake).
    """
    power = np.asarray(power, dtype=float) + eps
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def dominant_frequency(freqs: np.ndarray, power: np.ndarray) -> float:
    """Frequency of the peak of the power spectrum (Hz)."""
    return float(freqs[int(np.argmax(power))])


def spectral_centroid(freqs: np.ndarray, power: np.ndarray, eps: float = 1e-20) -> float:
    """Power-weighted mean frequency (Hz) — the spectral 'center of mass'."""
    power = np.asarray(power, dtype=float)
    return float(np.sum(freqs * power) / (np.sum(power) + eps))


def spectral_bandwidth(freqs: np.ndarray, power: np.ndarray, eps: float = 1e-20) -> float:
    """Power-weighted spectral spread (Hz) about the centroid."""
    c = spectral_centroid(freqs, power, eps)
    power = np.asarray(power, dtype=float)
    return float(np.sqrt(np.sum(((freqs - c) ** 2) * power) / (np.sum(power) + eps)))


def envelope(data: np.ndarray) -> np.ndarray:
    """Analytic-signal amplitude envelope."""
    data = np.asarray(data, dtype=float)
    return np.abs(hilbert(data - data.mean()))


def envelope_duration(data: np.ndarray, fs: float, frac: float = 0.8) -> float:
    """Duration (s) of the central window holding ``frac`` of cumulative envelope energy.

    Robust to emergent onsets: uses the (1-frac)/2 and (1+frac)/2 quantiles of the
    cumulative squared envelope. Thunder claps are long/emergent; an earthquake's
    energy is concentrated near the P/S arrivals.
    """
    env = envelope(data) ** 2
    c = np.cumsum(env)
    if c[-1] <= 0:
        return 0.0
    c /= c[-1]
    lo = np.searchsorted(c, (1 - frac) / 2)
    hi = np.searchsorted(c, (1 + frac) / 2)
    return float((hi - lo) / fs)


def band_energy_ratio(freqs, power, band) -> float:
    """Fraction of spectral power within ``band`` = (fmin, fmax)."""
    power = np.asarray(power, dtype=float)
    fmin, fmax = band
    m = (freqs >= fmin) & (freqs < fmax)
    total = np.sum(power)
    return float(np.sum(power[m]) / total) if total > 0 else 0.0


def trace_features(data: np.ndarray, fs: float) -> dict:
    """Compute the full per-trace feature dict used for class characterization."""
    data = np.asarray(data, dtype=float)
    freqs, power = power_spectrum(data, fs)
    return {
        "dominant_freq_hz": dominant_frequency(freqs, power),
        "spectral_centroid_hz": spectral_centroid(freqs, power),
        "spectral_bandwidth_hz": spectral_bandwidth(freqs, power),
        "spectral_flatness": spectral_flatness(power),
        "duration_80pct_s": envelope_duration(data, fs, 0.8),
        "kurtosis": float(kurtosis(data, fisher=True, bias=False)),
        "energy_1_5hz": band_energy_ratio(freqs, power, (1, 5)),
        "energy_5_10hz": band_energy_ratio(freqs, power, (5, 10)),
        "energy_10_20hz": band_energy_ratio(freqs, power, (10, 20)),
        "energy_20_45hz": band_energy_ratio(freqs, power, (20, 45)),
    }
