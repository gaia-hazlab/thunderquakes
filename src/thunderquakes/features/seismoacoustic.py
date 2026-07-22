"""Seismo-acoustic coupling: the physical discriminator for thunderquakes (WS1/WS4).

Thunder couples into the ground both directly (a weak, immediate elastic wave)
and via the airborne acoustic front (arriving at ~340 m/s). At a co-located
seismic+infrasound station, both channels record the SAME passing acoustic
front, so their smoothed envelopes correlate strongly at a small, physically
bounded lag. This is a much stronger verification signal than mere proximity to
a lightning strike (which only says a storm was nearby, not that the seismic
signal is genuinely thunder-coupled) — see WS1 (median envelope xcorr 0.85 at
~0s lag at PNW seismoacoustic sites) and its use as an independent QC signal
for GLM-triggered OK candidates (WS4 #11).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import correlate, correlation_lags

from thunderquakes.features.characterize import envelope


def smooth_env(data, fs, win_s: float = 1.0):
    """Amplitude envelope smoothed over a ``win_s``-second window."""
    return uniform_filter1d(envelope(data), size=max(1, int(win_s * fs)))


def resample_to(x, fs_in, fs_out, n_out):
    """Linearly resample ``x`` (sampled at ``fs_in``) onto ``n_out`` samples at ``fs_out``."""
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, x, left=0.0, right=0.0)


def seismo_acoustic_lag(seis, fs_s, infra, fs_i, max_lag_s: float = 30.0):
    """Envelope cross-correlation lag (s, infrasound relative to seismic) and peak coefficient.

    Search is restricted to ``|lag| <= max_lag_s``: a thunderquake's acoustic and
    air-coupled seismic arrivals are near-simultaneous, so larger apparent lags
    are spurious matches to unrelated bursts elsewhere in the window.
    """
    fs = 50.0
    n = int(min(len(seis) / fs_s, len(infra) / fs_i) * fs)
    es = resample_to(smooth_env(seis, fs_s), fs_s, fs, n)
    ei = resample_to(smooth_env(infra, fs_i), fs_i, fs, n)
    es = (es - es.mean()) / (es.std() + 1e-12)
    ei = (ei - ei.mean()) / (ei.std() + 1e-12)
    xc = correlate(ei, es, mode="full") / len(es)
    lags = correlation_lags(len(ei), len(es), mode="full") / fs
    win = np.abs(lags) <= max_lag_s
    xc, lags = xc[win], lags[win]
    j = int(np.argmax(xc))
    return float(lags[j]), float(xc[j])
