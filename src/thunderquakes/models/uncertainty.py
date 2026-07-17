"""Epistemic uncertainty: MC-dropout, deep ensembles, calibration (WS4).

The epistemic-uncertainty claim only holds if predictions are calibrated —
always report a reliability diagram / Expected Calibration Error alongside
the uncertainty estimates.
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE for multi-class predictions."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def mc_dropout_predict(model, x, n_samples: int = 30):
    """Run ``n_samples`` stochastic forward passes with dropout enabled."""
    raise NotImplementedError("WS4: enable dropout at eval and stack softmax outputs.")
