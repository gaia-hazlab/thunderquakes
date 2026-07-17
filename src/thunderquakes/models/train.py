"""Training / evaluation harness (WS4).

Splits are by EVENT and by STATION/REGION (never random) to avoid leakage.
Headline experiment: train PNW -> test OK (domain shift to the deployment
target). Report PR-AUC / precision@fixed-recall (severe class imbalance), a
per-region breakdown, and calibration.
"""
from __future__ import annotations


def make_splits(metadata, by: str = "region"):
    raise NotImplementedError("WS4: event/station/region-disjoint splits.")


def train(config: dict):
    raise NotImplementedError("WS4: training loop (seismic-only, then dual-branch).")
