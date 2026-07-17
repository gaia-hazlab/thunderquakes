"""Match model detections to lightning strikes and measure completeness gain (WS4).

A detection is a true positive if a catalog strike falls within (radius_km,
window_s). Because open catalogs are a LOWER bound on lightning activity, an
unmatched detection is not necessarily a false positive — report the RESULT as
completeness gain relative to the catalog, not a naive precision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from thunderquakes.geo import haversine_km


def match_detections_to_strikes(
    detections: pd.DataFrame,
    strikes: pd.DataFrame,
    radius_km: float = 30.0,
    window_s: float = 120.0,
) -> pd.DataFrame:
    """Attach the nearest space-time strike to each detection.

    ``detections`` needs: time_utc, latitude, longitude.
    ``strikes``    needs: time_utc, latitude, longitude (lightning COMMON_SCHEMA).
    Adds columns: matched (bool), dt_s, dist_km.
    """
    out = detections.copy()
    # Normalise both sides to tz-naive UTC datetime64 (avoids the np.datetime64
    # tz warning and keeps comparisons unambiguous).
    s_t = pd.to_datetime(strikes["time_utc"], utc=True).dt.tz_localize(None).values
    s_la = strikes["latitude"].values
    s_lo = strikes["longitude"].values
    order = np.argsort(s_t)
    s_t, s_la, s_lo = s_t[order], s_la[order], s_lo[order]
    det_t = pd.to_datetime(out["time_utc"], utc=True).dt.tz_localize(None).values
    win = np.timedelta64(int(window_s), "s")

    matched, dts, dists = [], [], []
    for row, t0 in zip(out.itertuples(index=False), det_t, strict=True):
        lo = np.searchsorted(s_t, t0 - win)
        hi = np.searchsorted(s_t, t0 + win, side="right")
        hit_matched, hit_dt, hit_dist = False, np.nan, np.nan
        if hi > lo:
            d = haversine_km(row.latitude, row.longitude, s_la[lo:hi], s_lo[lo:hi])
            j = int(np.argmin(d))
            if d[j] <= radius_km:
                hit_matched = True
                hit_dt = (s_t[lo:hi][j] - t0) / np.timedelta64(1, "s")
                hit_dist = float(d[j])
        matched.append(hit_matched)
        dts.append(hit_dt)
        dists.append(hit_dist)
    out["matched"] = matched
    out["dt_s"] = dts
    out["dist_km"] = dists
    return out
