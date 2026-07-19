"""On-disk waveform cache so analyses/figures never re-fetch from FDSN.

Every requested (network, station, channel, start, duration, band) window is
fetched once via :func:`thunderquakes.data.waveforms.fetch_window` and stored as
a small ``.npz`` under the cache dir (``$THUNDERQUAKES_DATA/cache`` by default,
which is gitignored). Subsequent calls load from disk — so re-running figures or
building training tensors is instant and offline. Empty results (no data at that
station/time) are recorded as ``.miss`` markers so dead traces aren't re-queried.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thunderquakes.config import DATA_ROOT, SEISMIC_BAND
from thunderquakes.data.waveforms import fetch_window


def default_cache_dir() -> Path:
    d = DATA_ROOT / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(net, sta, chan, t0, dur, band) -> str:
    ts = pd.Timestamp(t0).tz_convert("UTC") if pd.Timestamp(t0).tzinfo else pd.Timestamp(t0)
    btag = "raw" if band is None else f"{band[0]:g}-{band[1]:g}"
    return f"{net}.{sta}.{chan}.{ts:%Y%m%dT%H%M%S}.{int(round(dur))}s.{btag}"


def cached_waveform(
    net: str,
    sta: str,
    chan: str,
    t0,
    dur: float,
    band=SEISMIC_BAND,
    cache_dir: Path | None = None,
    client="IRIS",
    remove_response: bool = False,
):
    """Return ``(data, fs)`` for one window, from cache or FDSN. ``(None, None)`` on miss.

    ``chan`` is the exact channel code to fetch (e.g. ``"BHZ"``, ``"BDF"``).
    """
    cache_dir = cache_dir or default_cache_dir()
    stem = _key(net, sta, chan, t0, dur, band)
    npz = cache_dir / f"{stem}.npz"
    miss = cache_dir / f"{stem}.miss"

    if npz.exists():
        with np.load(npz) as z:
            return z["data"], float(z["fs"])
    if miss.exists():
        return None, None

    t_center = pd.Timestamp(t0) + pd.Timedelta(seconds=dur / 2)
    st, _ = fetch_window(
        net, sta, t_center, pre_s=dur / 2, post_s=dur / 2,
        channels=[chan], band=band, remove_response=remove_response, client=client,
    )
    if st is None:
        miss.touch()
        return None, None
    tr = st[0]
    data = tr.data.astype("float32")
    fs = float(tr.stats.sampling_rate)
    np.savez(npz, data=data, fs=fs)
    return data, fs
