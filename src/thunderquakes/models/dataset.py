"""Build labelled CNN inputs from PNWML metadata via the waveform cache (WS4 #8).

Turns each labelled trace into a fixed 50 s window (@100 Hz, the WS1-recommended
size) cropped around the envelope peak, then a log-spectrogram — the 2D-CNN input.
Cache-backed, so rebuilding the dataset is fast/offline after the first fetch.

Classes (Model A, seismic-only): thunder / sonic boom / surface event / noise.
Earthquake & explosion are handled by the existing QuakeXNet classifier, so they
are intentionally excluded here (this is the thunderquake-vs-confusers model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import spectrogram as _spec

from thunderquakes.data import cached_waveform
from thunderquakes.features.characterize import envelope

FETCH_WINDOW_S = 150.0  # PNWML trace length we fetch/cache; we crop the model window from it
CLASS_STORE = {
    "thunder": "exotic",
    "sonic boom": "exotic",
    "surface event": "exotic",
    "noise": "noise",
}


@dataclass(frozen=True)
class WindowConfig:
    win_s: float = 50.0
    fs: float = 100.0
    band: tuple = (1.0, 45.0)
    nperseg: int = 128
    noverlap: int = 96

    @property
    def n_samples(self) -> int:
        return int(round(self.win_s * self.fs))


def crop_around_peak(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Crop ``n_samples`` centred on the smoothed-envelope peak (clamped to bounds).

    Onset-free by design — PNWML's sample-7000 'onset' is a placeholder for thunder.
    """
    if len(data) <= n_samples:
        return np.pad(data, (0, n_samples - len(data)))
    env = envelope(data)
    peak = int(np.argmax(env))
    lo = int(np.clip(peak - n_samples // 2, 0, len(data) - n_samples))
    return data[lo : lo + n_samples]


def log_spectrogram_image(win: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    """Fixed-size log-power spectrogram (freq×time), band-limited and standardised."""
    f, _, sxx = _spec(win, fs=cfg.fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    band = (f >= cfg.band[0]) & (f <= cfg.band[1])
    img = 10.0 * np.log10(sxx[band] + 1e-12)
    img = (img - img.mean()) / (img.std() + 1e-6)  # per-image standardisation
    return img.astype("float32")


@dataclass
class Dataset:
    X: np.ndarray  # (N, 1, F, T) float32
    y: np.ndarray  # (N,) int
    meta: pd.DataFrame  # network, station, source_type, ...
    classes: list = field(default_factory=list)


def build_dataset(
    metadata: dict[str, pd.DataFrame],
    classes=tuple(CLASS_STORE),
    cfg: WindowConfig | None = None,
    n_per_class: int | None = None,
    client="IRIS",
    seed: int = 0,
) -> Dataset:
    """Assemble a labelled spectrogram dataset from PNWML metadata via the cache.

    ``metadata`` maps store name ('exotic'/'noise') -> DataFrame.
    """
    cfg = cfg or WindowConfig()
    classes = list(classes)
    imgs, labels, rows = [], [], []
    for ci, cls in enumerate(classes):
        store = CLASS_STORE[cls]
        sel = metadata[store]
        sel = sel[sel["source_type"] == cls]
        if n_per_class and len(sel) > n_per_class:
            sel = sel.sample(n_per_class, random_state=seed)
        for r in sel.itertuples(index=False):
            chan = str(r.station_channel_code) + "Z"
            data, fs = cached_waveform(
                r.station_network_code, r.station_code, chan,
                pd.Timestamp(r.trace_start_time), FETCH_WINDOW_S, band=cfg.band, client=client,
            )
            if data is None or len(data) < cfg.n_samples // 2:
                continue
            win = crop_around_peak(np.asarray(data, float), cfg.n_samples)
            imgs.append(log_spectrogram_image(win, cfg)[None])  # add channel dim
            labels.append(ci)
            rows.append({"source_type": cls, "network": r.station_network_code,
                         "station": r.station_code, "time": str(r.trace_start_time)})
    X = np.stack(imgs).astype("float32")
    y = np.asarray(labels, dtype="int64")
    return Dataset(X=X, y=y, meta=pd.DataFrame(rows), classes=classes)
