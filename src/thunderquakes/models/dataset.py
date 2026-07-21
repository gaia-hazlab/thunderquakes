"""Build labelled CNN inputs from PNWML metadata via the waveform cache (WS4 #8).

Two stages so augmentation can act on raw waveforms before the spectrogram:
  1. ``build_windows``  -> raw 50 s @100 Hz windows (WS1-recommended size), cropped
     around the envelope peak (onset-free — PNWML sample-7000 is a placeholder).
  2. ``to_spectrograms`` -> band-limited, per-image-standardised log-spectrograms
     (the 2D-CNN input), applied AFTER any augmentation.

Region-aware class sets (see :data:`REGION_CLASSES`): surface events are a real
confuser in PNW/AK but essentially absent as a source class in Oklahoma, so the OK
model uses a leaner negative set. Earthquake & explosion are handled by the
existing QuakeXNet classifier and excluded here.
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
# Which negative classes matter where (thunder is always the positive).
REGION_CLASSES = {
    "OK": ["thunder", "sonic boom", "noise"],            # no surface events in OK
    "PNW": ["thunder", "sonic boom", "surface event", "noise"],
    "AK": ["thunder", "sonic boom", "surface event", "noise"],
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


def crop_around_peak(data: np.ndarray, n_samples: int, offset: int = 0) -> np.ndarray:
    """Crop ``n_samples`` centred on the smoothed-envelope peak (+ optional jitter offset)."""
    if len(data) <= n_samples:
        return np.pad(data, (0, n_samples - len(data)))
    peak = int(np.argmax(envelope(data)))
    start = peak - n_samples // 2 + offset
    lo = int(np.clip(start, 0, len(data) - n_samples))
    return data[lo : lo + n_samples]


def log_spectrogram_image(win: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    """Fixed-size log-power spectrogram (freq×time), band-limited and standardised."""
    f, _, sxx = _spec(win, fs=cfg.fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
    band = (f >= cfg.band[0]) & (f <= cfg.band[1])
    img = 10.0 * np.log10(sxx[band] + 1e-12)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return img.astype("float32")


@dataclass
class RawDataset:
    """Raw labelled windows (pre-spectrogram) + metadata for augmentation & splits."""

    W: np.ndarray  # (N, n_samples) float32
    y: np.ndarray  # (N,) int
    meta: pd.DataFrame
    classes: list = field(default_factory=list)
    fs: float = 100.0


def build_windows(
    metadata: dict[str, pd.DataFrame],
    classes,
    cfg: WindowConfig | None = None,
    n_per_class: int | None = None,
    client="IRIS",
    seed: int = 0,
) -> RawDataset:
    """Assemble raw 50 s windows for the given classes from PNWML metadata via the cache."""
    cfg = cfg or WindowConfig()
    classes = list(classes)
    wins, labels, rows = [], [], []
    for ci, cls in enumerate(classes):
        sel = metadata[CLASS_STORE[cls]]
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
            wins.append(crop_around_peak(np.asarray(data, np.float32), cfg.n_samples))
            labels.append(ci)
            rows.append({"source_type": cls, "network": r.station_network_code,
                         "station": r.station_code, "time": str(r.trace_start_time)})
    return RawDataset(W=np.stack(wins).astype("float32"), y=np.asarray(labels, "int64"),
                      meta=pd.DataFrame(rows), classes=classes, fs=cfg.fs)


def to_spectrograms(W: np.ndarray, cfg: WindowConfig) -> np.ndarray:
    """Convert (N, n_samples) raw windows -> (N, 1, F, T) standardised log-spectrograms."""
    return np.stack([log_spectrogram_image(w, cfg)[None] for w in W]).astype("float32")
