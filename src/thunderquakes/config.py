"""Centralised, config-driven paths and constants.

Replaces the hardcoded ``/data/whd02/niyiyu_data/PNWML/...`` paths from the
prototype notebook. Resolution order for the data root:

1. ``THUNDERQUAKES_DATA`` environment variable, if set.
2. ``data_root`` in a ``config.yaml`` at the repo root, if present.
3. ``./data`` under the repo root (default).

PNWML sub-paths follow the published layout (comcat / exotic / noise, each with
``waveforms.hdf5`` + ``metadata.csv``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_data_root() -> Path:
    env = os.environ.get("THUNDERQUAKES_DATA")
    if env:
        return Path(env).expanduser().resolve()
    cfg = REPO_ROOT / "config.yaml"
    if cfg.exists():
        with cfg.open() as f:
            data = yaml.safe_load(f) or {}
        if data.get("data_root"):
            return Path(data["data_root"]).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()


DATA_ROOT = _resolve_data_root()


@dataclass(frozen=True)
class PNWMLPaths:
    """Locations of the three PNWML class stores under ``root``."""

    root: Path

    def _store(self, name: str) -> dict[str, Path]:
        base = self.root / name
        return {"waveforms": base / "waveforms.hdf5", "metadata": base / "metadata.csv"}

    @property
    def comcat(self) -> dict[str, Path]:
        return self._store("comcat")

    @property
    def exotic(self) -> dict[str, Path]:
        return self._store("exotic")

    @property
    def noise(self) -> dict[str, Path]:
        return self._store("noise")


def pnwml_paths(root: Path | str | None = None) -> PNWMLPaths:
    root = Path(root) if root is not None else DATA_ROOT / "PNWML"
    return PNWMLPaths(root=root)


# --- Region definitions (bounding boxes + FDSN networks) ---------------------
# Used by stations/ inventory and lightning/ catalog queries.
REGIONS = {
    "OK": {
        "priority": 1,
        "bbox": (33.5, 37.2, -103.1, -94.4),  # (lat_min, lat_max, lon_min, lon_max)
        "seismic_networks": ["OK", "GS", "N4", "TA"],
        "notes": "Densely instrumented (induced seismicity). TA/N4 carry BDF infrasound.",
    },
    "PNW": {
        "priority": 2,
        "bbox": (41.0, 49.5, -125.0, -116.5),
        "seismic_networks": ["UW", "UO", "CC"],
        "notes": "PNSN. Home of the PNWML analyst-labeled 'thunder' class.",
    },
    "AK": {
        "priority": 3,
        "bbox": (51.0, 71.5, -170.0, -130.0),
        "seismic_networks": ["AK", "TA", "AV"],
        "notes": "Alaska TA (2014-2021) carried BDF infrasound. GLM coverage is poor here.",
    },
}

# --- Signal / preprocessing defaults (see WS1 characterization) --------------
TARGET_SAMPLING_RATE = 100.0  # Hz, seismic
WINDOW_SECONDS = 30.0
SEISMIC_BAND = (1.0, 20.0)  # Hz

# Infrasound / pressure channels use SEED instrument code 'D' (2nd char of the
# channel code). Only high-rate bands resolve ~1-20 Hz thunder acoustics:
#   band codes F,G,D,C = very high rate; H = >=80 Hz; B = 10-80 Hz; E = short-period.
# Low-rate pressure (band L = 1 sps: LDF/LDM/LDO) is meteorological and too coarse
# for thunder claps, so it is recorded but NOT flagged as usable infrasound.
PRESSURE_INSTRUMENT_CODE = "D"
ACOUSTIC_BAND_CODES = frozenset("FGDCHBE")


def is_acoustic_channel(channel: str) -> bool:
    """True if a channel is a pressure sensor at a rate usable for thunder acoustics."""
    return (
        len(channel) >= 2
        and channel[1] == PRESSURE_INSTRUMENT_CODE
        and channel[0] in ACOUSTIC_BAND_CODES
    )


def is_pressure_channel(channel: str) -> bool:
    """True for any pressure/infrasound sensor (incl. low-rate met pressure)."""
    return len(channel) >= 2 and channel[1] == PRESSURE_INSTRUMENT_CODE
