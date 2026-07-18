"""WWLLN (World Wide Lightning Location Network) loader — WS3 backstop / AK truth.

WWLLN is a ground-based VLF network that locates individual lightning *strokes*
globally, including high latitudes where GLM detection efficiency collapses — so
it is the primary open option for Alaska and the only one that reaches before the
GLM era (pre-2018).

Access: WWLLN data are **not** openly downloadable; they require a data-sharing
agreement. WWLLN is operated by the University of Washington (Holzworth group,
Earth & Space Sciences) — the same institution as this project — so the archive
is reachable internally. Request the located-stroke files ("A-files": ``A*.loc``
plain-text, or ``AE*`` energy files) and point this loader at them.

File format (located-stroke / A-files), one stroke per comma-separated line:
    YYYY/MM/DD, HH:MM:SS.ssssss, latitude, longitude, residual_us, n_stations[, energy_J]
This loader tolerates the 6-column ``.loc`` form and the 7-column ``AE`` form.
WWLLN reports a relative energy estimate (J), not peak current, so
``peak_current`` is left NaN.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COMMON_SCHEMA = ["time_utc", "latitude", "longitude", "peak_current", "source"]

# Column names for the raw WWLLN located-stroke files (extra cols beyond 6 ignored).
_RAW_COLUMNS = ["date", "time", "latitude", "longitude", "residual_us", "n_stations"]


def parse_wwlln_file(path: str | Path) -> pd.DataFrame:
    """Parse one WWLLN located-stroke file into the common schema.

    Handles both the 6-column ``.loc`` and 7-column ``AE`` energy variants (the
    optional energy column is ignored here).
    """
    raw = pd.read_csv(
        path,
        header=None,
        names=_RAW_COLUMNS,
        usecols=range(6),
        skipinitialspace=True,
        dtype={"date": str, "time": str},
    )
    t = pd.to_datetime(
        raw["date"].str.strip() + " " + raw["time"].str.strip(),
        format="%Y/%m/%d %H:%M:%S.%f",
        utc=True,
        errors="coerce",
    )
    out = pd.DataFrame(
        {
            "time_utc": t,
            "latitude": pd.to_numeric(raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(raw["longitude"], errors="coerce"),
            "peak_current": pd.NA,  # WWLLN reports energy, not current
            "source": "WWLLN",
        }
    )
    return out.dropna(subset=["time_utc", "latitude", "longitude"])[COMMON_SCHEMA]


def load_wwlln_strikes(
    path: str | Path,
    t_start=None,
    t_end=None,
    bbox=None,
    pattern: str = "A*.loc",
) -> pd.DataFrame:
    """Load WWLLN strokes from a file or directory of A-files, filtered to a window/bbox.

    Parameters
    ----------
    path : a single located-stroke file, or a directory to glob with ``pattern``.
    t_start, t_end : optional UTC bounds.
    bbox : optional (lat_min, lat_max, lon_min, lon_max).
    """
    p = Path(path)
    files = sorted(p.glob(pattern)) if p.is_dir() else [p]
    if not files:
        raise FileNotFoundError(f"No WWLLN files at {p} (pattern {pattern!r})")

    frames = [parse_wwlln_file(f) for f in files]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COMMON_SCHEMA)

    if t_start is not None:
        df = df[df["time_utc"] >= pd.Timestamp(t_start, tz="UTC")]
    if t_end is not None:
        df = df[df["time_utc"] <= pd.Timestamp(t_end, tz="UTC")]
    if bbox is not None:
        lat0, lat1, lon0, lon1 = bbox
        df = df[df["latitude"].between(lat0, lat1) & df["longitude"].between(lon0, lon1)]
    return df.sort_values("time_utc").reset_index(drop=True)
