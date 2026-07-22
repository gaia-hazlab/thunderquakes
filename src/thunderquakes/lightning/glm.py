"""GOES-GLM (Geostationary Lightning Mapper) strike loader — WS3 primary truth.

GLM L2 data are public on AWS Open Data (no credentials):
    s3://noaa-goes{16,17,18,19}/GLM-L2-LCFA/<year>/<doy>/<hour>/*.nc
Each file is a ~20 s granule with event/group/flash records. For ground-truthing
we use *flash* centroids (``flash_lat``, ``flash_lon``, and the flash time, which
xarray decodes from ``flash_time_offset_of_first_event`` to absolute UTC).

GLM is an *optical* detector: it reports flash energy (J), not peak current, so
``peak_current`` is left NaN (that column is populated by ground networks such as
WWLLN/NLDN). Detection efficiency is high over CONUS (OK & PNW) but degrades
badly toward Alaska latitudes — treat the catalog as a lower bound.

GOES-East position (covers OK & PNW): GOES-16 through 2025-04-04, GOES-19 after.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

COMMON_SCHEMA = ["time_utc", "latitude", "longitude", "peak_current", "source"]

# GOES-19 replaced GOES-16 at the GOES-East slot on this date.
_GOES_EAST_SWITCH = datetime(2025, 4, 4, tzinfo=UTC)


def default_east_satellite(t: datetime) -> str:
    """GOES-East satellite id for a given time (covers OK & PNW)."""
    return "goes19" if t >= _GOES_EAST_SWITCH else "goes16"


def _parse_granule_start(fname: str) -> datetime:
    """Parse the ``sYYYYDDDHHMMSSt`` start token from a GLM L2 filename → UTC datetime.

    e.g. ``OR_GLM-L2-LCFA_G16_s20201520000000_e..._c....nc`` -> 2020-05-31 00:00:00Z
    """
    token = next(p for p in fname.split("_") if p.startswith("s") and p[1:].isdigit())
    digits = token[1:]  # YYYYDDDHHMMSSt (tenths of second at the end)
    year = int(digits[0:4])
    doy = int(digits[4:7])
    hour = int(digits[7:9])
    minute = int(digits[9:11])
    second = int(digits[11:13])
    base = datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=doy - 1)
    return base.replace(hour=hour, minute=minute, second=second)


def _flashes_from_dataset(ds, bbox=None) -> pd.DataFrame:
    """Extract flash centroids from an open GLM L2 dataset as the common schema."""
    lat = ds["flash_lat"].values
    lon = ds["flash_lon"].values
    t = ds["flash_time_offset_of_first_event"].values  # datetime64, UTC
    df = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(t, utc=True),
            "latitude": lat,
            "longitude": lon,
            "peak_current": pd.NA,  # GLM is optical; no current
            "source": ds.attrs.get("platform_ID", "GLM"),
        }
    )
    if bbox is not None:
        lat0, lat1, lon0, lon1 = bbox
        df = df[
            (df["latitude"] >= lat0)
            & (df["latitude"] <= lat1)
            & (df["longitude"] >= lon0)
            & (df["longitude"] <= lon1)
        ]
    return df[COMMON_SCHEMA]


def _hour_prefixes(t_start: datetime, t_end: datetime, bucket: str):
    """Yield ``bucket/GLM-L2-LCFA/YYYY/DDD/HH`` prefixes spanning the range."""
    cur = t_start.replace(minute=0, second=0, microsecond=0)
    while cur <= t_end:
        yield f"{bucket}/GLM-L2-LCFA/{cur.year}/{cur.timetuple().tm_yday:03d}/{cur.hour:02d}"
        cur += timedelta(hours=1)


def load_glm_strikes(
    t_start,
    t_end,
    bbox=None,
    satellite: str | None = None,
    progress: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Return GLM flashes in ``[t_start, t_end]`` (and optional bbox) as the common schema.

    Parameters
    ----------
    t_start, t_end : tz-aware datetimes (UTC assumed if naive).
    bbox : (lat_min, lat_max, lon_min, lon_max) or None.
    satellite : 'goes16'/'goes17'/'goes18'/'goes19'. Defaults to the GOES-East
        satellite for ``t_start`` (goes16 pre-2025-04-04, goes19 after).
    progress : show a tqdm bar over granules.
    max_workers : threads for concurrent granule reads (I/O-bound); 1 = sequential.

    Notes
    -----
    I/O heavy (~180 granules/hour). For repeated use, cache the returned frame
    (see ``scripts/fetch_glm.py``, which writes a CSV under ``outputs/``).
    """
    from concurrent.futures import ThreadPoolExecutor

    import s3fs
    import xarray as xr

    def _as_utc(t) -> datetime:
        ts = pd.Timestamp(t)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.to_pydatetime()

    t_start = _as_utc(t_start)
    t_end = _as_utc(t_end)
    sat = satellite or default_east_satellite(t_start)
    bucket = f"noaa-{sat}"

    def _retry(fn, *args, retries=3, **kwargs):
        """S3 listing/reads occasionally hit transient connection resets at scale;
        retry with backoff rather than aborting the whole fetch."""
        import time

        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except FileNotFoundError:
                raise
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)

    fs = s3fs.S3FileSystem(anon=True)
    granules = []
    for prefix in _hour_prefixes(t_start, t_end, bucket):
        try:
            listing = _retry(fs.ls, prefix)
        except FileNotFoundError:
            continue
        except OSError:
            continue  # give up on this hour after retries; don't abort the whole fetch
        for path in listing:
            gstart = _parse_granule_start(path.split("/")[-1])
            if t_start <= gstart <= t_end:
                granules.append(path)

    def _read(path):
        try:
            def _open_and_extract():
                with fs.open(path) as f:
                    ds = xr.open_dataset(f, engine="h5netcdf")
                    out = _flashes_from_dataset(ds, bbox)
                    ds.close()
                    return out

            return _retry(_open_and_extract)
        except (OSError, ValueError, KeyError):
            return None  # skip corrupt/empty/unreadable granule after retries

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(_read, granules)
            if progress:
                from tqdm import tqdm

                results = tqdm(results, total=len(granules), desc=f"GLM {sat}")
            frames = [r for r in results if r is not None]
    else:
        it = granules
        if progress:
            from tqdm import tqdm

            it = tqdm(granules, desc=f"GLM {sat}")
        frames = [r for r in (_read(p) for p in it) if r is not None]

    if not frames:
        return pd.DataFrame(columns=COMMON_SCHEMA)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("time_utc").reset_index(drop=True)
