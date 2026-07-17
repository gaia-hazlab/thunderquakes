"""GOES-GLM (Geostationary Lightning Mapper) strike loader — WS3 primary truth.

GLM L2 data are public on AWS Open Data (no credentials):
    s3://noaa-goes16/GLM-L2-LCFA/<year>/<doy>/<hour>/*.nc
Each file is a ~20 s granule with `event`, `group`, and `flash` records. For
ground-truthing we use flash centroids (flash_lat, flash_lon, flash_time).

TODO(WS3): implement the S3 listing + xarray read; this module documents the
target schema so downstream evaluation code can be written against it now.
"""

from __future__ import annotations

import pandas as pd

COMMON_SCHEMA = ["time_utc", "latitude", "longitude", "peak_current", "source"]


def load_glm_strikes(t_start, t_end, bbox=None, satellite: str = "goes16") -> pd.DataFrame:
    """Return GLM flashes in ``[t_start, t_end]`` (and optional bbox) as the common schema.

    Parameters
    ----------
    t_start, t_end : tz-aware datetimes
    bbox : (lat_min, lat_max, lon_min, lon_max) or None
    satellite : 'goes16' (East, covers OK/PNW) or 'goes18' (West).
    """
    raise NotImplementedError(
        "WS3: read GLM-L2-LCFA granules from s3://noaa-<sat>/GLM-L2-LCFA/ via s3fs+xarray, "
        "extract flash centroids, filter to bbox, return COMMON_SCHEMA columns."
    )
