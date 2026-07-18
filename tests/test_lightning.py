import datetime as dt

import pandas as pd

from thunderquakes.lightning import COMMON_SCHEMA
from thunderquakes.lightning.glm import (
    _flashes_from_dataset,
    _hour_prefixes,
    _parse_granule_start,
    default_east_satellite,
)
from thunderquakes.lightning.wwlln import parse_wwlln_file


class _FakeDS:
    """Minimal stand-in for an xarray GLM dataset (no network)."""

    def __init__(self, lat, lon, times):
        import numpy as np

        self._d = {
            "flash_lat": type("A", (), {"values": np.array(lat)}),
            "flash_lon": type("A", (), {"values": np.array(lon)}),
            "flash_time_offset_of_first_event": type(
                "A", (), {"values": pd.to_datetime(times).values}
            ),
        }
        self.attrs = {"platform_ID": "G16"}

    def __getitem__(self, k):
        return self._d[k]


def test_parse_granule_start():
    t = _parse_granule_start("OR_GLM-L2-LCFA_G16_s20201520000000_e20201520000200_c...nc")
    assert t == dt.datetime(2020, 5, 31, 0, 0, 0, tzinfo=dt.UTC)


def test_default_east_satellite_switch():
    assert default_east_satellite(dt.datetime(2020, 1, 1, tzinfo=dt.UTC)) == "goes16"
    assert default_east_satellite(dt.datetime(2025, 6, 1, tzinfo=dt.UTC)) == "goes19"


def test_hour_prefixes_span():
    t0 = dt.datetime(2020, 5, 31, 0, 30, tzinfo=dt.UTC)
    t1 = dt.datetime(2020, 5, 31, 2, 15, tzinfo=dt.UTC)
    pre = list(_hour_prefixes(t0, t1, "noaa-goes16"))
    assert pre[0].endswith("/2020/152/00")
    assert pre[-1].endswith("/2020/152/02")
    assert len(pre) == 3


def test_flashes_from_dataset_bbox_filter():
    ds = _FakeDS(
        lat=[35.0, 10.0, 36.5],
        lon=[-97.0, -50.0, -96.0],
        times=["2020-05-31T00:00:01", "2020-05-31T00:00:02", "2020-05-31T00:00:03"],
    )
    df = _flashes_from_dataset(ds, bbox=(33.5, 37.2, -103.1, -94.4))
    assert list(df.columns) == COMMON_SCHEMA
    assert len(df) == 2  # the lat=10 flash is outside OK bbox
    assert df["source"].iloc[0] == "G16"
    assert df["peak_current"].isna().all()


def test_parse_wwlln_file(tmp_path):
    f = tmp_path / "A20160601.loc"
    f.write_text(
        "2016/06/01, 00:00:01.123456, 35.5000, -97.5000, 12.5, 6\n"
        "2016/06/01, 00:00:02.654321, 10.0000, -50.0000, 20.1, 5\n"
    )
    df = parse_wwlln_file(f)
    assert list(df.columns) == COMMON_SCHEMA
    assert len(df) == 2
    assert df["source"].iloc[0] == "WWLLN"
    assert abs(df["latitude"].iloc[0] - 35.5) < 1e-6
    assert str(df["time_utc"].iloc[0]).startswith("2016-06-01 00:00:01.123456")
