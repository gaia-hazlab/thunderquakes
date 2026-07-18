import pandas as pd

from thunderquakes.config import is_acoustic_channel, is_pressure_channel
from thunderquakes.stations.inventory import station_summary


def test_acoustic_vs_lowrate_pressure():
    # High-rate infrasound mics -> usable acoustics
    assert is_acoustic_channel("BDF")
    assert is_acoustic_channel("BDO")
    assert is_acoustic_channel("HDF")
    # 1-sps meteorological pressure -> pressure but NOT usable acoustics
    assert not is_acoustic_channel("LDF")
    assert not is_acoustic_channel("LDM")
    assert not is_acoustic_channel("LDO")
    # Seismic channels are neither
    assert not is_acoustic_channel("HHZ")


def test_pressure_detection_by_instrument_code():
    for c in ("BDF", "LDF", "LDM", "BDO"):
        assert is_pressure_channel(c)
    for c in ("HHZ", "BHN", "HNZ"):
        assert not is_pressure_channel(c)


def test_station_summary_flags_seismoacoustic():
    inv = pd.DataFrame(
        {
            "region": "OK",
            "network": ["TA", "TA", "TA", "OK", "OK"],
            "station": ["A1", "A1", "A1", "B2", "B2"],
            "channel": ["HHZ", "BDF", "LDM", "HHZ", "HHN"],
            "latitude": [35.0, 35.0, 35.0, 36.0, 36.0],
            "longitude": [-97.0, -97.0, -97.0, -98.0, -98.0],
            "start": pd.to_datetime(["2010-01-01"] * 5, utc=True),
            "end": pd.to_datetime(["2015-01-01"] * 5, utc=True),
            "is_infrasound": [False, True, False, False, False],
            "is_pressure": [False, True, True, False, False],
        }
    )
    s = station_summary(inv).set_index("station")
    # A1 has BDF -> seismoacoustic; B2 is seismic-only
    assert s.loc["A1", "seismoacoustic"]
    assert not s.loc["B2", "seismoacoustic"]
    assert s.loc["A1", "infrasound_channels"] == "BDF"
    assert "LDM" in s.loc["A1", "pressure_channels"]
    assert s.loc["B2", "seismic_channels"] == "HHN,HHZ"


def test_station_summary_with_availability_and_datacenter():
    inv = pd.DataFrame(
        {
            "region": "OK",
            "datacenter": ["IRIS", "IRIS", "IRISPH5"],
            "network": ["N4", "N4", "YW"],
            "station": ["T35B", "T35B", "601"],
            "channel": ["HHZ", "BDF", "HDF"],
            "latitude": [36.9, 36.9, 36.6],
            "longitude": [-96.5, -96.5, -97.7],
            "start": pd.to_datetime(["2014-01-01"] * 3, utc=True),
            "end": pd.to_datetime(["2019-01-01"] * 3, utc=True),
            "is_infrasound": [False, True, True],
            "is_pressure": [False, True, True],
            "data_start": pd.to_datetime(["2014-02-19", "2014-02-19", None], utc=True),
            "data_end": pd.to_datetime(["2026-07-16", "2026-07-16", None], utc=True),
            "n_timespans": [7000, 224, None],
        }
    )
    s = station_summary(inv).set_index("station")
    assert "data_start" in s.columns and "n_timespans" in s.columns
    assert s.loc["601", "datacenter"] == "IRISPH5"       # nodal PH5 site
    assert s.loc["601", "seismoacoustic"]                # YW carries HDF
    assert s.loc["T35B", "seismoacoustic"]
    assert pd.notna(s.loc["T35B", "data_start"])          # availability merged
    assert pd.isna(s.loc["601", "data_start"])            # PH5 has no fdsnws availability
