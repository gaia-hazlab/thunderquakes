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
