import pandas as pd

from thunderquakes.evaluation.groundtruth import match_detections_to_strikes


def test_match_within_window_and_radius():
    t = pd.Timestamp("2020-06-01T12:00:00Z")
    det = pd.DataFrame({"time_utc": [t], "latitude": [35.0], "longitude": [-97.0]})
    strikes = pd.DataFrame({
        "time_utc": [t + pd.Timedelta(seconds=30)],
        "latitude": [35.05], "longitude": [-97.0],
    })
    out = match_detections_to_strikes(det, strikes, radius_km=30, window_s=120)
    assert bool(out.loc[0, "matched"]) is True
    assert abs(out.loc[0, "dt_s"] - 30) < 1


def test_no_match_when_too_far():
    t = pd.Timestamp("2020-06-01T12:00:00Z")
    det = pd.DataFrame({"time_utc": [t], "latitude": [35.0], "longitude": [-97.0]})
    strikes = pd.DataFrame({"time_utc": [t], "latitude": [40.0], "longitude": [-97.0]})
    out = match_detections_to_strikes(det, strikes, radius_km=30, window_s=120)
    assert bool(out.loc[0, "matched"]) is False
