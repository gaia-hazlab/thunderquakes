import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_inventory import first_year_at_least  # noqa: E402


def test_first_year_at_least_basic():
    df = pd.DataFrame({
        "data_start": ["2010-01-01", "2012-01-01", "2015-01-01"],
        "data_end": [None, None, None],
    })
    # by 2012, stations from 2010 and 2012 are active (2), not yet 3 (threshold)
    assert first_year_at_least(df, threshold=2) == 2012


def test_first_year_at_least_none_if_never_reached():
    df = pd.DataFrame({"data_start": ["2010-01-01"], "data_end": [None]})
    assert first_year_at_least(df, threshold=5) is None


def test_first_year_at_least_respects_end_date():
    df = pd.DataFrame({
        "data_start": ["2000-01-01", "2001-01-01"],
        "data_end": ["2001-06-01", None],
    })
    # station 1 ends mid-2001; both active through 2001, only station 2 in 2002
    assert first_year_at_least(df, threshold=2) == 2001
