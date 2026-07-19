import numpy as np

from thunderquakes.data import cache as C


def test_key_is_deterministic_and_band_tagged():
    import pandas as pd
    t = pd.Timestamp("2021-08-02T03:17:19Z")
    k1 = C._key("CC", "CPCO", "BHZ", t, 150, (1.0, 20.0))
    k2 = C._key("CC", "CPCO", "BHZ", t, 150, (1.0, 20.0))
    assert k1 == k2
    assert "CC.CPCO.BHZ" in k1 and "150s" in k1 and "1-20" in k1
    assert C._key("CC", "CPCO", "BHZ", t, 150, None).endswith("raw")


def test_cache_hit_roundtrip(tmp_path, monkeypatch):
    # Force a miss->fetch path with a stub fetch, then confirm the second call hits disk.
    calls = {"n": 0}

    class _Tr:
        def __init__(self):
            self.data = np.arange(10, dtype="float32")
            self.stats = type("S", (), {"sampling_rate": 100.0})()

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return [ _Tr() ], "BHZ"

    monkeypatch.setattr(C, "fetch_window", fake_fetch)
    import pandas as pd
    t = pd.Timestamp("2021-08-02T03:17:19Z")

    d1, fs1 = C.cached_waveform("CC", "CPCO", "BHZ", t, 10, cache_dir=tmp_path)
    d2, fs2 = C.cached_waveform("CC", "CPCO", "BHZ", t, 10, cache_dir=tmp_path)
    assert calls["n"] == 1               # second call served from cache
    assert fs1 == fs2 == 100.0
    assert np.array_equal(d1, d2)


def test_cache_miss_marker(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(*a, **k):
        calls["n"] += 1
        return None, None

    monkeypatch.setattr(C, "fetch_window", fake_fetch)
    import pandas as pd
    t = pd.Timestamp("2021-08-02T03:17:19Z")
    r1 = C.cached_waveform("CC", "XXXX", "BHZ", t, 10, cache_dir=tmp_path)
    r2 = C.cached_waveform("CC", "XXXX", "BHZ", t, 10, cache_dir=tmp_path)
    assert r1 == (None, None) and r2 == (None, None)
    assert calls["n"] == 1               # miss cached, not re-queried
