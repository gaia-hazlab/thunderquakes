import numpy as np

from thunderquakes.features.seismoacoustic import (
    resample_to,
    seismo_acoustic_lag,
    smooth_env,
)


def test_resample_to_preserves_endpoints_shape():
    x = np.sin(np.linspace(0, 10, 1000))
    y = resample_to(x, fs_in=100.0, fs_out=50.0, n_out=500)
    assert y.shape == (500,)


def test_smooth_env_nonnegative():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    env = smooth_env(x, fs=100.0)
    assert np.all(env >= 0)


def test_seismo_acoustic_lag_detects_known_shift():
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    burst = np.exp(-((t - 10) ** 2) / (2 * 0.5**2))
    seis = burst + 0.01 * np.random.default_rng(1).standard_normal(len(t))
    shift_s = 5.0
    infra = np.roll(burst, int(shift_s * fs)) + 0.01 * np.random.default_rng(2).standard_normal(
        len(t)
    )
    lag, coef = seismo_acoustic_lag(seis, fs, infra, fs, max_lag_s=10.0)
    assert abs(lag - shift_s) < 0.5
    assert coef > 0.8


def test_seismo_acoustic_lag_uncorrelated_gives_low_coefficient():
    rng = np.random.default_rng(3)
    fs = 100.0
    seis = rng.standard_normal(2000)
    infra = rng.standard_normal(2000)
    _, coef = seismo_acoustic_lag(seis, fs, infra, fs)
    assert coef < 0.5
