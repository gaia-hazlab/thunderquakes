import numpy as np

from thunderquakes.features.characterize import spectral_flatness


def test_flatness_white_noise_near_one():
    rng = np.random.default_rng(0)
    power = rng.random(1024)  # roughly flat
    assert spectral_flatness(power) > 0.5


def test_flatness_peaky_near_zero():
    power = np.full(1024, 1e-12)
    power[10] = 1.0  # single spike
    assert spectral_flatness(power) < 0.1


def test_dominant_frequency_of_sine():
    from thunderquakes.features.characterize import dominant_frequency, power_spectrum
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    x = np.sin(2 * np.pi * 7.0 * t)
    f, p = power_spectrum(x, fs)
    assert abs(dominant_frequency(f, p) - 7.0) < 0.2


def test_flatness_orders_noise_above_tone():
    from thunderquakes.features.characterize import power_spectrum, spectral_flatness
    fs = 100.0
    t = np.arange(0, 20, 1 / fs)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(t))
    tone = np.sin(2 * np.pi * 10 * t)
    _, pn = power_spectrum(noise, fs)
    _, pt = power_spectrum(tone, fs)
    assert spectral_flatness(pn) > spectral_flatness(pt)


def test_envelope_duration_longer_for_sustained_signal():
    from thunderquakes.features.characterize import envelope_duration
    fs = 100.0
    n = 2000
    rng = np.random.default_rng(1)
    # impulsive: energy in a short burst
    impulsive = np.zeros(n)
    impulsive[1000:1050] = rng.standard_normal(50) * 10
    # sustained: energy spread across the window
    sustained = rng.standard_normal(n)
    assert envelope_duration(sustained, fs) > envelope_duration(impulsive, fs)


def test_trace_features_keys():
    from thunderquakes.features.characterize import trace_features
    fs = 100.0
    x = np.random.default_rng(2).standard_normal(1500)
    feats = trace_features(x, fs)
    for k in ("dominant_freq_hz", "spectral_flatness", "duration_80pct_s", "kurtosis"):
        assert k in feats
    # band-energy fractions sum to <= 1
    frac = sum(feats[k] for k in feats if k.startswith("energy_"))
    assert 0.0 <= frac <= 1.0001
