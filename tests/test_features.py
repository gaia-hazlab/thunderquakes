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
