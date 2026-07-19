import numpy as np

from thunderquakes.models.dataset import WindowConfig, crop_around_peak, log_spectrogram_image


def test_crop_centers_on_peak():
    n = 15000
    x = np.zeros(n)
    x[9000:9100] = 5.0  # energy burst well away from center
    win = crop_around_peak(x, 5000)
    assert len(win) == 5000
    # the burst should be inside the crop and near its center
    peak_local = int(np.argmax(np.abs(win)))
    assert 2000 < peak_local < 3000


def test_crop_pads_short_input():
    x = np.ones(3000)
    win = crop_around_peak(x, 5000)
    assert len(win) == 5000
    assert win[4000] == 0.0  # padded tail


def test_log_spectrogram_shape_and_standardized():
    cfg = WindowConfig()
    rng = np.random.default_rng(0)
    win = rng.standard_normal(cfg.n_samples)
    img = log_spectrogram_image(win, cfg)
    assert img.ndim == 2
    assert img.dtype == np.float32
    # per-image standardized -> ~zero mean, ~unit std
    assert abs(img.mean()) < 0.1
    assert 0.7 < img.std() < 1.3
    # frequency axis limited to the band (1-45 Hz of 0-50 Hz nyquist)
    assert img.shape[0] < cfg.nperseg // 2 + 1
