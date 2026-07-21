import numpy as np

from thunderquakes.models.augment import add_real_noise, augment_training_set, augment_window


def test_add_real_noise_hits_target_snr():
    rng = np.random.default_rng(0)
    sig = np.sin(np.linspace(0, 50, 5000)).astype("float32")
    noise = rng.standard_normal(5000).astype("float32")
    out = add_real_noise(sig, noise, snr_db=10.0, rng=rng)
    added = out - sig
    snr = 20 * np.log10(np.sqrt(np.mean(sig**2)) / np.sqrt(np.mean(added**2)))
    assert abs(snr - 10.0) < 0.5


def test_augment_window_changes_signal_preserves_length():
    rng = np.random.default_rng(1)
    sig = np.ones(5000, dtype="float32")
    pool = rng.standard_normal((5, 5000)).astype("float32")
    out = augment_window(sig, rng, noise_pool=pool)
    assert out.shape == sig.shape
    assert out.dtype == np.float32
    assert not np.allclose(out, sig)


def test_augment_training_set_grows_and_keeps_labels():
    W = np.random.default_rng(2).standard_normal((10, 5000)).astype("float32")
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    classes = ["thunder", "sonic boom", "noise"]
    W2, y2 = augment_training_set(W, y, classes, n_aug=3, seed=0)
    assert len(W2) == len(W) * 4          # originals + 3 augmented copies
    assert np.bincount(y2).tolist() == (np.bincount(y) * 4).tolist()
