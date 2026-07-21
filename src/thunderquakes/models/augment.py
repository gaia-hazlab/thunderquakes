"""Waveform augmentation for training diversity (WS4 #8).

Acts on raw 50 s windows BEFORE the spectrogram, applied to the TRAIN split only
(never test). Two goals from the review:

- **Realistic noise:** mix real recorded noise (sampled from the noise-class
  windows in the *training* split) into signal windows at a random SNR, so the
  model learns thunder embedded in field noise rather than clean lab-like traces.
- **Signal diversity:** amplitude scaling, sub-window time jitter, polarity flip,
  and light band emphasis — broadening the per-class distribution the model sees.

All randomness flows through an explicit ``np.random.Generator`` for reproducibility.
"""

from __future__ import annotations

import numpy as np


def _rms(x):
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def add_real_noise(sig, noise, snr_db, rng):
    """Mix a real ``noise`` window into ``sig`` at a target SNR (dB)."""
    if len(noise) != len(sig):
        # random-crop / tile the noise to match length
        if len(noise) >= len(sig):
            s = rng.integers(0, len(noise) - len(sig) + 1)
            noise = noise[s : s + len(sig)]
        else:
            noise = np.resize(noise, len(sig))
    target = _rms(sig) / (10 ** (snr_db / 20.0))
    noise = noise * (target / _rms(noise))
    return sig + noise


def amplitude_scale(sig, rng, db_range=(-6.0, 6.0)):
    return sig * 10 ** (rng.uniform(*db_range) / 20.0)


def polarity_flip(sig, rng, p=0.5):
    return -sig if rng.random() < p else sig


def add_white_noise(sig, rng, snr_db):
    n = rng.standard_normal(len(sig)).astype(sig.dtype)
    return sig + n * (_rms(sig) / (10 ** (snr_db / 20.0)) / _rms(n))


def augment_window(
    sig,
    rng,
    noise_pool=None,
    p_real_noise=0.7,
    real_snr_db=(0.0, 15.0),
    p_white=0.3,
    white_snr_db=(10.0, 25.0),
    p_polarity=0.5,
    amp_db=(-6.0, 6.0),
):
    """Return one augmented copy of a raw window."""
    out = amplitude_scale(np.asarray(sig, np.float32), rng, amp_db)
    if noise_pool is not None and len(noise_pool) and rng.random() < p_real_noise:
        out = add_real_noise(out, noise_pool[rng.integers(len(noise_pool))],
                             rng.uniform(*real_snr_db), rng)
    if rng.random() < p_white:
        out = add_white_noise(out, rng, rng.uniform(*white_snr_db))
    out = polarity_flip(out, rng, p_polarity)
    return out.astype("float32")


def augment_training_set(W, y, classes, n_aug=3, seed=0, noise_class="noise"):
    """Return an augmented (W, y): originals plus ``n_aug`` augmented copies each.

    The noise pool is drawn from the noise-class windows in THIS set (train split),
    so no test data leaks in. Noise-class windows themselves are not augmented with
    injected noise (they are the noise).
    """
    rng = np.random.default_rng(seed)
    noise_idx = classes.index(noise_class) if noise_class in classes else None
    noise_pool = W[y == noise_idx] if noise_idx is not None else None

    out_W, out_y = [W], [y]
    for _ in range(n_aug):
        aug = np.empty_like(W)
        for i in range(len(W)):
            pool = None if (noise_idx is not None and y[i] == noise_idx) else noise_pool
            aug[i] = augment_window(W[i], rng, noise_pool=pool)
        out_W.append(aug)
        out_y.append(y)
    return np.concatenate(out_W).astype("float32"), np.concatenate(out_y)
