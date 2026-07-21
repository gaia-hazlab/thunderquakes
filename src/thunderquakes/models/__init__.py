"""CNN classifier, uncertainty estimation, and training (WS4).

Model A: seismic-only 2D CNN (anti-aliased BlurPool) — deployable everywhere.
Model B: seismic + infrasound dual-branch — the seismo-acoustic coupling (WS1:
median envelope xcorr 0.85 at ~0 s lag) is the discriminator that should break the
thunder<->sonic-boom / surface-event confusion seen in the Model A baseline.

Dataset/augment helpers import cleanly in the default env; cnn/train require the
'ml' env (torch) so import those modules explicitly.
"""

from thunderquakes.models.augment import augment_training_set, augment_window
from thunderquakes.models.dataset import (
    CLASS_STORE,
    REGION_CLASSES,
    RawDataset,
    WindowConfig,
    build_windows,
    to_spectrograms,
)

__all__ = [
    "CLASS_STORE",
    "REGION_CLASSES",
    "RawDataset",
    "WindowConfig",
    "augment_training_set",
    "augment_window",
    "build_windows",
    "to_spectrograms",
]
