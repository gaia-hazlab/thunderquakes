"""Spectrogram CNN — seismic-only and dual-branch (WS4).

Requires the 'ml' pixi environment (pytorch). Kept import-light so the default
env can still import the package: torch is imported lazily inside the builders.
"""
from __future__ import annotations


def build_seismic_cnn(n_classes: int = 5, dropout: float = 0.3):
    """Model A: single-branch spectrogram CNN over 3-C seismic input."""
    raise NotImplementedError("WS4: implement torch.nn.Module (seismic-only).")


def build_dual_branch_cnn(n_classes: int = 5, dropout: float = 0.3):
    """Model B: shared-encoder seismic + infrasound branches, late fusion.

    Setting the infrasound branch to zeros recovers Model A (single codebase).
    """
    raise NotImplementedError("WS4: implement dual-branch torch.nn.Module.")
