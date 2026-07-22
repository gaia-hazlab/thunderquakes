"""Small statistical helpers for evaluation (WS4 #11).

Kept dependency-light (scipy only, no torch) so it is importable and testable
from the default env.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """One-sided two-proportion z-test for p1 > p2. Returns (z, p_value).

    Used to test whether a model's candidate-detection match rate against an
    independent catalog (e.g. GLM strikes) exceeds the null baseline (the same
    match test applied to random windows) — the honest way to interpret a
    match rate when the reference catalog is very dense (see issue #11).
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return float(z), float(1 - norm.cdf(z))
