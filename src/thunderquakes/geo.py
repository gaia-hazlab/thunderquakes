"""Geospatial helpers (single source of truth for haversine).

The prototype defined ``haversine_km`` twice with slightly different code; this
consolidates it.
"""

from __future__ import annotations

import numpy as np

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km. Any argument may be array-like."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))
