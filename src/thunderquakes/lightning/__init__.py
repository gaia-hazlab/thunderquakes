"""Open lightning-strike catalogs used as independent ground truth (WS3).

Decision (locked): open-only sources.
  - GOES-GLM   -> primary truth for OK & PNW (2018-present; poor coverage in AK).
  - ALDN       -> primary truth for AK.
  - WWLLN      -> cross-region backstop and only pre-2018 option.

All loaders return a common schema:
    time_utc (tz-aware), latitude, longitude, peak_current (nullable), source.
Detection efficiency varies by network/region: treat every catalog as a LOWER
bound on true lightning activity when computing completeness.
"""

from thunderquakes.lightning.glm import load_glm_strikes
from thunderquakes.lightning.wwlln import load_wwlln_strikes, parse_wwlln_file

# Shared output schema for every lightning catalog loader (GLM / ALDN / WWLLN).
COMMON_SCHEMA = ["time_utc", "latitude", "longitude", "peak_current", "source"]

__all__ = ["COMMON_SCHEMA", "load_glm_strikes", "load_wwlln_strikes", "parse_wwlln_file"]
