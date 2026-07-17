"""WWLLN (World Wide Lightning Location Network) loader — WS3 backstop.

Global, works at high latitude and pre-2018 (unlike GLM). Requires a research
data agreement; returns the common schema.

TODO(WS3): parse WWLLN A-files / stroke files into COMMON_SCHEMA.
"""
from __future__ import annotations

import pandas as pd


def load_wwlln_strikes(t_start, t_end, bbox=None) -> pd.DataFrame:
    raise NotImplementedError("WS3: implement WWLLN loader (cross-region backstop).")
