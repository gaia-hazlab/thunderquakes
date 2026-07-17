"""Alaska Lightning Detection Network (ALDN) loader — WS3 primary truth for AK.

Operated for the BLM Alaska Fire Service / Alaska Interagency Coordination
Center (AICC). Returns the common schema (see ``lightning.__init__``).

TODO(WS3): wire up the AICC/BLM data access and normalise to COMMON_SCHEMA.
"""
from __future__ import annotations

import pandas as pd


def load_aldn_strikes(t_start, t_end, bbox=None) -> pd.DataFrame:
    raise NotImplementedError("WS3: implement ALDN access for Alaska ground truth.")
