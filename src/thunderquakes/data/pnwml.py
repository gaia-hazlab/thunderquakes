"""Load the PNWML dataset metadata (comcat / exotic / noise).

Refactored from cells 1-3 of the prototype notebook, with the hardcoded
``/data/whd02/...`` paths replaced by :mod:`thunderquakes.config`.

PNWML source: Ni et al., "A curated dataset of seismic waveforms for the
Pacific Northwest" (the 'comcat', 'exotic', and 'noise' stores). The 'exotic'
store carries the analyst 'thunder', 'surface event', etc. source types that
are the gold labels for this project.
"""

from __future__ import annotations

import pandas as pd

from thunderquakes.config import pnwml_paths

# PNWML source_type -> which store its metadata lives in.
_STORE_FOR_CLASS = {
    "earthquake": "comcat",
    "explosion": "comcat",
    "surface event": "exotic",
    "thunder": "exotic",
}


def load_pnwml_metadata(root=None) -> dict[str, pd.DataFrame]:
    """Return the three PNWML metadata tables keyed by store name.

    Adds a synthetic ``event_id`` to the noise table (as the prototype did),
    since noise windows have no catalog event id.
    """
    p = pnwml_paths(root)
    meta = {
        "comcat": pd.read_csv(p.comcat["metadata"]),
        "exotic": pd.read_csv(p.exotic["metadata"]),
        "noise": pd.read_csv(p.noise["metadata"]),
    }
    noise = meta["noise"]
    if "event_id" not in noise.columns and "trace_start_time" in noise.columns:
        noise["event_id"] = noise["trace_start_time"].astype(str) + "_noise"
    return meta


def load_class_metadata(source_type: str, root=None) -> pd.DataFrame:
    """Return metadata rows for a single PNWML ``source_type``.

    Example
    -------
    >>> thunder = load_class_metadata("thunder")
    """
    store = _STORE_FOR_CLASS.get(source_type)
    if store is None:
        raise ValueError(
            f"Unknown source_type {source_type!r}; "
            f"expected one of {sorted(_STORE_FOR_CLASS)}"
        )
    meta = load_pnwml_metadata(root)[store]
    return meta[meta["source_type"] == source_type].copy()
