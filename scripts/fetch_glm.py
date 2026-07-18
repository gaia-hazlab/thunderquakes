#!/usr/bin/env python
"""Fetch a GOES-GLM lightning catalog for a region + time window (WS3, issue #6).

Usage:
    pixi run -e default python scripts/fetch_glm.py \
        --region OK --start 2019-05-20T22:00:00 --end 2019-05-20T23:00:00

Writes outputs/glm_<REGION>_<START>_<END>.csv (regenerable; gitignored).
GLM is optical (no peak current) and its detection efficiency drops toward
Alaska latitudes — treat the catalog as a lower bound on true lightning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thunderquakes.config import REGIONS, REPO_ROOT
from thunderquakes.lightning import load_glm_strikes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--start", required=True, help="UTC ISO time, e.g. 2019-05-20T22:00:00")
    ap.add_argument("--end", required=True, help="UTC ISO time")
    ap.add_argument("--satellite", default=None, help="goes16/17/18/19 (default: East by date)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC")
    bbox = REGIONS[args.region]["bbox"]

    print(f"Fetching GLM for {args.region} {t0} → {t1} (bbox {bbox}) …")
    df = load_glm_strikes(t0, t1, bbox=bbox, satellite=args.satellite,
                          progress=True, max_workers=args.workers)

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.region}_{t0:%Y%m%dT%H%M}_{t1:%Y%m%dT%H%M}"
    out_path = out_dir / f"glm_{tag}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{len(df):,} flashes → {out_path.relative_to(REPO_ROOT)}")
    if len(df):
        print(f"  time  : {df['time_utc'].min()} → {df['time_utc'].max()}")
        print(f"  extent: lat {df.latitude.min():.2f}..{df.latitude.max():.2f}, "
              f"lon {df.longitude.min():.2f}..{df.longitude.max():.2f}")
        print(f"  source: {df['source'].unique().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
