#!/usr/bin/env python
"""Survey close-proximity GLM strike / OK station pairs (WS4 #11, sizing step).

Before building a GLM-triggered extraction pipeline, size the opportunity: for a
known-active convective window, how many (station, strike) pairs land within a
tight radius (where a strike is plausibly audible/coupled at the seismic station)?
This is INDEPENDENT of the model's own predictions — it uses only the lightning
catalog + station coordinates, avoiding the circularity risk of training on the
model's own flagged windows.

Usage:
    pixi run -e default python scripts/survey_ok_close_strikes.py \
        --start 2019-05-20T20:00:00 --end 2019-05-21T02:00:00

Outputs:
    outputs/ok_close_strikes_survey.csv    one row per (station, strike) within
                                            the widest radius checked (regenerable,
                                            large -- not committed)
    docs/figures/ok_close_strikes_map.png  station map, sized/colored by count
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REGIONS, REPO_ROOT  # noqa: E402
from thunderquakes.geo import haversine_km  # noqa: E402
from thunderquakes.lightning import load_glm_strikes  # noqa: E402

RADII_KM = [5.0, 10.0, 20.0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--region", default="OK")
    args = ap.parse_args()

    cat_dir = REPO_ROOT / "catalogs"
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig_dir = REPO_ROOT / "docs" / "figures"

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC")
    print(f"Fetching GLM strikes for {t0} -> {t1} …")
    strikes = load_glm_strikes(t0, t1, bbox=REGIONS[args.region]["bbox"], progress=True)
    print(f"  {len(strikes):,} flashes statewide")

    inv = pd.read_csv(cat_dir / f"stations_{args.region}_summary.csv")
    stations = inv[inv["datacenter"] == "IRIS"].drop_duplicates(["network", "station"]).copy()
    print(f"Checking proximity against {len(stations)} permanent stations …")

    s_lat = strikes["latitude"].to_numpy()
    s_lon = strikes["longitude"].to_numpy()
    s_time = strikes["time_utc"].to_numpy()

    max_r = max(RADII_KM)
    rows = []
    counts = dict.fromkeys(RADII_KM, 0)
    stations_hit = dict.fromkeys(RADII_KM, 0)
    per_station_max_r = []
    for st in stations.itertuples(index=False):
        d = haversine_km(st.latitude, st.longitude, s_lat, s_lon)
        n_at_max = int((d <= max_r).sum())
        per_station_max_r.append(n_at_max)
        if n_at_max:
            for i in np.flatnonzero(d <= max_r):
                rows.append({
                    "network": st.network, "station": st.station,
                    "sta_lat": st.latitude, "sta_lon": st.longitude,
                    "strike_time": s_time[i], "dist_km": float(d[i]),
                })
        for r in RADII_KM:
            n_at_r = int((d <= r).sum())
            counts[r] += n_at_r
            stations_hit[r] += n_at_r > 0

    survey = pd.DataFrame(rows).sort_values("dist_km") if rows else pd.DataFrame(rows)
    survey.to_csv(out_dir / "ok_close_strikes_survey.csv", index=False)
    stations[f"n_strikes_within_{max_r:g}km"] = per_station_max_r

    print("\n=== (station, strike) pairs within radius, this window ===")
    for r in RADII_KM:
        print(f"  <= {r:>5.1f} km : {counts[r]:>6,} pairs   "
              f"({stations_hit[r]}/{len(stations)} stations have >=1 pair)")

    print("\nTop 15 closest (station, strike) pairs:")
    if len(survey):
        cols = ["network", "station", "dist_km", "strike_time"]
        print(survey.head(15)[cols].to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 7))
    lat0, lat1, lon0, lon1 = REGIONS[args.region]["bbox"]
    sizes = 15 + 8 * np.sqrt(np.clip(stations[f"n_strikes_within_{max_r:g}km"], 0, None))
    sc = ax.scatter(stations["longitude"], stations["latitude"], s=sizes,
                    c=stations[f"n_strikes_within_{max_r:g}km"], cmap="inferno_r",
                    edgecolors="k", linewidths=0.3)
    fig.colorbar(sc, label=f"# strikes within {max_r:g} km")
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)", xlim=(lon0, lon1), ylim=(lat0, lat1),
          title=f"{args.region} close-strike survey (independent of model)\n"
                f"{t0:%Y-%m-%d %H:%M} → {t1:%H:%M} UTC, radius {max_r:g} km")
    fig.tight_layout()
    fig.savefig(fig_dir / "ok_close_strikes_map.png", dpi=150)

    print("\nsurvey -> outputs/ok_close_strikes_survey.csv")
    print("map    -> docs/figures/ok_close_strikes_map.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
