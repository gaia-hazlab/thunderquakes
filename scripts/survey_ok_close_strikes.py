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
from thunderquakes.plotting import (  # noqa: E402
    LETTER_WIDTH_IN,
    gray_relief_background,
    set_paper_style,
)

set_paper_style()

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

    lat0, lat1, lon0, lon1 = REGIONS[args.region]["bbox"]
    mean_lat_rad = np.radians((lat0 + lat1) / 2)
    aspect = 1.0 / max(np.cos(mean_lat_rad), 0.05)
    width = LETTER_WIDTH_IN
    height = min(9.5, width * (lat1 - lat0) / (lon1 - lon0) * aspect)
    fig, ax = plt.subplots(figsize=(width, height))
    gray_relief_background((lat0, lat1, lon0, lon1), ax)

    count_col = f"n_strikes_within_{max_r:g}km"
    is_sa = stations.get("seismoacoustic", pd.Series(False, index=stations.index)).fillna(False)
    vmin, vmax = 0, max(int(stations[count_col].max()), 1)

    def _sizes(counts):
        return 15 + 8 * np.sqrt(np.clip(counts, 0, None))

    sc = ax.scatter(
        stations.loc[~is_sa, "longitude"], stations.loc[~is_sa, "latitude"],
        s=_sizes(stations.loc[~is_sa, count_col]), c=stations.loc[~is_sa, count_col],
        cmap="inferno_r", vmin=vmin, vmax=vmax, marker="^",
        edgecolors="k", linewidths=0.3, zorder=3, label="seismic station",
    )
    if is_sa.any():
        ax.scatter(
            stations.loc[is_sa, "longitude"], stations.loc[is_sa, "latitude"],
            s=_sizes(stations.loc[is_sa, count_col]) * 1.8, c=stations.loc[is_sa, count_col],
            cmap="inferno_r", vmin=vmin, vmax=vmax, marker="*",
            edgecolors="k", linewidths=0.5, zorder=4,
            label=f"seismic + infrasound (n={int(is_sa.sum())})",
        )
    cbar = fig.colorbar(sc)
    cbar.set_label(f"Number of GLM strikes within {max_r:g} km", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    # Pad beyond the nominal bbox to the union with actual station coordinates,
    # so no marker (and no marker's radius) is clipped at the map edge.
    all_lon = pd.concat([stations["longitude"], pd.Series([lon0, lon1])])
    all_lat = pd.concat([stations["latitude"], pd.Series([lat0, lat1])])
    lon_pad = 0.04 * (all_lon.max() - all_lon.min() + 1e-6)
    lat_pad = 0.04 * (all_lat.max() - all_lat.min() + 1e-6)
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)",
          xlim=(all_lon.min() - lon_pad, all_lon.max() + lon_pad),
          ylim=(all_lat.min() - lat_pad, all_lat.max() + lat_pad))
    ax.set_aspect(aspect)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "ok_close_strikes_map.png", bbox_inches="tight")

    print("\nsurvey -> outputs/ok_close_strikes_survey.csv")
    print("map    -> docs/figures/ok_close_strikes_map.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
