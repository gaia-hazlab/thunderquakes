#!/usr/bin/env python
"""Build the seismic + infrasound station inventory for a region (WS2, issue #4).

Usage:
    pixi run -e default python scripts/build_inventory.py --region OK

Outputs:
    outputs/stations_<REGION>_channels.csv   full channel-epoch table (regenerable)
    catalogs/stations_<REGION>_summary.csv   one row per station (committed)
    catalogs/stations_<REGION>_infrasound.csv  seismoacoustic subset (committed)
    docs/figures/stations_<REGION>_map.png   coverage map (committed)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REGIONS, REPO_ROOT  # noqa: E402
from thunderquakes.plotting import (  # noqa: E402
    LETTER_WIDTH_IN,
    gray_relief_background,
    set_paper_style,
)
from thunderquakes.stations import add_availability, build_inventory, station_summary  # noqa: E402

set_paper_style()


def first_year_at_least(df: pd.DataFrame, threshold: int = 10) -> int | None:
    """First calendar year in which >= ``threshold`` stations in ``df`` were active.

    Uses actual archived-data extent (data_start/data_end) when available,
    falling back to the metadata deployment epoch (start/end) otherwise --
    metadata epochs can overstate real availability (see WS2 issue #4).
    """
    start_col = "data_start" if "data_start" in df.columns else "start"
    end_col = "data_end" if "data_end" in df.columns else "end"
    starts = pd.to_datetime(df[start_col], utc=True, errors="coerce")
    ends = pd.to_datetime(df[end_col], utc=True, errors="coerce")
    if not len(starts.dropna()):
        return None
    year0 = int(starts.dropna().dt.year.min())
    year1 = pd.Timestamp.now(tz="UTC").year
    for year in range(year0, year1 + 1):
        y_end = pd.Timestamp(f"{year}-12-31", tz="UTC")
        y_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        active = (starts <= y_end) & (ends.isna() | (ends >= y_start))
        if active.sum() >= threshold:
            return year
    return None


def plot_map(summary, region: str, out_path: Path) -> None:
    spec = REGIONS[region]
    lat0, lat1, lon0, lon1 = spec["bbox"]
    is_nodal = summary["datacenter"] == "IRISPH5"
    nodal = summary[is_nodal & ~summary["seismoacoustic"]]
    seismic = summary[~is_nodal & ~summary["seismoacoustic"]]
    seismoacoustic = summary[summary["seismoacoustic"]]

    # Equal physical aspect: at latitude L, 1 deg longitude spans cos(L) times
    # the physical distance of 1 deg latitude, so naive equal-degree axes
    # distort high-latitude regions (e.g. Alaska looks squeezed vertically).
    mean_lat_rad = np.radians((lat0 + lat1) / 2)
    aspect = 1.0 / max(np.cos(mean_lat_rad), 0.05)
    width = LETTER_WIDTH_IN
    height = min(9.5, width * (lat1 - lat0) / (lon1 - lon0) * aspect)
    fig, ax = plt.subplots(figsize=(width, height))
    gray_relief_background((lat0, lat1, lon0, lon1), ax)
    if len(nodal):
        ax.scatter(
            nodal["longitude"], nodal["latitude"],
            c="tan", s=6, marker="o", alpha=0.6,
            label=f"2016 nodal, LASSO/YW (n={len(nodal)})", zorder=2,
        )
    seismic_label = f"permanent seismic (n={len(seismic)}"
    if len(seismic) > 10:
        year = first_year_at_least(seismic, 10)
        if year:
            seismic_label += f", >=10 stations since {year}"
    seismic_label += ")"
    ax.scatter(
        seismic["longitude"], seismic["latitude"],
        c="0.4", s=20, marker="^", label=seismic_label, zorder=3,
    )
    sa_label = f"seismic + infrasound (n={len(seismoacoustic)}"
    if len(seismoacoustic) > 10:
        year = first_year_at_least(seismoacoustic, 10)
        if year:
            sa_label += f", >=10 stations since {year}"
    sa_label += ")"
    ax.scatter(
        seismoacoustic["longitude"], seismoacoustic["latitude"],
        c="crimson", s=110, marker="*",
        label=sa_label, zorder=4,
        edgecolors="k", linewidths=0.5,
    )
    # Pad beyond the nominal bbox to the union with actual station coordinates,
    # so no marker (and no marker's radius) is clipped at the map edge.
    all_lon = pd.concat([summary["longitude"], pd.Series([lon0, lon1])])
    all_lat = pd.concat([summary["latitude"], pd.Series([lat0, lat1])])
    lon_pad = 0.04 * (all_lon.max() - all_lon.min() + 1e-6)
    lat_pad = 0.04 * (all_lat.max() - all_lat.min() + 1e-6)
    ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)",
          xlim=(all_lon.min() - lon_pad, all_lon.max() + lon_pad),
          ylim=(all_lat.min() - lat_pad, all_lat.max() + lat_pad))
    ax.set_aspect(aspect)
    ax.grid(alpha=0.3)
    # Legend below the map (not overlapping it) so it never hides station markers.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=9.5,
             frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  map  -> {out_path.relative_to(REPO_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=list(REGIONS))
    ap.add_argument("--client", default="IRIS")
    ap.add_argument("--no-nodal", action="store_true", help="skip PH5 nodal networks (LASSO/YW)")
    ap.add_argument("--no-availability", action="store_true", help="skip fdsnws-availability query")
    args = ap.parse_args()
    region = args.region

    ph5 = REGIONS[region].get("ph5_networks", []) if not args.no_nodal else []
    print(f"Querying FDSN ({args.client}) for {region} "
          f"networks {REGIONS[region]['seismic_networks']} + PH5 {ph5} …")
    inv = build_inventory(region, client=args.client, include_nodal=not args.no_nodal)
    if not args.no_availability:
        print("Querying fdsnws-availability for actual archived-data extents …")
        inv = add_availability(inv)
    summary = station_summary(inv)
    infra = summary[summary["seismoacoustic"]]

    out_dir = REPO_ROOT / "outputs"
    cat_dir = REPO_ROOT / "catalogs"
    out_dir.mkdir(exist_ok=True)
    cat_dir.mkdir(exist_ok=True)

    ch_path = out_dir / f"stations_{region}_channels.csv"
    sum_path = cat_dir / f"stations_{region}_summary.csv"
    infra_path = cat_dir / f"stations_{region}_infrasound.csv"
    inv.to_csv(ch_path, index=False)
    summary.to_csv(sum_path, index=False)
    infra.to_csv(infra_path, index=False)

    print(f"\n{region}: {summary.shape[0]} stations, {inv.shape[0]} channel-epochs")
    print(f"  by network        : {inv.groupby('network')['station'].nunique().to_dict()}")
    print(f"  seismoacoustic    : {len(infra)} stations "
          f"(networks {infra['network'].value_counts().to_dict()})")
    print(f"  channels -> {ch_path.relative_to(REPO_ROOT)}")
    print(f"  summary  -> {sum_path.relative_to(REPO_ROOT)}")
    print(f"  infra    -> {infra_path.relative_to(REPO_ROOT)}")
    plot_map(summary, region, REPO_ROOT / "docs" / "figures" / f"stations_{region}_map.png")

    print("\nSeismoacoustic stations (seismic + usable infrasound):")
    cols = ["network", "station", "latitude", "longitude", "infrasound_channels", "start", "end"]
    print(infra[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
