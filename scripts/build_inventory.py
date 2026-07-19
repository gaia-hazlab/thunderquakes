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

from thunderquakes.config import REGIONS, REPO_ROOT  # noqa: E402
from thunderquakes.stations import add_availability, build_inventory, station_summary  # noqa: E402


def plot_map(summary, region: str, out_path: Path) -> None:
    spec = REGIONS[region]
    lat0, lat1, lon0, lon1 = spec["bbox"]
    is_nodal = summary["datacenter"] == "IRISPH5"
    nodal = summary[is_nodal & ~summary["seismoacoustic"]]
    seismic = summary[~is_nodal & ~summary["seismoacoustic"]]
    seismoacoustic = summary[summary["seismoacoustic"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    if len(nodal):
        ax.scatter(
            nodal["longitude"], nodal["latitude"],
            c="tan", s=4, marker="o", alpha=0.5,
            label=f"2016 nodal, LASSO/YW (n={len(nodal)})", zorder=1,
        )
    ax.scatter(
        seismic["longitude"], seismic["latitude"],
        c="0.55", s=18, marker="^", label=f"permanent seismic (n={len(seismic)})", zorder=2,
    )
    ax.scatter(
        seismoacoustic["longitude"], seismoacoustic["latitude"],
        c="crimson", s=90, marker="*",
        label=f"seismic + infrasound (n={len(seismoacoustic)})", zorder=3,
        edgecolors="k", linewidths=0.4,
    )
    # Only label seismoacoustic sites when sparse enough to stay readable.
    if len(seismoacoustic) <= 25:
        for r in seismoacoustic.itertuples():
            ax.annotate(f"{r.network}.{r.station}", (r.longitude, r.latitude),
                        fontsize=5, color="darkred", xytext=(2, 2), textcoords="offset points")
    ax.set(
        xlabel="Longitude", ylabel="Latitude", xlim=(lon0, lon1), ylim=(lat0, lat1),
        title=f"{region} station coverage — seismic + infrasound (WS2)\n"
        f"fdsnws: {', '.join(spec['seismic_networks'])}"
        + (f"   |   PH5: {', '.join(spec['ph5_networks'])}" if spec.get("ph5_networks") else ""),
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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
