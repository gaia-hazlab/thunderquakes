#!/usr/bin/env python
"""GLM-triggered extraction of candidate OK thunderquake waveforms (WS4 #11).

Builds a real, OK-native candidate training set WITHOUT touching the model's own
predictions (avoiding the circularity risk of self-labelling): starts from the
close-strike survey (scripts/survey_ok_close_strikes.py), clusters raw strikes
into distinct time-separated EVENTS per station (a storm cell passing overhead
produces many strikes within seconds — these are one physical episode, not many),
fetches the real seismic window at each event independent of any model, and
produces a waveform+spectrogram gallery for human verification before anything
is trusted as a label.

Usage:
    pixi run -e default python scripts/extract_ok_thunder_candidates.py \
        --radius-km 10 --gallery-n 24

Outputs:
    catalogs/ok_candidate_events.csv     one row per distinct (station, event)
    docs/figures/verify_ok_candidates_*.png   waveform+spectrogram galleries
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import spectrogram as _spec  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

EVENT_GAP_S = 90.0  # strikes more than this far apart start a new event (WS1: episodes ~89s median)
WINDOW_S = 90.0     # extraction window length (WS1 p90/p95 episode span)
GALLERY_NCOL = 2    # 2 columns keeps each panel wide enough for >=10pt labels at print width
GALLERY_MAX_PLOT = 8  # cap panels drawn even if more were successfully fetched


def cluster_events(survey: pd.DataFrame, radius_km: float, gap_s: float) -> pd.DataFrame:
    """Collapse (station, strike) pairs within ``radius_km`` into distinct events.

    An event = a run of strikes at the same station with consecutive gaps <= gap_s.
    Event time = earliest strike in the cluster (so the extraction window naturally
    covers the whole multi-clap episode, per the WS1 duration finding).
    """
    sub = survey[survey["dist_km"] <= radius_km].copy()
    sub["strike_time"] = pd.to_datetime(sub["strike_time"], utc=True)
    events = []
    for (net, sta), g in sub.groupby(["network", "station"]):
        g = g.sort_values("strike_time")
        t = g["strike_time"].values
        gap = np.diff(t) / np.timedelta64(1, "s")
        new_event = np.concatenate([[True], gap > gap_s])
        event_id = np.cumsum(new_event)
        g = g.assign(event_id=event_id)
        for _, ev in g.groupby("event_id"):
            events.append({
                "network": net, "station": sta,
                "event_start": ev["strike_time"].min(),
                "event_end": ev["strike_time"].max(),
                "n_strikes": len(ev),
                "min_dist_km": ev["dist_km"].min(),
                "sta_lat": ev["sta_lat"].iloc[0], "sta_lon": ev["sta_lon"].iloc[0],
            })
    out = pd.DataFrame(events)
    return out.sort_values("min_dist_km").reset_index(drop=True) if len(out) else out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius-km", type=float, default=10.0,
                    help="max strike-to-station distance to treat as plausibly coupled")
    ap.add_argument("--gap-s", type=float, default=EVENT_GAP_S)
    ap.add_argument("--gallery-n", type=int, default=24,
                    help="number of candidate events to fetch+plot for human QC")
    ap.add_argument("--max-strikes", type=int, default=20,
                    help="exclude events with more strikes than this from the gallery sample "
                    "-- very high counts mean a storm sat continuously overhead (sustained "
                    "noise), not a single discrete thunderclap episode like the PNWML labels")
    ap.add_argument("--client", default="IRIS")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cat_dir = REPO_ROOT / "catalogs"
    out_dir = REPO_ROOT / "outputs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    survey = pd.read_csv(out_dir / "ok_close_strikes_survey.csv")
    events = cluster_events(survey, args.radius_km, args.gap_s)
    events.to_csv(cat_dir / "ok_candidate_events.csv", index=False)

    n_raw = len(survey[survey.dist_km <= args.radius_km])
    print(f"Raw pairs within {args.radius_km:g} km: {n_raw:,}")
    print(f"Distinct events after clustering (gap>{args.gap_s:g}s): {len(events):,} "
          f"across {events[['network', 'station']].drop_duplicates().shape[0]} stations")
    print(f"  median n_strikes/event: {events['n_strikes'].median():.0f}  "
          f"(events with n_strikes>100, likely sustained storm-overhead noise "
          f"rather than a discrete episode: {(events['n_strikes'] > 100).sum()})")
    print("\nEvents per station (top 10):")
    print(events.groupby(["network", "station"]).size().sort_values(ascending=False).head(10))

    # Sample spread across stations, excluding sustained-overhead-storm clusters (see --max-strikes)
    from obspy.clients.fdsn import Client

    client = Client(args.client)
    pool = events[events["n_strikes"] <= args.max_strikes]
    print(f"\nGallery sample pool after n_strikes<={args.max_strikes} filter: "
          f"{len(pool):,}/{len(events):,} events")
    events = pool.reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(events), size=min(args.gallery_n, len(events)), replace=False)
    sample = events.iloc[sample_idx].sort_values("min_dist_km")

    print(f"\nFetching {len(sample)} candidate windows for verification gallery …")
    fetched = []
    for ev in sample.itertuples(index=False):
        # Acoustic travel time from strike to station: GLM is optical (~instantaneous);
        # the coupled seismic arrival lags by dist/c_sound. At the 10 km max radius that
        # is up to ~29s -- a fixed quarter-window offset pushes far events toward (or past)
        # the window edge, so center on the PHYSICALLY EXPECTED arrival instead.
        acoustic_delay_s = ev.min_dist_km / 0.34
        t_center = (pd.Timestamp(ev.event_start) + pd.Timedelta(seconds=acoustic_delay_s)
                   - pd.Timedelta(seconds=5))  # small pre-buffer for the direct/seismic arrival
        data, fs = cached_waveform(ev.network, ev.station, "HHZ", t_center, WINDOW_S, client=client)
        if data is None:
            data, fs = cached_waveform(ev.network, ev.station, "BHZ", t_center, WINDOW_S,
                                       client=client)
        if data is None:
            continue
        fetched.append((ev, data, fs))
    print(f"  got waveform data for {len(fetched)}/{len(sample)}")

    plotted = fetched[:GALLERY_MAX_PLOT]
    ncol = min(GALLERY_NCOL, len(plotted))
    nrow = int(np.ceil(len(plotted) / ncol)) if ncol else 0
    if nrow:
        col_w = LETTER_WIDTH_IN / GALLERY_NCOL
        fig, axes = plt.subplots(nrow * 2, ncol, figsize=(col_w * ncol, 2.2 * nrow),
                                 sharex="col", squeeze=False)
        spec_axes = []
        for k, (ev, data, fs) in enumerate(plotted):
            r, c = divmod(k, ncol)
            is_last_row = r == nrow - 1
            t = np.arange(len(data)) / fs
            aw = axes[r * 2][c]
            aw.plot(t, data, lw=0.5, color="steelblue")
            aw.set_title(f"{ev.network}.{ev.station}  d={ev.min_dist_km*1000:.0f}m  "
                        f"n_strikes={ev.n_strikes}\n{str(ev.event_start)[:16]}", fontsize=10)
            aw.tick_params(labelsize=10)
            if c == 0:
                aw.set_ylabel("Amp\n(counts)", fontsize=10)
            asp = axes[r * 2 + 1][c]
            f, tt, sxx = _spec(data.astype(float), fs=fs, nperseg=128, noverlap=96)
            band = (f >= 1) & (f <= 45)
            asp.pcolormesh(tt, f[band], 10 * np.log10(sxx[band] + 1e-12),
                          shading="gouraud", cmap="inferno")
            asp.tick_params(labelsize=10)
            if c == 0:
                asp.set_ylabel("Freq\n(Hz)", fontsize=10)
            else:
                asp.tick_params(labelleft=False)
            if is_last_row:
                asp.set_xlabel("Time (s)", fontsize=10)
            spec_axes.append(asp)
        for ax in spec_axes[1:]:
            ax.sharey(spec_axes[0])
        for k in range(len(plotted), nrow * ncol):
            r, c = divmod(k, ncol)
            axes[r * 2][c].axis("off")
            axes[r * 2 + 1][c].axis("off")
        fig.tight_layout(h_pad=0.3)
        out = fig_dir / f"verify_ok_candidates_r{args.radius_km:g}km.png"
        fig.savefig(out)
        print(f"\ngallery -> {out.relative_to(REPO_ROOT)} (showing {len(plotted)}/{len(fetched)})")

    print("events  -> catalogs/ok_candidate_events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
