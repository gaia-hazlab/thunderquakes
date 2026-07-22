#!/usr/bin/env python
"""Seismoacoustic moveout verification for OK GLM-triggered candidates (WS4 #9/#11).

The cross-correlation numbers in scripts/verify_seismoacoustic_candidates.py show
THAT seismic and infrasound envelopes correlate, but not whether the timing is
physically consistent with a real acoustic wave. A genuine thunderquake's envelope
lag should track strike-to-station distance at the speed of sound (~340 m/s) --
this is the same moveout logic used to confirm a real seismic phase arrival across
a network, applied here to a single station's two channels across many events at
different distances instead of many stations for one event.

Produces:
  1. A moveout scatter: observed envelope lag vs. strike distance for every
     candidate with a fetched seismic+infrasound pair, against the predicted
     travel-time line lag = distance / 340 m/s. Points near the line are strong
     physical confirmation; scatter far from it is not.
  2. Two concrete "record section" examples (raw seismic + infrasound waveforms,
     not just envelopes) for the best-coupled events, time-aligned to the strike
     and marked with the predicted acoustic arrival, so the propagation is visible
     directly rather than only in a summary statistic.

Usage:
    pixi run -e default python scripts/plot_seismoacoustic_moveout.py

Requires catalogs/ok_seismoacoustic_verification.csv (from
scripts/verify_seismoacoustic_candidates.py) to already exist.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

WINDOW_S = 90.0
SOUND_SPEED_M_S = 340.0
N_EXAMPLES = 2


def main() -> int:
    cat_dir = REPO_ROOT / "catalogs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    result = pd.read_csv(cat_dir / "ok_seismoacoustic_verification.csv")
    if not len(result):
        raise SystemExit("catalogs/ok_seismoacoustic_verification.csv is empty; "
                         "run scripts/verify_seismoacoustic_candidates.py first.")

    dist_m = result["min_dist_km"].to_numpy() * 1000.0
    predicted_s = dist_m / SOUND_SPEED_M_S
    residual_s = np.abs(result["lag_s"].to_numpy()) - predicted_s
    print(f"Moveout check: {len(result)} events, distance {dist_m.min():.0f}-{dist_m.max():.0f} m")
    print(f"  median |observed lag - predicted d/{SOUND_SPEED_M_S:g}m/s|: "
          f"{np.median(np.abs(residual_s)):.1f} s")

    # --- Figure 1: moveout scatter (distance vs observed lag, vs predicted line) ---
    fig, ax = plt.subplots(figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.72))
    sc = ax.scatter(dist_m, result["lag_s"].abs(), c=result["xcorr"], cmap="viridis",
                    s=45, edgecolors="k", linewidths=0.4, zorder=3)
    d_line = np.linspace(max(dist_m.min(), 1), dist_m.max(), 100)
    ax.plot(d_line, d_line / SOUND_SPEED_M_S, color="crimson", ls="--", lw=1.5, zorder=2,
           label=f"predicted acoustic travel time\n(distance / {SOUND_SPEED_M_S:g} m/s)")
    cbar = fig.colorbar(sc)
    cbar.set_label("Cross-correlation coefficient\n(dimensionless, -1 to 1)", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    ax.set(xlabel="Strike-to-station distance (m)",
          ylabel="Observed envelope lag,\n|infrasound - seismic| (s)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = fig_dir / "ok_seismoacoustic_moveout.png"
    fig.savefig(out1)
    print(f"moveout scatter -> {out1.relative_to(REPO_ROOT)}")

    # --- Figure 2: record-section examples for the best-coupled events ---
    best = result.sort_values("xcorr", ascending=False).head(N_EXAMPLES)
    from obspy.clients.fdsn import Client

    client = Client("IRIS")
    fig, axes = plt.subplots(N_EXAMPLES, 2, figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.85),
                             squeeze=False)
    for row_i, ev in enumerate(best.itertuples(index=False)):
        acoustic_delay_s = ev.min_dist_km * 1000.0 / SOUND_SPEED_M_S
        t_center = (pd.Timestamp(ev.event_start) + pd.Timedelta(seconds=acoustic_delay_s)
                   - pd.Timedelta(seconds=5))
        seis, fs_s = cached_waveform(ev.network, ev.station, "HHZ", t_center, WINDOW_S,
                                     band=(1.0, 20.0), client=client)
        if seis is None:
            seis, fs_s = cached_waveform(ev.network, ev.station, "BHZ", t_center, WINDOW_S,
                                         band=(1.0, 20.0), client=client)
        infra, fs_i = cached_waveform(ev.network, ev.station, "BDF", t_center, WINDOW_S,
                                      band=(1.0, 20.0), client=client)
        if seis is None or infra is None:
            continue
        # cached_waveform's t0 arg is the window START (t_center here despite the name),
        # and the window was built as [predicted_arrival - 5s, predicted_arrival + 85s]
        # (see extract/verify scripts), so sample i falls at (predicted_arrival - 5s) + i/fs:
        # t=0 below is exactly the PREDICTED acoustic arrival, independent of acoustic_delay_s.
        t_seis = np.arange(len(seis)) / fs_s - 5.0
        t_infra = np.arange(len(infra)) / fs_i - 5.0

        is_last_row = row_i == len(axes) - 1
        ax_s, ax_i = axes[row_i]
        ax_s.plot(t_seis, seis, lw=0.5, color="steelblue")
        ax_s.axvline(0, color="crimson", ls="--", lw=1.2,
                    label=f"predicted arrival\n(d={ev.min_dist_km*1000:.0f} m)")
        ax_s.set_title(f"{ev.network}.{ev.station}  {str(ev.event_start)[:19]}  "
                      f"xcorr={ev.xcorr:.2f}", fontsize=10)
        ax_s.set_ylabel("Seismic\namplitude (counts)", fontsize=10)
        ax_s.tick_params(labelsize=10)
        ax_s.legend(fontsize=10, loc="upper right")
        if is_last_row:
            ax_s.set_xlabel("Time rel. to predicted\nacoustic arrival (s)", fontsize=10)

        ax_i.plot(t_infra, infra, lw=0.5, color="darkorange")
        ax_i.axvline(0, color="crimson", ls="--", lw=1.2)
        ax_i.set_ylabel("Infrasound\namplitude (counts)", fontsize=10)
        ax_i.tick_params(labelsize=10)
        if is_last_row:
            ax_i.set_xlabel("Time rel. to predicted\nacoustic arrival (s)", fontsize=10)
    fig.tight_layout()
    out2 = fig_dir / "ok_seismoacoustic_moveout_examples.png"
    fig.savefig(out2)
    print(f"record-section examples -> {out2.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
