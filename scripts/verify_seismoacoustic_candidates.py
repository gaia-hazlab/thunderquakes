#!/usr/bin/env python
"""Seismoacoustic verification of GLM-triggered OK candidates (WS4 #11/#9).

Proximity to a GLM strike only says a storm was nearby; it does NOT confirm the
seismic signal is genuinely thunder-coupled. At a co-located seismic+infrasound
station, a real thunderquake shows a much stronger, physics-based signature: the
seismic and infrasound envelopes correlate strongly at a small, near-zero lag
(WS1: median xcorr 0.85 across PNW seismoacoustic sites). This script applies that
test to the subset of GLM-triggered OK candidates (scripts/extract_ok_thunder_
candidates.py) that fall at Oklahoma's permanent seismoacoustic stations, to see
how much infrasound can help VERIFY (not just detect) real thunderquakes when
building the training set.

Usage:
    pixi run -e default python scripts/verify_seismoacoustic_candidates.py

Outputs:
    catalogs/ok_seismoacoustic_verification.csv   xcorr/lag per candidate event
    docs/figures/ok_seismoacoustic_verification.png  paired seismic+infrasound gallery
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.features import seismo_acoustic_lag, smooth_env  # noqa: E402

WINDOW_S = 90.0
XCORR_THRESHOLD = 0.5  # WS1's working definition of "strong" seismoacoustic coupling


def main() -> int:
    cat_dir = REPO_ROOT / "catalogs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(cat_dir / "ok_candidate_events.csv")
    infra = pd.read_csv(cat_dir / "stations_OK_infrasound.csv")
    infra_set = set(zip(infra["network"], infra["station"], strict=True))

    events["is_seismoacoustic"] = [
        (n, s) in infra_set for n, s in zip(events["network"], events["station"], strict=True)
    ]
    sub = events[events["is_seismoacoustic"]].copy()
    print(f"Candidate events at OK permanent seismoacoustic stations: "
          f"{len(sub)}/{len(events)}")
    print(sub.groupby(["network", "station"]).size().to_string())

    from obspy.clients.fdsn import Client

    client = Client("IRIS")
    rows, gallery = [], []
    for ev in sub.itertuples(index=False):
        acoustic_delay_s = ev.min_dist_km / 0.34
        t_center = (pd.Timestamp(ev.event_start) + pd.Timedelta(seconds=acoustic_delay_s)
                   - pd.Timedelta(seconds=5))
        seis, fs_s = cached_waveform(ev.network, ev.station, "HHZ", t_center, WINDOW_S,
                                     band=(1.0, 20.0), client=client)
        if seis is None:
            seis, fs_s = cached_waveform(ev.network, ev.station, "BHZ", t_center, WINDOW_S,
                                         band=(1.0, 20.0), client=client)
        infra_data, fs_i = cached_waveform(ev.network, ev.station, "BDF", t_center, WINDOW_S,
                                          band=(1.0, 20.0), client=client)
        if seis is None or infra_data is None:
            continue
        lag, coef = seismo_acoustic_lag(seis, fs_s, infra_data, fs_i)
        rows.append({"network": ev.network, "station": ev.station,
                    "event_start": ev.event_start, "min_dist_km": ev.min_dist_km,
                    "n_strikes": ev.n_strikes, "lag_s": lag, "xcorr": coef})
        gallery.append((ev, seis, fs_s, infra_data, fs_i, lag, coef))

    result = pd.DataFrame(rows)
    result.to_csv(cat_dir / "ok_seismoacoustic_verification.csv", index=False)

    print(f"\nFetched seismic+infrasound pairs: {len(result)}/{len(sub)}")
    if len(result):
        strong = result["xcorr"] >= XCORR_THRESHOLD
        print(f"Strong coupling (xcorr >= {XCORR_THRESHOLD}): {strong.sum()}/{len(result)} "
              f"({100 * strong.mean():.0f}%)")
        print(f"Median xcorr: {result['xcorr'].median():.2f}  "
              f"median |lag|: {result['lag_s'].abs().median():.1f}s")
        print("\nBy n_strikes (single-strike events are the cleanest test):")
        stats = result.groupby(result["n_strikes"] == 1)["xcorr"].describe()
        print(stats[["count", "mean", "50%"]])

    # Paired seismic+infrasound gallery, sorted by coupling strength (best first)
    gallery.sort(key=lambda g: g[6], reverse=True)
    ncol = 4
    n = min(len(gallery), 16)
    nrow = int(np.ceil(n / ncol))
    if n:
        fig, axes = plt.subplots(nrow * 2, ncol, figsize=(3.6 * ncol, 2.6 * nrow), squeeze=False)
        for k in range(n):
            ev, seis, fs_s, infra_data, fs_i, lag, coef = gallery[k]
            r, c = divmod(k, ncol)
            ts = np.arange(len(seis)) / fs_s
            ti = np.arange(len(infra_data)) / fs_i
            aw = axes[r * 2][c]
            aw.plot(ts, smooth_env(seis, fs_s), lw=0.8, color="steelblue", label="seismic env")
            aw.set_title(f"{ev.network}.{ev.station}  xcorr={coef:.2f} lag={lag:+.1f}s\n"
                        f"d={ev.min_dist_km*1000:.0f}m  {str(ev.event_start)[:16]}", fontsize=6)
            aw.set_xlabel("Time (s)", fontsize=6)
            aw.set_ylabel("Seismic envelope\namplitude (counts)", fontsize=6)
            aw.tick_params(labelsize=6)
            ai = axes[r * 2 + 1][c]
            ai.plot(ti, smooth_env(infra_data, fs_i), lw=0.8, color="darkorange",
                   label="infrasound env")
            ai.set_xlabel("Time (s)", fontsize=6)
            ai.set_ylabel("Infrasound envelope\namplitude (counts)", fontsize=6)
            ai.tick_params(labelsize=6)
        for k in range(n, nrow * ncol):
            r, c = divmod(k, ncol)
            axes[r * 2][c].axis("off")
            axes[r * 2 + 1][c].axis("off")
        fig.suptitle("OK GLM candidates: seismic vs infrasound envelope coupling\n"
                    "(sorted best-to-worst by cross-correlation)", fontweight="bold")
        fig.tight_layout()
        out = fig_dir / "ok_seismoacoustic_verification.png"
        fig.savefig(out, dpi=130)
        print(f"\ngallery -> {out.relative_to(REPO_ROOT)}")

    print("result  -> catalogs/ok_seismoacoustic_verification.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
