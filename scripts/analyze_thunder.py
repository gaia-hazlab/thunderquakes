#!/usr/bin/env python
"""Thunderquake duration + seismo-acoustic delay (WS1, issue #3).

Cache-backed (see thunderquakes.data.cache): first run fetches from FDSN, later
runs are instant/offline. Answers two questions:

1. How long is a thunderquake signal actually? -> duration distribution above the
   pre-onset noise floor -> informs the CNN window length / stride.
2. Do seismic and co-located infrasound record the same passing acoustic front?
   -> envelope cross-correlation lag + coefficient at CC.CPCO/KWBU/SVIC. High
   correlation at small lag is the seismo-acoustic signature (feeds WS4 Model B).

Usage:
    pixi run -e default python scripts/analyze_thunder.py \
        --metadata-dir /path/to/pnwml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.features import seismo_acoustic_lag, smooth_env  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

WINDOW_S = 150.0
ONSET_S = 70.0  # PNWML exotic traces have the labelled onset at sample 7000 (70 s @ 100 Hz)
BAND = (1.0, 20.0)  # common band for seismic + infrasound comparison
INFRA_STATIONS = {  # co-located seismoacoustic thunder stations (from WS2 PNW inventory)
    ("CC", "CPCO"), ("CC", "KWBU"), ("CC", "SVIC"),
}


def episode_span(data, fs, k=3.0):
    """Thunderquake episode duration (s): first-to-last time above k×noise.

    NB: the PNWML ``trace_P_arrival_sample`` (=7000) is a placeholder for thunder,
    not the true onset, so we do NOT key off it. Noise is the 20th-percentile of
    the smoothed envelope (robust to the signal itself). Also returns the total
    active time. A thunderquake is a *sequence* of claps, so the span (not a
    single contiguous burst) is the physically meaningful signal duration.
    """
    env = smooth_env(data, fs)
    noise = np.percentile(env, 20)
    if not noise or noise <= 0:
        return np.nan, np.nan
    active = env > k * noise
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return 0.0, 0.0
    span = (idx[-1] - idx[0]) / fs
    total = idx.size / fs
    return span, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--client", default="IRIS")
    args = ap.parse_args()

    from obspy.clients.fdsn import Client

    client = Client(args.client)  # build once; passing the object avoids per-call discovery

    ex = pd.read_csv(Path(args.metadata_dir) / "exotic_metadata.csv", low_memory=False)
    th = ex[ex["source_type"] == "thunder"].copy()
    fig_dir = REPO_ROOT / "docs" / "figures"
    cat_dir = REPO_ROOT / "catalogs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Duration ────────────────────────────────────────────────────────
    spans, actives, examples = [], [], []
    for r in th.itertuples(index=False):
        chan = str(r.station_channel_code) + "Z"
        data, fs = cached_waveform(r.station_network_code, r.station_code, chan,
                                   pd.Timestamp(r.trace_start_time), WINDOW_S,
                                   band=(1.0, 45.0), client=client)
        if data is None:
            continue
        span, total = episode_span(data, fs)
        if np.isfinite(span):
            spans.append(span)
            actives.append(total)
            if len(examples) < 3 and span > 20:
                examples.append((f"{r.station_network_code}.{r.station_code}", data, fs, span))
    spans = np.array(spans)
    actives = np.array(actives)
    pct = np.percentile(spans, [50, 90, 95])
    print(f"Thunder episode span (s): median={pct[0]:.0f}  p90={pct[1]:.0f}  p95={pct[2]:.0f}  "
          f"(n={len(spans)})  | median active time={np.median(actives):.0f}s")

    fig, axes = plt.subplots(1, 2, figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.42))
    axes[0].hist(spans, bins=30, color="steelblue", edgecolor="white")
    for p, lab in zip(pct, ["median", "p90", "p95"], strict=True):
        axes[0].axvline(p, ls="--", lw=1, color="crimson")
        axes[0].text(p, axes[0].get_ylim()[1] * 0.9, f"{lab}={p:.0f}s", fontsize=10, rotation=90)
    axes[0].axvspan(0, 50, alpha=0.08, color="green")
    axes[0].set(xlabel="Episode span above 3x noise floor (s)", ylabel="Number of traces")
    for name, data, fs, span in examples:
        t = np.arange(len(data)) / fs
        e = smooth_env(data, fs)
        axes[1].plot(t, e / e.max(), lw=0.8, label=f"{name} (span {span:.0f}s)")
    axes[1].set(xlabel="Time in 150 s window (s)",
                ylabel="Normalized envelope amplitude (peak = 1, dimensionless)",
                xlim=(0, 150))
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "pnwml_thunder_duration.png")
    print(f"  fig -> {(fig_dir / 'pnwml_thunder_duration.png').relative_to(REPO_ROOT)}")

    # ── 2. Seismo-acoustic delay ───────────────────────────────────────────
    rows = []
    for r in th.itertuples(index=False):
        key = (r.station_network_code, r.station_code)
        if key not in INFRA_STATIONS:
            continue
        chan_z = str(r.station_channel_code) + "Z"
        t0 = pd.Timestamp(r.trace_start_time)
        seis, fs_s = cached_waveform(*key, chan_z, t0, WINDOW_S, band=BAND, client=client)
        infra, fs_i = cached_waveform(*key, "BDF", t0, WINDOW_S, band=BAND, client=client)
        if seis is None or infra is None:
            continue
        lag, coef = seismo_acoustic_lag(seis, fs_s, infra, fs_i)
        rows.append({"network": key[0], "station": key[1], "time": t0,
                     "lag_s": lag, "xcorr": coef})
    sa = pd.DataFrame(rows)
    sa.to_csv(cat_dir / "thunder_seismoacoustic.csv", index=False)
    med_lag = sa["lag_s"].abs().median()
    med_xc = sa["xcorr"].median()
    print(f"\nSeismo-acoustic pairs: {len(sa)}  "
          f"(median |lag|={med_lag:.1f}s, median xcorr={med_xc:.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.42))
    hi = sa[sa["xcorr"] >= 0.5]
    axes[0].scatter(sa["lag_s"], sa["xcorr"], s=25, c="0.6", label=f"all (n={len(sa)})")
    axes[0].scatter(hi["lag_s"], hi["xcorr"], s=30, c="crimson",
                    label=f"xcorr≥0.5 (n={len(hi)})")
    axes[0].axhline(0.5, ls=":", c="k", lw=0.8)
    axes[0].set(xlabel="Envelope lag: infrasound minus seismic (s)",
                ylabel="Peak cross-correlation coefficient (dimensionless, -1 to 1)")
    axes[0].legend(fontsize=10)
    axes[1].hist(sa["lag_s"], bins=25, color="teal", edgecolor="white")
    axes[1].axvline(0, color="crimson", ls="--", lw=1)
    axes[1].set(xlabel="Envelope lag: infrasound minus seismic (s)", ylabel="Number of events")
    fig.tight_layout()
    fig.savefig(fig_dir / "pnwml_thunder_seismoacoustic.png")
    print(f"  fig -> {(fig_dir / 'pnwml_thunder_seismoacoustic.png').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
