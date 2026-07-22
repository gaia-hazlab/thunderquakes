#!/usr/bin/env python
"""Waveform + spectrogram verification gallery per class (WS4 #8).

Lets a human eyeball-verify the labelled training/test data: for each class, a grid
of examples showing the 50 s model window (waveform) above its log-spectrogram, with
station/time/SNR labels. Cache-backed (offline after first fetch).

Usage:
    pixi run -e default python scripts/plot_waveforms.py \
        --metadata-dir /path/to/pnwml --region PNW --per-class 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import spectrogram as _spec  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.models.dataset import (  # noqa: E402
    REGION_CLASSES,
    WindowConfig,
    build_windows,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--region", default="PNW", choices=list(REGION_CLASSES))
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--client", default="IRIS")
    args = ap.parse_args()

    from obspy.clients.fdsn import Client

    client = Client(args.client)
    mdir = Path(args.metadata_dir)
    meta = {
        "exotic": pd.read_csv(mdir / "exotic_metadata.csv", low_memory=False),
        "noise": pd.read_csv(mdir / "noise_metadata.csv", low_memory=False),
    }
    classes = REGION_CLASSES[args.region]
    cfg = WindowConfig()
    ds = build_windows(meta, classes, cfg=cfg, n_per_class=args.per_class, client=client, seed=7)
    print(f"{args.region}: {len(ds.W)} windows across {classes}")

    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(cfg.n_samples) / cfg.fs

    for cls in classes:
        idx = np.flatnonzero(ds.y == classes.index(cls))[: args.per_class]
        ncol = min(4, len(idx))
        if ncol == 0:
            continue
        nrow = int(np.ceil(len(idx) / ncol))
        fig, axes = plt.subplots(nrow * 2, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                                 squeeze=False)
        for k, i in enumerate(idx):
            r, c = divmod(k, ncol)
            w = ds.W[i]
            aw = axes[r * 2][c]
            aw.plot(t, w, lw=0.4, color="steelblue")
            m = ds.meta.iloc[i]
            aw.set_title(f"{m.network}.{m.station}\n{m.time[:16]}", fontsize=7)
            aw.set_xlabel("Time (s)", fontsize=6)
            aw.set_ylabel("Amplitude (counts)", fontsize=6)
            aw.tick_params(labelsize=6)
            asp = axes[r * 2 + 1][c]
            f, tt, sxx = _spec(w, fs=cfg.fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
            band = (f >= cfg.band[0]) & (f <= cfg.band[1])
            asp.pcolormesh(tt, f[band], 10 * np.log10(sxx[band] + 1e-12),
                           shading="gouraud", cmap="inferno")
            asp.set_ylim(cfg.band)
            asp.set_xlabel("Time (s)", fontsize=6)
            asp.set_ylabel("Frequency (Hz)", fontsize=6)
            asp.tick_params(labelsize=6)
        # blank any unused axes
        for k in range(len(idx), nrow * ncol):
            r, c = divmod(k, ncol)
            axes[r * 2][c].axis("off")
            axes[r * 2 + 1][c].axis("off")
        fig.suptitle(f"{args.region} — class: {cls}  (50 s window @ {cfg.fs:g} Hz + spectrogram)",
                     fontweight="bold")
        fig.tight_layout()
        safe = cls.replace(" ", "_")
        out = fig_dir / f"verify_{args.region}_{safe}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  {cls:15s} -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
