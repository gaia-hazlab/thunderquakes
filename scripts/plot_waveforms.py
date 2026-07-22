#!/usr/bin/env python
"""Waveform + spectrogram verification gallery per class (WS4 #8).

Lets a human eyeball-verify the labelled training/test data: for each class, a grid
of examples showing the 50 s model window (waveform) above its log-spectrogram, with
station/time labels. Cache-backed (offline after first fetch).

Panel count is capped at GALLERY_NCOL columns so the figure fits US Letter width
with every label readable at >=10pt when printed. All panels share the time axis
(the model window is always 50 s) and all spectrograms share the frequency axis
(same band), so tick labels/axis labels are only drawn once per column/row to cut
whitespace -- see the report caption for the descriptive framing normally carried
by a figure title.

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
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

GALLERY_NCOL = 2   # 2 columns keeps each panel ~3.1in wide -> readable at >=10pt on Letter
GALLERY_MAX = 4    # examples actually drawn per class (2x2 grid of waveform+spectrogram pairs)


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
        idx = np.flatnonzero(ds.y == classes.index(cls))[:GALLERY_MAX]
        ncol = min(GALLERY_NCOL, len(idx))
        if ncol == 0:
            continue
        nrow = int(np.ceil(len(idx) / ncol))
        col_w = LETTER_WIDTH_IN / GALLERY_NCOL
        fig, axes = plt.subplots(nrow * 2, ncol, figsize=(col_w * ncol, 2.2 * nrow),
                                 sharex="col", squeeze=False)
        spec_axes = []
        for k, i in enumerate(idx):
            r, c = divmod(k, ncol)
            is_last_row = r == nrow - 1
            w = ds.W[i]
            aw = axes[r * 2][c]
            aw.plot(t, w, lw=0.5, color="steelblue")
            m = ds.meta.iloc[i]
            aw.set_title(f"{m.network}.{m.station}  {m.time[:16]}", fontsize=10)
            aw.tick_params(labelsize=10)
            if c == 0:
                aw.set_ylabel("Amp\n(counts)", fontsize=10)
            asp = axes[r * 2 + 1][c]
            f, tt, sxx = _spec(w, fs=cfg.fs, nperseg=cfg.nperseg, noverlap=cfg.noverlap)
            band = (f >= cfg.band[0]) & (f <= cfg.band[1])
            asp.pcolormesh(tt, f[band], 10 * np.log10(sxx[band] + 1e-12),
                           shading="gouraud", cmap="inferno")
            asp.set_ylim(cfg.band)
            asp.tick_params(labelsize=10)
            if c == 0:
                asp.set_ylabel("Freq\n(Hz)", fontsize=10)
            else:
                asp.tick_params(labelleft=False)
            if is_last_row:
                asp.set_xlabel("Time (s)", fontsize=10)
            spec_axes.append(asp)
        # Spectrograms all share the same band -> link their frequency axes.
        for ax in spec_axes[1:]:
            ax.sharey(spec_axes[0])
        # blank any unused axes
        for k in range(len(idx), nrow * ncol):
            r, c = divmod(k, ncol)
            axes[r * 2][c].axis("off")
            axes[r * 2 + 1][c].axis("off")
        fig.tight_layout(h_pad=0.3)
        safe = cls.replace(" ", "_")
        out = fig_dir / f"verify_{args.region}_{safe}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  {cls:15s} -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
