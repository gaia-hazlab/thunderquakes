#!/usr/bin/env python
"""Per-class PNWML signal characterization (WS1, issue #3).

Reconstructs labelled PNWML waveforms from FDSN (using the class metadata) and
computes per-class signal features to reveal what discriminates thunderquakes
from other sources. No multi-GB HDF5 needed — fully reproducible from metadata.

Usage:
    pixi run -e default python scripts/characterize_pnwml.py \
        --metadata-dir /path/to/pnwml --n-per-class 120

Outputs:
    catalogs/pnwml_class_features.csv   per-trace features (committed)
    catalogs/pnwml_class_summary.csv    per-class medians (committed)
    docs/figures/pnwml_spectra.png      median spectra by class
    docs/figures/pnwml_distributions.png feature distributions by class
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.features.characterize import power_spectrum, trace_features  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

# Classes to compare and where their metadata lives. comcat (earthquake/explosion)
# is added once that metadata is available; here we use the locally-complete
# exotic + noise stores. Sonic boom is a valuable acoustic-coupled contrast.
CLASS_STORE = {
    "thunder": "exotic",
    "sonic boom": "exotic",
    "surface event": "exotic",
    "noise": "noise",
}
WINDOW_S = 150.0
BAND = (1.0, 45.0)
LOGF = np.logspace(np.log10(1.0), np.log10(45.0), 120)  # common freq grid for median spectra


def _z_snr(snr_field) -> float:
    """Vertical-component SNR from a 'E|N|Z' pipe string (thunder etc.); NaN if absent."""
    try:
        return float(str(snr_field).split("|")[-1])
    except (ValueError, IndexError):
        return np.nan


def sample_class(meta: pd.DataFrame, source_type: str, n: int, seed: int = 0) -> pd.DataFrame:
    rows = meta[meta["source_type"] == source_type].copy()
    if "trace_snr_db" in rows.columns:
        rows["z_snr"] = rows["trace_snr_db"].map(_z_snr)
    if len(rows) > n:
        rows = rows.sample(n, random_state=seed)
    return rows


def process_row(row, client) -> dict | None:
    chan = str(row.station_channel_code) + "Z"
    t0 = pd.Timestamp(row.trace_start_time)
    data, fs = cached_waveform(
        row.station_network_code, row.station_code, chan, t0, WINDOW_S,
        band=BAND, client=client,
    )
    if data is None:
        return None
    data = data.astype(float)
    feats = trace_features(data, fs)
    feats.update(source_type=row.source_type, network=row.station_network_code,
                 station=row.station_code, channel=chan)
    # spectrum on the common log-freq grid (for median per-class spectra)
    f, p = power_spectrum(data, fs)
    p = p / (p.sum() + 1e-20)
    feats["_spec"] = np.interp(LOGF, f, p, left=np.nan, right=np.nan)
    return feats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", required=True,
                    help="dir containing exotic_metadata.csv and noise_metadata.csv")
    ap.add_argument("--n-per-class", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--client", default="IRIS")
    args = ap.parse_args()

    from obspy.clients.fdsn import Client

    client = Client(args.client)
    mdir = Path(args.metadata_dir)
    stores = {
        "exotic": pd.read_csv(mdir / "exotic_metadata.csv", low_memory=False),
        "noise": pd.read_csv(mdir / "noise_metadata.csv", low_memory=False),
    }

    rows = []
    for cls, store in CLASS_STORE.items():
        s = sample_class(stores[store], cls, args.n_per_class)
        print(f"{cls:15s}: sampling {len(s)} traces")
        rows.append(s)
    sample = pd.concat(rows, ignore_index=True)

    print(f"\nReconstructing {len(sample)} waveforms from FDSN ({args.workers} workers) …")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda r: process_row(r, client), sample.itertuples(index=False)))
    feats = [r for r in results if r is not None]
    print(f"  succeeded: {len(feats)}/{len(sample)}")

    df = pd.DataFrame(feats)
    specs = np.vstack(df.pop("_spec").values)
    feat_cols = [c for c in df.columns if c not in ("source_type", "network", "station", "channel")]

    cat_dir = REPO_ROOT / "catalogs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    cat_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(cat_dir / "pnwml_class_features.csv", index=False)
    summary = df.groupby("source_type")[feat_cols].median().round(3)
    summary["n"] = df.groupby("source_type").size()
    summary.to_csv(cat_dir / "pnwml_class_summary.csv")
    print("\n=== per-class median features ===")
    print(summary.to_string())

    classes = [c for c in CLASS_STORE if c in set(df["source_type"])]
    colors = dict(zip(classes, plt.cm.tab10(np.linspace(0, 1, len(classes))), strict=True))

    # --- Figure 1: median normalised spectra by class ---
    fig, ax = plt.subplots(figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.75))
    for cls in classes:
        idx = (df["source_type"] == cls).values
        med = np.nanmedian(specs[idx], axis=0)
        ax.loglog(LOGF, med, label=f"{cls} (n={idx.sum()})", color=colors[cls], lw=2)
    ax.set(xlabel="Frequency (Hz)", ylabel="Normalized power (median, dimensionless)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "pnwml_spectra.png")
    print(f"  fig -> {(fig_dir / 'pnwml_spectra.png').relative_to(REPO_ROOT)}")

    # --- Figure 2: feature distributions by class ---
    panels = [
        ("spectral_flatness", "Spectral flatness (dimensionless, 0-1)"),
        ("duration_80pct_s", "Duration containing 80% of energy (s)"),
        ("spectral_centroid_hz", "Spectral centroid (Hz)"),
        ("kurtosis", "Kurtosis (dimensionless)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(LETTER_WIDTH_IN, LETTER_WIDTH_IN * 0.85))
    for ax, (feat, ylabel) in zip(axes.ravel(), panels, strict=True):
        data = [df.loc[df["source_type"] == cls, feat].dropna().values for cls in classes]
        bp = ax.boxplot(data, tick_labels=classes, showfliers=False, patch_artist=True)
        for patch, cls in zip(bp["boxes"], classes, strict=True):
            patch.set_facecolor(colors[cls])
            patch.set_alpha(0.6)
        ax.set(xlabel="Class", ylabel=ylabel)
        ax.tick_params(axis="x", rotation=20)
        if feat == "spectral_centroid_hz":
            ax.axhspan(1, 20, alpha=0.05, color="k")
    fig.tight_layout()
    fig.savefig(fig_dir / "pnwml_distributions.png")
    print(f"  fig -> {(fig_dir / 'pnwml_distributions.png').relative_to(REPO_ROOT)}")

    print(f"\nfeatures -> {(cat_dir / 'pnwml_class_features.csv').relative_to(REPO_ROOT)}")
    print(f"summary  -> {(cat_dir / 'pnwml_class_summary.csv').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
