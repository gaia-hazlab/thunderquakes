#!/usr/bin/env python
"""Batch OK thunderquake detection across stations, with a statistical test (WS4 #11).

Scales up the single-station pilot (scripts/detect_ok_thunderquakes.py): runs the
OK-targeted Model A over CONTINUOUS data at several geographically spread Oklahoma
stations during the same known-convective window, matches candidate detections
against GOES-GLM strikes with a physically-motivated radius/window, and pools
results across stations for a two-proportion z-test against the null baseline
(same test on random windows). GLM is fetched ONCE per time window and reused
across all stations in that window (it does not depend on station).

Usage (run under the ml env):
    pixi run -e ml python scripts/detect_ok_thunderquakes_batch.py \
        --start 2019-05-20T20:00:00 --end 2019-05-21T02:00:00

Outputs:
    catalogs/ok_batch_detections.csv   all sliding-window probabilities, all stations
    catalogs/ok_batch_summary.csv      per-station + pooled match-rate summary
    docs/figures/ok_batch_summary.png  candidate vs null match rate, per station + pooled
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import norm  # noqa: E402

from thunderquakes.config import REGIONS, REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.evaluation import match_detections_to_strikes  # noqa: E402
from thunderquakes.lightning import load_glm_strikes  # noqa: E402
from thunderquakes.models.cnn import build_seismic_cnn  # noqa: E402
from thunderquakes.models.dataset import WindowConfig, log_spectrogram_image  # noqa: E402

STRIDE_S = 25.0
DEFAULT_STATIONS = [  # (network, station) — spread across the state's longitude range
    ("OK", "CROK"),  # north-central
    ("OK", "FNO"),   # Norman, S-central (heart of Tornado Alley)
    ("OK", "NOKA"),  # northwest
    ("N4", "T35B"),  # northeast, seismoacoustic site (BDF co-located)
    ("OK", "RLOK"),  # east
]


def two_proportion_ztest(k1, n1, k2, n2):
    """Two-proportion z-test; returns (z, p_one_sided) for p1 > p2."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return float(z), float(1 - norm.cdf(z))


def sliding_windows(data, n_samples, stride_samples):
    for start in range(0, len(data) - n_samples + 1, stride_samples):
        yield start, data[start : start + n_samples]


def score_station(net, sta, t0, t1, model, classes, cfg, client):
    dur = (t1 - t0).total_seconds()
    data, fs = cached_waveform(net, sta, "HHZ", t0, dur, band=tuple(cfg.band), client=client)
    if data is None:
        data, fs = cached_waveform(net, sta, "BHZ", t0, dur, band=tuple(cfg.band), client=client)
    if data is None:
        return None
    n_samples = cfg.n_samples
    stride_samples = int(STRIDE_S * cfg.fs)
    imgs, starts = [], []
    for start, win in sliding_windows(data, n_samples, stride_samples):
        imgs.append(log_spectrogram_image(win.astype(np.float32), cfg)[None])
        starts.append(start)
    if not imgs:
        return None
    X = torch.from_numpy(np.stack(imgs).astype("float32"))
    with torch.no_grad():
        p = torch.softmax(model(X), dim=1).numpy()
    ti = classes.index("thunder")
    times = [t0 + pd.Timedelta(seconds=(s + n_samples / 2) / fs) for s in starts]
    return pd.DataFrame({"network": net, "station": sta, "time_utc": times,
                         "thunder_prob": p[:, ti]})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model-region", default="OK")
    ap.add_argument("--match-radius-km", type=float, default=15.0)
    ap.add_argument("--match-window-s", type=float, default=20.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--client", default="IRIS")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir, cat_dir = REPO_ROOT / "outputs", REPO_ROOT / "catalogs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    cat_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    arch = json.loads((out_dir / f"model_a_seismic_{args.model_region}_arch.json").read_text())
    cfg = WindowConfig(**arch["cfg"])
    classes = arch["classes"]
    model = build_seismic_cnn(len(classes), width=arch["width"], depth=arch["depth"],
                              in_ch=arch["in_ch"])
    model.load_state_dict(torch.load(out_dir / f"model_a_seismic_{args.model_region}.pt",
                                     map_location="cpu"))
    model.eval()
    print(f"Loaded Model A ({args.model_region}): classes={classes}")

    from obspy.clients.fdsn import Client

    client = Client(args.client)
    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC")

    inv = pd.read_csv(cat_dir / f"stations_{args.model_region}_summary.csv")

    print(f"\nFetching GLM strikes ONCE for {t0} -> {t1} (reused across all stations) …")
    strikes = load_glm_strikes(t0, t1, bbox=REGIONS[args.model_region]["bbox"], progress=True)
    print(f"  {len(strikes)} GLM flashes statewide "
          f"(1 every {(t1-t0).total_seconds()/max(len(strikes),1):.2f}s statewide)")

    rng = np.random.default_rng(args.seed)
    all_det, rows = [], []
    for net, sta in DEFAULT_STATIONS:
        print(f"\n{net}.{sta}: scoring continuous data …")
        det = score_station(net, sta, t0, t1, model, classes, cfg, client)
        if det is None:
            print("  no data, skipping")
            continue
        row = inv[(inv.network == net) & (inv.station == sta)].iloc[0]
        det["latitude"], det["longitude"] = row.latitude, row.longitude
        all_det.append(det)

        cand = det[det["thunder_prob"] >= args.threshold].copy()
        n_null = min(len(cand), len(det))
        if n_null:
            null = det.iloc[rng.choice(len(det), size=n_null, replace=False)].copy()
        else:
            null = det.iloc[:0]

        k_cand = k_null = 0
        if len(strikes) and n_null:
            m_cand = match_detections_to_strikes(cand, strikes, radius_km=args.match_radius_km,
                                                 window_s=args.match_window_s)
            m_null = match_detections_to_strikes(null, strikes, radius_km=args.match_radius_km,
                                                 window_s=args.match_window_s)
            k_cand, k_null = int(m_cand["matched"].sum()), int(m_null["matched"].sum())
        print(f"  windows={len(det)}  candidates={len(cand)} (matched {k_cand})  "
              f"null n={n_null} (matched {k_null})")
        rows.append({"network": net, "station": sta, "n_windows": len(det),
                    "n_candidates": len(cand), "k_candidate_matched": k_cand,
                    "n_null": n_null, "k_null_matched": k_null})

    summary = pd.DataFrame(rows)
    K1, N1 = int(summary["k_candidate_matched"].sum()), int(summary["n_candidates"].sum())
    K2, N2 = int(summary["k_null_matched"].sum()), int(summary["n_null"].sum())
    z, p = two_proportion_ztest(K1, N1, K2, N2)
    cand_rate = K1 / N1 if N1 else float("nan")
    null_rate = K2 / N2 if N2 else float("nan")

    print(f"\n=== POOLED across {len(summary)} stations ===")
    print(f"Candidates: {K1}/{N1} matched ({cand_rate:.0%})")
    print(f"Null      : {K2}/{N2} matched ({null_rate:.0%})")
    print(f"Two-proportion z-test (candidate rate > null rate): z={z:.2f}, p={p:.4f}")

    pd.concat(all_det, ignore_index=True).to_csv(cat_dir / "ok_batch_detections.csv", index=False)
    summary.to_csv(cat_dir / "ok_batch_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(summary) + 1)
    n_cand_safe = summary["n_candidates"].replace(0, np.nan)
    n_null_safe = summary["n_null"].replace(0, np.nan)
    cand_rates = (summary["k_candidate_matched"] / n_cand_safe).tolist() + [cand_rate]
    null_rates = (summary["k_null_matched"] / n_null_safe).tolist() + [null_rate]
    sta_labels = [f"{n}.{s}" for n, s in zip(summary["network"], summary["station"], strict=True)]
    labels = sta_labels + ["POOLED"]
    w = 0.35
    ax.bar(x - w / 2, cand_rates, w, label="candidate (model-flagged)", color="crimson")
    ax.bar(x + w / 2, null_rates, w, label="null (random windows)", color="0.6")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set(ylabel="Match rate vs GLM strike", ylim=(0, 1),
           title=f"OK synchronous-lightning detection — candidate vs null match rate\n"
                 f"radius={args.match_radius_km:.0f} km, window=±{args.match_window_s:.0f}s  "
                 f"(pooled z={z:.2f}, p={p:.3f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "ok_batch_summary.png", dpi=150)

    print("\ndetections -> catalogs/ok_batch_detections.csv")
    print("summary    -> catalogs/ok_batch_summary.csv")
    print("figure     -> docs/figures/ok_batch_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
