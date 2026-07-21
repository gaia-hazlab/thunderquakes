#!/usr/bin/env python
"""OK thunderquake detection via synchronous lightning strikes (WS4 #11, first pass).

The PNWML labelled thunder class is entirely PNW (no OK examples), so this is the
real generalization test: run the PNW-trained Model A over CONTINUOUS Oklahoma
seismic data during a known thunderstorm, and check whether flagged windows
line up with independently-observed GOES-GLM lightning strikes. This closes the
loop the roadmap calls for — baseline first, then iterate the training set on
whatever this reveals (false positives -> hard negatives; misses -> augmentation).

Usage (run under the ml env, needs a trained OK model from scripts/train_cnn.py):
    pixi run -e ml python scripts/detect_ok_thunderquakes.py \
        --network OK --station CROK --start 2019-05-20T22:00:00 --end 2019-05-21T01:00:00

Outputs:
    catalogs/ok_detections_<STA>_<START>.csv   sliding-window thunder probability
    docs/figures/ok_detection_<STA>_<START>.png   probability trace vs GLM strikes
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

from thunderquakes.config import REGIONS, REPO_ROOT  # noqa: E402
from thunderquakes.data import cached_waveform  # noqa: E402
from thunderquakes.evaluation import match_detections_to_strikes  # noqa: E402
from thunderquakes.lightning import load_glm_strikes  # noqa: E402
from thunderquakes.models.cnn import build_seismic_cnn  # noqa: E402
from thunderquakes.models.dataset import WindowConfig, log_spectrogram_image  # noqa: E402

STRIDE_S = 25.0  # WS1-recommended: 50 s window, 50% overlap


def sliding_windows(data, n_samples, stride_samples):
    for start in range(0, len(data) - n_samples + 1, stride_samples):
        yield start, data[start : start + n_samples]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", default="OK")
    ap.add_argument("--station", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--model-region", default="OK",
                    help="which trained model to load (outputs/model_a_seismic_<region>.pt)")
    ap.add_argument("--match-radius-km", type=float, default=15.0,
                    help="thunder is rarely audible/coupled beyond ~15-20 km of the strike")
    ap.add_argument("--match-window-s", type=float, default=20.0,
                    help="near-simultaneous: sound/seismic lag from a 15 km strike is under ~1 min")
    ap.add_argument("--client", default="IRIS")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "outputs"
    cat_dir = REPO_ROOT / "catalogs"
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

    t0 = pd.Timestamp(args.start, tz="UTC")
    t1 = pd.Timestamp(args.end, tz="UTC")
    dur = (t1 - t0).total_seconds()
    client = Client(args.client)

    # station lat/lon for GLM matching (from the WS2 inventory)
    inv = pd.read_csv(cat_dir / f"stations_{args.model_region}_summary.csv")
    row = inv[(inv.network == args.network) & (inv.station == args.station)].iloc[0]
    lat, lon = row.latitude, row.longitude

    print(f"Fetching continuous data: {args.network}.{args.station}  {t0} -> {t1} "
          f"({dur/3600:.1f} h) …")
    data, fs = cached_waveform(args.network, args.station, "HHZ", t0, dur,
                               band=tuple(cfg.band), client=client)
    if data is None:
        data, fs = cached_waveform(args.network, args.station, "BHZ", t0, dur,
                                   band=tuple(cfg.band), client=client)
    if data is None:
        raise SystemExit(f"No continuous data for {args.network}.{args.station} in this window")
    print(f"  got {len(data)} samples @ {fs} Hz")

    n_samples = cfg.n_samples
    stride_samples = int(STRIDE_S * cfg.fs)
    times, probs = [], []
    imgs = []
    starts = []
    for start, win in sliding_windows(data, n_samples, stride_samples):
        imgs.append(log_spectrogram_image(win.astype(np.float32), cfg)[None])
        starts.append(start)
    X = torch.from_numpy(np.stack(imgs).astype("float32"))
    with torch.no_grad():
        p = torch.softmax(model(X), dim=1).numpy()
    ti = classes.index("thunder")
    for start, prob in zip(starts, p[:, ti], strict=True):
        times.append(t0 + pd.Timedelta(seconds=(start + n_samples / 2) / fs))
        probs.append(float(prob))

    det = pd.DataFrame({"time_utc": times, "thunder_prob": probs})
    det["latitude"] = lat
    det["longitude"] = lon
    tag = f"{args.station}_{t0:%Y%m%dT%H%M}"
    det.to_csv(cat_dir / f"ok_detections_{tag}.csv", index=False)

    print(f"\nFetching GLM strikes for the same window (OK bbox, radius={args.match_radius_km:.0f} "
          f"km, window=±{args.match_window_s:.0f}s around each candidate) …")
    strikes = load_glm_strikes(t0, t1, bbox=REGIONS["OK"]["bbox"], progress=True)
    print(f"  {len(strikes)} GLM flashes statewide in window "
          f"(rate: 1 every {(t1-t0).total_seconds()/max(len(strikes),1):.2f}s statewide — "
          "match rate is only meaningful next to the null baseline below)")

    cand = det[det["thunder_prob"] >= 0.5].copy()
    n_null = min(len(cand), len(det)) if len(cand) else 0
    rng = np.random.default_rng(0)
    null_idx = rng.choice(len(det), size=n_null, replace=False) if n_null else np.array([], int)
    null = det.iloc[null_idx].copy()

    if len(strikes) and (len(cand) or len(null)):
        m_cand = match_detections_to_strikes(cand, strikes, radius_km=args.match_radius_km,
                                             window_s=args.match_window_s) if len(cand) else cand
        m_null = match_detections_to_strikes(null, strikes, radius_km=args.match_radius_km,
                                             window_s=args.match_window_s) if len(null) else null
        cand_rate = m_cand["matched"].mean() if len(m_cand) else float("nan")
        null_rate = m_null["matched"].mean() if len(m_null) else float("nan")
        print(f"\nCandidate windows (prob>=0.5): {len(cand)}  "
              f"matched within {args.match_radius_km:.0f} km / {args.match_window_s:.0f}s: "
              f"{cand_rate:.0%}")
        print(f"Null baseline (same-size random sample of ALL windows): matched rate "
              f"{null_rate:.0%}")
        print("Candidate rate must exceed the null rate to show the model adds information "
              "beyond 'a storm was active nearby' — GLM is also a lower bound on true "
              "lightning, so misses there are not necessarily false positives either.")
    else:
        print(f"\nCandidate windows (prob>=0.5): {len(cand)}; strikes in window: {len(strikes)}")

    # GLM strike-rate histogram (density, not overlapping lines: 20k+ strikes saturate the plot)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(det["time_utc"], det["thunder_prob"], lw=1, color="steelblue", label="P(thunder)")
    if len(cand):
        ax.scatter(cand["time_utc"], cand["thunder_prob"], s=15, color="crimson", zorder=3,
                  label=f"candidates (n={len(cand)})")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set(ylabel="P(thunder)", ylim=(0, 1),
           title=f"{args.network}.{args.station} — Model A thunder probability vs GLM strike rate\n"
                 f"({t0:%Y-%m-%d %H:%M} → {t1:%H:%M} UTC)  [match radius "
                 f"{args.match_radius_km:.0f} km, window ±{args.match_window_s:.0f}s]")
    ax.legend(fontsize=8, loc="upper right")
    if len(strikes):
        counts, edges = np.histogram(
            strikes["time_utc"].astype("int64"),
            bins=int((t1 - t0).total_seconds() // 30),
        )
        centers = pd.to_datetime(edges[:-1], unit="ns", utc=True)
        ax2.bar(centers, counts, width=pd.Timedelta(seconds=28), color="orange", align="edge")
    ax2.set(xlabel="Time (UTC)", ylabel="GLM flashes\n/ 30s (statewide)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(fig_dir / f"ok_detection_{tag}.png", dpi=150)

    print(f"\ndetections -> catalogs/ok_detections_{tag}.csv")
    print(f"figure     -> docs/figures/ok_detection_{tag}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
