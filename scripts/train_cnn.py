#!/usr/bin/env python
"""Train the seismic-only thunderquake CNN (Model A, WS4 #8).

Cache-backed dataset build (WS1 50 s window @ 100 Hz) -> station-disjoint split ->
class-weighted 2D-CNN -> metrics + confusion matrix. Run under the ml env:

    pixi run -e ml python scripts/train_cnn.py --metadata-dir /path/to/pnwml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.models.dataset import WindowConfig, build_dataset  # noqa: E402
from thunderquakes.models.train import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--client", default="IRIS")
    args = ap.parse_args()

    from obspy.clients.fdsn import Client

    client = Client(args.client)
    mdir = Path(args.metadata_dir)
    meta = {
        "exotic": pd.read_csv(mdir / "exotic_metadata.csv", low_memory=False),
        "noise": pd.read_csv(mdir / "noise_metadata.csv", low_memory=False),
    }

    print("Building dataset from cache …")
    ds = build_dataset(meta, cfg=WindowConfig(), n_per_class=args.n_per_class, client=client)
    print(f"  X={ds.X.shape}  classes={ds.classes}  "
          f"counts={np.bincount(ds.y).tolist()}")

    print("Training Model A (seismic-only, station-disjoint split) …")
    out = train(ds, epochs=args.epochs)
    m = out["metrics"]

    print(f"\nn_train={m['n_train']}  n_test={m['n_test']}")
    if "thunder_ap" in m:
        print(f"thunder one-vs-rest AP (PR-AUC): {m['thunder_ap']:.3f}")
    print("per-class (test):")
    for c in ds.classes:
        r = m["report"][c]
        print(f"  {c:15s} P={r['precision']:.2f} R={r['recall']:.2f} "
              f"F1={r['f1-score']:.2f} n={int(r['support'])}")

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    fig_dir = REPO_ROOT / "docs" / "figures"
    (out_dir / "cnn_metrics.json").write_text(json.dumps(m, indent=2))

    cm = np.array(m["confusion"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ds.classes)), ds.classes, rotation=30, ha="right")
    ax.set_yticks(range(len(ds.classes)), ds.classes)
    for i in range(len(ds.classes)):
        for j in range(len(ds.classes)):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="predicted", ylabel="true",
           title=f"Model A confusion (station-disjoint test)\n"
                 f"thunder PR-AUC={m.get('thunder_ap', float('nan')):.2f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "cnn_confusion.png", dpi=150)
    print("\nmetrics -> outputs/cnn_metrics.json")
    print("figure  -> docs/figures/cnn_confusion.png")

    import torch
    torch.save(out["model"].state_dict(), out_dir / "model_a_seismic.pt")
    print("model   -> outputs/model_a_seismic.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
