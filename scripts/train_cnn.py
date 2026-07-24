#!/usr/bin/env python
"""Train the seismic-only thunderquake CNN (Model A, WS4 #8).

Cache-backed raw windows (WS1 50 s @ 100 Hz) -> station-disjoint split -> TRAIN-only
augmentation (realistic noise + diversity) -> anti-aliased (BlurPool) 2D CNN. Compares
short-and-fat vs long-and-skinny architectures. Run under the ml env:

    pixi run -e ml python scripts/train_cnn.py --metadata-dir /path/to/pnwml --region PNW
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
from thunderquakes.models.dataset import REGION_CLASSES, WindowConfig, build_windows  # noqa: E402
from thunderquakes.models.train import benchmark_inference, train  # noqa: E402
from thunderquakes.plotting import set_paper_style  # noqa: E402

set_paper_style()

# (label, width, depth): short-fat has few wide blocks; long-skinny many narrow ones.
ARCHS = [
    ("baseline_w32_d3", 32, 3),
    ("short_fat_w64_d3", 64, 3),
    ("long_skinny_w16_d5", 16, 5),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata-dir", required=True)
    ap.add_argument("--region", default="PNW", choices=list(REGION_CLASSES))
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-aug", type=int, default=3)
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
    print(f"Building windows from cache ({args.region}: {classes}) …")
    ds = build_windows(meta, classes, cfg=cfg, n_per_class=args.n_per_class, client=client)
    print(f"  W={ds.W.shape}  counts={np.bincount(ds.y).tolist()}")

    cat_dir = REPO_ROOT / "catalogs"
    cat_dir.mkdir(exist_ok=True)

    results = []
    for label, width, depth in ARCHS:
        print(f"\n=== {label} (width={width}, depth={depth}) + aug×{args.n_aug} ===")
        out = train(ds, cfg=cfg, epochs=args.epochs, width=width, depth=depth, n_aug=args.n_aug)
        ap_ = out["metrics"].get("thunder_ap", float("nan"))
        thr = out["metrics"]["report"]["thunder"]
        t_s = out["metrics"]["train_time_s"]
        print(f"  thunder PR-AUC={ap_:.3f}  P={thr['precision']:.2f} R={thr['recall']:.2f}"
              f"  train_time={t_s:.1f}s  params={out['metrics']['n_params']:,}")
        pd.DataFrame(out["history"]).to_csv(
            cat_dir / f"cnn_training_history_{args.region}_{label}.csv", index=False)
        results.append((label, ap_, out))

    results.sort(key=lambda r: (np.nan_to_num(r[1])), reverse=True)
    best_label, best_ap, best = results[0]
    print(f"\nBEST: {best_label}  thunder PR-AUC={best_ap:.3f}")

    # Realistic streaming-deployment inference throughput: one window at a time,
    # not one big batched forward pass over the whole test set.
    bench = benchmark_inference(best["model"], best["X_test"], best["metrics"]["device"],
                                batch_size=1)
    print(f"\nInference benchmark (batch_size=1, {bench['device']}): "
          f"{bench['windows_per_second']:.1f} windows/s "
          f"({bench['seconds_per_window'] * 1000:.2f} ms/window)")

    out_dir = REPO_ROOT / "outputs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    out_dir.mkdir(exist_ok=True)
    summary = {lab: {"thunder_ap": ap_, **o["metrics"]} for lab, ap_, o in results}
    summary["_best_model_inference_benchmark"] = bench
    (out_dir / f"cnn_arch_comparison_{args.region}.json").write_text(json.dumps(summary, indent=2))
    (cat_dir / f"cnn_arch_comparison_{args.region}.json").write_text(json.dumps(summary, indent=2))

    m = best["metrics"]
    cm = np.array(m["confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted class", ylabel="True class")
    fig.colorbar(im, fraction=0.046, label="Number of test-set windows")
    fig.tight_layout()
    fig.savefig(fig_dir / f"cnn_confusion_{args.region}.png")

    import torch

    model_path = out_dir / f"model_a_seismic_{args.region}.pt"
    torch.save(best["model"].state_dict(), model_path)
    arch_path = out_dir / f"model_a_seismic_{args.region}_arch.json"
    _, best_width, best_depth = next(a for a in ARCHS if a[0] == best_label)
    arch_path.write_text(json.dumps({
        "classes": classes, "width": best_width, "depth": best_depth,
        "in_ch": 1, "cfg": {"win_s": cfg.win_s, "fs": cfg.fs, "band": list(cfg.band),
                            "nperseg": cfg.nperseg, "noverlap": cfg.noverlap},
    }, indent=2))
    print(f"\narch comparison -> outputs/cnn_arch_comparison_{args.region}.json")
    print(f"confusion       -> docs/figures/cnn_confusion_{args.region}.png")
    print(f"best model      -> {model_path.relative_to(REPO_ROOT)}")
    print(f"arch metadata   -> {arch_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
