#!/usr/bin/env python
"""Learning curves for the three CNN architectures compared in WS4 #8.

Reads the per-epoch history CSVs written by ``scripts/train_cnn.py``
(``catalogs/cnn_training_history_{region}_{arch}.csv``) and plots train/test
loss and test accuracy vs. epoch, one column per architecture. There is no
separate validation split (station-disjoint train/test only, WS4 #8) so "test"
is plotted directly rather than invented as a "val" curve.

Usage:
    pixi run -e default python scripts/plot_learning_curves.py --region OK

Outputs:
    docs/figures/cnn_learning_curves_{region}.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

ARCHS = ["baseline_w32_d3", "short_fat_w64_d3", "long_skinny_w16_d5"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="OK")
    args = ap.parse_args()

    cat_dir = REPO_ROOT / "catalogs"
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, len(ARCHS), figsize=(LETTER_WIDTH_IN, 4.6),
                             sharex="col", sharey="row")
    for col, arch in enumerate(ARCHS):
        path = cat_dir / f"cnn_training_history_{args.region}_{arch}.csv"
        h = pd.read_csv(path)
        ax_loss, ax_acc = axes[0][col], axes[1][col]
        ax_loss.plot(h["epoch"], h["train_loss"], label="train", color="steelblue")
        ax_loss.plot(h["epoch"], h["test_loss"], label="test", color="darkorange")
        ax_loss.set_title(arch, fontsize=9)
        ax_acc.plot(h["epoch"], h["train_acc"], color="steelblue")
        ax_acc.plot(h["epoch"], h["test_acc"], color="darkorange")
        ax_acc.set_xlabel("Epoch")
        if col == 0:
            ax_loss.set_ylabel("Cross-entropy loss")
            ax_acc.set_ylabel("Accuracy")
        if col == len(ARCHS) - 1:
            ax_loss.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = fig_dir / f"cnn_learning_curves_{args.region}.png"
    fig.savefig(out)
    print(f"learning curves -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
