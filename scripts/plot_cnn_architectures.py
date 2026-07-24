#!/usr/bin/env python
"""Schematic diagram of the three CNN architectures compared in WS4 #8.

Draws each architecture (baseline_w32_d3, short_fat_w64_d3, long_skinny_w16_d5)
as a row of blocks (Conv->BN->ReLU->BlurPool, `thunderquakes.models.cnn._block`),
box width scaled to log(channels), labeled with the channel count and total
parameter count -- a visual complement to the numeric comparison table in the
report, not a replacement for it.

Usage:
    pixi run -e ml python scripts/plot_cnn_architectures.py

Outputs:
    docs/figures/cnn_architectures.png
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from thunderquakes.config import REPO_ROOT  # noqa: E402
from thunderquakes.models.cnn import build_seismic_cnn  # noqa: E402
from thunderquakes.plotting import LETTER_WIDTH_IN, set_paper_style  # noqa: E402

set_paper_style()

ARCHS = [
    ("baseline_w32_d3", 32, 3),
    ("short_fat_w64_d3", 64, 3),
    ("long_skinny_w16_d5", 16, 5),
]
MAX_CHANNELS = 256
N_CLASSES = 3  # thunder / sonic boom / noise -- matches the OK-targeted comparison


def block_channels(width: int, depth: int) -> list[int]:
    return [min(width * (2**i), MAX_CHANNELS) for i in range(depth)]


def main() -> int:
    fig_dir = REPO_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(ARCHS), 1, figsize=(LETTER_WIDTH_IN, 5.4))
    for ax, (label, width, depth) in zip(axes, ARCHS, strict=True):
        model = build_seismic_cnn(N_CLASSES, width=width, depth=depth)
        n_params = sum(p.numel() for p in model.parameters())
        chans = block_channels(width, depth)

        x = 0.0
        box_h = 0.8
        gap = 0.15
        # input box
        w_in = 0.5
        ax.add_patch(FancyBboxPatch((x, 0), w_in, box_h, boxstyle="round,pad=0.02",
                                    fc="0.92", ec="0.4"))
        ax.text(x + w_in / 2, box_h / 2, "in\n1ch", ha="center", va="center", fontsize=8)
        x += w_in + gap
        for c in chans:
            w = 0.35 + 0.55 * math.log2(c) / math.log2(MAX_CHANNELS)
            ax.add_patch(FancyBboxPatch((x, 0), w, box_h, boxstyle="round,pad=0.02",
                                        fc="#4c72b0", ec="0.2", alpha=0.85))
            ax.text(x + w / 2, box_h / 2, f"{c}ch", ha="center", va="center",
                    color="white", fontsize=9, weight="bold")
            x += w + gap
        # head
        w_head = 0.6
        ax.add_patch(FancyBboxPatch((x, 0), w_head, box_h, boxstyle="round,pad=0.02",
                                    fc="0.92", ec="0.4"))
        ax.text(x + w_head / 2, box_h / 2, f"avgpool\n+fc({N_CLASSES})", ha="center",
                va="center", fontsize=7)
        x += w_head

        ax.set_xlim(-0.1, x + 0.1)
        ax.set_ylim(-0.1, box_h + 0.1)
        ax.axis("off")
        ax.text(-0.1, box_h + 0.35,
                f"{label}  (width={width}, depth={depth}, {n_params:,} params)",
                fontsize=10, weight="bold", ha="left", va="bottom")

    fig.tight_layout(h_pad=1.5)
    out = fig_dir / "cnn_architectures.png"
    fig.savefig(out)
    print(f"architecture diagram -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
