"""Spectrogram CNN — seismic-only (Model A) and dual-branch (Model B) (WS4).

Requires the 'ml' pixi env (pytorch). Not imported by ``models/__init__`` so the
default env can still import the package; import this module explicitly under -e ml.

Design:
- **Anti-aliased downsampling (BlurPool, Zhang 2019).** Plain max/stride pooling
  aliases high-frequency spectrogram structure; we use MaxBlurPool (MaxPool stride 1
  then a fixed binomial low-pass BlurPool stride 2) so downsampling is shift-robust.
- **Configurable width/depth** to explore short-and-fat (few wide blocks) vs
  long-and-skinny (many narrow blocks).
- Dropout between blocks + before the head so the net supports MC-dropout (WS4 #10).
- AdaptiveAvgPool makes the head independent of exact (freq×time) input size.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BlurPool2d(nn.Module):
    """Anti-aliased downsampling: depthwise fixed binomial low-pass, stride 2."""

    def __init__(self, channels: int, stride: int = 2):
        super().__init__()
        a = torch.tensor([1.0, 2.0, 1.0])
        k = torch.outer(a, a)
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(x, self.kernel, stride=self.stride, groups=self.channels)


def _block(cin: int, cout: int, dropout: float) -> nn.Sequential:
    # Conv -> BN -> ReLU -> MaxPool(stride1) -> BlurPool(stride2)  (= anti-aliased MaxPool)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=1),
        BlurPool2d(cout, stride=2),
        nn.Dropout(dropout),
    )


class SeismicCNN(nn.Module):
    """2D CNN over a log-spectrogram (Model A, seismic-only).

    ``width`` = channels in the first block; each block doubles channels (capped).
    ``depth`` = number of blocks. Short-fat: large width, small depth. Long-skinny:
    small width, large depth.
    """

    def __init__(
        self,
        n_classes: int,
        width: int = 32,
        depth: int = 3,
        dropout: float = 0.3,
        in_ch: int = 1,
        max_channels: int = 256,
    ):
        super().__init__()
        blocks, cin = [], in_ch
        for i in range(depth):
            cout = min(width * (2**i), max_channels)
            blocks.append(_block(cin, cout, dropout))
            cin = cout
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_seismic_cnn(
    n_classes: int, width: int = 32, depth: int = 3, dropout: float = 0.3, in_ch: int = 1
) -> SeismicCNN:
    """Model A factory: single-branch anti-aliased spectrogram CNN."""
    return SeismicCNN(n_classes, width=width, depth=depth, dropout=dropout, in_ch=in_ch)
