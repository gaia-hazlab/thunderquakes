"""Training / evaluation harness for the seismic-only CNN (WS4 #8).

Pipeline: raw windows -> STATION-disjoint split (GroupShuffleSplit; no station in
both sides, the honest precursor to train-PNW/test-OK) -> augment the TRAIN split
only (realistic noise + signal diversity) -> log-spectrograms -> class-weighted
CNN. Headline metric is thunder one-vs-rest average precision (PR-AUC), the right
choice under class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thunderquakes.models.augment import augment_training_set
from thunderquakes.models.cnn import build_seismic_cnn
from thunderquakes.models.dataset import WindowConfig, to_spectrograms


@dataclass
class SplitIdx:
    tr: np.ndarray
    te: np.ndarray


def station_split_idx(ds, test_frac: float = 0.3, seed: int = 0) -> SplitIdx:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(ds.W, ds.y, ds.meta["station"].values))
    return SplitIdx(tr, te)


def _loader(X, y, batch=32, shuffle=False):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch, shuffle=shuffle)


def train(
    ds,
    cfg: WindowConfig | None = None,
    epochs: int = 40,
    lr: float = 1e-3,
    batch: int = 32,
    dropout: float = 0.3,
    width: int = 32,
    depth: int = 3,
    n_aug: int = 3,
    test_frac: float = 0.3,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train Model A on a RawDataset and return trained model + metrics."""
    cfg = cfg or WindowConfig()
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(ds.classes)
    sp = station_split_idx(ds, test_frac, seed)

    # augment TRAIN windows only, then spectrogram both splits
    Wtr, ytr = augment_training_set(ds.W[sp.tr], ds.y[sp.tr], ds.classes, n_aug=n_aug, seed=seed)
    Xtr = to_spectrograms(Wtr, cfg)
    Xte = to_spectrograms(ds.W[sp.te], cfg)
    yte = ds.y[sp.te]

    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.clip(counts, 1, None)),
                           dtype=torch.float32, device=device)
    model = build_seismic_cnn(n_classes, width=width, depth=depth, dropout=dropout,
                              in_ch=Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=weights)

    tr_loader = _loader(Xtr, ytr, batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  train_loss={tot / len(ytr):.3f}")

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(Xte).to(device)), dim=1).cpu().numpy()
    pred = probs.argmax(1)

    report = classification_report(yte, pred, target_names=ds.classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=range(n_classes))
    metrics = {"classes": ds.classes, "report": report, "confusion": cm.tolist(),
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "width": width, "depth": depth, "n_aug": n_aug}
    if "thunder" in ds.classes:
        ti = ds.classes.index("thunder")
        metrics["thunder_ap"] = float(
            average_precision_score((yte == ti).astype(int), probs[:, ti]))
    return {"model": model, "metrics": metrics, "probs": probs, "split": sp,
            "y_test": yte, "pred": pred}
