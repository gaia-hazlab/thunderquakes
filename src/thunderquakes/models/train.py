"""Training / evaluation harness for the seismic-only CNN (WS4 #8).

Splits are by STATION (GroupShuffleSplit) so no station appears in both train and
test — this avoids the leakage that inflates random-split scores and is the honest
precursor to the WS4 train-PNW / test-OK generalization experiment. Loss is
class-weighted (severe imbalance: noise >> thunder). Reports macro P/R/F1, a
confusion matrix, and thunder one-vs-rest average precision (PR-AUC), the right
headline metric under imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thunderquakes.models.cnn import build_seismic_cnn


@dataclass
class SplitData:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


def station_split(ds, test_frac: float = 0.3, seed: int = 0) -> SplitData:
    """Station-disjoint train/test split (no station in both sides)."""
    groups = ds.meta["station"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(gss.split(ds.X, ds.y, groups))
    return SplitData(ds.X[tr], ds.y[tr], ds.X[te], ds.y[te])


def _loader(X, y, batch=32, shuffle=False):
    tX = torch.from_numpy(X)
    ty = torch.from_numpy(y)
    return DataLoader(TensorDataset(tX, ty), batch_size=batch, shuffle=shuffle)


def train(
    ds,
    epochs: int = 30,
    lr: float = 1e-3,
    batch: int = 32,
    dropout: float = 0.3,
    test_frac: float = 0.3,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    """Train Model A on ``ds`` and return trained model + metrics."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = station_split(ds, test_frac, seed)
    n_classes = len(ds.classes)

    counts = np.bincount(sp.ytr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.clip(counts, 1, None)),
                           dtype=torch.float32, device=device)
    model = build_seismic_cnn(n_classes, dropout=dropout, in_ch=ds.X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(weight=weights)

    tr_loader = _loader(sp.Xtr, sp.ytr, batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train_loss={tot / len(sp.ytr):.3f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sp.Xte).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred = probs.argmax(1)

    report = classification_report(sp.yte, pred, target_names=ds.classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(sp.yte, pred, labels=range(n_classes))
    metrics = {"classes": ds.classes, "report": report, "confusion": cm.tolist(),
               "n_train": int(len(sp.ytr)), "n_test": int(len(sp.yte))}
    if "thunder" in ds.classes:
        ti = ds.classes.index("thunder")
        metrics["thunder_ap"] = float(
            average_precision_score((sp.yte == ti).astype(int), probs[:, ti])
        )
    return {"model": model, "metrics": metrics, "probs": probs, "split": sp}
