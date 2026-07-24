"""Training / evaluation harness for the seismic-only CNN (WS4 #8).

Pipeline: raw windows -> STATION-disjoint split (GroupShuffleSplit; no station in
both sides, the honest precursor to train-PNW/test-OK) -> augment the TRAIN split
only (realistic noise + signal diversity) -> log-spectrograms -> class-weighted
CNN. Headline metric is thunder one-vs-rest average precision (PR-AUC), the right
choice under class imbalance.
"""

from __future__ import annotations

import time
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


def benchmark_inference(model, X: np.ndarray, device: str, batch_size: int = 1,
                        n_repeats: int = 3) -> dict:
    """Wall-clock inference throughput at a realistic streaming batch size.

    Continuous deployment scores one sliding window at a time (batch_size=1 by
    default here), which is much slower per-window than one large batched forward
    pass over the whole test set -- report this, not the training-loop batch
    timing, as the number that should inform a deployment cost estimate.
    """
    model.eval()
    Xt = torch.from_numpy(X).to(device)
    n = len(Xt)
    with torch.no_grad():
        model(Xt[:min(batch_size, n)])  # warm-up (lazy CUDA init, cudnn autotune)
        times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            for i in range(0, n, batch_size):
                model(Xt[i:i + batch_size])
            times.append(time.perf_counter() - start)
    best = min(times)
    return {"batch_size": batch_size, "n_windows": n, "device": device,
            "seconds_total": best, "seconds_per_window": best / n,
            "windows_per_second": n / best}


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

    Xte_t = torch.from_numpy(Xte).to(device)
    yte_t = torch.from_numpy(yte).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    tr_loader = _loader(Xtr, ytr, batch, shuffle=True)
    history = []
    train_start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        tot, correct = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
        train_loss = tot / len(ytr)
        train_acc = correct / len(ytr)

        model.eval()
        with torch.no_grad():
            te_out = model(Xte_t)
            test_loss = crit(te_out, yte_t).item()
            test_acc = (te_out.argmax(1) == yte_t).float().mean().item()
        history.append({"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  train_loss={train_loss:.3f}  test_loss={test_loss:.3f}"
                  f"  test_acc={test_acc:.3f}")
    train_time_s = time.perf_counter() - train_start

    model.eval()
    infer_start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=1).cpu().numpy()
    infer_time_s = time.perf_counter() - infer_start
    pred = probs.argmax(1)

    report = classification_report(yte, pred, target_names=ds.classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=range(n_classes))
    metrics = {"classes": ds.classes, "report": report, "confusion": cm.tolist(),
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "width": width, "depth": depth, "n_aug": n_aug,
               "n_params": int(n_params), "device": device,
               "train_time_s": train_time_s, "epochs": epochs,
               "infer_time_s_per_test_set": infer_time_s,
               "infer_s_per_window": infer_time_s / max(len(yte), 1)}
    if "thunder" in ds.classes:
        ti = ds.classes.index("thunder")
        metrics["thunder_ap"] = float(
            average_precision_score((yte == ti).astype(int), probs[:, ti]))
    return {"model": model, "metrics": metrics, "probs": probs, "split": sp,
            "y_test": yte, "pred": pred, "history": history, "X_test": Xte}
