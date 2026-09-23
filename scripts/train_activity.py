#!/usr/bin/env python3
"""Train the deep activity-recognition models (Bi-LSTM or TCN) on a pose-sequence dataset.

    python scripts/train_activity.py --data data/activity/dataset.npz --model lstm
    python scripts/train_activity.py --data data/activity/dataset.npz --model tcn --epochs 80

The split is made per source video (no clip appears in two splits). The best epoch on the
validation macro-F1 is kept; test-set metrics are printed and stored in the checkpoint.
Evaluate / compare all algorithms afterwards with:  python scripts/evaluate.py activity --data ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activity.dataset import build_features, load_npz, split_indices  # noqa: E402
from src.activity.models import build_model  # noqa: E402
from src.evaluation.metrics import classification_report  # noqa: E402


def predict(model, X, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs])).argmax(1).numpy())
    return np.concatenate(out) if out else np.array([], int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["lstm", "tcn"], default="lstm")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="checkpoint path (default models/activity_<model>.pt)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = load_npz(args.data)
    classes = [str(c) for c in d["classes"]]
    tr, va, te = split_indices(d["groups"], seed=args.seed)
    print(f"windows: train {len(tr)}  val {len(va)}  test {len(te)}  (videos: {len(set(d['groups'].tolist()))})")
    Xtr, ytr = build_features(d, tr, augment=True, seed=args.seed)
    Xva, yva = build_features(d, va)
    Xte, yte = build_features(d, te)
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    norm = lambda X: ((X - mean) / std).astype(np.float32)
    cidx = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([cidx[str(v)] for v in y], np.int64)

    counts = np.bincount(enc(ytr), minlength=len(classes)).astype(float)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(classes), dtype=torch.float32)
    print("train class counts:", dict(zip(classes, counts.astype(int))))

    model = build_model(args.model, len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    dl = DataLoader(TensorDataset(torch.from_numpy(norm(Xtr)), torch.from_numpy(enc(ytr))),
                    batch_size=args.batch, shuffle=True)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        Xv, yv = (Xva, yva) if len(Xva) else (Xtr, ytr)
        rep = classification_report([str(v) for v in yv], [classes[i] for i in predict(model, norm(Xv))], classes)
        print(f"epoch {ep:3d}  loss {tot / len(dl.dataset):.4f}  val acc {rep['accuracy']:.3f}  "
              f"val F1 {rep['f1']:.3f}  ({time.time() - t0:.1f}s)")
        if rep["f1"] > best:
            best, bad = rep["f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    test = classification_report([str(v) for v in yte], [classes[i] for i in predict(model, norm(Xte))], classes) \
        if len(Xte) else {}
    if test:
        print(f"\nTEST  accuracy {test['accuracy']:.3f}  precision {test['precision']:.3f}  "
              f"recall {test['recall']:.3f}  F1 {test['f1']:.3f}")
    out = Path(args.out or f"models/activity_{args.model}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "window": int(d["times"].shape[1]),
                "mean": mean.tolist(), "std": std.tolist(), "kind": args.model, "val_f1": best, "test": test,
                "data": str(args.data), "seed": args.seed}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
