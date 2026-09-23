#!/usr/bin/env python3
"""Fine-tune the full-body gender classifier (MobileNetV3-Small, ImageNet-pretrained).

Data layout (person crops; e.g. exported from PA-100K / PETA with the 'Female' attribute):
    data/gender/train/male/*.jpg   data/gender/train/female/*.jpg
    data/gender/val/male/*.jpg     data/gender/val/female/*.jpg

    python scripts/train_gender.py --data data/gender --epochs 15
Tip: scripts/prepare_pa100k.py converts the PA-100K annotation file into this layout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluation.metrics import classification_report  # noqa: E402
from src.gender.body_model import CLASSES, INPUT_SIZE, MEAN, STD, build_body_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/gender")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="models/gender_body_mobilenetv3.pt")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tf_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02), transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0),
                                                                                  ratio=(0.45, 0.55)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.RandomErasing(p=0.3)])
    tf_eval = transforms.Compose([transforms.Resize(INPUT_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train = datasets.ImageFolder(Path(args.data) / "train", tf_train)
    val = datasets.ImageFolder(Path(args.data) / "val", tf_eval)
    assert train.classes == sorted(CLASSES), f"expected folders {sorted(CLASSES)}, got {train.classes}"
    # ImageFolder sorts folders alphabetically (female=0, male=1); the model outputs [male, female]
    remap = torch.tensor([CLASSES.index(c) for c in train.classes])
    dl_tr = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    dl_va = DataLoader(val, batch_size=args.batch, num_workers=args.workers)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_body_model(pretrained=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl_tr))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), remap[y].to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
            sched.step()
        model.eval()
        yt, yp = [], []
        with torch.no_grad():
            for x, y in dl_va:
                yp += model(x.to(dev)).argmax(1).cpu().tolist()
                yt += remap[y].tolist()
        rep = classification_report([CLASSES[i] for i in yt], [CLASSES[i] for i in yp], CLASSES)
        print(f"epoch {ep}: val acc {rep['accuracy']:.4f}  F1 {rep['f1']:.4f}")
        if rep["f1"] > best:
            best = rep["f1"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out)
    print(f"best val F1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
