#!/usr/bin/env python3
"""Convert the PA-100K pedestrian-attribute dataset into male/female folders for train_gender.py
and evaluate.py gender.

    python scripts/prepare_pa100k.py --root /path/to/PA-100K --out data/gender
Expects <root>/annotation.mat and <root>/data/*.jpg (the official release layout).
"""
import argparse
import shutil
from pathlib import Path

from scipy.io import loadmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/gender")
    ap.add_argument("--link", action="store_true", help="symlink instead of copying")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    m = loadmat(root / "annotation.mat")
    attrs = [str(x[0][0]) for x in m["attributes"]]
    fi = attrs.index("Female")
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        names, labels = m[f"{key}_images_name"], m[f"{key}_label"]
        for n, lab in zip(names, labels):
            name = str(n[0][0])
            cls = "female" if lab[fi] == 1 else "male"
            dst = out / split / cls / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = root / "data" / name
            if a.link:
                if not dst.exists():
                    dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        print(split, len(names))


if __name__ == "__main__":
    main()
