#!/usr/bin/env python3
"""Sort the PA-100K pedestrian dataset into male/female folders for train_gender.py and
evaluate.py gender.

    python scripts/prepare_pa100k.py --root /path/to/PA-100K --out data/gender

Expects <root>/annotation.mat and <root>/data/*.jpg, as in the official download.
"""
import argparse
import shutil
from pathlib import Path

from scipy.io import loadmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="data/gender")
    parser.add_argument("--link", action="store_true", help="create symlinks instead of copying")
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.out)
    annotation = loadmat(root / "annotation.mat")
    attributes = [str(a[0][0]) for a in annotation["attributes"]]
    female = attributes.index("Female")

    for split in ("train", "val", "test"):
        names = annotation[f"{split}_images_name"]
        labels = annotation[f"{split}_label"]
        for entry, label in zip(names, labels):
            name = str(entry[0][0])
            gender = "female" if label[female] == 1 else "male"
            dest = out / split / gender / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = root / "data" / name
            if args.link:
                if not dest.exists():
                    dest.symlink_to(source.resolve())
            else:
                shutil.copy2(source, dest)
        print(split, len(names))


if __name__ == "__main__":
    main()
