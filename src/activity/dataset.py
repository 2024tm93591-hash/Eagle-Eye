"""Pose-sequence dataset utilities shared by extraction, training and evaluation.

Dataset file (.npz) produced by scripts/extract_pose_dataset.py:
    times  (N, W)          timestamps (s)
    bboxes (N, W, 4)       person boxes (px)
    kps    (N, W, 17, 3)   COCO-17 keypoints (x, y, conf); conf = 0 where missing
    nbr    (N, W)          nearest-other-person distance (body heights)
    refh   (N,)            reference standing height (px) at the end of the window
    labels (N,)            activity label (str)
    groups (N,)            source video id - splits are made per video to avoid leakage
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core import Detection
from ..tracking.track_state import TrackState
from .features import sequence_features

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def split_indices(groups, seed: int = 42, val: float = 0.15, test: float = 0.15):
    """Group-wise (per video) train / val / test split, deterministic for a given seed."""
    ug = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    n = len(ug)
    n_test = max(1, int(round(n * test))) if n >= 3 else 0
    n_val = max(1, int(round(n * val))) if n >= 3 else 0
    test_g, val_g = set(ug[:n_test]), set(ug[n_test:n_test + n_val])
    idx = np.arange(len(groups))
    te = idx[np.isin(groups, list(test_g))]
    va = idx[np.isin(groups, list(val_g))]
    tr = idx[~np.isin(groups, list(test_g | val_g))]
    return tr, va, te


def window_features(d: dict, i: int, flip: bool = False, noise: float = 0.0, rng=None) -> np.ndarray:
    b = d["bboxes"][i].astype(float).copy()
    k = d["kps"][i].astype(float).copy()
    if flip:
        cx = (b[:, 0].min() + b[:, 2].max()) / 2
        b[:, [0, 2]] = 2 * cx - b[:, [2, 0]]
        k = k[:, FLIP_IDX]
        k[..., 0] = np.where(k[..., 2] > 0, 2 * cx - k[..., 0], 0)
    if noise and rng is not None:
        k[..., :2] += rng.normal(0, noise * d["refh"][i], size=k[..., :2].shape) * (k[..., 2:3] > 0)
    kl = [None if (kk[:, 2] == 0).all() else kk for kk in k]
    return sequence_features(d["times"][i], b, kl, d["nbr"][i], float(d["refh"][i]))


def build_features(d: dict, idx, augment: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i in idx:
        X.append(window_features(d, i))
        y.append(d["labels"][i])
        if augment:
            X.append(window_features(d, i, flip=True, noise=0.01, rng=rng))
            y.append(d["labels"][i])
    return np.stack(X).astype(np.float32), np.array(y)


def replay_rule_based(rules, d: dict, i: int, smoothing: int = 7) -> str:
    """Feed one window frame-by-frame through a TrackState so the rule-based classifier sees exactly
    what it would see live, and return its final (smoothed) label."""
    tr = TrackState(1, first_seen=float(d["times"][i][0]), last_seen=float(d["times"][i][0]))
    tr.activity_hist = deque(maxlen=smoothing)
    label = "standing"
    for t, b, k, nd in zip(d["times"][i], d["bboxes"][i], d["kps"][i], d["nbr"][i]):
        kp = None if (k[:, 2] == 0).all() else k.astype(float)
        tr.add(float(t), Detection(bbox=b.astype(float), conf=1.0, track_id=1, keypoints=kp))
        tr.nbr_dist.append(float(nd))
        if len(tr.history) >= 8:
            label, conf, _ = rules.predict(tr, float(t))
            tr.set_activity(label, conf, float(t))
    return tr.activity
