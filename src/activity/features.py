"""Scale-invariant per-frame features for activity recognition.

The *same* function is used when building the training set (scripts/extract_pose_dataset.py)
and at run time, which guarantees train/test consistency.

Per frame (F = 76):
  34  keypoint (x, y) relative to the box centre, divided by the reference body height
  34  keypoint velocities (body-heights / second)
   2  box-centre velocity (vx, vy)         (body-heights / second)
   1  box aspect ratio (w / h)
   1  box height / reference height        (drops when a person falls or crouches)
   2  torso orientation (sin, cos of angle from vertical)
   2  nearest-neighbour distance (body heights, clipped) and its rate of change
"""
from __future__ import annotations

import numpy as np

FEATURE_DIM = 76
KP_CONF = 0.3


def torso_angle(kps: np.ndarray | None) -> float | None:
    """Angle (deg) between the hip->shoulder vector and the vertical axis."""
    if kps is None:
        return None
    sh, hp = kps[[5, 6]], kps[[11, 12]]
    if (sh[:, 2] < KP_CONF).all() or (hp[:, 2] < KP_CONF).all():
        return None
    s = sh[sh[:, 2] >= KP_CONF, :2].mean(0)
    h = hp[hp[:, 2] >= KP_CONF, :2].mean(0)
    dx, dy = s - h
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def sequence_features(times, bboxes, kps_list, nbr_dist=None, ref_h: float | None = None) -> np.ndarray:
    """times (T,), bboxes (T,4), kps_list: list of (17,3) or None, nbr_dist (T,) -> (T, 76)."""
    times = np.asarray(times, dtype=float)
    b = np.asarray(bboxes, dtype=float)
    T = len(times)
    if T == 0:
        return np.zeros((0, FEATURE_DIM), np.float32)
    heights = np.maximum(b[:, 3] - b[:, 1], 1.0)
    widths = np.maximum(b[:, 2] - b[:, 0], 1.0)
    ref_h = float(ref_h or np.percentile(heights, 75))
    centers = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)

    rel = np.zeros((T, 17, 2))
    conf = np.zeros((T, 17))
    for i, k in enumerate(kps_list):
        if k is not None:
            rel[i] = (k[:, :2] - centers[i]) / ref_h
            conf[i] = k[:, 2]
    rel[conf < KP_CONF] = 0.0

    dt = np.diff(times, prepend=times[0] - 1 / 25.0)
    dt = np.clip(dt, 1e-3, 1.0)[:, None]
    kvel = np.diff(rel.reshape(T, -1), axis=0, prepend=rel.reshape(T, -1)[:1]) / dt
    cvel = np.diff(centers, axis=0, prepend=centers[:1]) / ref_h / dt

    ang = np.array([torso_angle(k) if k is not None else np.nan for k in kps_list], dtype=float)
    # fall back to box shape when the torso is not visible
    ang = np.where(np.isnan(ang), np.degrees(np.arctan(np.clip(widths / heights, 0, 5) * 0.8)), ang)
    ang = np.radians(ang)

    nd = np.full(T, 5.0) if nbr_dist is None else np.clip(np.asarray(nbr_dist, float)[-T:], 0, 5.0)
    if len(nd) < T:
        nd = np.concatenate([np.full(T - len(nd), nd[0] if len(nd) else 5.0), nd])
    ndv = np.diff(nd, prepend=nd[:1]) / dt[:, 0]

    feats = np.concatenate([
        rel.reshape(T, -1), np.clip(kvel, -10, 10), np.clip(cvel, -10, 10),
        (widths / heights)[:, None], (heights / ref_h)[:, None],
        np.sin(ang)[:, None], np.cos(ang)[:, None],
        nd[:, None], np.clip(ndv, -10, 10)[:, None],
    ], axis=1)
    return feats.astype(np.float32)


def pad_window(feats: np.ndarray, window: int) -> np.ndarray:
    """Take the last `window` frames, left-padding by repeating the first frame."""
    if len(feats) >= window:
        return feats[-window:]
    pad = np.repeat(feats[:1], window - len(feats), axis=0)
    return np.concatenate([pad, feats], 0)


def track_features(track, window: int) -> np.ndarray:
    hist = list(track.history)[-window:]
    nbr = list(track.nbr_dist)[-len(hist):]
    return pad_window(sequence_features([h[0] for h in hist], [h[1] for h in hist],
                                        [h[2] for h in hist], nbr, track.body_height()), window)
