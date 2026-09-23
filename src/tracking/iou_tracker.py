"""A light SORT-style tracker (IoU + centre distance, Hungarian assignment).

Used for detector back-ends that do not come with their own tracker (HOG baseline).
The YOLO back-end uses Ultralytics' ByteTrack instead.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]
    b = b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


class _Track:
    def __init__(self, tid: int, det: Detection):
        self.id = tid
        self.bbox = det.bbox.astype(float)
        self.velocity = np.zeros(4)
        self.hits = 1
        self.missed = 0

    def predict(self) -> np.ndarray:
        return self.bbox + self.velocity

    def update(self, det: Detection):
        new = det.bbox.astype(float)
        self.velocity = 0.6 * self.velocity + 0.4 * (new - self.bbox)
        self.bbox = new
        self.hits += 1
        self.missed = 0


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.tracks: list[_Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        preds = np.array([t.predict() for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        boxes = np.array([d.bbox for d in dets]) if dets else np.zeros((0, 4))
        iou = iou_matrix(preds, boxes)
        matched_t, matched_d = set(), set()
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    self.tracks[r].update(dets[c])
                    dets[c].track_id = self.tracks[r].id
                    matched_t.add(r)
                    matched_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.missed += 1
                t.bbox = t.predict()
        for j, d in enumerate(dets):
            if j not in matched_d:
                t = _Track(self._next_id, d)
                self._next_id += 1
                self.tracks.append(t)
                d.track_id = t.id
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        confirmed = {t.id for t in self.tracks if t.hits >= self.min_hits}
        return [d for d in dets if d.track_id in confirmed]
