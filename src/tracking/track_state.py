"""What we remember about each tracked person over time.

The gender, activity and scene modules all read from and write to these objects.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from ..core import Detection


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=90))   # (t, bbox, keypoints or None)
    gender_scores: np.ndarray = field(default_factory=lambda: np.zeros(2))  # summed [male, female]
    gender_votes: int = 0
    gender: str = "unknown"
    gender_conf: float = 0.0
    activity: str = "standing"
    activity_conf: float = 0.0
    activity_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    activity_since: float = 0.0
    nbr_dist: deque = field(default_factory=lambda: deque(maxlen=90))   # distance to nearest person
    pos_log: deque = field(default_factory=lambda: deque(maxlen=600))   # (t, x, y), twice a second
    fall_time: float | None = None
    last_det: Detection | None = None

    def add(self, t: float, det: Detection):
        keypoints = None if det.keypoints is None else det.keypoints.copy()
        self.history.append((t, det.bbox.copy(), keypoints))
        self.last_seen = t
        self.last_det = det
        if not self.pos_log or t - self.pos_log[-1][0] >= 0.5:
            foot = det.foot
            self.pos_log.append((t, float(foot[0]), float(foot[1])))

    @property
    def dwell(self) -> float:
        return self.last_seen - self.first_seen

    def body_height(self) -> float:
        # 75th percentile of recent box heights, so a person lying down
        # for a moment doesn't shrink their "standing" height
        heights = [box[3] - box[1] for _, box, _ in self.history]
        return float(np.percentile(heights, 75)) if heights else 1.0

    def centers(self) -> np.ndarray:
        return np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b, _ in self.history])

    def times(self) -> np.ndarray:
        return np.array([t for t, _, _ in self.history])

    def speed(self, seconds: float = 1.0) -> np.ndarray:
        """Average velocity in pixels per second over the last `seconds`."""
        if len(self.history) < 2:
            return np.zeros(2)
        times, centers = self.times(), self.centers()
        start = int(np.searchsorted(times, times[-1] - seconds))
        start = min(start, len(times) - 2)
        elapsed = max(times[-1] - times[start], 1e-3)
        return (centers[-1] - centers[start]) / elapsed

    def set_activity(self, label: str, conf: float, t: float):
        # majority vote over the last few predictions stops the label flickering
        self.activity_hist.append(label)
        smoothed = Counter(self.activity_hist).most_common(1)[0][0]
        if smoothed != self.activity:
            self.activity_since = t
        self.activity = smoothed
        self.activity_conf = conf

    def add_gender(self, probs: np.ndarray, min_votes: int, min_conf: float):
        self.gender_scores += probs
        self.gender_votes += 1
        share = self.gender_scores / max(self.gender_scores.sum(), 1e-9)
        best = int(np.argmax(share))
        self.gender_conf = float(share[best])
        if self.gender_votes >= min_votes and share[best] >= min_conf:
            self.gender = ["male", "female"][best]
        else:
            self.gender = "unknown"


class TrackManager:
    def __init__(self, stale_seconds: float = 2.0):
        self.tracks: dict[int, TrackState] = {}
        self.stale_seconds = stale_seconds
        self.total_unique = 0

    def update(self, t: float, persons: list[Detection]) -> list[TrackState]:
        active = []
        for det in persons:
            track = self.tracks.get(det.track_id)
            if track is None:
                track = TrackState(det.track_id, first_seen=t, last_seen=t, activity_since=t)
                self.tracks[det.track_id] = track
                self.total_unique += 1
            track.add(t, det)
            active.append(track)

        gone = [tid for tid, track in self.tracks.items() if t - track.last_seen > self.stale_seconds]
        for tid in gone:
            del self.tracks[tid]
        return active
