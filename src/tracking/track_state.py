"""Per-person temporal memory shared by the gender, activity and scene modules."""
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
    history: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, bbox, keypoints|None)
    gender_scores: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [male, female] summed probs
    gender_votes: int = 0
    gender: str = "unknown"
    gender_conf: float = 0.0
    activity: str = "standing"
    activity_conf: float = 0.0
    activity_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    activity_since: float = 0.0
    zone_enter: dict = field(default_factory=dict)          # zone name -> entry time
    nbr_dist: deque = field(default_factory=lambda: deque(maxlen=90))  # nearest-person distance (body h)
    pos_log: deque = field(default_factory=lambda: deque(maxlen=600))  # (t, x, y) sampled at 2 Hz
    fall_time: float | None = None
    last_det: Detection | None = None

    # ------------------------------------------------------------------ helpers
    def add(self, t: float, det: Detection):
        self.history.append((t, det.bbox.copy(), None if det.keypoints is None else det.keypoints.copy()))
        self.last_seen = t
        self.last_det = det
        if not self.pos_log or t - self.pos_log[-1][0] >= 0.5:
            c = det.foot
            self.pos_log.append((t, float(c[0]), float(c[1])))

    @property
    def dwell(self) -> float:
        return self.last_seen - self.first_seen

    def body_height(self) -> float:
        """Robust standing-height estimate (median of recent box heights, ignores lying poses)."""
        hs = [b[3] - b[1] for _, b, _ in self.history]
        return float(np.percentile(hs, 75)) if hs else 1.0

    def centers(self) -> np.ndarray:
        return np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b, _ in self.history])

    def times(self) -> np.ndarray:
        return np.array([t for t, _, _ in self.history])

    def speed(self, seconds: float = 1.0) -> np.ndarray:
        """Average velocity (px/s) over the last `seconds`."""
        if len(self.history) < 2:
            return np.zeros(2)
        ts, cs = self.times(), self.centers()
        i = int(np.searchsorted(ts, ts[-1] - seconds))
        i = min(i, len(ts) - 2)
        dt = max(ts[-1] - ts[i], 1e-3)
        return (cs[-1] - cs[i]) / dt

    def set_activity(self, label: str, conf: float, t: float):
        self.activity_hist.append(label)
        smooth = Counter(self.activity_hist).most_common(1)[0][0]
        if smooth != self.activity:
            self.activity_since = t
        self.activity = smooth
        self.activity_conf = conf

    def add_gender(self, probs: np.ndarray, min_votes: int, min_conf: float):
        self.gender_scores += probs
        self.gender_votes += 1
        p = self.gender_scores / max(self.gender_scores.sum(), 1e-9)
        k = int(np.argmax(p))
        self.gender_conf = float(p[k])
        if self.gender_votes >= min_votes and p[k] >= min_conf:
            self.gender = ["male", "female"][k]
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
            st = self.tracks.get(det.track_id)
            if st is None:
                st = TrackState(det.track_id, first_seen=t, last_seen=t, activity_since=t)
                self.tracks[det.track_id] = st
                self.total_unique += 1
            st.add(t, det)
            active.append(st)
        for tid in [k for k, v in self.tracks.items() if t - v.last_seen > self.stale_seconds]:
            del self.tracks[tid]
        return active
