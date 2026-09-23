"""Data classes passed between the pipeline stages."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# COCO keypoint order used by YOLOv8-pose
KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_shoulder=5, r_shoulder=6,
          l_elbow=7, r_elbow=8, l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
          l_knee=13, r_knee=14, l_ankle=15, r_ankle=16)

# pairs of keypoints joined by a line when the skeleton is drawn
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class Detection:
    bbox: np.ndarray                        # x1, y1, x2, y2 in pixels
    conf: float
    track_id: int = -1
    cls: str = "person"
    keypoints: Optional[np.ndarray] = None  # 17 rows of (x, y, confidence)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def foot(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2])

    @property
    def height(self) -> float:
        return float(max(1.0, self.bbox[3] - self.bbox[1]))

    @property
    def width(self) -> float:
        return float(max(1.0, self.bbox[2] - self.bbox[0]))


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    persons: list[Detection]
    objects: list[Detection] = field(default_factory=list)


@dataclass
class SceneEvent:
    """Something the scene analyser noticed, e.g. a fall or a fight."""
    type: str
    track_ids: tuple[int, ...]
    description: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    confidence: float = 1.0
    location: Optional[tuple[float, float]] = None   # pixel position, used for the overlay
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        # the same kind of event involving the same people is treated as one situation
        ids = ",".join(str(i) for i in sorted(self.track_ids))
        return f"{self.type}|{ids}"


@dataclass
class ThreatAlert:
    event: SceneEvent
    score: float
    level: str
    reasons: list[str]
    camera: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    snapshot: Optional[str] = None
    acknowledged: bool = False

    def to_dict(self) -> dict:
        event = self.event
        return {
            "id": self.id,
            "type": event.type,
            "description": event.description,
            "track_ids": list(event.track_ids),
            "timestamp": event.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp)),
            "duration": round(event.duration, 1),
            "score": round(self.score, 1),
            "level": self.level,
            "reasons": self.reasons,
            "camera": self.camera,
            "snapshot": self.snapshot,
            "acknowledged": self.acknowledged,
        }
