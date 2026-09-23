"""Shared data structures passed between the pipeline stages."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# COCO-17 keypoint indices (YOLOv8-pose order)
KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_shoulder=5, r_shoulder=6,
          l_elbow=7, r_elbow=8, l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
          l_knee=13, r_knee=14, l_ankle=15, r_ankle=16)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class Detection:
    """A single person (or object) detection in one frame."""
    bbox: np.ndarray                       # [x1, y1, x2, y2] pixels
    conf: float
    track_id: int = -1
    cls: str = "person"
    keypoints: Optional[np.ndarray] = None  # (17, 3) -> x, y, confidence

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    @property
    def foot(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2])

    @property
    def height(self) -> float:
        return float(max(1.0, self.bbox[3] - self.bbox[1]))

    @property
    def width(self) -> float:
        return float(max(1.0, self.bbox[2] - self.bbox[0]))


@dataclass
class FrameResult:
    """Everything the detector stage produced for one frame."""
    frame_idx: int
    timestamp: float
    persons: list[Detection]
    objects: list[Detection] = field(default_factory=list)


@dataclass
class SceneEvent:
    """An observation produced by the scene-understanding module."""
    type: str
    track_ids: tuple[int, ...]
    description: str
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    zone: Optional[str] = None
    zone_sensitivity: float = 1.0
    confidence: float = 1.0
    location: Optional[tuple[float, float]] = None   # pixel coords (for the overlay)
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        ids = ",".join(str(i) for i in sorted(self.track_ids))
        return f"{self.type}|{ids}|{self.zone or ''}"


@dataclass
class ThreatAlert:
    """A scored scene event that is shown / notified / stored."""
    event: SceneEvent
    score: float
    level: str
    reasons: list[str]
    camera: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    snapshot: Optional[str] = None
    acknowledged: bool = False

    def to_dict(self) -> dict:
        e = self.event
        return {
            "id": self.id,
            "type": e.type,
            "description": e.description,
            "track_ids": list(e.track_ids),
            "timestamp": e.timestamp,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(e.timestamp)),
            "duration": round(e.duration, 1),
            "zone": e.zone,
            "score": round(self.score, 1),
            "level": self.level,
            "reasons": self.reasons,
            "camera": self.camera,
            "snapshot": self.snapshot,
            "acknowledged": self.acknowledged,
        }
