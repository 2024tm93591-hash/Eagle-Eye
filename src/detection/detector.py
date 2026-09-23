"""Person detection + multi-object tracking back-ends.

* YoloPoseDetector : Ultralytics YOLOv8-pose -> boxes + 17 keypoints, ByteTrack IDs.
                     Optionally a second YOLOv8 detector for bags (abandoned objects).
* HogDetector      : OpenCV HOG + linear SVM people detector with IoU tracker (classic baseline).
* SimulatedDetector: reads ground truth from the synthetic scene (pipeline self-test).
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from ..config import resolve_path
from ..core import Detection, FrameResult
from ..tracking.iou_tracker import IOUTracker, iou_matrix

log = logging.getLogger(__name__)

COCO_BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy non-maximum suppression."""
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        if order:
            ious = iou_matrix(boxes[[i]], boxes[order])[0]
            order = [j for j, v in zip(order, ious) if v < iou_thr]
    return keep


class BaseDetector:
    name = "base"
    has_keypoints = False

    def __call__(self, frame: np.ndarray, frame_idx: int, meta: dict | None = None) -> FrameResult:
        raise NotImplementedError


class YoloPoseDetector(BaseDetector):
    has_keypoints = True

    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # imported lazily so the rest works without it

        d = cfg["detection"]
        self.cfg = d
        weights = str(resolve_path(d["yolo_model"]))
        self.model = YOLO(weights)
        self.name = f"YOLOv8-pose ({resolve_path(d['yolo_model']).stem})"
        self.obj_model = None
        if d.get("object_model"):
            try:
                self.obj_model = YOLO(str(resolve_path(d["object_model"])))
            except Exception as exc:  # object model is optional
                log.warning("Object model not loaded (%s) - abandoned-object detection disabled", exc)
        self._last_objects: list[Detection] = []

    def __call__(self, frame, frame_idx, meta=None) -> FrameResult:
        d = self.cfg
        res = self.model.track(frame, persist=True, classes=[0], conf=d["conf_threshold"],
                               iou=d["iou_threshold"], tracker=d.get("tracker", "bytetrack.yaml"),
                               device=d.get("device") or None, imgsz=d.get("imgsz", 640), verbose=False)[0]
        persons: list[Detection] = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
            kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
            for i in range(len(xyxy)):
                if ids[i] < 0:
                    continue  # not yet confirmed by the tracker
                persons.append(Detection(bbox=xyxy[i], conf=float(conf[i]), track_id=int(ids[i]),
                                         keypoints=None if kps is None else kps[i]))
        if self.obj_model is not None and frame_idx % max(1, d.get("object_every_n_frames", 5)) == 0:
            r = self.obj_model.predict(frame, classes=list(COCO_BAG_CLASSES), conf=0.35,
                                       device=d.get("device") or None, verbose=False)[0]
            self._last_objects = []
            if r.boxes is not None:
                for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy().astype(int)):
                    self._last_objects.append(Detection(bbox=b, conf=float(c), cls=COCO_BAG_CLASSES[k]))
        return FrameResult(frame_idx, time.time(), persons, list(self._last_objects))


class HogDetector(BaseDetector):
    """Dalal-Triggs HOG + SVM. Fast on CPU, no keypoints -> activity uses box motion only."""
    name = "HOG + SVM (baseline)"

    def __init__(self, cfg: dict):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker = IOUTracker()
        self.conf = cfg["detection"].get("hog_threshold", 0.3)

    def detect(self, frame):
        scale = 640.0 / max(frame.shape[1], 1)
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        s = scale if scale < 1 else 1.0
        rects, weights = self.hog.detectMultiScale(small, hitThreshold=min(0.0, self.conf), winStride=(8, 8),
                                                    padding=(8, 8), scale=1.05)
        boxes, scores = [], []
        for (x, y, w, h), wt in zip(rects, np.ravel(weights) if len(rects) else []):
            if wt >= self.conf:
                # the 64x128 HOG window contains a margin around the person: shrink to a tight box
                px, py = 0.15 * w, 0.06 * h
                boxes.append([(x + px) / s, (y + py) / s, (x + w - px) / s, (y + h - py) / s])
                scores.append(float(wt))
        keep = nms(np.array(boxes, float).reshape(-1, 4), np.array(scores), 0.45)
        return [Detection(bbox=np.array(boxes[i], dtype=float), conf=min(1.0, scores[i] / 2.0)) for i in keep]

    def __call__(self, frame, frame_idx, meta=None) -> FrameResult:
        return FrameResult(frame_idx, time.time(), self.tracker.update(self.detect(frame)))


class SimulatedDetector(BaseDetector):
    """Returns the synthetic scene's ground truth (with small noise) - lets the whole
    pipeline, alerting and dashboard run without a camera or deep-learning models."""
    name = "Simulated ground truth"
    has_keypoints = True

    def __init__(self, cfg: dict, noise: float = 1.0):
        self.noise = noise
        self.rng = np.random.default_rng(0)

    def __call__(self, frame, frame_idx, meta=None) -> FrameResult:
        meta = meta or {}
        persons = []
        for p in meta.get("persons", []):
            kp = p["keypoints"].copy()
            kp[:, :2] += self.rng.normal(0, self.noise, size=(17, 2))
            persons.append(Detection(bbox=p["bbox"].copy(), conf=0.95, track_id=p["id"], keypoints=kp))
        objects = [Detection(bbox=o["bbox"].copy(), conf=0.9, cls=o["cls"]) for o in meta.get("objects", [])]
        return FrameResult(frame_idx, time.time(), persons, objects)


def build_detector(cfg: dict, backend: str | None = None) -> BaseDetector:
    if backend is None:
        backend = "simulated" if cfg["source"]["uri"] == "simulated" else cfg["detection"]["backend"]
    if backend == "simulated":
        return SimulatedDetector(cfg)
    if backend == "yolo":
        try:
            return YoloPoseDetector(cfg)
        except ImportError:
            log.error("ultralytics is not installed - falling back to the HOG baseline detector")
    return HogDetector(cfg)
