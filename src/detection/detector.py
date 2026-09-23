"""Person detection and tracking.

YoloPoseDetector gives person boxes, 17 body keypoints and ByteTrack ids, and can run a
second YOLOv8 model to spot bags. HogDetector is the classic OpenCV HOG + SVM people
detector, kept as a CPU-only baseline. It has no keypoints and uses our own IoU tracker.
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

# COCO class ids we treat as bags for the abandoned-object check
COCO_BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy non-maximum suppression. Returns the indices to keep."""
    order = list(np.argsort(-scores))
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        if order:
            ious = iou_matrix(boxes[[best]], boxes[order])[0]
            order = [i for i, overlap in zip(order, ious) if overlap < iou_thr]
    return keep


class YoloPoseDetector:
    has_keypoints = True

    def __init__(self, cfg: dict):
        from ultralytics import YOLO   # imported here so the HOG baseline works without it

        self.cfg = cfg["detection"]
        weights = resolve_path(self.cfg["yolo_model"])
        self.model = YOLO(str(weights))
        self.name = f"YOLOv8-pose ({weights.stem})"

        self.bag_model = None
        if self.cfg.get("object_model"):
            try:
                self.bag_model = YOLO(str(resolve_path(self.cfg["object_model"])))
            except Exception as exc:
                log.warning("Object model not loaded (%s) - abandoned-object detection disabled", exc)
        self._last_bags: list[Detection] = []

    def reset(self):
        # dropping the predictor also drops ByteTrack's state, so ids start fresh
        self.model.predictor = None
        self._last_bags = []

    def __call__(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        c = self.cfg
        device = c.get("device") or None
        result = self.model.track(frame, persist=True, classes=[0], conf=c["conf_threshold"],
                                  iou=c["iou_threshold"], tracker=c.get("tracker", "bytetrack.yaml"),
                                  device=device, imgsz=c.get("imgsz", 640), verbose=False)[0]

        persons = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
            keypoints = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for i in range(len(xyxy)):
                if ids[i] < 0:
                    continue   # the tracker hasn't confirmed this person yet
                persons.append(Detection(bbox=xyxy[i], conf=float(conf[i]), track_id=int(ids[i]),
                                         keypoints=None if keypoints is None else keypoints[i]))

        # bags don't move much, so looking for them every few frames is enough
        if self.bag_model is not None and frame_idx % max(1, c.get("object_every_n_frames", 5)) == 0:
            self._last_bags = self._find_bags(frame, device)
        return FrameResult(frame_idx, time.time(), persons, list(self._last_bags))

    def _find_bags(self, frame, device) -> list[Detection]:
        result = self.bag_model.predict(frame, classes=list(COCO_BAG_CLASSES), conf=0.35,
                                        device=device, verbose=False)[0]
        if result.boxes is None:
            return []
        bags = []
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int)):
            bags.append(Detection(bbox=box, conf=float(conf), cls=COCO_BAG_CLASSES[cls]))
        return bags


class HogDetector:
    """Dalal-Triggs HOG + linear SVM. Fast on a CPU, but no keypoints."""
    name = "HOG + SVM (baseline)"
    has_keypoints = False

    def __init__(self, cfg: dict):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.tracker = IOUTracker()
        self.threshold = cfg["detection"].get("hog_threshold", 0.3)

    def reset(self):
        self.tracker = IOUTracker()

    def detect(self, frame) -> list[Detection]:
        scale = min(1.0, 640.0 / max(frame.shape[1], 1))
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        rects, weights = self.hog.detectMultiScale(small, hitThreshold=min(0.0, self.threshold),
                                                   winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes, scores = [], []
        for (x, y, w, h), weight in zip(rects, np.ravel(weights) if len(rects) else []):
            if weight < self.threshold:
                continue
            # the 64x128 HOG window has a margin around the person, so shrink it to a tight box
            pad_x, pad_y = 0.15 * w, 0.06 * h
            boxes.append([(x + pad_x) / scale, (y + pad_y) / scale,
                          (x + w - pad_x) / scale, (y + h - pad_y) / scale])
            scores.append(float(weight))

        keep = nms(np.array(boxes, float).reshape(-1, 4), np.array(scores), 0.45)
        return [Detection(bbox=np.array(boxes[i], dtype=float), conf=min(1.0, scores[i] / 2.0)) for i in keep]

    def __call__(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        return FrameResult(frame_idx, time.time(), self.tracker.update(self.detect(frame)))


def build_detector(cfg: dict, backend: str | None = None):
    backend = backend or cfg["detection"]["backend"]
    if backend == "yolo":
        try:
            return YoloPoseDetector(cfg)
        except ImportError:
            log.error("ultralytics is not installed - falling back to the HOG detector")
    return HogDetector(cfg)
