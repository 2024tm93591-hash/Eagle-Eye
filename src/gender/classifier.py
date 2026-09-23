"""Gender estimation back-ends.

Each back-end returns, for a person crop, a probability vector [p_male, p_female] or None
when it cannot decide (e.g. no face visible). Predictions are accumulated per track over
time (TrackState.add_gender) so a single bad frame does not flip the label.

Note: appearance-based gender estimation is probabilistic, binary and can be biased; it is
reported as an aggregate statistic ("estimated gender") and never used to score threats.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ..config import resolve_path

log = logging.getLogger(__name__)


class FaceDnnGender:
    """OpenCV res10-SSD face detector + Levi & Hassner (2015) gender CaffeNet."""
    name = "Face DNN (Levi-Hassner CaffeNet)"
    MEAN = (78.4263377603, 87.7689143744, 114.895847746)

    def __init__(self, cfg: dict):
        g = cfg["gender"]
        self.face_net = cv2.dnn.readNetFromCaffe(str(resolve_path(g["face_proto"])),
                                                 str(resolve_path(g["face_model"])))
        self.gender_net = cv2.dnn.readNetFromCaffe(str(resolve_path(g["gender_proto"])),
                                                   str(resolve_path(g["gender_model"])))

    def find_face(self, crop: np.ndarray):
        h, w = crop.shape[:2]
        head = crop[: max(1, int(h * 0.45))]          # face is in the upper part of a person box
        hh, hw = head.shape[:2]
        if hh < 20 or hw < 20:
            return None
        blob = cv2.dnn.blobFromImage(cv2.resize(head, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        det = self.face_net.forward()[0, 0]
        best = None
        for row in det:
            if row[2] > 0.6 and (best is None or row[2] > best[2]):
                best = row
        if best is None:
            return None
        x1, y1, x2, y2 = (best[3:7] * [hw, hh, hw, hh]).astype(int)
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(hw, x2 + pad), min(hh, y2 + pad)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return head[y1:y2, x1:x2]

    def predict(self, crop: np.ndarray):
        face = self.find_face(crop)
        if face is None:
            return None
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MEAN, swapRB=False)
        self.gender_net.setInput(blob)
        return self.gender_net.forward()[0].astype(float)  # [male, female]


class BodyCnnGender:
    """MobileNetV3 on full-body crops (trained with scripts/train_gender.py)."""
    name = "Body CNN (MobileNetV3-Small)"

    def __init__(self, cfg: dict):
        import torch

        from .body_model import build_body_model
        self.torch = torch
        path = resolve_path(cfg["gender"]["body_model"])
        self.model = build_body_model(pretrained=False)
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()

    def predict(self, crop: np.ndarray):
        from .body_model import preprocess_bgr
        if crop.shape[0] < 32 or crop.shape[1] < 16:
            return None
        with self.torch.no_grad():
            return self.torch.softmax(self.model(preprocess_bgr([crop])), 1)[0].numpy()


class HybridGender:
    name = "Hybrid (face DNN -> body CNN)"

    def __init__(self, face, body):
        self.face, self.body = face, body

    def predict(self, crop):
        p = self.face.predict(crop) if self.face else None
        if p is None and self.body is not None:
            p = self.body.predict(crop)
        return p


def build_gender(cfg: dict, backend: str | None = None):
    backend = backend or cfg["gender"]["backend"]
    if backend == "none":
        return None
    face = body = None
    if backend in ("face_dnn", "hybrid"):
        try:
            face = FaceDnnGender(cfg)
        except Exception as exc:
            log.warning("Face gender model unavailable (%s). Run scripts/download_models.py", exc)
    if backend in ("body_cnn", "hybrid"):
        try:
            body = BodyCnnGender(cfg)
        except Exception as exc:
            log.warning("Body gender model unavailable (%s). Train it with scripts/train_gender.py", exc)
    if backend == "face_dnn":
        return face
    if backend == "body_cnn":
        return body
    if face and body:
        return HybridGender(face, body)
    return face or body


def crop_person(frame: np.ndarray, bbox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]
