"""Activity recognition stage: updates `activity` of every active track."""
from __future__ import annotations

import logging

import numpy as np

from ..config import resolve_path
from .features import track_features
from .rule_based import RuleBasedActivity

log = logging.getLogger(__name__)


def pick_label(probs, classes, fight_confidence: float = 0.0) -> tuple[str, float]:
    """The most likely class, except that an unsure "fighting" gives way to the next guess.
    A fight raises a serious alert, so it needs more evidence than the other classes."""
    ranked = np.argsort(-probs)
    best = ranked[0]
    if classes[best] == "fighting" and probs[best] < fight_confidence:
        best = ranked[1]
    return classes[best], float(probs[best])


class LearnedActivity:
    """Wraps a trained LSTM / TCN checkpoint produced by scripts/train_activity.py."""

    def __init__(self, cfg: dict, kind: str):
        import torch

        from .models import build_model
        self.torch = torch
        path = resolve_path(cfg["activity"][f"{kind}_model"])
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.window = ckpt.get("window", cfg["activity"]["window"])
        self.fps = ckpt.get("fps") or cfg["activity"]["sequence_fps"]
        self.mean = np.asarray(ckpt["mean"], np.float32)
        self.std = np.asarray(ckpt["std"], np.float32)
        self.fight_confidence = cfg["activity"].get("fight_min_confidence", 0.0)
        self.model = build_model(kind, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.name = {"lstm": "Bi-LSTM + attention (pose sequence)", "tcn": "TCN (pose sequence)"}[kind]

    def predict_batch(self, tracks) -> list[tuple[str, float]]:
        x = np.stack([(track_features(tr, self.window, self.fps) - self.mean) / self.std for tr in tracks])
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(self.torch.from_numpy(x.astype(np.float32))), 1).numpy()
        return [pick_label(probs, self.classes, self.fight_confidence) for probs in p]


class ActivityRecognizer:
    def __init__(self, cfg: dict, backend: str | None = None):
        self.cfg = cfg
        backend = backend or cfg["activity"]["backend"]
        self.rules = RuleBasedActivity(cfg)
        self.model = None
        if backend in ("lstm", "tcn"):
            try:
                self.model = LearnedActivity(cfg, backend)
            except Exception as exc:
                log.warning("Could not load %s activity model (%s) - using rule-based classifier", backend, exc)
        self.name = self.model.name if self.model else self.rules.name
        self.min_frames = 8

    @staticmethod
    def update_neighbours(tracks):
        """Nearest-person distance (in body heights) for every active track."""
        if not tracks:
            return
        c = np.array([tr.last_det.center for tr in tracks])
        h = np.array([tr.body_height() for tr in tracks])
        for i, tr in enumerate(tracks):
            if len(tracks) == 1:
                tr.nbr_dist.append(99.0)
                continue
            d = np.linalg.norm(c - c[i], axis=1) / ((h + h[i]) / 2)
            d[i] = np.inf
            tr.nbr_dist.append(float(d.min()))

    def update(self, tracks, t: float):
        self.update_neighbours(tracks)
        ready = [tr for tr in tracks if len(tr.history) >= self.min_frames]
        preds = {}
        # the rule-based model always runs: it also maintains fall timing used by the scene module
        for tr in ready:
            preds[tr.track_id] = self.rules.predict(tr, t)[:2]
        if self.model is not None and ready:
            for tr, p in zip(ready, self.model.predict_batch(ready)):
                preds[tr.track_id] = p
                if p[0] == "falling" and tr.fall_time is None:
                    tr.fall_time = t
                elif p[0] != "falling":
                    tr.fall_time = None
        smoothing = self.cfg["activity"].get("smoothing", 5)
        for tr in ready:
            if tr.activity_hist.maxlen != smoothing:
                tr.activity_hist = type(tr.activity_hist)(tr.activity_hist, maxlen=smoothing)
            label, conf = preds[tr.track_id]
            tr.set_activity(label, conf, t)
