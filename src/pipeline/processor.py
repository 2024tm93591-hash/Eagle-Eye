"""The real-time video analytics pipeline.

frame -> person detection + tracking -> gender estimation -> activity recognition
      -> scene / interaction analysis -> threat assessment -> alerts + annotated stream
"""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

from ..activity.recognizer import ActivityRecognizer
from ..config import resolve_path
from ..detection.detector import build_detector
from ..gender.classifier import build_gender, crop_person
from ..scene.analyzer import SceneAnalyzer
from ..storage.db import EventStore
from ..threat.alerts import AlertManager
from ..threat.assessment import ThreatAssessor
from ..tracking.track_state import TrackManager
from .draw import annotate
from .simulator import SimulatedGender, SimulatedSource

log = logging.getLogger("threatvision.pipeline")
STAGES = ["detection", "gender", "activity", "scene", "threat", "render"]


def open_source(uri):
    if uri == "simulated":
        return SimulatedSource(), False
    if isinstance(uri, int) or (isinstance(uri, str) and uri.isdigit()):
        return cv2.VideoCapture(int(uri)), True
    p = resolve_path(uri) if not str(uri).startswith(("rtsp://", "http://", "https://")) else uri
    return cv2.VideoCapture(str(p)), str(uri).startswith(("rtsp://", "http://", "https://"))


class Analytics:
    """Synchronous analytics core (no threading, no I/O). Used by the live processor,
    the evaluation scripts and the tests."""

    def __init__(self, cfg: dict, detector_backend=None, activity_backend=None, gender_backend=None):
        self.cfg = cfg
        simulated = cfg["source"]["uri"] == "simulated"
        self.detector = build_detector(cfg, detector_backend)
        self.gender = SimulatedGender() if simulated and gender_backend is None else build_gender(cfg, gender_backend)
        self.activity = ActivityRecognizer(cfg, activity_backend)
        self.scene = SceneAnalyzer(cfg)
        self.assessor = ThreatAssessor(cfg, cfg["source"].get("camera_name", ""))
        self.tracks = TrackManager()
        self.timing = {s: 0.0 for s in STAGES}          # EMA of stage latency (ms)
        self.active_tracks = []
        self.objects = []

    def _t(self, stage, t0):
        ms = (time.perf_counter() - t0) * 1000
        self.timing[stage] = ms if self.timing[stage] == 0 else 0.9 * self.timing[stage] + 0.1 * ms
        return time.perf_counter()

    def process(self, frame, frame_idx: int, t: float, meta: dict | None = None):
        """Returns (active_alerts, new_alerts)."""
        g = self.cfg["gender"]
        t0 = time.perf_counter()
        res = self.detector(frame, frame_idx, meta)
        tracks = self.tracks.update(t, res.persons)
        self.active_tracks, self.objects = tracks, res.objects
        t0 = self._t("detection", t0)

        if self.gender is not None:
            n = max(1, g.get("every_n_frames", 3))
            for tr in tracks:
                if (frame_idx + tr.track_id) % n or tr.gender_votes >= 30:
                    continue
                if isinstance(self.gender, SimulatedGender):
                    self.gender.meta = meta or {}
                    p = self.gender.predict_track(tr.track_id)
                else:
                    p = self.gender.predict(crop_person(frame, tr.last_det.bbox))
                if p is not None:
                    tr.add_gender(np.asarray(p, float), g["min_votes"], g["min_confidence"])
        t0 = self._t("gender", t0)

        self.activity.update(tracks, t)
        t0 = self._t("activity", t0)
        events = self.scene.update(tracks, res.objects, t, frame.shape)
        t0 = self._t("scene", t0)
        active, new = self.assessor.assess(events, now=t)
        self._t("threat", t0)
        return active, new


class VideoProcessor(threading.Thread):
    """Background thread: reads the source in real time, runs Analytics, publishes results."""

    def __init__(self, cfg: dict, store: EventStore, alerts: AlertManager):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.store, self.alerts = store, alerts
        self.lock = threading.Lock()
        self.new_frame = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.running = True
        self.pending_source = None
        self.stats: dict = {}
        self.fps_hist: deque = deque(maxlen=30)
        self.frame_idx = 0
        self._build(cfg["source"]["uri"])

    def _build(self, uri):
        self.cfg["source"]["uri"] = uri
        self.analytics = Analytics(self.cfg)
        self.cap, self.is_live = open_source(uri)
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not self.src_fps or self.src_fps > 120:
            self.src_fps = 25.0
        self.frame_idx = 0
        self.uri = uri
        log.info("Source %s opened (%.1f fps) | detector: %s | activity: %s", uri, self.src_fps,
                 self.analytics.detector.name, self.analytics.activity.name)

    def switch_source(self, uri):
        self.pending_source = uri

    def stop(self):
        self.running = False

    def run(self):
        last_stats_db = 0.0
        t_start = time.time()
        while self.running:
            if self.pending_source is not None:
                uri, self.pending_source = self.pending_source, None
                self.cap.release()
                self._build(uri)
                t_start = time.time()
            loop0 = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                if not self.is_live and self.cfg["source"].get("loop_video", True):
                    self.cap.release()
                    self.cap, _ = open_source(self.uri)
                    continue
                time.sleep(0.5)
                continue
            rw = self.cfg["source"].get("resize_width", 0)
            if rw and frame.shape[1] != rw:
                s = rw / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s)
                meta = getattr(self.cap, "last_meta", None)
                if meta and s != 1:
                    meta = _scale_meta(meta, s)
            else:
                meta = getattr(self.cap, "last_meta", None)
            t = time.time() if self.is_live else self.frame_idx / self.src_fps
            a = self.analytics
            active, new = a.process(frame, self.frame_idx, t, meta)
            t0 = time.perf_counter()
            fps = float(np.mean(self.fps_hist)) if self.fps_hist else 0.0
            vis = annotate(frame, a.active_tracks, a.objects, active, a.scene.zone_pixels(frame.shape),
                           a.scene.zones, self.cfg["source"].get("camera_name", ""), fps, len(a.active_tracks))
            for al in new:
                self.alerts.raise_alert(al, vis)
            ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["dashboard"].get("jpeg_quality", 80)])
            a._t("render", t0)
            proc = time.perf_counter() - loop0
            self.fps_hist.append(1.0 / max(proc, 1e-6))
            with self.new_frame:
                self.latest_jpeg = jpg.tobytes()
                self.new_frame.notify_all()
            self._update_stats(active, fps, proc, t_start)
            if time.time() - last_stats_db > 5:
                s = self.stats
                self.store.add_stats(s["camera"], s["count"], s["genders"].get("male", 0),
                                     s["genders"].get("female", 0), s["genders"].get("unknown", 0),
                                     s["activities"], s["fps"])
                last_stats_db = time.time()
            self.frame_idx += 1
            if not self.is_live:  # play files / simulation at their native speed
                time.sleep(max(0.0, 1.0 / self.src_fps - (time.perf_counter() - loop0)))

    def _update_stats(self, active, fps, proc, t_start):
        a = self.analytics
        tracks = a.active_tracks
        summ = a.scene.summary
        self.stats = {
            "camera": self.cfg["source"].get("camera_name", ""),
            "source": str(self.uri),
            "count": len(tracks),
            "unique_total": a.tracks.total_unique,
            "genders": dict(Counter(tr.gender for tr in tracks)),
            "activities": dict(Counter(tr.activity for tr in tracks)),
            "scene_text": summ.get("text", ""),
            "fps": round(fps, 1),
            "latency_ms": round(proc * 1000, 1),
            "stage_ms": {k: round(v, 2) for k, v in a.timing.items()},
            "models": {"detector": a.detector.name,
                       "gender": getattr(a.gender, "name", "disabled") if a.gender else "disabled",
                       "activity": a.activity.name},
            "persons": [{"id": tr.track_id, "gender": tr.gender, "gender_conf": round(tr.gender_conf, 2),
                         "activity": tr.activity, "activity_conf": round(tr.activity_conf, 2),
                         "dwell": round(tr.dwell, 1), "zones": list(tr.zone_enter)} for tr in tracks],
            "active_threats": [x.to_dict() for x in active if x.level != "LOW"][:10],
            "uptime": round(time.time() - t_start, 0),
            "frame": self.frame_idx,
        }


def _scale_meta(meta, s):
    out = dict(meta)
    out["persons"] = [dict(p, bbox=p["bbox"] * s, keypoints=np.concatenate([p["keypoints"][:, :2] * s,
                                                                          p["keypoints"][:, 2:]], 1))
                      for p in meta["persons"]]
    out["objects"] = [dict(o, bbox=o["bbox"] * s) for o in meta["objects"]]
    return out
