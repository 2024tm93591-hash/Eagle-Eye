"""The real-time pipeline.

Each frame goes through: person detection + tracking -> gender estimation -> activity
recognition -> scene analysis -> threat assessment. The annotated result is published for
the dashboard and any new alerts are handed to the AlertManager.
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
from ..threat.alerts import AlertManager
from ..threat.assessment import ThreatAssessor
from ..tracking.track_state import TrackManager
from .draw import annotate

log = logging.getLogger("threatvision.pipeline")

STAGES = ["detection", "gender", "activity", "scene", "threat", "render"]
STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def open_source(uri):
    """Returns (capture, is_live). Webcams and network streams are live, video files are not."""
    uri = str(uri)
    if uri.isdigit():
        return cv2.VideoCapture(int(uri)), True
    if uri.startswith(STREAM_PREFIXES):
        return cv2.VideoCapture(uri), True
    return cv2.VideoCapture(str(resolve_path(uri))), False


class Analytics:
    """Runs every analysis stage on one frame.

    There are no threads or I/O in here, so the evaluation script and the tests can use it
    directly on their own frames.
    """

    def __init__(self, cfg: dict, detector_backend=None, activity_backend=None, gender_backend=None):
        self.cfg = cfg
        self.detector = build_detector(cfg, detector_backend)
        self.gender = build_gender(cfg, gender_backend)
        self.activity = ActivityRecognizer(cfg, activity_backend)
        self.timing = {stage: 0.0 for stage in STAGES}   # smoothed milliseconds per stage
        self.reset()

    def reset(self):
        """Forget everyone seen so far, e.g. after switching to another video source."""
        self.detector.reset()
        self.tracks = TrackManager()
        self.scene = SceneAnalyzer(self.cfg)
        self.assessor = ThreatAssessor(self.cfg, self.cfg["source"].get("camera_name", ""))
        self.active_tracks = []
        self.objects = []

    def record_time(self, stage: str, started: float) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1000
        previous = self.timing[stage]
        self.timing[stage] = elapsed_ms if previous == 0 else 0.9 * previous + 0.1 * elapsed_ms
        return time.perf_counter()

    def process(self, frame, frame_idx: int, t: float):
        """Analyse one frame. Returns (alerts active in this frame, alerts that are new)."""
        started = time.perf_counter()
        result = self.detector(frame, frame_idx)
        self.active_tracks = self.tracks.update(t, result.persons)
        self.objects = result.objects
        started = self.record_time("detection", started)

        self._estimate_gender(frame, frame_idx)
        started = self.record_time("gender", started)

        self.activity.update(self.active_tracks, t)
        started = self.record_time("activity", started)

        events = self.scene.update(self.active_tracks, self.objects, t)
        started = self.record_time("scene", started)

        active, new = self.assessor.assess(events, now=t)
        self.record_time("threat", started)
        return active, new

    def _estimate_gender(self, frame, frame_idx):
        if self.gender is None:
            return
        settings = self.cfg["gender"]
        every = max(1, settings.get("every_n_frames", 3))
        for track in self.active_tracks:
            # spread the work over several frames, and stop once a person has plenty of votes
            if (frame_idx + track.track_id) % every or track.gender_votes >= 30:
                continue
            probs = self.gender.predict(crop_person(frame, track.last_det.bbox))
            if probs is not None:
                track.add_gender(np.asarray(probs, float), settings["min_votes"], settings["min_confidence"])


class VideoProcessor(threading.Thread):
    """Background thread that reads the video source, runs Analytics and publishes the results."""

    def __init__(self, cfg: dict, alerts: AlertManager):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.alerts = alerts
        self.analytics = Analytics(cfg)
        self.uri = cfg["source"]["uri"]
        self.cap = None
        self.is_live = True
        self.source_fps = 25.0
        self.frame_idx = 0
        self.started_at = time.time()
        self.running = True
        self.feed_on = True
        self.pending_source = None
        self.fps_history: deque = deque(maxlen=30)

        self.new_frame = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.stats: dict = {}
        self._update_stats(0.0)

    # called from the web server's threads; the processing thread picks the change up
    def switch_source(self, uri):
        self.pending_source = uri

    def set_feed(self, on: bool):
        self.feed_on = on
        self.stats = dict(self.stats, feed_on=on)

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            if self.pending_source is not None:
                self.uri, self.pending_source = self.pending_source, None
                self._close()

            if not self.feed_on:
                self._close()
                time.sleep(0.2)
                continue
            if self.cap is None:
                self._open()

            loop_start = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                self._handle_read_failure()
                continue

            frame = self._resize(frame)
            t = time.time() if self.is_live else self.frame_idx / self.source_fps
            active, new = self.analytics.process(frame, self.frame_idx, t)
            self._publish(frame, active, new)

            elapsed = time.perf_counter() - loop_start
            self.fps_history.append(1.0 / max(elapsed, 1e-6))
            self._update_stats(elapsed, active)
            self.frame_idx += 1
            if not self.is_live:
                # play video files at their normal speed instead of as fast as we can
                time.sleep(max(0.0, 1.0 / self.source_fps - (time.perf_counter() - loop_start)))
        self._close()

    def _open(self):
        self.cap, self.is_live = open_source(self.uri)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if 0 < fps <= 120 else 25.0
        self.frame_idx = 0
        self.started_at = time.time()
        a = self.analytics
        log.info("Source %s opened (%.1f fps) | detector: %s | activity: %s",
                 self.uri, self.source_fps, a.detector.name, a.activity.name)

    def _close(self):
        """Release the camera or file and clear everything we knew about the scene."""
        if self.cap is None:
            return
        self.cap.release()
        self.cap = None
        self.analytics.reset()
        self.fps_history.clear()
        self._update_stats(0.0)
        log.info("Source %s closed", self.uri)

    def _handle_read_failure(self):
        if not self.is_live and self.cfg["source"].get("loop_video", True):
            self.cap.release()
            self.cap, _ = open_source(self.uri)   # start the file again from the beginning
        else:
            time.sleep(0.5)

    def _resize(self, frame):
        width = self.cfg["source"].get("resize_width", 0)
        if not width or frame.shape[1] == width:
            return frame
        scale = width / frame.shape[1]
        return cv2.resize(frame, None, fx=scale, fy=scale)

    def _current_fps(self) -> float:
        return float(np.mean(self.fps_history)) if self.fps_history else 0.0

    def _publish(self, frame, active, new):
        started = time.perf_counter()
        a = self.analytics
        image = annotate(frame, a.active_tracks, a.objects, active,
                         self.cfg["source"].get("camera_name", ""), self._current_fps())
        for alert in new:
            self.alerts.raise_alert(alert, image)

        quality = self.cfg["dashboard"].get("jpeg_quality", 80)
        _, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        a.record_time("render", started)
        with self.new_frame:
            self.latest_jpeg = jpg.tobytes()
            self.new_frame.notify_all()

    def _update_stats(self, frame_seconds: float, active=()):
        a = self.analytics
        tracks = a.active_tracks
        genders = Counter(track.gender for track in tracks)
        persons = [{
            "id": track.track_id,
            "gender": track.gender,
            "gender_conf": round(track.gender_conf, 2),
            "activity": track.activity,
            "activity_conf": round(track.activity_conf, 2),
            "dwell": round(track.dwell, 1),
        } for track in tracks]

        self.stats = {
            "camera": self.cfg["source"].get("camera_name", ""),
            "source": str(self.uri),
            "feed_on": self.feed_on,
            "count": len(tracks),
            "male": genders.get("male", 0),
            "female": genders.get("female", 0),
            "unknown": genders.get("unknown", 0),
            "unique_total": a.tracks.total_unique,
            "activities": dict(Counter(track.activity for track in tracks)),
            "scene_text": a.scene.summary.get("text", ""),
            "fps": round(self._current_fps(), 1),
            "latency_ms": round(frame_seconds * 1000, 1),
            "stage_ms": {stage: round(ms, 2) for stage, ms in a.timing.items()},
            "models": {
                "detector": a.detector.name,
                "gender": getattr(a.gender, "name", "disabled") if a.gender else "disabled",
                "activity": a.activity.name,
            },
            "persons": persons,
            "active_threats": [alert.to_dict() for alert in active if alert.level != "LOW"][:10],
            "uptime": round(time.time() - self.started_at),
            "frame": self.frame_idx,
        }
