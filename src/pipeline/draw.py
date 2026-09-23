"""Draws boxes, skeletons, counts and active threats on top of a frame."""
from __future__ import annotations

import time
from collections import Counter

import cv2

from ..core import SKELETON

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (230, 230, 230)

# colours are BGR
LEVEL_COLOR = {"LOW": (160, 160, 160), "MEDIUM": (0, 200, 255), "HIGH": (0, 120, 255), "CRITICAL": (40, 40, 230)}
ACTIVITY_COLOR = {"standing": (200, 200, 200), "walking": (120, 220, 120), "running": (0, 200, 255),
                  "fighting": (60, 60, 240), "falling": (230, 80, 230)}
GENDER_LETTER = {"male": "M", "female": "F", "unknown": "?"}


def draw_label(img, text, origin, color, scale=0.5):
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(origin[0]), int(origin[1])
    cv2.rectangle(img, (x, y - h - baseline - 2), (x + w + 4, y + 1), color, -1)
    cv2.putText(img, text, (x + 2, y - baseline), FONT, scale, (20, 20, 20), 1, cv2.LINE_AA)


def worst_level_per_person(alerts) -> dict[int, str]:
    """For everyone involved in a MEDIUM or worse alert, the most serious level they're part of."""
    worst: dict[int, str] = {}
    levels = list(LEVEL_COLOR)
    for alert in alerts:
        if alert.level == "LOW":
            continue
        for tid in alert.event.track_ids:
            if tid not in worst or levels.index(alert.level) > levels.index(worst[tid]):
                worst[tid] = alert.level
    return worst


def annotate(frame, tracks, objects, active_alerts, camera: str, fps: float):
    img = frame.copy()
    threat_level = worst_level_per_person(active_alerts)

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        draw_label(img, obj.cls, (x1, y1 - 2), (255, 220, 120), 0.4)

    for track in tracks:
        det = track.last_det
        x1, y1, x2, y2 = det.bbox.astype(int)
        in_threat = track.track_id in threat_level
        # people involved in an alert are drawn in the alert colour
        color = LEVEL_COLOR[threat_level[track.track_id]] if in_threat else ACTIVITY_COLOR.get(track.activity, WHITE)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if in_threat else 2)

        if det.keypoints is not None:
            kp = det.keypoints
            for a, b in SKELETON:
                if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                    cv2.line(img, tuple(kp[a, :2].astype(int)), tuple(kp[b, :2].astype(int)),
                             (255, 255, 0), 1, cv2.LINE_AA)
        draw_label(img, f"#{track.track_id} {GENDER_LETTER.get(track.gender, '?')} {track.activity}",
                   (x1, y1 - 2), color, 0.45)

    draw_header(img, tracks, camera, fps)

    # the four most serious threats, listed from the bottom of the frame upwards
    h = img.shape[0]
    y = h - 10
    for alert in active_alerts[:4]:
        draw_label(img, f"{alert.level}: {alert.event.description}", (8, y), LEVEL_COLOR[alert.level], 0.5)
        y -= 24
        if alert.event.location is not None and alert.level in ("HIGH", "CRITICAL"):
            cv2.circle(img, tuple(int(v) for v in alert.event.location), 6, LEVEL_COLOR[alert.level], -1)
    return img


def draw_header(img, tracks, camera, fps):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 26), (25, 25, 25), -1)
    cv2.putText(img, f"{camera}   {time.strftime('%Y-%m-%d %H:%M:%S')}", (8, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)

    genders = Counter(track.gender for track in tracks)
    counts = (f"People: {len(tracks)}   Male: {genders.get('male', 0)}   "
              f"Female: {genders.get('female', 0)}   FPS: {fps:.1f}")
    (text_w, _), _ = cv2.getTextSize(counts, FONT, 0.5, 1)
    cv2.putText(img, counts, (w - text_w - 8, 18), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
