"""Frame annotation (boxes, skeletons, zones, threat banners)."""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..core import SKELETON

LEVEL_BGR = {"LOW": (160, 160, 160), "MEDIUM": (0, 200, 255), "HIGH": (0, 120, 255), "CRITICAL": (40, 40, 230)}
ACT_BGR = {"standing": (200, 200, 200), "walking": (120, 220, 120), "running": (0, 200, 255),
           "fighting": (60, 60, 240), "falling": (230, 80, 230)}
G_SHORT = {"male": "M", "female": "F", "unknown": "?"}


def _label(img, text, org, color, scale=0.5):
    (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - b - 2), (x + w + 4, y + 1), color, -1)
    cv2.putText(img, text, (x + 2, y - b), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def annotate(frame, tracks, objects, active_alerts, zones_px, zone_defs, camera: str, fps: float, count: int):
    img = frame.copy()
    # zones
    over = img.copy()
    for poly, z in zip(zones_px, zone_defs):
        cv2.fillPoly(over, [poly], (0, 0, 180) if z["type"] == "restricted" else (180, 120, 0))
    img = cv2.addWeighted(over, 0.18, img, 0.82, 0)
    for poly, z in zip(zones_px, zone_defs):
        cv2.polylines(img, [poly], True, (0, 0, 220) if z["type"] == "restricted" else (220, 160, 0), 2)
        _label(img, z["name"], poly[0] + [4, 18], (180, 180, 255))

    # people involved in serious situations are drawn in the alert colour
    threat_of: dict[int, str] = {}
    for a in active_alerts:
        for tid in a.event.track_ids:
            if a.level in ("HIGH", "CRITICAL", "MEDIUM") and (tid not in threat_of or a.score > 0):
                prev = threat_of.get(tid)
                if prev is None or list(LEVEL_BGR).index(a.level) > list(LEVEL_BGR).index(prev):
                    threat_of[tid] = a.level

    for o in objects:
        x1, y1, x2, y2 = o.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        _label(img, o.cls, (x1, y1 - 2), (255, 220, 120), 0.4)

    for tr in tracks:
        d = tr.last_det
        x1, y1, x2, y2 = d.bbox.astype(int)
        col = LEVEL_BGR[threat_of[tr.track_id]] if tr.track_id in threat_of else ACT_BGR.get(tr.activity, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 if tr.track_id not in threat_of else 3)
        if d.keypoints is not None:
            k = d.keypoints
            for a, b in SKELETON:
                if k[a, 2] > 0.3 and k[b, 2] > 0.3:
                    cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), (255, 255, 0), 1, cv2.LINE_AA)
        _label(img, f"#{tr.track_id} {G_SHORT.get(tr.gender, '?')} {tr.activity}", (x1, y1 - 2), col, 0.45)

    # header
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 26), (25, 25, 25), -1)
    cv2.putText(img, f"{camera}   {time.strftime('%Y-%m-%d %H:%M:%S')}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)
    txt = f"Persons: {count}   FPS: {fps:4.1f}"
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, txt, (w - tw - 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # active threats (top 4)
    y = h - 10
    for a in active_alerts[:4]:
        _label(img, f"{a.level}: {a.event.description}", (8, y), LEVEL_BGR[a.level], 0.5)
        y -= 24
        if a.event.location is not None and a.level in ("HIGH", "CRITICAL"):
            cv2.circle(img, tuple(int(v) for v in a.event.location), 6, LEVEL_BGR[a.level], -1)
    return img
