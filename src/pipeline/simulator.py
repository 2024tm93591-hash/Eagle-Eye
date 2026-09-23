"""Synthetic CCTV scene for self-testing the full pipeline without a camera or trained models.

A scripted 100-second loop plays out walking, running, loitering, a fight followed by a fall
(person down), a restricted-zone intrusion, a chase, an abandoned bag and a group gathering
that disperses in panic. The simulator renders stick-figure frames *and* returns exact ground
truth (boxes, COCO-17 keypoints, gender, activity) so the modules downstream can be exercised
and sanity-checked. Results obtained on this scene are NOT a substitute for evaluation on real
footage - see scripts/evaluate.py.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

W, H, FPS, CYCLE = 960, 540, 20, 100.0

# base pose in units of body height, origin = feet centre, y up is negative
BASE = np.array([
    [0.00, -0.93], [-0.02, -0.95], [0.02, -0.95], [-0.04, -0.94], [0.04, -0.94],   # nose eyes ears
    [-0.11, -0.80], [0.11, -0.80], [-0.13, -0.63], [0.13, -0.63], [-0.13, -0.47], [0.13, -0.47],
    [-0.07, -0.50], [0.07, -0.50], [-0.07, -0.27], [0.07, -0.27], [-0.07, -0.02], [0.07, -0.02],
])


def body_h(y: float) -> float:
    """Perspective: people further away (smaller y) look smaller."""
    return 70 + (y - 200) * 0.45


# (t, x, y, mode) keyframes; mode applies from that keyframe to the next one
SCRIPT = [
    dict(id=1, gender="male", kf=[(0, -40, 430, "walk"), (13, 1000, 430, "walk")]),
    dict(id=2, gender="female", kf=[(5, 1000, 330, "walk"), (20, -40, 330, "walk")]),
    dict(id=3, gender="male", kf=[(6, 1000, 430, "walk"), (11, 620, 290, "stand"), (20, 620, 290, "walk"),
                                  (22, 580, 300, "stand"), (32, 580, 300, "walk"), (34, 630, 285, "stand"),
                                  (46, 630, 285, "walk"), (48, 600, 300, "stand"), (58, 600, 300, "walk"),
                                  (64, 1000, 390, "walk")]),
    dict(id=4, gender="male", kf=[(28, -40, 400, "walk"), (36, 470, 400, "fight"), (44, 470, 400, "run"),
                                  (46.5, -60, 360, "run")]),
    dict(id=5, gender="male", kf=[(28, 1000, 400, "walk"), (36, 540, 400, "fight"), (44, 540, 400, "fall"),
                                  (45, 540, 400, "lie"), (60, 540, 400, "stand"), (62, 540, 400, "walk"),
                                  (72, 1000, 420, "walk")]),
    dict(id=6, gender="female", kf=[(18, 1000, 330, "walk"), (22, 830, 270, "stand"), (36, 830, 270, "walk"),
                                    (40, 1000, 330, "walk")]),
    dict(id=7, gender="male", kf=[(60.8, -60, 480, "run"), (64.8, 1080, 480, "run")]),
    dict(id=8, gender="female", kf=[(60, -40, 480, "run"), (64, 1080, 480, "run")]),
    dict(id=13, gender="female", kf=[(44, -40, 235, "walk"), (48, 120, 235, "stand"), (50, 120, 235, "walk"),
                                     (55, -40, 225, "walk")]),
] + [
    dict(id=9 + i, gender=g, kf=[(70 + i, sx, 460, "walk"), (76 + i * 0.5, gx, gy, "stand"),
                                 (88, gx, gy, "run"), (90.5, ex, ey, "run")])
    for i, (g, sx, gx, gy, ex, ey) in enumerate([
        ("male", -40, 220, 450, -400, 540), ("female", -40, 290, 440, 700, 900),
        ("female", 1000, 250, 490, 500, 1100), ("male", 1000, 320, 470, 1300, 470)])
]
BAG = dict(cls="backpack", start=49.0, end=95.0, x=135, y=240)


def _interp(kf, t):
    for (t0, x0, y0, m0), (t1, x1, y1, _) in zip(kf, kf[1:]):
        if t0 <= t < t1:
            a = (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0), m0, t - t0, (x1 - x0) / (t1 - t0 + 1e-9)
    return None


def _pose(mode, t, local_t, vx, h, facing):
    p = BASE.copy()
    if mode in ("walk", "run"):
        ph = t * (2 * math.pi) * (1.1 if mode == "walk" else 2.2)
        amp = 0.14 if mode == "walk" else 0.28
        s = math.sin(ph)
        p[[15, 16], 0] += [amp * s, -amp * s]
        p[[13, 14], 0] += [amp * s * 0.5, -amp * s * 0.5]
        p[[9, 10], 0] += [-amp * s * 0.8, amp * s * 0.8]
        p[[7, 8], 0] += [-amp * s * 0.4, amp * s * 0.4]
        if mode == "run":
            p[:, 0] += 0.08 * facing * (-p[:, 1])   # lean forward
    elif mode == "fight":
        for k, (w, e) in enumerate([(9, 7), (10, 8)]):
            ph = t * 2 * math.pi * 3.0 + k * math.pi
            reach = 0.08 + 0.30 * max(0.0, math.sin(ph))
            p[w] = [p[5 + k, 0] + facing * reach, -0.78 + 0.05 * math.cos(ph)]
            p[e] = [(p[5 + k, 0] + p[w, 0]) / 2, -0.70]
        p[:, 0] += 0.03 * math.sin(t * 2 * math.pi * 2.0)
    elif mode in ("fall", "lie"):
        a = math.radians(85 * (min(1.0, local_t / 0.8) if mode == "fall" else 1.0))
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = p @ rot.T * np.array([facing, 1])
        p[:, 1] = np.minimum(p[:, 1], -0.01)
    return p * h


def scene_at(t: float) -> dict:
    t = t % CYCLE
    persons = []
    for a in SCRIPT:
        r = _interp(a["kf"], t)
        if r is None:
            continue
        x, y, mode, lt, vx = r
        h = body_h(y)
        facing = 1 if vx >= 0 else -1
        if mode == "fight":
            facing = 1 if a["id"] == 4 else -1
        kp2 = _pose(mode, t, lt, vx, h, facing) + [x, y]
        kps = np.concatenate([kp2, np.ones((17, 1))], 1)
        pad = 0.06 * h
        top = kp2[:, 1].min() - 0.07 * h
        bbox = np.array([kp2[:, 0].min() - pad, top, kp2[:, 0].max() + pad, kp2[:, 1].max() + pad * 0.3])
        if bbox[2] < 0 or bbox[0] > W:
            continue
        act = {"walk": "walking", "run": "running", "stand": "standing", "fight": "fighting",
               "fall": "falling", "lie": "falling"}[mode]
        persons.append(dict(id=a["id"], bbox=bbox, keypoints=kps, gender=a["gender"], activity=act))
    objects = []
    if BAG["start"] <= t < BAG["end"]:
        objects.append(dict(cls=BAG["cls"], bbox=np.array([BAG["x"] - 14, BAG["y"] - 30, BAG["x"] + 14, BAG["y"]],
                                                          dtype=float)))
    return dict(persons=persons, objects=objects, t=t)


def _background():
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        c = 60 + int(70 * y / H)
        bg[y] = (c, c, c - 10)
    cv2.rectangle(bg, (0, 0), (W, 180), (95, 85, 80), -1)                       # back wall
    for x in range(40, W, 160):
        cv2.rectangle(bg, (x, 40), (x + 90, 150), (130, 120, 110), -1)          # windows
    for i in range(0, W, 60):
        cv2.line(bg, (i, 180), (int(W / 2 + (i - W / 2) * 2.5), H), (85, 85, 80), 1)
    return bg


_BG = None


def render(scene: dict) -> np.ndarray:
    global _BG
    if _BG is None:
        _BG = _background()
    img = _BG.copy()
    for o in scene["objects"]:
        x1, y1, x2, y2 = o["bbox"].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 70, 140), -1)
        cv2.rectangle(img, (x1 + 5, y1 - 6), (x2 - 5, y1 + 3), (30, 50, 100), 2)
    for p in sorted(scene["persons"], key=lambda q: q["bbox"][3]):
        k = p["keypoints"][:, :2].astype(int)
        h = p["bbox"][3] - p["bbox"][1]
        col = (170, 120, 60) if p["gender"] == "male" else (120, 80, 170)
        th = max(2, int(h / 22))
        torso = np.array([k[5], k[6], k[12], k[11]])
        cv2.fillConvexPoly(img, torso, col)
        for a, b in [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]:
            cv2.line(img, tuple(k[a]), tuple(k[b]), (50, 50, 60) if a >= 11 else col, th, cv2.LINE_AA)
        cv2.circle(img, tuple(k[0]), max(3, int(h * 0.07)), (150, 180, 220), -1, cv2.LINE_AA)
    return img


class SimulatedSource:
    """Frame source with the same interface as cv2.VideoCapture.read()."""

    def __init__(self, fps: int = FPS):
        self.fps = fps
        self.idx = 0
        self.last_meta: dict = {}

    def read(self):
        t = self.idx / self.fps
        self.idx += 1
        self.last_meta = scene_at(t)
        return True, render(self.last_meta)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H}.get(prop, 0)

    def release(self):
        pass


class SimulatedGender:
    """Returns the scene's ground-truth gender with noise (keeps the pipeline testable)."""
    name = "Simulated ground truth"

    def __init__(self):
        self.rng = np.random.default_rng(1)
        self.meta: dict = {}

    def predict_track(self, track_id):
        g = next((p["gender"] for p in self.meta.get("persons", []) if p["id"] == track_id), None)
        if g is None:
            return None
        p = 0.8 + 0.15 * self.rng.random()
        return np.array([p, 1 - p]) if g == "male" else np.array([1 - p, p])
