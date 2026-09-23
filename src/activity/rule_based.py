"""Interpretable rule-based activity classifier (baseline and zero-training fallback).

All thresholds are expressed in *body heights per second* so they are independent of the
camera distance and resolution.
"""
from __future__ import annotations

import numpy as np

from .features import KP_CONF, torso_angle


class RuleBasedActivity:
    name = "Rule-based (pose geometry + motion)"

    def __init__(self, cfg: dict):
        self.r = cfg["activity"]["rules"]

    @staticmethod
    def _joint_speed(track, body_h: float, joints=(9, 10), seconds: float = 0.6) -> float:
        """75th-percentile speed of the given joints relative to the hips (body-heights/s)."""
        hist = [h for h in track.history if h[2] is not None][-int(seconds * 30):]
        if len(hist) < 4:
            return 0.0
        rel = []
        for t, b, k in hist:
            hips = k[[11, 12]]
            c = hips[hips[:, 2] >= KP_CONF, :2].mean(0) if (hips[:, 2] >= KP_CONF).any() \
                else np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            w = k[list(joints)]
            rel.append(np.where(w[:, 2:3] >= KP_CONF, (w[:, :2] - c) / body_h, np.nan))
        rel = np.array(rel)                          # (T, 2, 2)
        ts = np.array([h[0] for h in hist])
        d = np.linalg.norm(np.diff(rel, axis=0), axis=2) / np.clip(np.diff(ts), 1e-3, None)[:, None]
        d = d[~np.isnan(d)]
        return float(np.percentile(d, 75)) if d.size else 0.0

    @staticmethod
    def _box_agitation(track, body_h: float) -> float:
        """Without keypoints: mean absolute acceleration of the box centre (body-heights/s^2 / 10)."""
        cs, ts = track.centers()[-15:], track.times()[-15:]
        if len(cs) < 5:
            return 0.0
        v = np.diff(cs, axis=0) / np.clip(np.diff(ts), 1e-3, None)[:, None]
        a = np.linalg.norm(np.diff(v, axis=0), axis=1)
        return float(a.mean() / body_h / 10.0)

    def predict(self, track, t: float) -> tuple[str, float, dict]:
        r = self.r
        body_h = track.body_height()
        det = track.last_det
        v = track.speed(0.7) / body_h
        speed = float(np.linalg.norm(v))
        drop = float(track.speed(0.4)[1] / body_h)          # +ve = moving down
        aspect = det.width / det.height
        ang = torso_angle(det.keypoints)
        lying = (ang is not None and ang > r["fall_torso_angle"]) or aspect > r["fall_aspect_ratio"]
        info = dict(speed=round(speed, 2), drop=round(drop, 2), aspect=round(aspect, 2),
                    torso=None if ang is None else round(ang, 1))

        # 1) falling / fallen
        if lying:
            if track.fall_time is None:
                track.fall_time = t                            # moment the person went down
            return "falling", 0.9 if drop > r["fall_drop_speed"] else 0.75, info
        if drop > r["fall_drop_speed"] * 1.5 and det.height < 0.75 * body_h:
            return "falling", 0.6, info
        track.fall_time = None

        # 2) fighting: fast limb motion while very close to another person
        nd = track.nbr_dist[-1] if track.nbr_dist else 99.0
        if det.keypoints is not None:
            limb = self._joint_speed(track, body_h, (9, 10))
            legs = self._joint_speed(track, body_h, (15, 16))
        else:
            limb, legs = self._box_agitation(track, body_h) * 2.0, 0.0
        info.update(limb=round(limb, 2), legs=round(legs, 2), nbr=round(float(nd), 2))
        # fighters keep their feet roughly planted: fast arm swing *with* a leg-swing gait
        # (people walking / running side by side) is locomotion, not fighting
        if nd <= r["fight_distance"] and limb >= r["fight_limb_speed"] and speed < r["run_speed"] * 0.8 \
                and legs < 0.5 * limb:
            return "fighting", min(1.0, 0.5 + limb / (4 * r["fight_limb_speed"])), info

        # 3) locomotion
        if speed >= r["run_speed"]:
            return "running", min(1.0, 0.6 + (speed - r["run_speed"]) / 2), info
        if speed >= r["walk_speed"]:
            return "walking", 0.8, info
        return "standing", 0.8, info
