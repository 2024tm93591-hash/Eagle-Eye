"""Rule-based activity classifier. Needs no training, and the thresholds are easy to explain.

All speeds are measured in body heights per second, so the rules work the same whether a
person is close to the camera or far away.
"""
from __future__ import annotations

import numpy as np

from .features import KP_CONF, torso_angle

WRISTS = (9, 10)
ANKLES = (15, 16)
HIPS = [11, 12]


class RuleBasedActivity:
    name = "Rule-based (pose geometry + motion)"

    def __init__(self, cfg: dict):
        self.rules = cfg["activity"]["rules"]

    @staticmethod
    def _joint_speed(track, body_h: float, joints, seconds: float = 0.6) -> float:
        """How fast the given joints move relative to the hips (75th percentile, body heights/s)."""
        recent = [h for h in track.history if h[2] is not None][-int(seconds * 30):]
        if len(recent) < 4:
            return 0.0

        offsets = []
        for _, box, kp in recent:
            hips = kp[HIPS]
            visible = hips[:, 2] >= KP_CONF
            if visible.any():
                center = hips[visible, :2].mean(0)
            else:
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            joint = kp[list(joints)]
            offsets.append(np.where(joint[:, 2:3] >= KP_CONF, (joint[:, :2] - center) / body_h, np.nan))

        offsets = np.array(offsets)   # (frames, joints, xy)
        times = np.array([h[0] for h in recent])
        speeds = np.linalg.norm(np.diff(offsets, axis=0), axis=2) / np.clip(np.diff(times), 1e-3, None)[:, None]
        speeds = speeds[~np.isnan(speeds)]
        return float(np.percentile(speeds, 75)) if speeds.size else 0.0

    @staticmethod
    def _box_agitation(track, body_h: float) -> float:
        """Used when there are no keypoints: how jerky the box movement is."""
        centers, times = track.centers()[-15:], track.times()[-15:]
        if len(centers) < 5:
            return 0.0
        velocity = np.diff(centers, axis=0) / np.clip(np.diff(times), 1e-3, None)[:, None]
        acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
        return float(acceleration.mean() / body_h / 10.0)

    def predict(self, track, t: float) -> tuple[str, float, dict]:
        """Returns (label, confidence, the measurements the decision was based on)."""
        rules = self.rules
        body_h = track.body_height()
        det = track.last_det
        speed = float(np.linalg.norm(track.speed(0.7) / body_h))
        drop = float(track.speed(0.4)[1] / body_h)   # positive when moving down
        aspect = det.width / det.height
        angle = torso_angle(det.keypoints)
        lying = (angle is not None and angle > rules["fall_torso_angle"]) or aspect > rules["fall_aspect_ratio"]
        info = dict(speed=round(speed, 2), drop=round(drop, 2), aspect=round(aspect, 2),
                    torso=None if angle is None else round(angle, 1))

        # falling, or already on the ground
        if lying:
            if track.fall_time is None:
                track.fall_time = t   # remember when they went down
            return "falling", 0.9 if drop > rules["fall_drop_speed"] else 0.75, info
        if drop > rules["fall_drop_speed"] * 1.5 and det.height < 0.75 * body_h:
            return "falling", 0.6, info
        track.fall_time = None

        # fighting: fast arm movement while very close to someone else
        neighbour = track.nbr_dist[-1] if track.nbr_dist else 99.0
        if det.keypoints is not None:
            arms = self._joint_speed(track, body_h, WRISTS)
            legs = self._joint_speed(track, body_h, ANKLES)
        else:
            arms, legs = self._box_agitation(track, body_h) * 2.0, 0.0
        info.update(limb=round(arms, 2), legs=round(legs, 2), nbr=round(float(neighbour), 2))

        # People fighting mostly keep their feet in place. Swinging arms together with a
        # walking leg movement is just two people walking or running side by side.
        if (neighbour <= rules["fight_distance"] and arms >= rules["fight_limb_speed"]
                and speed < rules["run_speed"] * 0.8 and legs < 0.5 * arms):
            return "fighting", min(1.0, 0.5 + arms / (4 * rules["fight_limb_speed"])), info

        # otherwise it's about how fast the person moves
        if speed >= rules["run_speed"]:
            return "running", min(1.0, 0.6 + (speed - rules["run_speed"]) / 2), info
        if speed >= rules["walk_speed"]:
            return "walking", 0.8, info
        return "standing", 0.8, info
