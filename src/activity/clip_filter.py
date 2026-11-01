"""Finds who in a labelled clip is really doing the clip's activity, and when.

A clip in the "falling" folder usually has bystanders in it, and most of its six seconds
are the person standing before the fall. Labelling everyone for the whole clip would teach
the model that standing around is falling, so samples are only kept for the frames chosen here.

Records are the per-frame tuples from extract_pose_dataset.track_video:
(frame, t, bbox, keypoints, nearest person distance, body height)
"""
from __future__ import annotations

import numpy as np

from .features import KP_CONF, torso_angle
from .rule_based import HIPS, WRISTS, whole_body_visible

MIN_DOWN_SECONDS = 0.4     # shorter than this is a detection glitch, not someone on the ground
DOWN_HEIGHT = 0.7          # someone lying along the view looks shorter, not horizontal
MOTION_SECONDS = 0.7       # movement is measured over this long, like the rule-based classifier
MAX_TRACK_GAP = 0.5        # a new track starting this soon after another ended may be the same person
MIN_FIGHT_SECONDS = 0.5    # shorter fragments have jittery keypoints and can't be judged


def stitch_tracks(records_by_track: dict) -> dict:
    """Join tracks that are really one person.

    The tracker often gives someone a new id when they drop to the floor or pass behind
    another person, and a person cut into short tracks gives few usable samples. A track
    that starts just after another ended, about where that one stopped, is treated as its
    continuation. Tracks that overlap in time are always different people.
    """
    def foot(rec):
        return np.array([(rec[2][0] + rec[2][2]) / 2, rec[2][3]])

    joined: dict = {}
    for tid, records in sorted(records_by_track.items(), key=lambda item: item[1][0][1]):
        first = records[0]
        best, best_dist = None, float("inf")
        for other, earlier in joined.items():
            last = earlier[-1]
            if not 0 < first[1] - last[1] <= MAX_TRACK_GAP:
                continue
            height = max(last[2][3] - last[2][1], first[2][3] - first[2][1], 1.0)
            dist = np.linalg.norm(foot(first) - foot(last)) / height
            if dist < best_dist:
                best, best_dist = other, dist
        if best is not None and best_dist <= 1.0:
            joined[best] = joined[best] + list(records)
        else:
            joined[tid] = list(records)
    return joined


def _full_height(records) -> float:
    """Standing height for the whole clip. Taken high because the person spends time on the ground."""
    heights = [rec[2][3] - rec[2][1] for rec in records]
    return max(float(np.percentile(heights, 90)), 1.0)


def _earlier(times: np.ndarray, seconds: float) -> np.ndarray:
    """For each time, the index of the record about `seconds` before it."""
    return np.clip(np.searchsorted(times, times - seconds), 0, len(times) - 1)


def _motion(records) -> np.ndarray:
    """Movement in body heights per second. The change in box height is included so that
    walking towards or away from the camera counts as movement too."""
    times = np.array([rec[1] for rec in records], float)
    boxes = np.array([rec[2] for rec in records], float)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    before = _earlier(times, MOTION_SECONDS)
    moved = np.linalg.norm(centers - centers[before], axis=1) + np.abs(heights - heights[before])
    return moved / np.maximum(times - times[before], 0.1) / _full_height(records)


def _is_down(rec, full_height: float, rules: dict) -> bool:
    box, keypoints = rec[2], rec[3]
    if not whole_body_visible(keypoints):
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    angle = torso_angle(keypoints)
    return bool((angle is not None and angle > rules["fall_torso_angle"])
                or width / max(height, 1.0) > rules["fall_aspect_ratio"]
                or height < DOWN_HEIGHT * full_height)


def _falling_frames(records, rules: dict) -> set:
    """The frames from when this person goes down until they get up. Empty if they never do."""
    if len(records) < 3:
        return set()
    full_height = _full_height(records)
    down = np.array([_is_down(rec, full_height, rules) for rec in records], float)
    frame_time = np.median(np.diff([rec[1] for rec in records]))
    size = max(1, int(round(MIN_DOWN_SECONDS / max(frame_time, 1e-3))))
    sustained = np.flatnonzero(np.convolve(down, np.ones(size) / size, mode="same") >= 0.6)
    if not len(sustained):
        return set()
    return {rec[0] for rec in records[sustained[0]:sustained[-1] + 1]}


def _arm_speed(records) -> float:
    """How fast this person's wrists move relative to their hips (75th percentile)."""
    offsets = []
    for rec in records:
        box, keypoints = rec[2], rec[3]
        hips = keypoints[HIPS]
        visible = hips[:, 2] >= KP_CONF
        center = hips[visible, :2].mean(0) if visible.any() else (box[:2] + box[2:]) / 2
        wrists = keypoints[list(WRISTS)]
        offsets.append(np.where(wrists[:, 2:3] >= KP_CONF, wrists[:, :2] - center, np.nan))
    offsets = np.array(offsets, float)

    times = np.array([rec[1] for rec in records], float)
    before = _earlier(times, 0.1)
    elapsed = np.maximum(times - times[before], 1e-3)[:, None]
    speeds = np.linalg.norm(offsets - offsets[before], axis=2) / elapsed / _full_height(records)
    speeds = speeds[~np.isnan(speeds)]
    return float(np.percentile(speeds, 75)) if speeds.size else 0.0


def _fighting_frames(records_by_track: dict, rules: dict) -> dict:
    """People close to someone else, whose arms move about as much as the busiest person's.
    Comparing within the clip means onlookers drop out without a fixed speed threshold."""
    limit = rules["fight_distance"] * 1.3
    close = {tid: [rec for rec in recs if rec[4] <= limit] for tid, recs in records_by_track.items()}

    def long_enough(tid):
        return len(close[tid]) >= 2 and close[tid][-1][1] - close[tid][0][1] >= MIN_FIGHT_SECONDS

    activity = {tid: _arm_speed(recs) for tid, recs in records_by_track.items() if long_enough(tid)}
    busiest = max(activity.values(), default=0.0)
    return {tid: {rec[0] for rec in close[tid]} if tid in activity and activity[tid] >= 0.35 * busiest else set()
            for tid in records_by_track}


def frames_showing_activity(label: str, records_by_track: dict, rules: dict):
    """Returns {track id: frames to keep}, or None when `label` has no check."""
    if label == "falling":
        return {tid: _falling_frames(recs, rules) for tid, recs in records_by_track.items()}
    if label == "fighting":
        return _fighting_frames(records_by_track, rules)

    limits = {"standing": (0.0, 2 * rules["walk_speed"]),
              "walking": (0.6 * rules["walk_speed"], float("inf")),
              "running": (0.5 * rules["run_speed"], float("inf"))}
    if label not in limits:
        return None
    low, high = limits[label]
    return {tid: {rec[0] for rec, speed in zip(recs, _motion(recs)) if low <= speed <= high}
            for tid, recs in records_by_track.items()}
