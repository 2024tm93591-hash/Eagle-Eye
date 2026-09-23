"""Scene understanding: turns what each tracked person is doing into scene events.

Events come from single people (fall, person down, running, loitering), pairs of people
(fights, chases), groups (crowds, gatherings, panic running), bags left behind and the
time of day.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter

import numpy as np

from ..core import Detection, SceneEvent


def in_quiet_hours(hours, now: dt.datetime | None = None) -> bool:
    """`hours` looks like ["22:00", "06:00"]. The range may wrap past midnight."""
    if not hours:
        return False
    now = (now or dt.datetime.now()).time()
    start, end = (dt.datetime.strptime(h, "%H:%M").time() for h in hours)
    if start > end:
        return now >= start or now < end
    return start <= now < end


def center_of(det: Detection) -> tuple[float, float]:
    return float(det.center[0]), float(det.center[1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_groups(points: np.ndarray, link_dist: np.ndarray) -> list[list[int]]:
    """Single-linkage clustering. link_dist[i] is how close point i must be to join a group."""
    parent = list(range(len(points)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if np.linalg.norm(points[i] - points[j]) <= (link_dist[i] + link_dist[j]) / 2:
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


class TrackedObject:
    """A bag seen over several frames, and the person it belongs to (if we know)."""

    def __init__(self, det: Detection, t: float):
        self.det = det
        self.first_seen = t
        self.last_seen = t
        self.unattended_since: float | None = None
        self.owner: int | None = None
        self.id = id(self) % 100000


class SceneAnalyzer:
    def __init__(self, cfg: dict):
        self.settings = cfg["scene"]
        self.rules = cfg["activity"]["rules"]
        self.objects: list[TrackedObject] = []
        self.chase_start: dict[tuple[int, int], float] = {}
        self.summary: dict = {}
        self.quiet_override: bool | None = None   # tests set this so results don't depend on the clock

    def update(self, tracks, objects: list[Detection], t: float) -> list[SceneEvent]:
        events = []
        events += self._single_person_events(tracks, t)
        events += self._fight_events(tracks, t)
        events += self._chase_events(tracks, t)
        events += self._group_events(tracks)
        events += self._abandoned_object_events(objects, tracks, t)
        if tracks and self._is_quiet_time():
            events.append(SceneEvent("after_hours_presence", (),
                                     f"{len(tracks)} person(s) present during quiet hours",
                                     extra={"count": len(tracks)}))
        self._summarise(tracks, events)
        return events

    def _is_quiet_time(self) -> bool:
        if self.quiet_override is not None:
            return self.quiet_override
        return in_quiet_hours(self.settings.get("quiet_hours"))

    def _single_person_events(self, tracks, t):
        events = []
        for track in tracks:
            pid = track.track_id
            where = center_of(track.last_det)

            if track.activity == "falling":
                down_for = t - track.fall_time if track.fall_time is not None else 0.0
                if down_for >= self.settings["person_down_seconds"]:
                    events.append(SceneEvent("person_down", (pid,),
                                             f"Person #{pid} has been on the ground for {down_for:.0f}s",
                                             duration=down_for, location=where,
                                             confidence=track.activity_conf))
                else:
                    events.append(SceneEvent("fall", (pid,), f"Person #{pid} fell down",
                                             duration=down_for, location=where,
                                             confidence=track.activity_conf))
            elif track.activity == "running":
                events.append(SceneEvent("running", (pid,), f"Person #{pid} is running",
                                         duration=t - track.activity_since, location=where,
                                         confidence=track.activity_conf))

            if self._is_loitering(track, t):
                events.append(SceneEvent("loitering", (pid,), f"Person #{pid} loitering for {track.dwell:.0f}s",
                                         duration=track.dwell, location=where))
        return events

    def _is_loitering(self, track, t) -> bool:
        """Someone who has been around for a long time without really going anywhere."""
        window = self.settings["loiter_seconds"]
        if track.dwell < window or track.activity not in ("standing", "walking"):
            return False
        recent = np.array([(x, y) for (ts, x, y) in track.pos_log if ts >= t - window])
        if len(recent) < 4:
            return False
        spread = np.linalg.norm(recent.max(0) - recent.min(0)) / track.body_height()
        return spread <= self.settings["loiter_radius"]

    def _fight_events(self, tracks, t):
        min_seconds = self.settings.get("min_fight_seconds", 0.6)
        fighters = [tr for tr in tracks if tr.activity == "fighting" and t - tr.activity_since >= min_seconds]
        events, reported = [], set()

        for fighter in fighters:
            other, dist = self._nearest_person(fighter, tracks)
            both_fighting = other is not None and other.activity == "fighting"
            # a passive victim has to be much closer than someone who is fighting back
            limit = self.rules["fight_distance"] * (1.3 if both_fighting else 0.7)

            if other is None or dist > limit:
                events.append(SceneEvent("fighting", (fighter.track_id,),
                                         f"Person #{fighter.track_id} shows violent motion",
                                         duration=t - fighter.activity_since,
                                         location=center_of(fighter.last_det)))
                continue

            pair = tuple(sorted((fighter.track_id, other.track_id)))
            if pair in reported:
                continue
            reported.add(pair)
            since = min(fighter.activity_since, other.activity_since) if both_fighting else fighter.activity_since
            middle = (fighter.last_det.center + other.last_det.center) / 2
            events.append(SceneEvent("physical_altercation", pair,
                                     f"Physical altercation between #{pair[0]} and #{pair[1]}",
                                     duration=t - since, location=(float(middle[0]), float(middle[1])),
                                     confidence=0.9 if both_fighting else 0.7))
        return events

    @staticmethod
    def _nearest_person(track, tracks):
        """The closest other person and their distance, measured in body heights."""
        nearest, nearest_dist = None, float("inf")
        for other in tracks:
            if other is track:
                continue
            avg_height = (track.body_height() + other.body_height()) / 2
            dist = np.linalg.norm(track.last_det.center - other.last_det.center) / avg_height
            if dist < nearest_dist:
                nearest, nearest_dist = other, dist
        return nearest, nearest_dist

    def _chase_events(self, tracks, t):
        fast = self.rules["run_speed"] * 0.8
        movers = [tr for tr in tracks if np.linalg.norm(tr.speed(0.7)) / tr.body_height() >= fast]
        events, still_chasing = [], set()

        for target in movers:
            for chaser in movers:
                if chaser is target:
                    continue
                gap = target.last_det.center - chaser.last_det.center
                dist = np.linalg.norm(gap) / chaser.body_height()
                same_direction = cosine(target.speed(0.7), chaser.speed(0.7))
                heading_for_target = cosine(chaser.speed(0.7), gap)
                if dist >= 8 or same_direction <= 0.7 or heading_for_target <= 0.7:
                    continue

                key = (chaser.track_id, target.track_id)
                still_chasing.add(key)
                start = self.chase_start.setdefault(key, t)
                if t - start >= self.settings["chase_min_seconds"]:
                    events.append(SceneEvent("chasing", key,
                                             f"Person #{chaser.track_id} is chasing #{target.track_id}",
                                             duration=t - start, location=center_of(target.last_det)))

        self.chase_start = {k: v for k, v in self.chase_start.items() if k in still_chasing}
        return events

    def _group_events(self, tracks):
        s = self.settings
        events = []

        count = len(tracks)
        if count >= s["crowd_threshold"]:
            events.append(SceneEvent("crowd_gathering", (), f"Crowd of {count} people in view",
                                     extra={"count": count}))

        calm = [tr for tr in tracks if tr.activity in ("standing", "walking")]
        if len(calm) >= s["group_size"]:
            feet = np.array([tr.last_det.foot for tr in calm])
            link = np.array([tr.body_height() * s["cluster_distance"] for tr in calm])
            for group in find_groups(feet, link):
                if len(group) < s["group_size"]:
                    continue
                ids = tuple(sorted(calm[i].track_id for i in group))
                middle = feet[group].mean(0)
                events.append(SceneEvent("group_gathering", ids, f"Group of {len(group)} people gathering",
                                         location=(float(middle[0]), float(middle[1])),
                                         extra={"count": len(group)}))

        runners = [tr.track_id for tr in tracks if tr.activity == "running"]
        if len(runners) >= s["group_running"]:
            events.append(SceneEvent("panic_running", (),
                                     f"{len(runners)} people running simultaneously (possible panic / stampede)",
                                     extra={"count": len(runners), "ids": runners}))
        return events

    def _abandoned_object_events(self, detections, tracks, t):
        s = self.settings
        for det in detections:
            size = max(det.width, det.height)
            obj = next((o for o in self.objects
                        if o.det.cls == det.cls and np.linalg.norm(o.det.center - det.center) < 0.6 * size), None)
            if obj is None:
                obj = TrackedObject(det, t)
                self.objects.append(obj)
            obj.det, obj.last_seen = det, t
        # forget bags that haven't been detected for a few seconds
        self.objects = [o for o in self.objects if t - o.last_seen < 3.0]

        present = {tr.track_id for tr in tracks}
        events = []
        for obj in self.objects:
            dists = {tr.track_id: np.linalg.norm(tr.last_det.foot - obj.det.center) / tr.body_height()
                     for tr in tracks}

            # The owner is whoever was closest when the bag first showed up. Only the owner
            # counts as looking after it - strangers walking past don't.
            if obj.owner is None and dists and t - obj.first_seen < 2.0:
                pid, closest = min(dists.items(), key=lambda item: item[1])
                if closest <= s["abandoned_radius"] * 1.5:
                    obj.owner = pid

            if obj.owner is not None:
                attended = obj.owner in present and dists[obj.owner] <= s["abandoned_radius"]
            else:
                attended = bool(dists) and min(dists.values()) <= s["abandoned_radius"] * 0.5
            if attended:
                obj.unattended_since = None
                continue

            obj.unattended_since = obj.unattended_since or t
            alone_for = t - obj.unattended_since
            if alone_for >= s["abandoned_seconds"]:
                owner_note = f" (owner #{obj.owner} left)" if obj.owner is not None else ""
                events.append(SceneEvent("abandoned_object", (obj.id,),
                                         f"Unattended {obj.det.cls} for {alone_for:.0f}s{owner_note}",
                                         duration=alone_for, location=center_of(obj.det)))
        return events

    def _summarise(self, tracks, events):
        genders = Counter(tr.gender for tr in tracks)
        activities = Counter(tr.activity for tr in tracks)
        count = len(tracks)

        text = f"{count} {'person' if count == 1 else 'people'} in view"
        if count:
            text += (f" ({genders.get('male', 0)} male, {genders.get('female', 0)} female, "
                     f"{genders.get('unknown', 0)} undetermined)")
            text += " - " + ", ".join(f"{n} {name}" for name, n in activities.most_common())
        notable = sorted({e.type.replace("_", " ") for e in events if e.type != "running"})
        if notable:
            text += " - observed: " + ", ".join(notable)

        self.summary = dict(count=count, genders=dict(genders), activities=dict(activities), text=text)
