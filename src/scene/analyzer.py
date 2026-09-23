"""Scene understanding: interactions + context -> suspicious / abnormal events.

Individual cues   : fall, person down (no recovery), running, fighting, loitering, zone intrusion
Pairwise cues     : physical altercation (two close people, violent limb motion), chasing
Group cues        : crowd formation, tight group gathering, panic running (several people fleeing)
Object context    : unattended / abandoned bag (no person nearby for N seconds)
Temporal context  : presence during configured quiet hours
"""
from __future__ import annotations

import datetime as dt
from collections import Counter

import cv2
import numpy as np

from ..core import Detection, SceneEvent


def _in_quiet_hours(spec, now: dt.datetime | None = None) -> bool:
    if not spec:
        return False
    now = (now or dt.datetime.now()).time()
    a, b = (dt.datetime.strptime(s, "%H:%M").time() for s in spec)
    return (a <= now or now < b) if a > b else (a <= now < b)


class _TrackedObject:
    def __init__(self, det: Detection, t: float):
        self.det, self.first_seen, self.last_seen = det, t, t
        self.unattended_since: float | None = None
        self.owner: int | None = None
        self.id = id(self) % 100000


class SceneAnalyzer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = cfg["scene"]
        self.rules = cfg["activity"]["rules"]
        self.zones = []
        for z in self.s.get("zones", []):
            self.zones.append(dict(name=z["name"], type=z.get("type", "restricted"),
                                   sensitivity=float(z.get("sensitivity", 1.0)),
                                   poly=np.array(z["polygon"], dtype=np.float32)))
        self.objects: list[_TrackedObject] = []
        self.chase_start: dict[tuple[int, int], float] = {}
        self.summary: dict = {}
        self.quiet_override: bool | None = None    # for tests / demos

    # ------------------------------------------------------------------ utilities
    def zone_pixels(self, shape) -> list[np.ndarray]:
        h, w = shape[:2]
        return [(z["poly"] * [w, h]).astype(np.int32) for z in self.zones]

    @staticmethod
    def _clusters(points: np.ndarray, thresh: np.ndarray) -> list[list[int]]:
        """Single-linkage clustering; thresh[i] is the linking distance for point i."""
        n = len(points)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(points[i] - points[j]) <= (thresh[i] + thresh[j]) / 2:
                    parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    # ------------------------------------------------------------------ main entry
    def update(self, tracks, objects: list[Detection], t: float, frame_shape) -> list[SceneEvent]:
        ev: list[SceneEvent] = []
        s, H, W = self.s, frame_shape[0], frame_shape[1]
        zone_px = self.zone_pixels(frame_shape)
        by_id = {tr.track_id: tr for tr in tracks}

        # ---------------- individual cues
        for tr in tracks:
            d = tr.last_det
            loc = tuple(float(v) for v in d.center)
            foot = d.foot
            bh = tr.body_height()
            for z, poly in zip(self.zones, zone_px):
                inside = cv2.pointPolygonTest(poly, (float(foot[0]), float(foot[1])), False) >= 0
                if inside:
                    tr.zone_enter.setdefault(z["name"], t)
                    if z["type"] == "restricted":
                        ev.append(SceneEvent("restricted_zone_intrusion", (tr.track_id,),
                                             f"Person #{tr.track_id} entered {z['name']}",
                                             duration=t - tr.zone_enter[z["name"]], zone=z["name"],
                                             zone_sensitivity=z["sensitivity"], location=loc))
                else:
                    tr.zone_enter.pop(z["name"], None)

            if tr.activity == "falling":
                down_for = t - tr.fall_time if tr.fall_time is not None else 0.0
                if down_for >= s["person_down_seconds"]:
                    ev.append(SceneEvent("person_down", (tr.track_id,),
                                         f"Person #{tr.track_id} has been on the ground for {down_for:.0f}s",
                                         duration=down_for, location=loc, confidence=tr.activity_conf))
                else:
                    ev.append(SceneEvent("fall", (tr.track_id,), f"Person #{tr.track_id} fell down",
                                         duration=down_for, location=loc, confidence=tr.activity_conf))
            elif tr.activity == "running":
                ev.append(SceneEvent("running", (tr.track_id,), f"Person #{tr.track_id} is running",
                                     duration=t - tr.activity_since, location=loc, confidence=tr.activity_conf))

            # loitering: long presence with little overall displacement
            if tr.dwell >= s["loiter_seconds"] and len(tr.pos_log) >= 4 and tr.activity in ("standing", "walking"):
                recent = np.array([(x, y) for (tt, x, y) in tr.pos_log if tt >= t - s["loiter_seconds"]])
                if len(recent) >= 4:
                    spread = float(np.linalg.norm(recent.max(0) - recent.min(0)) / bh)
                    if spread <= s["loiter_radius"]:
                        zone = next((z for z in self.zones if z["name"] in tr.zone_enter), None)
                        ev.append(SceneEvent("loitering", (tr.track_id,),
                                             f"Person #{tr.track_id} loitering for {tr.dwell:.0f}s"
                                             + (f" in {zone['name']}" if zone else ""),
                                             duration=tr.dwell, location=loc,
                                             zone=zone["name"] if zone else None,
                                             zone_sensitivity=zone["sensitivity"] if zone else 1.0))

        # ---------------- pairwise cues
        fighters = [tr for tr in tracks if tr.activity == "fighting" and t - tr.activity_since >= s.get("min_fight_seconds", 0.6)]
        paired = set()
        for a in fighters:
            best, best_d = None, 1e9
            for b in tracks:
                if b is a:
                    continue
                dd = np.linalg.norm(a.last_det.center - b.last_det.center) / ((a.body_height() + b.body_height()) / 2)
                if dd < best_d:
                    best, best_d = b, dd
            both = best is not None and best.activity == "fighting"
            limit = self.rules["fight_distance"] * (1.3 if both else 0.7)   # a passive victim must be very close
            if best is not None and best_d <= limit:
                pair = tuple(sorted((a.track_id, best.track_id)))
                if pair in paired:
                    continue
                paired.add(pair)
                since = min(a.activity_since, best.activity_since) if both else a.activity_since
                loc = tuple(float(v) for v in (a.last_det.center + best.last_det.center) / 2)
                ev.append(SceneEvent("physical_altercation", pair,
                                     f"Physical altercation between #{pair[0]} and #{pair[1]}",
                                     duration=t - since, location=loc, confidence=0.9 if both else 0.7))
            else:
                ev.append(SceneEvent("fighting", (a.track_id,), f"Person #{a.track_id} shows violent motion",
                                     duration=t - a.activity_since,
                                     location=tuple(float(v) for v in a.last_det.center)))

        movers = [tr for tr in tracks if np.linalg.norm(tr.speed(0.7)) / tr.body_height() >= self.rules["run_speed"] * 0.8]
        live_pairs = set()
        for a in movers:
            va = a.speed(0.7)
            for b in movers:
                if a is b:
                    continue
                vb = b.speed(0.7)
                ab = a.last_det.center - b.last_det.center          # b -> a
                dist = np.linalg.norm(ab) / b.body_height()
                cos_dir = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
                cos_tgt = vb @ ab / (np.linalg.norm(vb) * np.linalg.norm(ab) + 1e-9)
                if dist < 8 and cos_dir > 0.7 and cos_tgt > 0.7:      # b runs behind a, towards a
                    key = (b.track_id, a.track_id)
                    live_pairs.add(key)
                    start = self.chase_start.setdefault(key, t)
                    if t - start >= s["chase_min_seconds"]:
                        ev.append(SceneEvent("chasing", key, f"Person #{b.track_id} is chasing #{a.track_id}",
                                             duration=t - start,
                                             location=tuple(float(v) for v in a.last_det.center)))
        self.chase_start = {k: v for k, v in self.chase_start.items() if k in live_pairs}

        # ---------------- group cues
        n = len(tracks)
        if n >= s["crowd_threshold"]:
            ev.append(SceneEvent("crowd_gathering", (), f"Crowd of {n} people in view", extra={"count": n}))
        still = [tr for tr in tracks if tr.activity in ("standing", "walking")]
        if len(still) >= s["group_size"]:
            pts = np.array([tr.last_det.foot for tr in still])
            th = np.array([tr.body_height() * s["cluster_distance"] for tr in still])
            for g in self._clusters(pts, th):
                if len(g) >= s["group_size"]:
                    ids = tuple(sorted(still[i].track_id for i in g))
                    c = pts[g].mean(0)
                    ev.append(SceneEvent("group_gathering", ids, f"Group of {len(g)} people gathering",
                                         location=(float(c[0]), float(c[1])), extra={"count": len(g)}))
        runners = [tr.track_id for tr in tracks if tr.activity == "running"]
        if len(runners) >= s["group_running"]:
            ev.append(SceneEvent("panic_running", (), f"{len(runners)} people running simultaneously "
                                 "(possible panic / stampede)", extra={"count": len(runners), "ids": runners}))

        # ---------------- object context: abandoned bags
        ev += self._update_objects(objects, tracks, t)

        # ---------------- temporal context
        quiet = self.quiet_override if self.quiet_override is not None else _in_quiet_hours(s.get("quiet_hours"))
        if quiet and n:
            ev.append(SceneEvent("after_hours_presence", (), f"{n} person(s) present during quiet hours",
                                 extra={"count": n}))

        self._summarise(tracks, ev)
        return ev

    def _update_objects(self, objects, tracks, t):
        ev = []
        s = self.s
        matched = set()
        for det in objects:
            best = None
            for o in self.objects:
                if o.det.cls == det.cls and np.linalg.norm(o.det.center - det.center) < 0.6 * max(det.width, det.height):
                    best = o
                    break
            if best is None:
                best = _TrackedObject(det, t)
                self.objects.append(best)
            best.det, best.last_seen = det, t
            matched.add(id(best))
        self.objects = [o for o in self.objects if t - o.last_seen < 3.0]
        by_id = {tr.track_id: tr for tr in tracks}
        for o in self.objects:
            # owner = the person nearest to the object when it first appeared; passers-by do not
            # "attend" a bag, only its owner does (classic left-luggage logic)
            dists = {tr.track_id: np.linalg.norm(tr.last_det.foot - o.det.center) / tr.body_height()
                     for tr in tracks}
            if o.owner is None and dists and t - o.first_seen < 2.0:
                tid, dmin = min(dists.items(), key=lambda kv: kv[1])
                if dmin <= s["abandoned_radius"] * 1.5:
                    o.owner = tid
            if o.owner is not None:
                near = o.owner in by_id and dists[o.owner] <= s["abandoned_radius"]
            else:
                near = bool(dists) and min(dists.values()) <= s["abandoned_radius"] * 0.5
            if near:
                o.unattended_since = None
            else:
                o.unattended_since = o.unattended_since or t
                if t - o.unattended_since >= s["abandoned_seconds"]:
                    who = f" (owner #{o.owner} left)" if o.owner is not None else ""
                    ev.append(SceneEvent("abandoned_object", (o.id,),
                                         f"Unattended {o.det.cls} for {t - o.unattended_since:.0f}s{who}",
                                         duration=t - o.unattended_since,
                                         location=tuple(float(v) for v in o.det.center)))
        return ev

    def _summarise(self, tracks, events):
        genders = Counter(tr.gender for tr in tracks)
        acts = Counter(tr.activity for tr in tracks)
        n = len(tracks)
        parts = [f"{n} {'person' if n == 1 else 'people'} in view"]
        if n:
            parts.append(f"({genders.get('male', 0)} male, {genders.get('female', 0)} female, "
                         f"{genders.get('unknown', 0)} undetermined)")
            parts.append("- " + ", ".join(f"{v} {k}" for k, v in acts.most_common()))
        in_zones = sum(1 for tr in tracks if tr.zone_enter)
        if in_zones:
            parts.append(f"- {in_zones} inside monitored zone(s)")
        notable = sorted({e.type.replace('_', ' ') for e in events if e.type not in ("running",)})
        if notable:
            parts.append("- observed: " + ", ".join(notable))
        self.summary = dict(count=n, genders=dict(genders), activities=dict(acts), text=" ".join(parts))
