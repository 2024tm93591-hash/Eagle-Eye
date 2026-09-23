"""Real-time threat assessment: scene events -> scored, levelled, de-duplicated alerts.

score = base(type)
        + duration bonus        (persisting situations are more serious)
        + people involved bonus (group events)
        + zone bonus            (sensitive / restricted areas)
        + quiet-hours bonus     (activity at night)
        + correlation bonus     (e.g. a fall right after an altercation, intrusion while running)
      x  detection confidence  (0.7 .. 1.0)
The score (0-100) is mapped to LOW / MEDIUM / HIGH / CRITICAL.
"""
from __future__ import annotations

import time
from collections import deque

from ..core import SceneEvent, ThreatAlert
from ..scene.analyzer import _in_quiet_hours

LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
DURATION_RATE = {  # score points per second, capped at +15
    "loitering": 0.25, "restricted_zone_intrusion": 1.0, "person_down": 1.0, "physical_altercation": 2.0,
    "fighting": 2.0, "abandoned_object": 0.3, "chasing": 2.0, "group_gathering": 0.2, "crowd_gathering": 0.2,
}


def level_rank(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0


class ThreatAssessor:
    def __init__(self, cfg: dict, camera: str = ""):
        self.cfg = cfg
        self.t = cfg["threat"]
        self.quiet_hours = cfg["scene"].get("quiet_hours")
        self.camera = camera
        self.last_alert: dict[str, tuple[float, float, str]] = {}   # key -> (time, score, level)
        self.recent: deque = deque(maxlen=200)                      # (time, event) for correlation
        self.active: list[ThreatAlert] = []                          # everything scored in the last frame
        self.quiet_override: bool | None = None

    def level_for(self, score: float) -> str:
        lv = "LOW"
        for name in LEVEL_ORDER:
            if score >= self.t["levels"][name]:
                lv = name
        return lv

    def score(self, e: SceneEvent, now: float) -> tuple[float, list[str]]:
        base = float(self.t["base_scores"].get(e.type, 30))
        reasons = [f"{e.type.replace('_', ' ')} (base {base:.0f})"]
        s = base
        rate = DURATION_RATE.get(e.type, 0.0)
        if rate and e.duration > 0:
            b = min(15.0, e.duration * rate)
            if b >= 1:
                s += b
                reasons.append(f"+{b:.0f} persisting {e.duration:.0f}s")
        n = e.extra.get("count", len(e.track_ids))
        if n > 2 and e.type in ("group_gathering", "crowd_gathering", "panic_running"):
            b = min(15.0, 3.0 * (n - 2))
            s += b
            reasons.append(f"+{b:.0f} {n} people involved")
        if e.zone:
            b = 10.0 * e.zone_sensitivity
            s += b
            reasons.append(f"+{b:.0f} inside {e.zone}")
        quiet = self.quiet_override if self.quiet_override is not None else _in_quiet_hours(self.quiet_hours)
        if quiet and e.type != "after_hours_presence":
            s += 15
            reasons.append("+15 quiet hours")
        # correlation with recent events on the same people
        ids = set(e.track_ids)
        recent_types = {ev.type for (tt, ev) in self.recent if now - tt < 30 and ids & set(ev.track_ids)}
        if e.type in ("fall", "person_down") and recent_types & {"physical_altercation", "fighting"}:
            s += 20
            reasons.append("+20 fall after an altercation")
        if e.type == "restricted_zone_intrusion" and recent_types & {"running", "chasing"}:
            s += 10
            reasons.append("+10 entered zone while running")
        if e.type == "chasing" and recent_types & {"physical_altercation", "fighting"}:
            s += 10
            reasons.append("+10 chase linked to altercation")
        s *= 0.7 + 0.3 * max(0.0, min(1.0, e.confidence))
        return max(0.0, min(100.0, s)), reasons

    def assess(self, events: list[SceneEvent], now: float | None = None) -> tuple[list[ThreatAlert], list[ThreatAlert]]:
        """Returns (all_scored_events_this_frame, new_alerts_to_raise)."""
        now = now if now is not None else time.time()
        scored, new = [], []
        for e in events:
            sc, reasons = self.score(e, now)
            alert = ThreatAlert(e, sc, self.level_for(sc), reasons, camera=self.camera)
            scored.append(alert)
            prev = self.last_alert.get(e.key)
            if level_rank(alert.level) < level_rank(self.t.get("min_alert_level", "LOW")):
                continue
            fire = prev is None or now - prev[0] > self.t["cooldown_seconds"] \
                or sc >= prev[1] + self.t["escalation_step"] or level_rank(alert.level) > level_rank(prev[2])
            if fire:
                if prev is not None and now - prev[0] <= self.t["cooldown_seconds"]:
                    alert.reasons.append("escalated")
                self.last_alert[e.key] = (now, sc, alert.level)
                new.append(alert)
            else:  # keep the situation "alive" so it is not re-raised while it continues
                self.last_alert[e.key] = (now, prev[1], prev[2])
        for e in events:
            self.recent.append((now, e))
        self.active = sorted(scored, key=lambda a: -a.score)
        return self.active, new
