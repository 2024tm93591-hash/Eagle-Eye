"""Turns scene events into scored alerts.

The score starts from a base value per event type (config.yaml) and then gets extra points
when the situation goes on for a while, involves many people, happens during quiet hours or
follows a related event (for example a fall straight after a fight). It is then scaled a
little by how confident the detection was, and mapped to LOW / MEDIUM / HIGH / CRITICAL.
"""
from __future__ import annotations

import time
from collections import deque

from ..core import SceneEvent, ThreatAlert
from ..scene.analyzer import in_quiet_hours

LEVEL_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# extra points per second an event keeps going (capped at +15)
DURATION_RATE = {
    "loitering": 0.25,
    "person_down": 1.0,
    "physical_altercation": 2.0,
    "fighting": 2.0,
    "abandoned_object": 0.3,
    "chasing": 2.0,
    "group_gathering": 0.2,
    "crowd_gathering": 0.2,
}


def level_rank(level: str) -> int:
    return LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 0


class ThreatAssessor:
    def __init__(self, cfg: dict, camera: str = ""):
        self.settings = cfg["threat"]
        self.quiet_hours = cfg["scene"].get("quiet_hours")
        self.camera = camera
        self.last_alert: dict[str, tuple[float, float, str]] = {}   # event key -> (time, score, level)
        self.recent: deque = deque(maxlen=200)                      # (time, event), for linking events
        self.active: list[ThreatAlert] = []
        self.quiet_override: bool | None = None

    def level_for(self, score: float) -> str:
        level = "LOW"
        for name in LEVEL_ORDER:
            if score >= self.settings["levels"][name]:
                level = name
        return level

    def score(self, event: SceneEvent, now: float) -> tuple[float, list[str]]:
        base = float(self.settings["base_scores"].get(event.type, 30))
        score = base
        reasons = [f"{event.type.replace('_', ' ')} (base {base:.0f})"]

        rate = DURATION_RATE.get(event.type, 0.0)
        if rate and event.duration > 0:
            bonus = min(15.0, event.duration * rate)
            if bonus >= 1:
                score += bonus
                reasons.append(f"+{bonus:.0f} persisting {event.duration:.0f}s")

        people = event.extra.get("count", len(event.track_ids))
        if people > 2 and event.type in ("group_gathering", "crowd_gathering", "panic_running"):
            bonus = min(15.0, 3.0 * (people - 2))
            score += bonus
            reasons.append(f"+{bonus:.0f} {people} people involved")

        quiet = self.quiet_override if self.quiet_override is not None else in_quiet_hours(self.quiet_hours)
        if quiet and event.type != "after_hours_presence":
            score += 15
            reasons.append("+15 quiet hours")

        # events involving the same people in the last 30 seconds
        ids = set(event.track_ids)
        related = {old.type for (when, old) in self.recent if now - when < 30 and ids & set(old.track_ids)}
        violence = {"physical_altercation", "fighting"}
        if event.type in ("fall", "person_down") and related & violence:
            score += 20
            reasons.append("+20 fall after an altercation")
        if event.type == "chasing" and related & violence:
            score += 10
            reasons.append("+10 chase linked to altercation")

        score *= 0.7 + 0.3 * max(0.0, min(1.0, event.confidence))
        return max(0.0, min(100.0, score)), reasons

    def assess(self, events: list[SceneEvent], now: float | None = None):
        """Returns (every event scored this frame, alerts that should be raised now)."""
        now = now if now is not None else time.time()
        cooldown = self.settings["cooldown_seconds"]
        min_level = level_rank(self.settings.get("min_alert_level", "LOW"))
        scored, new = [], []

        for event in events:
            score, reasons = self.score(event, now)
            alert = ThreatAlert(event, score, self.level_for(score), reasons, camera=self.camera)
            scored.append(alert)
            if level_rank(alert.level) < min_level:
                continue   # shown on the video overlay only

            previous = self.last_alert.get(event.key)
            if previous is None:
                fire = True
            else:
                last_time, last_score, last_level = previous
                fire = (now - last_time > cooldown
                        or score >= last_score + self.settings["escalation_step"]
                        or level_rank(alert.level) > level_rank(last_level))

            if fire:
                if previous is not None and now - previous[0] <= cooldown:
                    alert.reasons.append("escalated")
                self.last_alert[event.key] = (now, score, alert.level)
                new.append(alert)
            else:
                # refresh the time so an ongoing situation isn't raised again when the cooldown ends
                self.last_alert[event.key] = (now, previous[1], previous[2])

        for event in events:
            self.recent.append((now, event))
        self.active = sorted(scored, key=lambda a: -a.score)
        return self.active, new
