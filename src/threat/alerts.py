"""Alert dispatch: snapshot, persistence, live dashboard push and external notifications."""
from __future__ import annotations

import json
import logging
import os
import queue
import smtplib
import threading
import urllib.request
from email.message import EmailMessage
from pathlib import Path

import cv2

from ..config import resolve_path
from ..core import ThreatAlert
from ..storage.db import EventStore
from .assessment import level_rank

log = logging.getLogger("threatvision.alerts")


class AlertManager:
    def __init__(self, cfg: dict, store: EventStore):
        self.cfg = cfg["alerts"]
        self.store = store
        self.snap_dir = resolve_path(self.cfg["snapshot_dir"])
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.subscribers: list[queue.Queue] = []
        self.sub_lock = threading.Lock()
        self.outbox: queue.Queue = queue.Queue()
        threading.Thread(target=self._notify_worker, daemon=True).start()

    # ---------------------------------------------------------------- live push (SSE)
    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=200)
        with self.sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def broadcast(self, msg: dict):
        with self.sub_lock:
            for q in list(self.subscribers):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    # ---------------------------------------------------------------- main entry
    def raise_alert(self, alert: ThreatAlert, frame=None):
        if frame is not None:
            name = f"{alert.id}_{alert.event.type}.jpg"
            cv2.imwrite(str(self.snap_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            alert.snapshot = name
        d = alert.to_dict()
        self.store.add_alert(d)
        self.broadcast({"kind": "alert", "alert": d})
        if level_rank(alert.level) >= level_rank(self.cfg.get("min_level_console", "MEDIUM")):
            log.warning("[%s] %s (score %.0f) - %s", alert.level, d["description"], alert.score,
                        "; ".join(alert.reasons))
        self.outbox.put(d)

    # ---------------------------------------------------------------- external channels
    def _notify_worker(self):
        while True:
            d = self.outbox.get()
            for name, fn in (("email", self._email), ("webhook", self._webhook), ("telegram", self._telegram)):
                c = self.cfg.get(name, {})
                if c.get("enabled") and level_rank(d["level"]) >= level_rank(c.get("min_level", "HIGH")):
                    try:
                        fn(c, d)
                    except Exception as exc:
                        log.error("%s notification failed: %s", name, exc)

    def _text(self, d):
        return (f"[{d['level']}] {d['description']}\nCamera: {d['camera']}\nTime: {d['time']}\n"
                f"Threat score: {d['score']}\nWhy: {'; '.join(d['reasons'])}")

    def _email(self, c, d):
        msg = EmailMessage()
        msg["Subject"] = f"ThreatVision {d['level']} alert: {d['type'].replace('_', ' ')}"
        msg["From"] = c["username"]
        msg["To"] = ", ".join(c["to"])
        msg.set_content(self._text(d))
        if d.get("snapshot"):
            p = self.snap_dir / d["snapshot"]
            if p.exists():
                msg.add_attachment(p.read_bytes(), maintype="image", subtype="jpeg", filename=p.name)
        with smtplib.SMTP(c["smtp_host"], c["smtp_port"], timeout=15) as s:
            s.starttls()
            s.login(c["username"], os.environ.get(c.get("password_env", ""), ""))
            s.send_message(msg)

    def _webhook(self, c, d):
        body = json.dumps({"text": self._text(d), "alert": d}).encode()
        req = urllib.request.Request(c["url"], data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()

    def _telegram(self, c, d):
        token = os.environ.get(c.get("bot_token_env", ""), "")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        body = json.dumps({"chat_id": c["chat_id"], "text": self._text(d)}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()


def snapshot_path(cfg: dict, name: str) -> Path:
    return resolve_path(cfg["alerts"]["snapshot_dir"]) / name
