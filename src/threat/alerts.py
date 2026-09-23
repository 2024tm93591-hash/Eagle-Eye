"""Sends alerts out: saves a snapshot, stores the alert, pushes it to open dashboards and
notifies the external channels (e-mail, webhook, Telegram) that are enabled in config.yaml."""
from __future__ import annotations

import json
import logging
import os
import queue
import smtplib
import threading
import urllib.request
from email.message import EmailMessage

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
        # external notifications can be slow, so a worker thread sends them
        self.outbox: queue.Queue = queue.Queue()
        threading.Thread(target=self._notify_worker, daemon=True).start()

    # each open dashboard page gets its own queue (see /api/stream)
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
                    pass   # that page has stopped reading; drop the message

    def raise_alert(self, alert: ThreatAlert, frame=None):
        if frame is not None:
            name = f"{alert.id}_{alert.event.type}.jpg"
            cv2.imwrite(str(self.snap_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            alert.snapshot = name

        data = alert.to_dict()
        self.store.add_alert(data)
        self.broadcast({"kind": "alert", "alert": data})
        if level_rank(alert.level) >= level_rank(self.cfg.get("min_level_console", "MEDIUM")):
            log.warning("[%s] %s (score %.0f) - %s", alert.level, data["description"], alert.score,
                        "; ".join(alert.reasons))
        self.outbox.put(data)

    def _notify_worker(self):
        senders = {"email": self._email, "webhook": self._webhook, "telegram": self._telegram}
        while True:
            alert = self.outbox.get()
            for channel, send in senders.items():
                settings = self.cfg.get(channel, {})
                if not settings.get("enabled"):
                    continue
                if level_rank(alert["level"]) < level_rank(settings.get("min_level", "HIGH")):
                    continue
                try:
                    send(settings, alert)
                except Exception as exc:
                    log.error("%s notification failed: %s", channel, exc)

    @staticmethod
    def _message(alert):
        return (f"[{alert['level']}] {alert['description']}\n"
                f"Camera: {alert['camera']}\n"
                f"Time: {alert['time']}\n"
                f"Threat score: {alert['score']}\n"
                f"Why: {'; '.join(alert['reasons'])}")

    def _email(self, settings, alert):
        msg = EmailMessage()
        msg["Subject"] = f"ThreatVision {alert['level']} alert: {alert['type'].replace('_', ' ')}"
        msg["From"] = settings["username"]
        msg["To"] = ", ".join(settings["to"])
        msg.set_content(self._message(alert))
        if alert.get("snapshot"):
            snapshot = self.snap_dir / alert["snapshot"]
            if snapshot.exists():
                msg.add_attachment(snapshot.read_bytes(), maintype="image", subtype="jpeg", filename=snapshot.name)

        password = os.environ.get(settings.get("password_env", ""), "")
        with smtplib.SMTP(settings["smtp_host"], settings["smtp_port"], timeout=15) as smtp:
            smtp.starttls()
            smtp.login(settings["username"], password)
            smtp.send_message(msg)

    def _webhook(self, settings, alert):
        self._post_json(settings["url"], {"text": self._message(alert), "alert": alert})

    def _telegram(self, settings, alert):
        token = os.environ.get(settings.get("bot_token_env", ""), "")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        self._post_json(url, {"chat_id": settings["chat_id"], "text": self._message(alert)})

    @staticmethod
    def _post_json(url, payload):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=10).read()
