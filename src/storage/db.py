"""Alert history, kept in SQLite."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    ts REAL,
    time TEXT,
    camera TEXT,
    type TEXT,
    level TEXT,
    score REAL,
    description TEXT,
    track_ids TEXT,
    duration REAL,
    reasons TEXT,
    snapshot TEXT,
    acknowledged INTEGER DEFAULT 0,
    ack_by TEXT,
    ack_time REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
"""


class EventStore:
    def __init__(self, path):
        self.path = str(path)
        self.lock = threading.Lock()   # the connection is shared between the pipeline and web threads
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add_alert(self, alert: dict):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO alerts (id, ts, time, camera, type, level, score, description,"
                " track_ids, duration, reasons, snapshot, acknowledged) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (alert["id"], alert["timestamp"], alert["time"], alert["camera"], alert["type"], alert["level"],
                 alert["score"], alert["description"], json.dumps(alert["track_ids"]), alert["duration"],
                 json.dumps(alert["reasons"]), alert["snapshot"], int(alert["acknowledged"])))
            self.conn.commit()

    def acknowledge(self, alert_id: str, user: str = "operator") -> bool:
        with self.lock:
            cur = self.conn.execute("UPDATE alerts SET acknowledged=1, ack_by=?, ack_time=? WHERE id=?",
                                    (user, time.time(), alert_id))
            self.conn.commit()
            return cur.rowcount > 0

    def query_alerts(self, level=None, type_=None, since=None, until=None, search=None, limit=500, offset=0,
                     unacked_only=False) -> list[dict]:
        sql = "SELECT * FROM alerts WHERE 1=1"
        args = []
        if level:
            levels = level if isinstance(level, (list, tuple)) else [level]
            sql += f" AND level IN ({','.join('?' * len(levels))})"
            args += list(levels)
        if type_:
            sql += " AND type=?"
            args.append(type_)
        if since:
            sql += " AND ts>=?"
            args.append(since)
        if until:
            sql += " AND ts<=?"
            args.append(until)
        if search:
            sql += " AND description LIKE ?"
            args.append(f"%{search}%")
        if unacked_only:
            sql += " AND acknowledged=0"
        sql += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]

        with self.lock:
            rows = self.conn.execute(sql, args).fetchall()
        alerts = []
        for row in rows:
            alert = dict(row)
            alert["track_ids"] = json.loads(alert["track_ids"] or "[]")
            alert["reasons"] = json.loads(alert["reasons"] or "[]")
            alert["acknowledged"] = bool(alert["acknowledged"])
            alerts.append(alert)
        return alerts

    def unacknowledged_count(self, since: float = 0) -> int:
        with self.lock:
            row = self.conn.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0 AND ts>=?", (since,)).fetchone()
        return row[0]
