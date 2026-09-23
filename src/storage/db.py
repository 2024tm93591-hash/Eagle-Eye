"""SQLite event history (alerts) and periodic scene statistics."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY, ts REAL, time TEXT, camera TEXT, type TEXT, level TEXT, score REAL,
    description TEXT, track_ids TEXT, zone TEXT, duration REAL, reasons TEXT, snapshot TEXT,
    acknowledged INTEGER DEFAULT 0, ack_by TEXT, ack_time REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
CREATE TABLE IF NOT EXISTS stats (
    ts REAL, camera TEXT, persons INTEGER, male INTEGER, female INTEGER, unknown INTEGER,
    activities TEXT, fps REAL
);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON stats(ts);
"""


class EventStore:
    def __init__(self, path: str):
        self.path = str(path)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add_alert(self, a: dict):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO alerts (id, ts, time, camera, type, level, score, description, track_ids,"
                " zone, duration, reasons, snapshot, acknowledged) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (a["id"], a["timestamp"], a["time"], a["camera"], a["type"], a["level"], a["score"],
                 a["description"], json.dumps(a["track_ids"]), a["zone"], a["duration"],
                 json.dumps(a["reasons"]), a["snapshot"], int(a["acknowledged"])))
            self.conn.commit()

    def acknowledge(self, alert_id: str, user: str = "operator") -> bool:
        with self.lock:
            cur = self.conn.execute("UPDATE alerts SET acknowledged=1, ack_by=?, ack_time=? WHERE id=?",
                                    (user, time.time(), alert_id))
            self.conn.commit()
            return cur.rowcount > 0

    def query_alerts(self, level=None, type_=None, since=None, until=None, search=None, limit=500, offset=0,
                     unacked_only=False) -> list[dict]:
        q, args = "SELECT * FROM alerts WHERE 1=1", []
        if level:
            levels = level if isinstance(level, (list, tuple)) else [level]
            q += f" AND level IN ({','.join('?' * len(levels))})"
            args += list(levels)
        if type_:
            q += " AND type=?"
            args.append(type_)
        if since:
            q += " AND ts>=?"
            args.append(since)
        if until:
            q += " AND ts<=?"
            args.append(until)
        if search:
            q += " AND description LIKE ?"
            args.append(f"%{search}%")
        if unacked_only:
            q += " AND acknowledged=0"
        q += " ORDER BY ts DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self.lock:
            rows = self.conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["track_ids"] = json.loads(d["track_ids"] or "[]")
            d["reasons"] = json.loads(d["reasons"] or "[]")
            d["acknowledged"] = bool(d["acknowledged"])
            out.append(d)
        return out

    def alert_summary(self, since: float | None = None) -> dict:
        since = since or 0
        with self.lock:
            lv = self.conn.execute("SELECT level, COUNT(*) c FROM alerts WHERE ts>=? GROUP BY level", (since,)).fetchall()
            ty = self.conn.execute("SELECT type, COUNT(*) c FROM alerts WHERE ts>=? GROUP BY type ORDER BY c DESC",
                                   (since,)).fetchall()
            hr = self.conn.execute("SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) h, COUNT(*) c "
                                   "FROM alerts WHERE ts>=? GROUP BY h", (since,)).fetchall()
            un = self.conn.execute("SELECT COUNT(*) FROM alerts WHERE acknowledged=0 AND ts>=?", (since,)).fetchone()[0]
        return {"by_level": {r["level"]: r["c"] for r in lv}, "by_type": {r["type"]: r["c"] for r in ty},
                "by_hour": {int(r["h"]): r["c"] for r in hr}, "unacknowledged": un}

    def add_stats(self, camera, persons, male, female, unknown, activities: dict, fps):
        with self.lock:
            self.conn.execute("INSERT INTO stats VALUES (?,?,?,?,?,?,?,?)",
                              (time.time(), camera, persons, male, female, unknown, json.dumps(activities), fps))
            self.conn.commit()

    def stats_since(self, since: float) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM stats WHERE ts>=? ORDER BY ts", (since,)).fetchall()
        return [dict(r) | {"activities": json.loads(r["activities"] or "{}")} for r in rows]
