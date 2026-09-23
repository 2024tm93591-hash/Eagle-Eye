"""The monitoring dashboard (Flask): live video, alerts, event history and analytics."""
from __future__ import annotations

import csv
import io
import json
import logging
import queue
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from ..config import PROJECT_ROOT, resolve_path
from ..evaluation.report import algorithm_report
from ..pipeline.processor import VideoProcessor
from ..storage.db import EventStore
from ..threat.alerts import AlertManager

log = logging.getLogger("threatvision.dashboard")

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# what the Analytics page shows, in order, and how to produce each part if it's missing
REPORT_SECTIONS = [
    ("detection", "Person detection",
     "python scripts/evaluate.py detection --images data/eval/coco128"),
    ("gender", "Gender estimation",
     "python scripts/evaluate.py gender --images data/gender/test"),
    ("activity", "Activity recognition (standing, walking, running, fighting, falling)",
     "python scripts/evaluate.py activity --data data/activity/dataset.npz"),
    ("threat", "Threat / suspicious event detection",
     "python scripts/evaluate.py threat --videos data/eval/threat --annotations data/eval/threat/events.json"),
]


def create_app(cfg: dict, start_processor: bool = True) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024   # 1 GB video uploads

    db_path = resolve_path(cfg["alerts"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = EventStore(db_path)
    alerts = AlertManager(cfg, store)
    proc = VideoProcessor(cfg, alerts)
    if start_processor:
        proc.start()

    @app.context_processor
    def page_globals():
        return {"camera": cfg["source"].get("camera_name", ""), "page": request.endpoint}

    @app.route("/")
    def live():
        return render_template("live.html")

    @app.route("/history")
    def history():
        return render_template("history.html", types=list(cfg["threat"]["base_scores"]))

    @app.route("/analytics")
    def analytics():
        results = load_metrics(cfg)
        return render_template("analytics.html", sections=build_report(results),
                               evaluated_at=results.get("generated_at"))

    @app.route("/settings")
    def settings():
        videos = sorted(p.name for p in VIDEO_DIR.glob("*") if p.suffix.lower() in ALLOWED_VIDEO)
        shown = {k: cfg[k] for k in ("detection", "gender", "activity", "scene", "threat")}
        return render_template("settings.html", videos=videos, current=str(proc.uri),
                               cfg_text=json.dumps(shown, indent=2))

    @app.route("/video_feed")
    def video_feed():
        # MJPEG: the browser keeps this response open and swaps in each new JPEG
        def frames():
            last = None
            while True:
                with proc.new_frame:
                    proc.new_frame.wait(timeout=1.0)
                    jpg = proc.latest_jpeg
                if jpg is None or jpg is last:
                    continue
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snapshot.jpg")
    def current_snapshot():
        if proc.latest_jpeg is None:
            abort(404)
        return Response(proc.latest_jpeg, mimetype="image/jpeg")

    @app.route("/snapshots/<path:name>")
    def snapshots(name):
        return send_from_directory(resolve_path(cfg["alerts"]["snapshot_dir"]), name)

    @app.route("/api/stream")
    def stream():
        """Server-sent events: new alerts as they happen, plus the live stats once a second."""
        q = alerts.subscribe()

        def events():
            try:
                yield "retry: 3000\n\n"
                last_stats = 0.0
                while True:
                    try:
                        yield f"data: {json.dumps(q.get(timeout=1.0))}\n\n"
                    except queue.Empty:
                        pass
                    if time.time() - last_stats >= 1.0:
                        last_stats = time.time()
                        yield f"data: {json.dumps({'kind': 'stats', 'stats': proc.stats})}\n\n"
            finally:
                alerts.unsubscribe(q)

        return Response(events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/stats")
    def api_stats():
        return jsonify(proc.stats)

    @app.route("/api/feed", methods=["POST"])
    def api_feed():
        on = bool((request.get_json(silent=True) or {}).get("on", True))
        proc.set_feed(on)
        return jsonify({"ok": True, "on": on})

    @app.route("/api/alerts")
    def api_alerts():
        args = request.args
        levels = [level for level in args.get("level", "").split(",") if level]
        rows = store.query_alerts(level=levels or None, type_=args.get("type") or None,
                                  since=parse_date(args.get("since")), until=parse_date(args.get("until"), end=True),
                                  search=args.get("q") or None, limit=int(args.get("limit", 200)),
                                  offset=int(args.get("offset", 0)), unacked_only=args.get("unacked") == "1")
        return jsonify(rows)

    @app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
    def api_ack(alert_id):
        user = (request.get_json(silent=True) or {}).get("user", "operator")
        ok = store.acknowledge(alert_id, user)
        alerts.broadcast({"kind": "ack", "id": alert_id})
        return jsonify({"ok": ok})

    @app.route("/api/alerts/unacknowledged")
    def api_unacknowledged():
        return jsonify({"count": store.unacknowledged_count(time.time() - 24 * 3600)})

    @app.route("/api/export.csv")
    def api_export():
        columns = ["time", "camera", "level", "score", "type", "description", "track_ids", "duration",
                   "reasons", "acknowledged", "snapshot"]
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(columns)
        for row in store.query_alerts(limit=100000):
            writer.writerow([json.dumps(row[c]) if isinstance(row[c], list) else row[c] for c in columns])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=threatvision_events.csv"})

    @app.route("/api/source", methods=["POST"])
    def api_source():
        uri = (request.get_json(silent=True) or {}).get("uri", "").strip()
        if not uri:
            return jsonify({"ok": False, "error": "empty source"}), 400
        if not uri.isdigit() and "://" not in uri:
            # a plain name means a file in data/videos
            path = VIDEO_DIR / Path(uri).name
            if not path.exists():
                return jsonify({"ok": False, "error": f"video not found: {uri}"}), 404
            uri = str(path.relative_to(PROJECT_ROOT))
        proc.switch_source(uri)
        return jsonify({"ok": True, "uri": uri})

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        upload = request.files.get("video")
        if not upload or Path(upload.filename).suffix.lower() not in ALLOWED_VIDEO:
            return jsonify({"ok": False, "error": "please upload an .mp4/.avi/.mov/.mkv/.webm file"}), 400
        dest = VIDEO_DIR / secure_filename(upload.filename)
        upload.save(dest)
        proc.switch_source(str(dest.relative_to(PROJECT_ROOT)))
        return jsonify({"ok": True, "uri": dest.name})

    return app


def load_metrics(cfg) -> dict:
    path = resolve_path(cfg["dashboard"]["metrics_file"])
    return json.loads(path.read_text()) if path.exists() else {}


def build_report(results: dict) -> list[dict]:
    sections = []
    for key, title, command in REPORT_SECTIONS:
        data = results.get(key) or {}
        algorithms = [(name, algorithm_report(metrics, data.get("classes")))
                      for name, metrics in (data.get("algorithms") or {}).items()]
        sections.append({"title": title, "command": command, "dataset": data.get("dataset"),
                         "notes": data.get("notes"), "algorithms": algorithms})
    return sections


def parse_date(text, end=False):
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' to epoch seconds. A bare date used as the end of a
    range means the end of that day."""
    if not text:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            t = time.mktime(time.strptime(text, fmt))
        except ValueError:
            continue
        return t + 86399 if end and fmt == "%Y-%m-%d" else t
    return None
