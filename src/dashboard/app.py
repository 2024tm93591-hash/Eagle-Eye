"""Flask monitoring dashboard: live video, scene statistics, threat alerts, history, metrics."""
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
from ..pipeline.processor import VideoProcessor
from ..storage.db import EventStore
from ..threat.alerts import AlertManager

log = logging.getLogger("threatvision.dashboard")
ALLOWED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_app(cfg: dict, start_processor: bool = True) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    db_path = resolve_path(cfg["alerts"]["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = EventStore(db_path)
    alerts = AlertManager(cfg, store)
    proc = VideoProcessor(cfg, store, alerts)
    if start_processor:
        proc.start()
    app.extensions["tv"] = dict(cfg=cfg, store=store, alerts=alerts, proc=proc)

    @app.context_processor
    def inject():
        return {"camera": cfg["source"].get("camera_name", ""), "page": request.endpoint}

    # ------------------------------------------------------------------ pages
    @app.route("/")
    def live():
        return render_template("live.html")

    @app.route("/history")
    def history():
        return render_template("history.html", types=list(cfg["threat"]["base_scores"]))

    @app.route("/analytics")
    def analytics():
        return render_template("analytics.html")

    @app.route("/metrics")
    def metrics():
        return render_template("metrics.html")

    @app.route("/settings")
    def settings():
        vids = sorted(p.name for p in (PROJECT_ROOT / "data" / "videos").glob("*") if p.suffix.lower() in ALLOWED_VIDEO)
        return render_template("settings.html", videos=vids, current=str(proc.uri),
                               cfg_text=json.dumps({k: cfg[k] for k in ("detection", "gender", "activity", "scene",
                                                                        "threat")}, indent=2))

    # ------------------------------------------------------------------ live video (MJPEG)
    @app.route("/video_feed")
    def video_feed():
        def gen():
            last = None
            while True:
                with proc.new_frame:
                    proc.new_frame.wait(timeout=1.0)
                    jpg = proc.latest_jpeg
                if jpg is None or jpg is last:
                    continue
                last = jpg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snapshot.jpg")
    def current_snapshot():
        if proc.latest_jpeg is None:
            abort(404)
        return Response(proc.latest_jpeg, mimetype="image/jpeg")

    @app.route("/snapshots/<path:name>")
    def snapshots(name):
        return send_from_directory(resolve_path(cfg["alerts"]["snapshot_dir"]), name)

    # ------------------------------------------------------------------ live push (Server-Sent Events)
    @app.route("/api/stream")
    def stream():
        q = alerts.subscribe()

        def gen():
            try:
                yield "retry: 3000\n\n"
                last_stats = 0.0
                while True:
                    try:
                        msg = q.get(timeout=1.0)
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        pass
                    if time.time() - last_stats >= 1.0:
                        last_stats = time.time()
                        yield f"data: {json.dumps({'kind': 'stats', 'stats': proc.stats})}\n\n"
            finally:
                alerts.unsubscribe(q)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ REST API
    @app.route("/api/stats")
    def api_stats():
        return jsonify(proc.stats)

    @app.route("/api/alerts")
    def api_alerts():
        a = request.args
        levels = [x for x in a.get("level", "").split(",") if x]
        rows = store.query_alerts(level=levels or None, type_=a.get("type") or None,
                                  since=_ts(a.get("since")), until=_ts(a.get("until"), end=True),
                                  search=a.get("q") or None, limit=int(a.get("limit", 200)),
                                  offset=int(a.get("offset", 0)), unacked_only=a.get("unacked") == "1")
        return jsonify(rows)

    @app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
    def api_ack(alert_id):
        ok = store.acknowledge(alert_id, (request.json or {}).get("user", "operator") if request.is_json else "operator")
        alerts.broadcast({"kind": "ack", "id": alert_id})
        return jsonify({"ok": ok})

    @app.route("/api/summary")
    def api_summary():
        hours = float(request.args.get("hours", 24))
        return jsonify(store.alert_summary(time.time() - hours * 3600))

    @app.route("/api/occupancy")
    def api_occupancy():
        minutes = float(request.args.get("minutes", 60))
        return jsonify(store.stats_since(time.time() - minutes * 60))

    @app.route("/api/metrics")
    def api_metrics():
        p = resolve_path(cfg["dashboard"]["metrics_file"])
        data = json.loads(p.read_text()) if p.exists() else {}
        data["live"] = {"fps": proc.stats.get("fps"), "latency_ms": proc.stats.get("latency_ms"),
                        "stage_ms": proc.stats.get("stage_ms"), "models": proc.stats.get("models")}
        return jsonify(data)

    @app.route("/api/export.csv")
    def api_export():
        rows = store.query_alerts(limit=100000)
        buf = io.StringIO()
        w = csv.writer(buf)
        cols = ["time", "camera", "level", "score", "type", "description", "track_ids", "zone", "duration",
                "reasons", "acknowledged", "snapshot"]
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r[c]) if isinstance(r[c], list) else r[c] for c in cols])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=threatvision_events.csv"})

    @app.route("/api/source", methods=["POST"])
    def api_source():
        uri = (request.json or {}).get("uri", "").strip()
        if not uri:
            return jsonify({"ok": False, "error": "empty source"}), 400
        if uri not in ("simulated",) and not uri.isdigit() and "://" not in uri:
            cand = PROJECT_ROOT / "data" / "videos" / Path(uri).name
            if not cand.exists():
                return jsonify({"ok": False, "error": f"video not found: {uri}"}), 404
            uri = str(cand.relative_to(PROJECT_ROOT))
        proc.switch_source(uri)
        return jsonify({"ok": True, "uri": uri})

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        f = request.files.get("video")
        if not f or Path(f.filename).suffix.lower() not in ALLOWED_VIDEO:
            return jsonify({"ok": False, "error": "please upload an .mp4/.avi/.mov/.mkv/.webm file"}), 400
        name = secure_filename(f.filename)
        dest = PROJECT_ROOT / "data" / "videos" / name
        f.save(dest)
        proc.switch_source(str(dest.relative_to(PROJECT_ROOT)))
        return jsonify({"ok": True, "uri": name})

    return app


def _ts(s, end=False):
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' -> epoch seconds."""
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            t = time.mktime(time.strptime(s, fmt))
            return t + (86399 if end and fmt == "%Y-%m-%d" else 0)
        except ValueError:
            pass
    return None
