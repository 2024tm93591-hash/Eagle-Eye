#!/usr/bin/env python3
"""Start ThreatVision: the analytics pipeline + the monitoring dashboard.

Examples
  python run.py                                   # uses config.yaml
  python run.py --source data/videos/mall.mp4     # analyse a video file
  python run.py --source 0                        # webcam
  python run.py --source rtsp://192.168.1.10/live # IP camera
  python run.py --source simulated                # built-in synthetic scene (no models needed)
  python run.py --activity lstm --detector yolo --port 8080
"""
import argparse
import logging

from src.config import load_config
from src.dashboard.app import create_app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", help="webcam index, video path, RTSP/HTTP URL or 'simulated'")
    ap.add_argument("--detector", choices=["yolo", "hog"])
    ap.add_argument("--activity", choices=["rule_based", "lstm", "tcn"])
    ap.add_argument("--gender", choices=["face_dnn", "body_cnn", "hybrid", "none"])
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()

    ov = {}
    if args.source is not None:
        ov.setdefault("source", {})["uri"] = args.source
    if args.detector:
        ov.setdefault("detection", {})["backend"] = args.detector
    if args.activity:
        ov.setdefault("activity", {})["backend"] = args.activity
    if args.gender:
        ov.setdefault("gender", {})["backend"] = args.gender
    if args.host:
        ov.setdefault("dashboard", {})["host"] = args.host
    if args.port:
        ov.setdefault("dashboard", {})["port"] = args.port

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    cfg = load_config(args.config, ov)
    app = create_app(cfg)
    d = cfg["dashboard"]
    print(f"\n  ThreatVision dashboard -> http://localhost:{d['port']}\n")
    app.run(host=d["host"], port=d["port"], threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
