#!/usr/bin/env python3
"""Start ThreatVision: the video analytics pipeline and the monitoring dashboard.

Examples
  python run.py                                   # settings from config.yaml (webcam 0 by default)
  python run.py --source data/videos/mall.mp4     # analyse a video file
  python run.py --source 1                        # second webcam
  python run.py --source rtsp://192.168.1.10/live # IP camera
  python run.py --activity lstm --detector yolo --port 8080
"""
import argparse
import logging

from src.config import load_config
from src.dashboard.app import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", help="webcam number, video file or RTSP/HTTP URL")
    parser.add_argument("--detector", choices=["yolo", "hog"])
    parser.add_argument("--activity", choices=["rule_based", "lstm", "tcn"])
    parser.add_argument("--gender", choices=["face_dnn", "body_cnn", "hybrid", "none"])
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    # command-line options override config.yaml
    overrides = {}
    if args.source is not None:
        overrides.setdefault("source", {})["uri"] = args.source
    if args.detector:
        overrides.setdefault("detection", {})["backend"] = args.detector
    if args.activity:
        overrides.setdefault("activity", {})["backend"] = args.activity
    if args.gender:
        overrides.setdefault("gender", {})["backend"] = args.gender
    if args.host:
        overrides.setdefault("dashboard", {})["host"] = args.host
    if args.port:
        overrides.setdefault("dashboard", {})["port"] = args.port

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    cfg = load_config(args.config, overrides)
    app = create_app(cfg)
    dashboard = cfg["dashboard"]
    print(f"\n  ThreatVision dashboard -> http://localhost:{dashboard['port']}\n")
    app.run(host=dashboard["host"], port=dashboard["port"], threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
