#!/usr/bin/env python3
"""Build a pose-sequence dataset for activity recognition.

Mode 1 - real videos, one sub-folder per class (clip-level labels):
    data/activity_videos/
        walking/*.mp4   standing/*.mp4   running/*.mp4   fighting/*.mp4   falling/*.mp4
    python scripts/extract_pose_dataset.py --videos data/activity_videos --out data/activity/dataset.npz

Mode 2 - real videos with frame-level annotations (recommended for fall / fight datasets where
the action occupies only part of the clip). CSV columns: video,start_frame,end_frame,label[,track_id]
    python scripts/extract_pose_dataset.py --videos data/raw --annotations data/raw/labels.csv --out ...

Mode 3 - the synthetic scene (pipeline self-test only, NOT a substitute for real data):
    python scripts/extract_pose_dataset.py --simulated 3 --out data/activity/sim_dataset.npz

Suggested public sources: Le2i / UR Fall Detection (falling), RWF-2000, Hockey Fight, Surveillance
Camera Fight (fighting), KTH & UCF-101 / own CCTV recordings (walking, running, standing).
Every tracked person is run through YOLOv8-pose + ByteTrack; windows of --window frames are taken
with --stride and labelled with the label covering the window's last frame.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activity.recognizer import ActivityRecognizer  # noqa: E402
from src.config import load_config  # noqa: E402
from src.detection.detector import build_detector  # noqa: E402
from src.pipeline.simulator import FPS, SimulatedSource, scene_at  # noqa: E402
from src.tracking.track_state import TrackManager  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".webm"}


def load_annotations(path):
    ann = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ann[Path(r["video"]).name].append((int(r["start_frame"]), int(r["end_frame"]), r["label"].strip(),
                                               int(r["track_id"]) if r.get("track_id") else None))
    return ann


def track_video(frames_iter, detector, fps, max_frames=None, meta_iter=None):
    """Run detection+tracking over a clip and return per-track lists of per-frame records."""
    tm = TrackManager(stale_seconds=1e9)
    recs = defaultdict(list)          # tid -> [(frame, t, bbox, kps, nbr, refh)]
    for fi, frame in enumerate(frames_iter):
        if max_frames and fi >= max_frames:
            break
        meta = next(meta_iter) if meta_iter else None
        t = fi / fps
        res = detector(frame, fi, meta)
        active = tm.update(t, res.persons)
        ActivityRecognizer.update_neighbours(active)
        for tr in active:
            d = tr.last_det
            kp = d.keypoints if d.keypoints is not None else np.zeros((17, 3))
            recs[tr.track_id].append((fi, t, d.bbox.copy(), kp.copy(), tr.nbr_dist[-1], tr.body_height()))
    return recs


def windows_from_tracks(recs, label_fn, window, stride, group):
    out = []
    for tid, r in recs.items():
        # split into contiguous runs (a track may be lost for a few frames)
        runs, cur = [], [r[0]] if r else []
        for a, b in zip(r, r[1:]):
            if b[0] - a[0] > 3:
                runs.append(cur)
                cur = []
            cur.append(b)
        if cur:
            runs.append(cur)
        for run in runs:
            for end in range(window, len(run) + 1, stride):
                w = run[end - window:end]
                lab = label_fn(w[-1][0], tid)
                if lab is None:
                    continue
                out.append(dict(times=[x[1] for x in w], bboxes=[x[2] for x in w], kps=[x[3] for x in w],
                                nbr=[x[4] for x in w], refh=w[-1][5], label=lab, group=group))
    return out


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f
    cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", help="folder with class sub-folders or raw clips")
    ap.add_argument("--annotations", help="CSV with frame-level labels")
    ap.add_argument("--simulated", type=int, default=0, help="number of synthetic 100-s cycles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["activity"]["classes"]
    samples = []

    if args.simulated:
        cfg["source"]["uri"] = "simulated"
        det = build_detector(cfg, "simulated")
        for c in range(args.simulated):
            src = SimulatedSource()
            n = int(100 * FPS)
            gt = {}

            def frames():
                for _ in range(n):
                    ok, f = src.read()
                    gt[src.idx - 1] = {p["id"]: p["activity"] for p in src.last_meta["persons"]}
                    yield f

            def metas():
                while True:
                    yield src.last_meta
            det.noise = 1.0 + c  # a bit more keypoint noise every cycle
            recs = track_video(frames(), det, FPS, meta_iter=metas())
            samples += windows_from_tracks(recs, lambda fi, tid: gt.get(fi, {}).get(tid), args.window, args.stride,
                                           f"sim_cycle{c}")
    else:
        root = Path(args.videos)
        det = build_detector(cfg, cfg["detection"]["backend"] if cfg["detection"]["backend"] != "simulated" else "yolo")
        ann = load_annotations(args.annotations) if args.annotations else None
        vids = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in VIDEO_EXT]
        print(f"{len(vids)} videos found")
        for vi, v in enumerate(vids):
            fps = cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FPS) or 25.0
            if hasattr(det, "model"):          # reset ByteTrack between clips
                det.model.predictor = None
            recs = track_video(video_frames(v), det, fps)
            if ann is not None:
                spans = ann.get(v.name, [])

                def label_fn(fi, tid, spans=spans):
                    for s, e, lab, t_only in spans:
                        if s <= fi <= e and (t_only is None or t_only == tid):
                            return lab
                    return None
            else:
                folder = v.parent.name
                label_fn = (lambda fi, tid, f=folder: f) if folder in classes else (lambda fi, tid: None)
            new = windows_from_tracks(recs, label_fn, args.window, args.stride, v.stem)
            samples += new
            print(f"[{vi + 1}/{len(vids)}] {v.name}: {len(recs)} tracks -> {len(new)} windows")

    samples = [s for s in samples if s["label"] in classes]
    if not samples:
        sys.exit("No labelled windows extracted - check folder names / annotations.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        times=np.array([s["times"] for s in samples], np.float32),
                        bboxes=np.array([s["bboxes"] for s in samples], np.float32),
                        kps=np.array([s["kps"] for s in samples], np.float32),
                        nbr=np.array([s["nbr"] for s in samples], np.float32),
                        refh=np.array([s["refh"] for s in samples], np.float32),
                        labels=np.array([s["label"] for s in samples]),
                        groups=np.array([s["group"] for s in samples]),
                        classes=np.array(classes))
    counts = {c: sum(s["label"] == c for s in samples) for c in classes}
    print(f"Saved {len(samples)} windows to {out}  class counts: {counts}")


if __name__ == "__main__":
    main()
