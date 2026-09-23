#!/usr/bin/env python3
"""Build a pose-sequence dataset for training and testing activity recognition.

Clip-level labels - one sub-folder per class:
    data/activity_videos/
        walking/*.mp4   standing/*.mp4   running/*.mp4   fighting/*.mp4   falling/*.mp4
    python scripts/extract_pose_dataset.py --videos data/activity_videos --out data/activity/dataset.npz

Frame-level labels - better for fall and fight clips, where the action is only part of the
video. CSV columns: video,start_frame,end_frame,label[,track_id]
    python scripts/extract_pose_dataset.py --videos data/raw --annotations data/raw/labels.csv --out ...

Useful public data: Le2i / UR Fall Detection (falling), RWF-2000, Hockey Fight and Surveillance
Camera Fight (fighting), KTH, UCF-101 or your own CCTV footage (walking, running, standing).

Every person is tracked with YOLOv8-pose + ByteTrack. Windows of --window frames are cut every
--stride frames and get the label that covers the window's last frame.
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
from src.tracking.track_state import TrackManager  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".webm"}


def load_annotations(path):
    spans = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            track = int(row["track_id"]) if row.get("track_id") else None
            spans[Path(row["video"]).name].append(
                (int(row["start_frame"]), int(row["end_frame"]), row["label"].strip(), track))
    return spans


def track_video(frames, detector, fps):
    """Detect and track everyone in a clip. Returns {track id: [per-frame records]}."""
    tracks = TrackManager(stale_seconds=1e9)
    records = defaultdict(list)   # (frame, t, bbox, keypoints, nearest distance, body height)
    for frame_idx, frame in enumerate(frames):
        t = frame_idx / fps
        active = tracks.update(t, detector(frame, frame_idx).persons)
        ActivityRecognizer.update_neighbours(active)
        for track in active:
            det = track.last_det
            keypoints = det.keypoints if det.keypoints is not None else np.zeros((17, 3))
            records[track.track_id].append((frame_idx, t, det.bbox.copy(), keypoints.copy(),
                                            track.nbr_dist[-1], track.body_height()))
    return records


def split_runs(records, max_gap=3):
    """Split a track into stretches without gaps (a track can be lost for a few frames)."""
    runs, current = [], records[:1]
    for prev, rec in zip(records, records[1:]):
        if rec[0] - prev[0] > max_gap:
            runs.append(current)
            current = []
        current.append(rec)
    if current:
        runs.append(current)
    return runs


def windows_from_tracks(records, label_for, window, stride, group):
    samples = []
    for track_id, recs in records.items():
        for run in split_runs(recs):
            for end in range(window, len(run) + 1, stride):
                chunk = run[end - window:end]
                label = label_for(chunk[-1][0], track_id)
                if label is None:
                    continue
                samples.append(dict(times=[r[1] for r in chunk], bboxes=[r[2] for r in chunk],
                                    kps=[r[3] for r in chunk], nbr=[r[4] for r in chunk],
                                    refh=chunk[-1][5], label=label, group=group))
    return samples


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="folder with class sub-folders or raw clips")
    parser.add_argument("--annotations", help="CSV with frame-level labels")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["activity"]["classes"]
    detector = build_detector(cfg)
    annotations = load_annotations(args.annotations) if args.annotations else None
    videos = [p for p in sorted(Path(args.videos).rglob("*")) if p.suffix.lower() in VIDEO_EXT]
    print(f"{len(videos)} videos found")

    samples = []
    for n, video in enumerate(videos, 1):
        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 25.0
        detector.reset()   # fresh track ids for every clip
        records = track_video(video_frames(video), detector, fps)

        if annotations is not None:
            spans = annotations.get(video.name, [])

            def label_for(frame_idx, track_id, spans=spans):
                for start, end, label, only_track in spans:
                    if start <= frame_idx <= end and only_track in (None, track_id):
                        return label
                return None
        else:
            folder = video.parent.name
            def label_for(frame_idx, track_id, folder=folder):
                return folder if folder in classes else None

        new = windows_from_tracks(records, label_for, args.window, args.stride, video.stem)
        samples += new
        print(f"[{n}/{len(videos)}] {video.name}: {len(records)} tracks -> {len(new)} windows")

    samples = [s for s in samples if s["label"] in classes]
    if not samples:
        sys.exit("No labelled windows extracted - check the folder names or the annotations.")

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
    print(f"Saved {len(samples)} windows to {out}. Windows per class: {counts}")


if __name__ == "__main__":
    main()
