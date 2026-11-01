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

Every person is tracked with YOLOv8-pose + ByteTrack. Each sample is --window poses taken
--fps times a second (the same way the live system samples them, whatever the clip's frame
rate), cut every --stride seconds, and labelled with the label that covers its last frame.

With folder labels, samples are only kept for people who really show the activity, so
bystanders in a falling clip aren't labelled "falling" (src/activity/clip_filter.py).
Use --no-filter to keep everyone. Frame-level CSV labels are used as they are.
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
from src.activity.clip_filter import frames_showing_activity, stitch_tracks  # noqa: E402
from src.activity.features import sample_by_time  # noqa: E402
from src.activity.recognizer import ActivityRecognizer  # noqa: E402
from src.config import load_config  # noqa: E402
from src.detection.detector import build_detector  # noqa: E402
from src.tracking.track_state import TrackManager  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".webm"}
MIN_HISTORY_SECONDS = 0.5   # the live system also starts predicting before it has a full sequence
MAX_GAP_SECONDS = 0.6       # a longer gap in a track splits it into separate runs


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


def split_runs(records):
    """Split a track wherever the person was lost for too long."""
    runs, current = [], records[:1]
    for prev, rec in zip(records, records[1:]):
        if rec[1] - prev[1] > MAX_GAP_SECONDS:
            runs.append(current)
            current = []
        current.append(rec)
    if current:
        runs.append(current)
    return runs


def windows_from_tracks(records, label_for, window, fps, stride_seconds, group):
    samples = []
    for track_id, recs in records.items():
        for run in split_runs(recs):
            times = [r[1] for r in run]
            next_end = times[0] + MIN_HISTORY_SECONDS
            for end, t in enumerate(times):
                if t < next_end:
                    continue
                next_end = t + stride_seconds
                label = label_for(run[end][0], track_id)
                if label is None:
                    continue
                chunk = [run[i] for i in sample_by_time(times[:end + 1], window, fps)]
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
    parser.add_argument("--window", type=int, help="poses per sample (default: activity.window)")
    parser.add_argument("--fps", type=float, help="poses per second (default: activity.sequence_fps)")
    parser.add_argument("--stride", type=float, default=0.25, help="seconds between samples")
    parser.add_argument("--no-filter", action="store_true", help="label everyone in a clip with its folder name")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg["activity"]["classes"]
    window = args.window or cfg["activity"]["window"]
    sample_fps = args.fps or cfg["activity"]["sequence_fps"]
    detector = build_detector(cfg)
    annotations = load_annotations(args.annotations) if args.annotations else None
    videos = [p for p in sorted(Path(args.videos).rglob("*")) if p.suffix.lower() in VIDEO_EXT]
    print(f"{len(videos)} videos found")

    samples = []
    empty_clips = []
    for n, video in enumerate(videos, 1):
        clip_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 25.0
        detector.reset()   # fresh track ids for every clip
        records = track_video(video_frames(video), detector, clip_fps)

        if annotations is not None:
            spans = annotations.get(video.name, [])

            def label_for(frame_idx, track_id, spans=spans):
                for start, end, label, only_track in spans:
                    if start <= frame_idx <= end and only_track in (None, track_id):
                        return label
                return None
        else:
            folder = video.parent.name
            rules = cfg["activity"]["rules"]
            records = stitch_tracks(records)
            keep = None if args.no_filter else frames_showing_activity(folder, records, rules)

            def label_for(frame_idx, track_id, folder=folder, keep=keep):
                if folder not in classes or (keep is not None and frame_idx not in keep[track_id]):
                    return None
                return folder

        new = windows_from_tracks(records, label_for, window, sample_fps, args.stride, video.stem)
        samples += new
        if not new:
            empty_clips.append(str(video.relative_to(args.videos)))
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
                        classes=np.array(classes),
                        fps=np.float32(sample_fps))
    counts = {c: sum(s["label"] == c for s in samples) for c in classes}
    print(f"Saved {len(samples)} windows to {out}. Windows per class: {counts}")
    if empty_clips:
        print(f"\n{len(empty_clips)} clip(s) gave no samples, so check or re-record them:")
        for clip in empty_clips:
            print(f"  {clip}")


if __name__ == "__main__":
    main()
