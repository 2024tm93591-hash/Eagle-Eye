#!/usr/bin/env python3
"""Record labelled activity clips from a webcam for training the LSTM / TCN models.

    python scripts/record_clips.py --label walking
    python scripts/record_clips.py --label fighting --count 10 --seconds 8

Each clip is saved to data/activity_videos/<label>/, ready for extract_pose_dataset.py.
A window shows the camera with a countdown before every clip, so you can step back into
position and act it out without touching the keyboard.

Keys in the window:  q / Esc = stop    d = delete the last clip (a bad take)    space = skip the wait

Tips: act only the chosen activity for the whole clip, keep other people out of view (except
your partner in fighting clips), and vary where you stand, which way you face and what you wear.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402

WINDOW = "Eagle Eye - record clips"
RED, WHITE, AMBER = (40, 40, 230), (240, 240, 240), (19, 164, 242)


def put(img, text, y, color=WHITE, scale=0.8):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def save_clip(frames, elapsed: float, path: Path) -> float:
    """Write the frames with the frame rate the camera really delivered.

    Webcams often run slower than they report, and the training script works out speeds
    from frame number / fps, so a wrong fps would make every movement look faster or slower.
    """
    fps = len(frames) / max(elapsed, 1e-6)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return fps


def wait(cap, seconds, lines, color=AMBER):
    """Show the live view for `seconds` with a countdown. Returns a key ('q', 'd') or None."""
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            return "q"
        view = frame.copy()
        for i, line in enumerate(lines):
            put(view, line, 40 + 35 * i)
        put(view, f"{end - time.time():.0f}", view.shape[0] // 2, color, 3.0)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "q"
        if key == ord("d"):
            return "d"
        if key == ord(" "):
            return None
    return None


def record(cap, seconds, label, number, total):
    """Record one clip. Returns (frames, elapsed seconds), or None if the user stopped."""
    frames = []
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)   # the overlay goes on a copy, never into the saved clip
        view = frame.copy()
        cv2.circle(view, (view.shape[1] - 40, 40), 14, RED, -1)
        put(view, f"REC  {label}  clip {number}/{total}", 40, RED)
        put(view, f"{seconds - (time.time() - start):.0f}s left", 75)
        cv2.imshow(WINDOW, view)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            return None
    return frames, time.time() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="activity to record (one of activity.classes in config.yaml)")
    parser.add_argument("--count", type=int, default=5, help="number of clips to record")
    parser.add_argument("--seconds", type=float, default=6, help="length of each clip (3 to 20)")
    parser.add_argument("--countdown", type=float, default=5, help="seconds to get into position before each clip")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--out", default="data/activity_videos")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    classes = load_config(args.config)["activity"]["classes"]
    if args.label not in classes:
        sys.exit(f"--label must be one of {classes}")
    if not 3 <= args.seconds <= 20:
        # a sample covers 3 s of movement, and long clips are held in memory until saved
        sys.exit("--seconds must be between 3 and 20")

    folder = resolve_path(args.out) / args.label
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Is Eagle Eye (run.py) still using it?")

    session = time.strftime("%Y%m%d_%H%M%S")
    saved = []
    number = 1
    try:
        while number <= args.count:
            key = wait(cap, args.countdown, [f"Next: {args.label}  (clip {number}/{args.count})",
                                             "Get into position", "q = stop   d = delete last clip"])
            if key == "q":
                break
            if key == "d":
                if saved:
                    gone = saved.pop()
                    gone.unlink(missing_ok=True)
                    number -= 1
                    print(f"deleted {gone.name}")
                continue

            result = record(cap, args.seconds, args.label, number, args.count)
            if result is None:
                break
            frames, elapsed = result
            if len(frames) < 10:
                print("camera stopped sending frames - clip not saved")
                break
            path = folder / f"{args.label}_{session}_{number:02d}.mp4"
            fps = save_clip(frames, elapsed, path)
            saved.append(path)
            print(f"saved {path.name}  ({len(frames)} frames, {fps:.1f} fps)")
            number += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\n{len(saved)} clip(s) saved to {folder}")
    print("Clips per activity so far:")
    for name in classes:
        count = len(list((resolve_path(args.out) / name).glob("*.mp4")))
        print(f"  {name:<10} {count}")


if __name__ == "__main__":
    main()
