#!/usr/bin/env python3
"""Evaluate the algorithms and save the results to results/metrics.json (shown on the Analytics page).

  detection  person detection and counting on YOLO-format labelled images
             python scripts/evaluate.py detection --images data/eval/coco128 --algorithms yolov8n-pose,yolov8s-pose,hog
  gender     person crops sorted into <dir>/male and <dir>/female (e.g. the PA-100K test split)
             python scripts/evaluate.py gender --images data/gender/test --algorithms face_dnn,body_cnn,hybrid
  activity   the test split of a pose-sequence dataset (same per-video split as training)
             python scripts/evaluate.py activity --data data/activity/dataset.npz --algorithms rule_based,lstm,tcn
  threat     annotated clips, with a JSON file like {"clip.mp4": [{"type": "fall", "start": 3.0, "end": 9.5}]}
             python scripts/evaluate.py threat --videos data/eval/threat --annotations data/eval/threat/events.json
  speed      end-to-end frames per second for different pipeline setups
             python scripts/evaluate.py speed --video data/videos/clip.mp4 --configs yolo+rule_based,hog+rule_based

Every command reports accuracy, precision, recall, F1 score, the confusion matrix and speed.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.evaluation.metrics import classification_report, detection_metrics, event_metrics  # noqa: E402
from src.evaluation.report import algorithm_report  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# event types that can be matched against annotations
EVAL_TYPES = ["loitering", "physical_altercation", "fighting", "fall", "person_down", "chasing",
              "abandoned_object", "group_gathering", "crowd_gathering", "panic_running"]


def save_section(cfg, section: str, payload: dict):
    path = resolve_path(cfg["dashboard"]["metrics_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = payload
    data["generated_at"] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved to {path} [{section}]")


def print_results(title, results: dict, labels=None):
    print(f"\n{title}")
    print("=" * len(title))
    for name, metrics in results.items():
        print(f"\n{name}\n{'-' * len(name)}")
        print(algorithm_report(metrics, labels))


def resize_to(frame, width):
    if not width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale)


def read_yolo_labels(label_file: Path, w, h):
    """Person boxes (class 0) from a YOLO label file, converted to pixel x1, y1, x2, y2."""
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5 and int(float(parts[0])) == 0:
                cx, cy, bw, bh = (float(x) for x in parts[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, float).reshape(-1, 4)


def detection_algorithms(cfg, names):
    """Returns {display name: (detect function, operating threshold)}."""
    algorithms = {}
    for name in names:
        if name == "hog":
            from src.detection.detector import HogDetector
            low_cfg = copy.deepcopy(cfg)
            low_cfg["detection"]["hog_threshold"] = -0.5   # keep weak boxes too, AP needs them
            hog = HogDetector(low_cfg)

            def detect(img, hog=hog):
                found = hog.detect(img)
                return np.array([d.bbox for d in found]).reshape(-1, 4), np.array([d.conf * 2 for d in found])

            algorithms["HOG + linear SVM (Dalal-Triggs)"] = (detect, cfg["detection"].get("hog_threshold", 0.3))
        else:
            from ultralytics import YOLO
            weights = name if name.endswith(".pt") else f"{name}.pt"
            local = resolve_path(f"models/{weights}")
            model = YOLO(str(local) if local.exists() else weights)
            d = cfg["detection"]

            def detect(img, model=model):
                result = model.predict(img, classes=[0], conf=0.01, iou=d["iou_threshold"], imgsz=d["imgsz"],
                                       device=d.get("device") or None, verbose=False)[0]
                return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()

            algorithms[Path(weights).stem.replace("yolov8", "YOLOv8")] = (detect, d["conf_threshold"])
    return algorithms


def eval_detection(cfg, args):
    root = Path(args.images)
    image_dir = root / "images" if (root / "images").exists() else root
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)
    if args.limit:
        images = images[:args.limit]
    print(f"{len(images)} images")

    results = {}
    for name, (detect, threshold) in detection_algorithms(cfg, args.algorithms.split(",")).items():
        detect(cv2.imread(str(images[0])))   # warm-up, so model loading isn't counted in the speed
        per_image, total_time = [], 0.0
        for path in images:
            img = cv2.imread(str(path))
            h, w = img.shape[:2]
            label_file = Path(str(path).replace("images", "labels")).with_suffix(".txt")
            started = time.perf_counter()
            boxes, scores = detect(img)
            total_time += time.perf_counter() - started
            per_image.append((boxes, scores, read_yolo_labels(label_file, w, h)))

        metrics = detection_metrics(per_image, threshold)
        metrics["fps"] = round(len(images) / total_time, 2)
        metrics["latency_ms"] = round(1000 * total_time / len(images), 1)
        metrics["conf_threshold"] = threshold
        results[name] = metrics

    print_results("PERSON DETECTION", results)
    gt_people = next(iter(results.values()))["gt_persons"] if results else 0
    save_section(cfg, "detection", {
        "dataset": args.name or f"{root.name} ({len(images)} images, {gt_people} people)",
        "samples": len(images),
        "notes": "A detection is correct when it overlaps a labelled person with IoU >= 0.5. "
                 "Detection has no true negatives, so accuracy here is TP / (TP + FP + FN)",
        "algorithms": results,
    })


def eval_gender(cfg, args):
    from src.gender.classifier import build_gender

    root = Path(args.images)
    items = [(p, label) for label in ("male", "female")
             for p in sorted((root / label).glob("*")) if p.suffix.lower() in IMAGE_EXT]
    if args.limit:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.choice(len(items), min(args.limit, len(items)), replace=False)]
    print(f"{len(items)} crops")

    names = {"face_dnn": "Face DNN (Levi-Hassner)", "body_cnn": "Body CNN (MobileNetV3)",
             "hybrid": "Hybrid face -> body"}
    results = {}
    for backend in args.algorithms.split(","):
        classifier = build_gender(cfg, backend)
        if classifier is None:
            print(f"skipping {backend}: model not available")
            continue
        actual, predicted, total_time = [], [], 0.0
        for path, label in items:
            img = cv2.imread(str(path))
            started = time.perf_counter()
            probs = classifier.predict(img)
            total_time += time.perf_counter() - started
            if probs is None:
                continue   # e.g. no face found
            actual.append(label)
            predicted.append(["male", "female"][int(np.argmax(probs))])

        metrics = classification_report(actual, predicted, ["male", "female"])
        metrics["coverage"] = round(len(predicted) / max(1, len(items)), 4)
        metrics["fps"] = round(len(items) / max(total_time, 1e-9), 1)
        metrics.pop("per_class", None)
        results[names.get(backend, backend)] = metrics

    print_results("GENDER ESTIMATION", results)
    save_section(cfg, "gender", {
        "dataset": args.name or f"{root.name} ({len(items)} person crops)",
        "samples": len(items),
        "notes": "Scores are over the crops where the model gave an answer (see coverage in metrics.json)",
        "algorithms": results,
    })


def eval_activity(cfg, args):
    from src.activity.dataset import build_features, load_npz, replay_rule_based, split_indices
    from src.activity.rule_based import RuleBasedActivity

    data = load_npz(args.data)
    classes = [str(c) for c in data["classes"]]
    _, _, test = split_indices(data["groups"], seed=args.seed)
    if args.all or len(test) == 0:
        test = np.arange(len(data["labels"]))
    actual = [str(v) for v in data["labels"][test]]
    window = data["times"].shape[1]
    print(f"{len(test)} test windows")

    results = {}
    for backend in args.algorithms.split(","):
        if backend == "rule_based":
            rules = RuleBasedActivity(cfg)
            started = time.perf_counter()
            predicted = [replay_rule_based(rules, data, i, cfg["activity"].get("smoothing", 7)) for i in test]
            elapsed = time.perf_counter() - started
            frames = len(test) * window   # the rules look at every frame of every window
            name = "Rule-based (pose geometry + motion)"
        else:
            path = resolve_path(cfg["activity"][f"{backend}_model"])
            if not path.exists():
                print(f"skipping {backend}: {path} not found (train it with scripts/train_activity.py)")
                continue
            import torch
            from src.activity.models import build_model

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(backend, len(checkpoint["classes"]))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            X, _ = build_features(data, test)
            X = ((X - np.asarray(checkpoint["mean"], np.float32)) / np.asarray(checkpoint["std"], np.float32))
            started = time.perf_counter()
            with torch.no_grad():
                out = model(torch.from_numpy(X.astype(np.float32))).argmax(1).numpy()
            elapsed = time.perf_counter() - started
            frames = len(test)   # live, the model runs once per frame on the latest window
            predicted = [checkpoint["classes"][i] for i in out]
            name = {"lstm": "Bi-LSTM + attention", "tcn": "Temporal ConvNet (TCN)"}[backend]

        metrics = classification_report(actual, predicted, classes)
        metrics["fps"] = round(frames / max(elapsed, 1e-9), 1)
        results[name] = metrics

    print_results("ACTIVITY RECOGNITION", results, classes)
    save_section(cfg, "activity", {
        "dataset": args.name or f"{Path(args.data).stem} ({len(test)} test windows)",
        "samples": len(test),
        "classes": classes,
        "notes": "Test videos are kept apart from training videos. Precision, recall and F1 are "
                 "averaged over the classes",
        "algorithms": results,
    })


def run_clip(cfg, frames, fps, activity_backend=None):
    from src.pipeline.processor import Analytics

    analytics = Analytics(cfg, activity_backend=activity_backend)
    analytics.scene.quiet_override = False     # the clock of the test machine shouldn't matter
    analytics.assessor.quiet_override = False
    predictions = []
    for i, frame in enumerate(frames):
        _, new = analytics.process(frame, i, i / fps)
        predictions += [dict(type=a.event.type, t=i / fps, level=a.level) for a in new]
    return predictions, analytics


def clip_frames(path, width):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield resize_to(frame, width)
    cap.release()


def eval_threat(cfg, args):
    annotations = json.loads(Path(args.annotations).read_text())
    results = {}
    for backend in args.algorithms.split(","):
        predictions, truth, name = [], [], None
        for clip, events in annotations.items():
            path = Path(args.videos) / clip
            fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS) or 25.0
            found, analytics = run_clip(cfg, clip_frames(path, cfg["source"].get("resize_width", 0)), fps, backend)
            name = f"Scene analyser + {analytics.activity.name}"
            predictions += [dict(p, clip=clip) for p in found]
            truth += [dict(e, clip=clip) for e in events]

        # a one-person "fighting" event and a two-person "physical_altercation" count as the same thing
        for p in predictions:
            if p["type"] == "fighting":
                p["type"] = "physical_altercation"
        types = sorted({g["type"] for g in truth} | ({p["type"] for p in predictions} & set(EVAL_TYPES)))
        results[name] = event_metrics([p for p in predictions if p["type"] in types], truth,
                                      tolerance=args.tolerance, types=types)

    print_results("THREAT / SUSPICIOUS EVENT DETECTION", results)
    save_section(cfg, "threat", {
        "dataset": args.name or f"{Path(args.videos).name} ({len(annotations)} clips)",
        "notes": f"An alert counts as correct when it falls within {args.tolerance}s of an annotated event",
        "algorithms": results,
    })


def eval_speed(cfg, args):
    from src.pipeline.processor import Analytics

    results = {}
    for setup in args.configs.split(","):
        detector, activity = setup.split("+")
        cap = cv2.VideoCapture(str(resolve_path(args.video)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = cfg["source"].get("resize_width", 0)

        def read():
            ok, frame = cap.read()
            if not ok:   # loop the clip if it is shorter than --frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            return resize_to(frame, width)

        analytics = Analytics(copy.deepcopy(cfg), detector_backend=detector, activity_backend=activity)
        times, people = [], []
        for i in range(args.frames + 10):
            frame = read()
            started = time.perf_counter()
            analytics.process(frame, i, i / fps)
            if i >= 10:   # the first frames include warm-up
                times.append(time.perf_counter() - started)
                people.append(len(analytics.active_tracks))
        cap.release()

        mean = float(np.mean(times))
        results[f"{analytics.detector.name} + {analytics.activity.name}"] = {
            "fps": round(1 / mean, 2),
            "latency_ms": round(1000 * mean, 2),
            "p95_latency_ms": round(1000 * float(np.percentile(times, 95)), 2),
            "realtime_factor": round((1 / mean) / fps, 2),
            "avg_persons": round(float(np.mean(people)), 1),
            "stage_ms": {k: round(v, 2) for k, v in analytics.timing.items() if k != "render"},
        }

    print_results("END-TO-END SPEED", results)
    save_section(cfg, "system", {
        "dataset": args.name or str(args.video),
        "samples": args.frames,
        "notes": f"{platform.processor() or platform.machine()}, {args.frames} frames, analytics only",
        "algorithms": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detection")
    p.add_argument("--images", required=True)
    p.add_argument("--algorithms", default="yolov8n-pose,yolov8s-pose,hog")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--name")

    p = sub.add_parser("gender")
    p.add_argument("--images", required=True)
    p.add_argument("--algorithms", default="face_dnn,body_cnn,hybrid")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--name")

    p = sub.add_parser("activity")
    p.add_argument("--data", required=True)
    p.add_argument("--algorithms", default="rule_based,lstm,tcn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="evaluate on every window, not only the test split")
    p.add_argument("--name")

    p = sub.add_parser("threat")
    p.add_argument("--videos", required=True)
    p.add_argument("--annotations", required=True)
    p.add_argument("--algorithms", default="rule_based")
    p.add_argument("--tolerance", type=float, default=2.0)
    p.add_argument("--name")

    p = sub.add_parser("speed")
    p.add_argument("--video", required=True)
    p.add_argument("--configs", default="yolo+rule_based,hog+rule_based")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--name")

    args = parser.parse_args()
    cfg = load_config(args.config)
    commands = {"detection": eval_detection, "gender": eval_gender, "activity": eval_activity,
                "threat": eval_threat, "speed": eval_speed}
    commands[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
