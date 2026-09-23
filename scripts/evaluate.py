#!/usr/bin/env python3
"""Evaluate every implemented algorithm and write results/metrics.json (shown on the dashboard).

  detection : person detection + counting on YOLO-format labelled images
              python scripts/evaluate.py detection --images data/eval/coco128 --algorithms yolov8n-pose,yolov8s-pose,hog
  gender    : person crops in <dir>/male and <dir>/female (e.g. PA-100K test split)
              python scripts/evaluate.py gender --images data/gender/test --algorithms face_dnn,body_cnn,hybrid
  activity  : pose-sequence test split (same per-video split as training)
              python scripts/evaluate.py activity --data data/activity/dataset.npz --algorithms rule_based,lstm,tcn
  threat    : annotated clips; JSON {"clip.mp4": [{"type": "fall", "start": 3.0, "end": 9.5}, ...]}
              python scripts/evaluate.py threat --videos data/eval/threat --annotations data/eval/threat/events.json
              python scripts/evaluate.py threat --simulated           (synthetic self-test scene)
  speed     : end-to-end FPS / latency per pipeline configuration
              python scripts/evaluate.py speed --video data/videos/clip.mp4 --configs yolo+rule_based,yolo+lstm,hog+rule_based

Metrics: accuracy, precision, recall, F1-score (macro-averaged for multi-class), AP@0.5 and counting
accuracy for detection, and processing speed (FPS / latency) for every algorithm.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config, resolve_path  # noqa: E402
from src.evaluation.metrics import classification_report, detection_metrics, event_metrics  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------- results file
def save_section(cfg, section: str, payload: dict):
    p = resolve_path(cfg["dashboard"]["metrics_file"])
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[section] = payload
    data["generated_at"] = time.strftime("%Y-%m-%d %H:%M")
    p.write_text(json.dumps(data, indent=2))
    print(f"\n-> results written to {p} [{section}]")


def print_table(title, algos: dict, keys):
    print(f"\n{title}")
    print(f"{'algorithm':42s}" + "".join(f"{k:>12s}" for k in keys))
    for n, m in algos.items():
        print(f"{n[:42]:42s}" + "".join(f"{m.get(k, float('nan')):12.4f}" if isinstance(m.get(k), (int, float))
                                        else f"{'-':>12s}" for k in keys))


# ----------------------------------------------------------------------------- detection
def read_yolo_labels(lbl: Path, w, h):
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5 and int(float(p[0])) == 0:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, float).reshape(-1, 4)


def detection_algorithms(cfg, names):
    algos = {}
    for n in names:
        if n == "hog":
            from src.detection.detector import HogDetector
            c = copy.deepcopy(cfg)
            c["detection"]["hog_threshold"] = -0.5          # keep low-score boxes for the AP curve
            det = HogDetector(c)
            thr = cfg["detection"].get("hog_threshold", 0.3)

            def fn(img, det=det):
                ds = det.detect(img)
                return np.array([d.bbox for d in ds]).reshape(-1, 4), np.array([d.conf * 2 for d in ds])
            algos["HOG + linear SVM (Dalal-Triggs)"] = (fn, thr)
        else:
            from ultralytics import YOLO
            w = n if n.endswith(".pt") else f"{n}.pt"
            wp = resolve_path(f"models/{w}")
            model = YOLO(str(wp) if wp.exists() else w)

            def fn(img, model=model):
                r = model.predict(img, classes=[0], conf=0.01, iou=cfg["detection"]["iou_threshold"],
                                  imgsz=cfg["detection"]["imgsz"], device=cfg["detection"].get("device") or None,
                                  verbose=False)[0]
                return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            algos[Path(w).stem.replace("yolov8", "YOLOv8")] = (fn, cfg["detection"]["conf_threshold"])
    return algos


def eval_detection(cfg, a):
    root = Path(a.images)
    img_dir = root / "images" if (root / "images").exists() else root
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if a.limit:
        imgs = imgs[:a.limit]
    print(f"{len(imgs)} images")
    results = {}
    for name, (fn, thr) in detection_algorithms(cfg, a.algorithms.split(",")).items():
        per, t_total = [], 0.0
        fn(cv2.imread(str(imgs[0])))  # warm-up
        for p in imgs:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            lbl = Path(str(p).replace("images", "labels")).with_suffix(".txt")
            t0 = time.perf_counter()
            pb, ps = fn(img)
            t_total += time.perf_counter() - t0
            per.append((pb, ps, read_yolo_labels(lbl, w, h)))
        m = detection_metrics(per, thr)
        m["fps"] = round(len(imgs) / t_total, 2)
        m["latency_ms"] = round(1000 * t_total / len(imgs), 1)
        m["conf_threshold"] = thr
        results[name] = m
    print_table("PERSON DETECTION", results, ["precision", "recall", "f1", "ap50", "count_accuracy", "fps"])
    save_section(cfg, "detection", {"dataset": a.name or root.name, "samples": len(imgs),
                                    "notes": "IoU >= 0.5; P/R/F1 at the operating confidence threshold",
                                    "algorithms": results})


# ----------------------------------------------------------------------------- gender
def eval_gender(cfg, a):
    from src.gender.classifier import build_gender
    root = Path(a.images)
    items = [(p, c) for c in ("male", "female") for p in sorted((root / c).glob("*")) if p.suffix.lower() in IMG_EXT]
    if a.limit:
        rng = np.random.default_rng(0)
        items = [items[i] for i in rng.choice(len(items), min(a.limit, len(items)), replace=False)]
    print(f"{len(items)} crops")
    names = {"face_dnn": "Face DNN (Levi-Hassner)", "body_cnn": "Body CNN (MobileNetV3)", "hybrid": "Hybrid face->body"}
    results = {}
    for b in a.algorithms.split(","):
        clf = build_gender(cfg, b)
        if clf is None:
            print(f"skip {b}: model not available")
            continue
        yt, yp, t_total, n_pred = [], [], 0.0, 0
        for p, c in items:
            img = cv2.imread(str(p))
            t0 = time.perf_counter()
            pr = clf.predict(img)
            t_total += time.perf_counter() - t0
            if pr is None:
                continue
            n_pred += 1
            yt.append(c)
            yp.append(["male", "female"][int(np.argmax(pr))])
        m = classification_report(yt, yp, ["male", "female"])
        m["coverage"] = round(n_pred / max(1, len(items)), 4)
        m["fps"] = round(len(items) / max(t_total, 1e-9), 1)
        m["note"] = "metrics on crops where the model produced a prediction; coverage = share of crops"
        m.pop("per_class", None)
        results[names.get(b, b)] = m
    print_table("GENDER", results, ["accuracy", "precision", "recall", "f1", "coverage", "fps"])
    save_section(cfg, "gender", {"dataset": a.name or root.name, "samples": len(items), "algorithms": results})


# ----------------------------------------------------------------------------- activity
def eval_activity(cfg, a):
    from src.activity.dataset import build_features, load_npz, replay_rule_based, split_indices
    from src.activity.rule_based import RuleBasedActivity
    d = load_npz(a.data)
    classes = [str(c) for c in d["classes"]]
    _, _, te = split_indices(d["groups"], seed=a.seed)
    if a.all or len(te) == 0:
        te = np.arange(len(d["labels"]))
    y = [str(v) for v in d["labels"][te]]
    print(f"{len(te)} test windows")
    results = {}
    for b in a.algorithms.split(","):
        if b == "rule_based":
            rules = RuleBasedActivity(cfg)
            t0 = time.perf_counter()
            pred = [replay_rule_based(rules, d, i, cfg["activity"].get("smoothing", 7)) for i in te]
            dt = time.perf_counter() - t0
            name = "Rule-based (pose geometry + motion)"
        else:
            path = resolve_path(cfg["activity"][f"{b}_model"])
            if not path.exists():
                print(f"skip {b}: {path} not found (train it with scripts/train_activity.py)")
                continue
            import torch
            from src.activity.models import build_model
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model = build_model(b, len(ck["classes"]))
            model.load_state_dict(ck["state_dict"])
            model.eval()
            X, _ = build_features(d, te)
            X = ((X - np.asarray(ck["mean"], np.float32)) / np.asarray(ck["std"], np.float32)).astype(np.float32)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(torch.from_numpy(X)).argmax(1).numpy()
            dt = time.perf_counter() - t0
            pred = [ck["classes"][i] for i in out]
            name = {"lstm": "Bi-LSTM + attention", "tcn": "Temporal ConvNet (TCN)"}[b]
        m = classification_report(y, pred, classes)
        m["fps"] = round(len(te) / max(dt, 1e-9), 1)
        m["latency_ms"] = round(1000 * dt / len(te), 3)
        results[name] = m
    print_table("ACTIVITY RECOGNITION", results, ["accuracy", "precision", "recall", "f1", "fps"])
    save_section(cfg, "activity", {"dataset": a.name or Path(a.data).stem, "samples": len(te), "classes": classes,
                                   "notes": "per-video test split; macro-averaged precision/recall/F1",
                                   "algorithms": results})


# ----------------------------------------------------------------------------- threat events
SIM_GT = [  # ground truth of the scripted synthetic scene (src/pipeline/simulator.py), seconds
    dict(type="restricted_zone_intrusion", start=19.0, end=40.0),
    dict(type="loitering", start=31.0, end=64.0),
    dict(type="physical_altercation", start=36.0, end=45.0),
    dict(type="fall", start=44.0, end=50.0),
    dict(type="person_down", start=50.0, end=60.5),
    dict(type="chasing", start=61.0, end=65.0),
    dict(type="abandoned_object", start=63.0, end=95.0),
    dict(type="group_gathering", start=75.0, end=88.5),
    dict(type="panic_running", start=88.0, end=91.0),
]
EVAL_TYPES = ["restricted_zone_intrusion", "loitering", "physical_altercation", "fighting", "fall", "person_down",
              "chasing", "abandoned_object", "group_gathering", "crowd_gathering", "panic_running"]


def run_clip(cfg, frames, fps, activity_backend=None, meta_fn=None):
    from src.pipeline.processor import Analytics
    an = Analytics(cfg, activity_backend=activity_backend)
    an.scene.quiet_override = False
    an.assessor.quiet_override = False
    preds = []
    for i, f in enumerate(frames):
        _, new = an.process(f, i, i / fps, meta_fn() if meta_fn else None)
        preds += [dict(type=x.event.type, t=i / fps, level=x.level) for x in new]
    return preds, an


def eval_threat(cfg, a):
    results = {}
    backends = a.algorithms.split(",")
    for b in backends:
        c = copy.deepcopy(cfg)
        preds, gts = [], []
        if a.simulated:
            from src.pipeline.simulator import FPS, SimulatedSource
            c["source"]["uri"] = "simulated"
            src = SimulatedSource()

            def frames():
                for _ in range(int(a.cycles * 100 * FPS)):
                    yield src.read()[1]
            p, an = run_clip(c, frames(), FPS, b, lambda: src.last_meta)
            name = f"Scene analyser + {an.activity.name}"
            for k in range(a.cycles):
                gts += [dict(g, start=g["start"] + 100 * k, end=g["end"] + 100 * k) for g in SIM_GT]
            preds += p
            dataset = f"Synthetic self-test scene ({a.cycles} x 100 s) - not real footage"
        else:
            ann = json.loads(Path(a.annotations).read_text())
            name = None
            for clip, events in ann.items():
                path = Path(a.videos) / clip
                cap = cv2.VideoCapture(str(path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                def frames(cap=cap):
                    while True:
                        ok, f = cap.read()
                        if not ok:
                            break
                        rw = c["source"].get("resize_width", 0)
                        yield cv2.resize(f, None, fx=rw / f.shape[1], fy=rw / f.shape[1]) if rw else f
                p, an = run_clip(c, frames(), fps, b)
                name = f"Scene analyser + {an.activity.name}"
                preds += [dict(x, clip=clip) for x in p]
                gts += [dict(g, clip=clip) for g in events]
            dataset = a.name or Path(a.videos).name
        # single-person "fighting" and two-person "physical_altercation" are one violence category
        preds = [dict(x, type="physical_altercation" if x["type"] == "fighting" else x["type"]) for x in preds]
        types = sorted({g["type"] for g in gts} | ({x["type"] for x in preds} & set(EVAL_TYPES)))
        m = event_metrics([x for x in preds if x["type"] in types], gts, tolerance=a.tolerance, types=types)
        results[name] = m
        print(f"\n{name}")
        for ty, v in m["per_type"].items():
            print(f"  {ty:28s} P {v['precision']:.2f}  R {v['recall']:.2f}  F1 {v['f1']:.2f}  (tp {v['tp']} fp {v['fp']} fn {v['fn']})")
    print_table("THREAT / SUSPICIOUS EVENT DETECTION", results, ["precision", "recall", "f1"])
    save_section(cfg, "threat", {"dataset": dataset, "notes": f"event-level matching, +/-{a.tolerance}s tolerance",
                                 "algorithms": results})


# ----------------------------------------------------------------------------- speed
def eval_speed(cfg, a):
    from src.pipeline.processor import Analytics
    results = {}
    for conf in a.configs.split(","):
        det, act = conf.split("+")
        c = copy.deepcopy(cfg)
        if a.video == "simulated":
            from src.pipeline.simulator import FPS, SimulatedSource
            c["source"]["uri"] = "simulated"
            src, fps = SimulatedSource(), FPS
            read = lambda: src.read()[1]
            meta = lambda: src.last_meta
        else:
            c["source"]["uri"] = a.video
            cap = cv2.VideoCapture(str(resolve_path(a.video)))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            def read(cap=cap):
                ok, f = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, f = cap.read()
                rw = c["source"].get("resize_width", 0)
                return cv2.resize(f, None, fx=rw / f.shape[1], fy=rw / f.shape[1]) if rw else f
            meta = lambda: None
        an = Analytics(c, detector_backend=det, activity_backend=act)
        times, n_persons = [], []
        for i in range(a.frames + 10):
            f = read()
            t0 = time.perf_counter()
            an.process(f, i, i / fps, meta())
            if i >= 10:  # skip warm-up
                times.append(time.perf_counter() - t0)
                n_persons.append(len(an.active_tracks))
        mean = float(np.mean(times))
        name = f"{an.detector.name} + {an.activity.name}"
        results[name] = {"fps": round(1 / mean, 2), "latency_ms": round(1000 * mean, 2),
                         "p95_latency_ms": round(1000 * float(np.percentile(times, 95)), 2),
                         "realtime_factor": round((1 / mean) / fps, 2), "avg_persons": round(float(np.mean(n_persons)), 1),
                         "stage_ms": {k: round(v, 2) for k, v in an.timing.items() if k != "render"}}
    print_table("END-TO-END SPEED", results, ["fps", "latency_ms", "p95_latency_ms", "realtime_factor"])
    import platform
    save_section(cfg, "system", {"dataset": a.name or str(a.video), "samples": a.frames,
                                 "notes": f"{platform.processor() or platform.machine()}, {a.frames} frames, "
                                          "analytics only (no video encoding)", "algorithms": results})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("detection")
    s.add_argument("--images", required=True)
    s.add_argument("--algorithms", default="yolov8n-pose,yolov8s-pose,hog")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--name")
    s = sub.add_parser("gender")
    s.add_argument("--images", required=True)
    s.add_argument("--algorithms", default="face_dnn,body_cnn,hybrid")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--name")
    s = sub.add_parser("activity")
    s.add_argument("--data", required=True)
    s.add_argument("--algorithms", default="rule_based,lstm,tcn")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--all", action="store_true", help="evaluate on every window, not only the test split")
    s.add_argument("--name")
    s = sub.add_parser("threat")
    s.add_argument("--videos")
    s.add_argument("--annotations")
    s.add_argument("--simulated", action="store_true")
    s.add_argument("--cycles", type=int, default=1)
    s.add_argument("--algorithms", default="rule_based,lstm,tcn")
    s.add_argument("--tolerance", type=float, default=2.0)
    s.add_argument("--name")
    s = sub.add_parser("speed")
    s.add_argument("--video", default="simulated")
    s.add_argument("--configs", default="yolo+rule_based,yolo+lstm,yolo+tcn,hog+rule_based")
    s.add_argument("--frames", type=int, default=300)
    s.add_argument("--name")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"detection": eval_detection, "gender": eval_gender, "activity": eval_activity, "threat": eval_threat,
     "speed": eval_speed}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
