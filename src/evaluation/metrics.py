"""Standard evaluation metrics (implemented with NumPy so they are transparent and dependency-free).

* classification_report : accuracy, macro precision / recall / F1, per-class scores, confusion matrix
* detection_metrics     : IoU-matched precision / recall / F1, AP@0.5 (VOC all-point), counting accuracy
* event_metrics         : temporal matching of predicted alerts against annotated events
"""
from __future__ import annotations

import numpy as np

from ..tracking.iou_tracker import iou_matrix


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification_report(y_true, y_pred, labels) -> dict:
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    per = {}
    for i, l in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r, f = _prf(tp, fp, fn)
        per[l] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "support": int(cm[i].sum())}
    present = [l for l in labels if per[l]["support"] > 0]
    mean = lambda k: float(np.mean([per[l][k] for l in present])) if present else 0.0
    total = cm.sum()
    return {"accuracy": round(float(np.trace(cm) / total) if total else 0.0, 4),
            "precision": round(mean("precision"), 4), "recall": round(mean("recall"), 4), "f1": round(mean("f1"), 4),
            "per_class": per, "confusion": cm.tolist(), "labels": list(labels), "samples": int(total)}


def average_precision(scores, tp_flags, n_gt) -> float:
    """VOC-style all-point interpolated AP."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp_flags, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def match_boxes(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy (score-ordered) matching; returns a TP flag per prediction."""
    flags = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return flags
    ious = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    for i in np.argsort(-scores):
        j = int(np.argmax(np.where(used, -1, ious[i])))
        if ious[i, j] >= iou_thr and not used[j]:
            used[j] = True
            flags[i] = True
    return flags


def detection_metrics(per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]], conf_thr: float,
                      iou_thr: float = 0.5) -> dict:
    """per_image: list of (pred_boxes (N,4), pred_scores (N,), gt_boxes (M,4))."""
    all_s, all_tp, n_gt = [], [], 0
    tp = fp = fn = 0
    exact, abs_err = 0, []
    for pb, ps, gb in per_image:
        n_gt += len(gb)
        flags = match_boxes(pb, ps, gb, iou_thr)
        all_s += list(ps)
        all_tp += list(flags)
        keep = ps >= conf_thr
        k_flags = match_boxes(pb[keep], ps[keep], gb, iou_thr)
        tp += int(k_flags.sum())
        fp += int((~k_flags).sum())
        fn += len(gb) - int(k_flags.sum())
        exact += int(keep.sum() == len(gb))
        abs_err.append(abs(int(keep.sum()) - len(gb)))
    p, r, f = _prf(tp, fp, fn)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "ap50": round(average_precision(all_s, all_tp, n_gt), 4),
            "count_accuracy": round(exact / max(1, len(per_image)), 4),
            "count_mae": round(float(np.mean(abs_err)) if abs_err else 0.0, 3),
            "tp": tp, "fp": fp, "fn": fn, "images": len(per_image), "gt_persons": n_gt}


def event_metrics(predicted: list[dict], ground_truth: list[dict], tolerance: float = 2.0, types=None) -> dict:
    """predicted: [{type, t}], ground_truth: [{type, start, end}] (all times in seconds of the same clip set;
    include a 'clip' key when evaluating several clips). A prediction is a TP when it falls inside an
    unmatched-or-matched GT interval of the same type (+/- tolerance). Repeated alerts for an already-detected
    GT event are not counted as false positives; a GT event with no prediction is a false negative."""
    types = sorted(types or {g["type"] for g in ground_truth})
    per = {}
    T = {"tp": 0, "fp": 0, "fn": 0}
    for ty in types:
        gts = [g for g in ground_truth if g["type"] == ty]
        preds = [p for p in predicted if p["type"] == ty]
        hit = [False] * len(gts)
        fp = 0
        for p in preds:
            ok = False
            for i, g in enumerate(gts):
                if g.get("clip") == p.get("clip") and g["start"] - tolerance <= p["t"] <= g["end"] + tolerance:
                    hit[i] = True
                    ok = True
                    break
            fp += not ok
        tp = sum(hit)
        fn = len(gts) - tp
        pr, rc, f1 = _prf(tp, fp, fn)
        per[ty] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
        T["tp"] += tp
        T["fp"] += fp
        T["fn"] += fn
    p, r, f = _prf(T["tp"], T["fp"], T["fn"])
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), **T, "per_type": per}
