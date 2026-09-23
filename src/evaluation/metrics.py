"""Evaluation metrics, written with plain NumPy so it's easy to see exactly what is computed.

classification_report - accuracy, macro precision / recall / F1, per-class scores, confusion matrix
detection_metrics     - IoU-matched precision / recall / F1, AP@0.5, counting accuracy
event_metrics         - predicted alerts matched in time against annotated events
"""
from __future__ import annotations

import numpy as np

from ..tracking.iou_tracker import iou_matrix


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def hit_rate(tp, fp, fn):
    """Accuracy when there are no true negatives to count (detection, event spotting):
    the share of all decisions that were right, TP / (TP + FP + FN)."""
    return tp / (tp + fp + fn) if tp + fp + fn else 0.0


def classification_report(y_true, y_pred, labels) -> dict:
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        if actual in index and predicted in index:
            cm[index[actual], index[predicted]] += 1

    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p, r, f = precision_recall_f1(tp, fp, fn)
        per_class[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                            "support": int(cm[i].sum())}

    # macro average over the classes that actually appear in the test data
    present = [label for label in labels if per_class[label]["support"] > 0]

    def macro(key):
        return float(np.mean([per_class[label][key] for label in present])) if present else 0.0

    total = cm.sum()
    return {
        "accuracy": round(float(np.trace(cm) / total) if total else 0.0, 4),
        "precision": round(macro("precision"), 4),
        "recall": round(macro("recall"), 4),
        "f1": round(macro("f1"), 4),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "labels": list(labels),
        "samples": int(total),
    }


def average_precision(scores, tp_flags, n_gt) -> float:
    """Pascal VOC style AP (all-point interpolation)."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp_flags, float)[order]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = cum_tp / n_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def match_boxes(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy matching, highest score first. Returns a true-positive flag per prediction."""
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


def detection_metrics(per_image, conf_thr: float, iou_thr: float = 0.5) -> dict:
    """per_image is a list of (predicted boxes, predicted scores, ground-truth boxes)."""
    all_scores, all_flags, n_gt = [], [], 0
    tp = fp = fn = 0
    exact_counts, count_errors = 0, []

    for boxes, scores, gt in per_image:
        n_gt += len(gt)
        # AP uses every prediction, whatever its score
        all_scores += list(scores)
        all_flags += list(match_boxes(boxes, scores, gt, iou_thr))

        # the other metrics use only the predictions above the operating threshold
        keep = scores >= conf_thr
        flags = match_boxes(boxes[keep], scores[keep], gt, iou_thr)
        tp += int(flags.sum())
        fp += int((~flags).sum())
        fn += len(gt) - int(flags.sum())
        exact_counts += int(keep.sum() == len(gt))
        count_errors.append(abs(int(keep.sum()) - len(gt)))

    p, r, f = precision_recall_f1(tp, fp, fn)
    return {
        "accuracy": round(hit_rate(tp, fp, fn), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "ap50": round(average_precision(all_scores, all_flags, n_gt), 4),
        "count_accuracy": round(exact_counts / max(1, len(per_image)), 4),
        "count_mae": round(float(np.mean(count_errors)) if count_errors else 0.0, 3),
        "tp": tp, "fp": fp, "fn": fn,
        # there are no true negatives in detection (every empty patch of background would be one)
        "confusion": [[tp, fn], [fp, None]],
        "labels": ["person", "background"],
        "images": len(per_image),
        "gt_persons": n_gt,
    }


def event_metrics(predicted: list[dict], ground_truth: list[dict], tolerance: float = 2.0, types=None) -> dict:
    """predicted: [{type, t}], ground_truth: [{type, start, end}], times in seconds.

    Add a 'clip' key to both when several clips are evaluated together. A prediction is a
    true positive when it falls inside an annotated event of the same type (give or take
    `tolerance` seconds). Repeated alerts for an event that was already found are not
    counted as false positives. An annotated event with no alert at all is a false negative.
    """
    types = sorted(types or {g["type"] for g in ground_truth})
    per_type = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}

    for event_type in types:
        gts = [g for g in ground_truth if g["type"] == event_type]
        preds = [p for p in predicted if p["type"] == event_type]
        found = [False] * len(gts)
        fp = 0
        for pred in preds:
            match = next((i for i, g in enumerate(gts)
                          if g.get("clip") == pred.get("clip")
                          and g["start"] - tolerance <= pred["t"] <= g["end"] + tolerance), None)
            if match is None:
                fp += 1
            else:
                found[match] = True
        tp = sum(found)
        fn = len(gts) - tp
        p, r, f = precision_recall_f1(tp, fp, fn)
        per_type[event_type] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                                "tp": tp, "fp": fp, "fn": fn}
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p, r, f = precision_recall_f1(tp, fp, fn)
    return {
        "accuracy": round(hit_rate(tp, fp, fn), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        **totals,
        "confusion": [[tp, fn], [fp, None]],
        "labels": ["event", "no event"],
        "per_type": per_type,
    }
