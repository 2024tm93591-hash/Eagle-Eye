"""Tests for the parts of the pipeline that don't need a camera or the deep-learning models.

Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activity.features import FEATURE_DIM, sequence_features  # noqa: E402
from src.config import load_config  # noqa: E402
from src.core import Detection, SceneEvent  # noqa: E402
from src.evaluation.metrics import (average_precision, classification_report,  # noqa: E402
                                    detection_metrics, event_metrics)
from src.evaluation.report import algorithm_report  # noqa: E402
from src.scene.analyzer import SceneAnalyzer  # noqa: E402
from src.threat.assessment import ThreatAssessor  # noqa: E402
from src.tracking.iou_tracker import IOUTracker, iou_matrix  # noqa: E402
from src.tracking.track_state import TrackState  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_track(track_id, box, activity, t=10.0):
    track = TrackState(track_id, first_seen=0.0, last_seen=t, activity=activity, activity_since=0.0)
    track.add(t, Detection(np.array(box, float), 0.9, track_id))
    return track


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_tracker_keeps_ids():
    tracker = IOUTracker(min_hits=1)
    ids = []
    for step in range(5):
        out = tracker.update([Detection(np.array([10 + 3 * step, 10, 50 + 3 * step, 110.0]), 0.9)])
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1


def test_classification_report():
    report = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert report["accuracy"] == 0.75
    assert report["per_class"]["a"]["precision"] == 1.0
    assert report["per_class"]["a"]["recall"] == 0.5
    assert report["confusion"] == [[1, 1], [0, 2]]


def test_detection_metrics_and_ap():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30.0]])
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60.0]])
    metrics = detection_metrics([(pred, np.array([0.9, 0.8]), gt)], conf_thr=0.5)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 1)
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5
    assert metrics["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["confusion"] == [[1, 1], [1, None]]
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_event_metrics():
    gt = [dict(type="fall", start=10, end=15)]
    metrics = event_metrics([dict(type="fall", t=11), dict(type="fall", t=12), dict(type="fall", t=40)], gt)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 0)


def test_text_report_has_every_metric():
    report = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    report["fps"] = 12.5
    text = algorithm_report(report)
    for label in ["Accuracy", "Precision", "Recall", "F1 score", "Frames per second", "Confusion matrix"]:
        assert label in text


def test_features_shape():
    T = 30
    t = np.arange(T) / 25
    boxes = np.tile([100, 100, 150, 250.0], (T, 1))
    keypoints = [np.concatenate([np.random.rand(17, 2) * 50 + 100, np.ones((17, 1))], 1) for _ in range(T)]
    assert sequence_features(t, boxes, keypoints, np.ones(T)).shape == (T, FEATURE_DIM)
    assert sequence_features(t, boxes, [None] * T).shape == (T, FEATURE_DIM)


def test_threat_levels_and_dedup(cfg):
    assessor = ThreatAssessor(cfg)
    assessor.quiet_override = False
    down = SceneEvent("person_down", (1,), "down", duration=8)
    _, new = assessor.assess([down], now=0.0)
    assert new and new[0].level == "CRITICAL"

    _, again = assessor.assess([down], now=1.0)   # same situation, so no second alert
    assert not again

    active, new = assessor.assess([SceneEvent("running", (2,), "run")], now=1.0)
    assert active[0].level == "LOW" and not new   # below min_alert_level: overlay only


def test_scene_reports_fight_between_two_close_people(cfg):
    scene = SceneAnalyzer(cfg)
    scene.quiet_override = False
    a = make_track(1, [100, 100, 150, 250], "fighting")
    b = make_track(2, [140, 100, 190, 250], "fighting")
    events = scene.update([a, b], [], t=10.0)
    assert [e.type for e in events] == ["physical_altercation"]
    assert events[0].track_ids == (1, 2)


def test_scene_reports_person_down_after_a_fall(cfg):
    scene = SceneAnalyzer(cfg)
    scene.quiet_override = False
    track = make_track(1, [100, 200, 250, 260], "falling")
    track.fall_time = 2.0   # went down 8 seconds ago
    events = scene.update([track], [], t=10.0)
    assert events[0].type == "person_down"
