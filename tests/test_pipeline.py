"""Unit + integration tests (run with:  python -m pytest -q). No GPU / deep-learning models needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activity.features import FEATURE_DIM, sequence_features  # noqa: E402
from src.config import load_config  # noqa: E402
from src.core import SceneEvent  # noqa: E402
from src.evaluation.metrics import (average_precision, classification_report, detection_metrics,  # noqa: E402
                                    event_metrics)
from src.pipeline.processor import Analytics  # noqa: E402
from src.pipeline.simulator import SimulatedSource  # noqa: E402
from src.threat.assessment import ThreatAssessor  # noqa: E402
from src.tracking.iou_tracker import IOUTracker, iou_matrix  # noqa: E402
from src.core import Detection  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides={"source": {"uri": "simulated"}})


def test_iou():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


def test_tracker_keeps_ids():
    tr = IOUTracker(min_hits=1)
    ids = []
    for k in range(5):
        out = tr.update([Detection(np.array([10 + 3 * k, 10, 50 + 3 * k, 110.0]), 0.9)])
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1


def test_classification_report():
    r = classification_report(["a", "a", "b", "b"], ["a", "b", "b", "b"], ["a", "b"])
    assert r["accuracy"] == 0.75
    assert r["per_class"]["a"]["precision"] == 1.0 and r["per_class"]["a"]["recall"] == 0.5
    assert r["confusion"] == [[1, 1], [0, 2]]


def test_detection_metrics_and_ap():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30.0]])
    pred = np.array([[0, 0, 10, 10], [50, 50, 60, 60.0]])
    m = detection_metrics([(pred, np.array([0.9, 0.8]), gt)], conf_thr=0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert average_precision([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_event_metrics():
    gt = [dict(type="fall", start=10, end=15)]
    m = event_metrics([dict(type="fall", t=11), dict(type="fall", t=12), dict(type="fall", t=40)], gt)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_features_shape():
    T = 30
    t = np.arange(T) / 25
    b = np.tile([100, 100, 150, 250.0], (T, 1))
    k = [np.concatenate([np.random.rand(17, 2) * 50 + 100, np.ones((17, 1))], 1) for _ in range(T)]
    assert sequence_features(t, b, k, np.ones(T)).shape == (T, FEATURE_DIM)
    assert sequence_features(t, b, [None] * T).shape == (T, FEATURE_DIM)


def test_threat_levels_and_dedup(cfg):
    ta = ThreatAssessor(cfg)
    ta.quiet_override = False
    e = SceneEvent("person_down", (1,), "down", duration=8)
    active, new = ta.assess([e], now=0.0)
    assert new and new[0].level == "CRITICAL"
    _, new2 = ta.assess([e], now=1.0)       # same situation -> no duplicate alert
    assert not new2
    active3, new3 = ta.assess([SceneEvent("running", (2,), "run")], now=1.0)
    assert active3[0].level == "LOW" and not new3       # below min_alert_level -> overlay only


def test_simulated_scene_end_to_end(cfg):
    """The scripted scene must produce each of the key threat types."""
    an = Analytics(cfg)
    an.scene.quiet_override = False
    an.assessor.quiet_override = False
    src = SimulatedSource()
    seen = set()
    for i in range(100 * 20):
        _, f = src.read()
        _, new = an.process(f, i, i / 20, src.last_meta)
        seen |= {a.event.type for a in new}
    for t in ["restricted_zone_intrusion", "loitering", "physical_altercation", "fall", "person_down", "chasing",
              "abandoned_object", "group_gathering", "panic_running"]:
        assert t in seen, t
