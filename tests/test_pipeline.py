"""Tests for the parts of the pipeline that don't need a camera or the deep-learning models.

Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.activity.clip_filter import frames_showing_activity, stitch_tracks  # noqa: E402
from src.activity.features import FEATURE_DIM, sequence_features  # noqa: E402
from src.activity.rule_based import RuleBasedActivity  # noqa: E402
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


def pose(points):
    """17 keypoints, all hidden except the ones given as {index: (x, y)}."""
    kp = np.zeros((17, 3))
    for i, (x, y) in points.items():
        kp[i] = [x, y, 0.9]
    return kp


def track_with_pose(box, keypoints):
    track = make_track(1, box, "standing")
    track.last_det.keypoints = keypoints
    return track


def test_half_visible_person_is_not_a_fall(cfg):
    # head and shoulders close to the camera: a wide box, nothing below the chest in view
    kp = pose({0: (300, 150), 5: (200, 300), 6: (400, 300)})
    label, _, _ = RuleBasedActivity(cfg).predict(track_with_pose([100, 100, 500, 400], kp), 10.0)
    assert label == "standing"


def test_seated_person_is_not_a_fall(cfg):
    # upright torso, knees out in front, so the box is wider than it is tall
    kp = pose({5: (200, 100), 6: (240, 100), 11: (205, 220), 12: (245, 220), 13: (330, 230), 14: (360, 230)})
    label, _, _ = RuleBasedActivity(cfg).predict(track_with_pose([180, 80, 400, 260], kp), 10.0)
    assert label == "standing"


def test_person_lying_down_is_a_fall(cfg):
    # torso horizontal, legs in view
    kp = pose({5: (100, 300), 6: (100, 320), 11: (250, 305), 12: (250, 325), 13: (350, 310), 15: (450, 315)})
    label, _, _ = RuleBasedActivity(cfg).predict(track_with_pose([80, 280, 470, 340], kp), 10.0)
    assert label == "falling"


@pytest.mark.parametrize("fps", [10, 30])
def test_joint_speed_looks_back_the_same_time_at_any_frame_rate(fps):
    # the wrist swings for one second, then stays still for the last second
    track = TrackState(1, first_seen=0.0, last_seen=0.0)
    for i in range(3 * fps):
        t = i / fps
        moving = 1.0 <= t < 2.0
        wrist = (100 + (40 if moving and i % 2 else 0), 150)
        kp = pose({9: wrist, 11: (100, 200), 12: (130, 200)})
        track.add(t, Detection(np.array([80, 100, 150, 250.0]), 0.9, 1, keypoints=kp))
    assert RuleBasedActivity._joint_speed(track, 150.0, (9,)) > 0


def test_sequences_cover_the_same_time_at_any_frame_rate():
    from src.activity.features import sample_by_time
    fast = np.arange(0, 3.0, 1 / 30)   # 3 s of history at 30 fps
    slow = np.arange(0, 3.0, 1 / 10)   # the same 3 s at 10 fps
    for times in (fast, slow):
        picked = times[sample_by_time(times, 30, 10)]
        assert picked[-1] == times[-1]
        assert picked[-1] - picked[0] == pytest.approx(2.9, abs=0.05)
    short = np.arange(0, 0.5, 1 / 10)   # not enough history yet: the first frame repeats
    assert sample_by_time(short, 30, 10)[0] == 0


@pytest.mark.parametrize("kind", ["lstm", "tcn"])
def test_learned_activity_model_runs_on_a_live_track(cfg, tmp_path, kind):
    torch = pytest.importorskip("torch")
    from src.activity.models import build_model
    from src.activity.recognizer import ActivityRecognizer

    classes = cfg["activity"]["classes"]
    path = tmp_path / f"activity_{kind}.pt"
    torch.save({"state_dict": build_model(kind, len(classes)).state_dict(), "classes": classes,
                "window": 30, "fps": 10.0, "mean": [0.0] * FEATURE_DIM, "std": [1.0] * FEATURE_DIM}, path)
    model_cfg = load_config(overrides={"activity": {f"{kind}_model": str(path)}})

    recognizer = ActivityRecognizer(model_cfg, kind)
    assert recognizer.model is not None   # loaded the checkpoint instead of falling back to rules

    track = TrackState(1, first_seen=0.0, last_seen=0.0)
    for i in range(40):
        t = i / 10
        track.add(t, Detection(np.array([100 + 5 * i, 100, 150 + 5 * i, 250.0]), 0.9, 1,
                               keypoints=pose({5: (120, 140)})))
        recognizer.update([track], t)
    assert track.activity in classes


def test_unsure_fight_uses_the_next_guess():
    from src.activity.recognizer import pick_label

    classes = ["standing", "fighting"]
    assert pick_label(np.array([0.4, 0.6]), classes, 0.7)[0] == "standing"
    assert pick_label(np.array([0.2, 0.8]), classes, 0.7)[0] == "fighting"
    assert pick_label(np.array([0.9, 0.1]), classes, 0.7)[0] == "standing"


def clip_record(frame, box, keypoints=None, nearest=5.0):
    keypoints = pose({}) if keypoints is None else keypoints
    return (frame, frame / 30, np.array(box, float), keypoints, nearest, 150.0)


def upright(x, y=100):
    return pose({5: (x, y + 30), 6: (x + 30, y + 30), 11: (x, y + 90), 12: (x + 30, y + 90), 15: (x, y + 145)})


def test_filter_keeps_only_the_person_who_falls_and_when(cfg):
    rules = cfg["activity"]["rules"]
    faller, bystander = [], []
    for f in range(180):
        bystander.append(clip_record(f, [400, 100, 450, 250], upright(405)))
        if f < 90:
            faller.append(clip_record(f, [100, 100, 150, 250], upright(105)))
        else:   # on the ground from 3 s onwards: torso horizontal
            lying = pose({5: (60, 240), 6: (60, 250), 11: (150, 240), 12: (150, 250), 15: (240, 245)})
            faller.append(clip_record(f, [50, 225, 250, 265], lying))

    keep = frames_showing_activity("falling", {1: faller, 2: bystander}, rules)
    assert not keep[2]
    assert 92 in keep[1] and 170 in keep[1]
    assert 40 not in keep[1]


def test_filter_drops_an_onlooker_from_a_fight(cfg):
    rules = cfg["activity"]["rules"]
    fighter, onlooker = [], []
    for f in range(120):
        swing = 60 if (f // 3) % 2 else 0   # wrist darts back and forth
        arms = upright(100)
        arms[9] = [120 + swing, 150, 0.9]
        fighter.append(clip_record(f, [100, 100, 150, 250], arms, nearest=0.5))
        still = upright(160)
        still[9] = [180, 150, 0.9]
        onlooker.append(clip_record(f, [160, 100, 210, 250], still, nearest=0.5))

    keep = frames_showing_activity("fighting", {1: fighter, 2: onlooker}, rules)
    assert len(keep[1]) == 120
    assert not keep[2]


def test_filter_separates_runners_from_people_standing_still(cfg):
    rules = cfg["activity"]["rules"]
    runner = [clip_record(f, [100 + 8 * f, 100, 150 + 8 * f, 250]) for f in range(60)]   # 240 px/s
    still = [clip_record(f, [400, 100, 450, 250]) for f in range(60)]

    keep = frames_showing_activity("running", {1: runner, 2: still}, rules)
    assert 50 in keep[1]
    assert not keep[2]
    assert frames_showing_activity("dancing", {1: runner}, rules) is None


def test_stitching_joins_a_broken_track_but_not_two_people(cfg):
    before = [clip_record(f, [100, 100, 150, 250]) for f in range(60)]
    after = [clip_record(f, [90, 210, 240, 255]) for f in range(65, 120)]   # same spot, now on the floor
    other = [clip_record(f, [500, 100, 550, 250]) for f in range(65, 120)]  # same time, far away
    joined = stitch_tracks({1: before, 2: after, 3: other})
    assert sorted(len(recs) for recs in joined.values()) == [55, 115]
