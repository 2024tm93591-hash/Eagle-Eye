# ThreatVision — AI-Based Scene Understanding and Threat Detection Using Video Analytics

A real-time video analytics system that:

1. **Counts people and estimates their gender** — YOLOv8-pose detection + ByteTrack tracking, face-DNN / body-CNN gender estimation with per-track voting.
2. **Recognises human activities** — *standing, walking, running, fighting, falling* from skeleton sequences (rule-based baseline, Bi-LSTM + attention, Temporal ConvNet).
3. **Understands interactions and context** — physical altercations, chasing, loitering, restricted-zone intrusion, person down (no recovery), group gathering, crowding, panic running, abandoned bags, presence during quiet hours.
4. **Assesses threats in real time** — explainable 0–100 threat score → LOW / MEDIUM / HIGH / CRITICAL, de-duplication and escalation, snapshots, notifications (dashboard, sound, e-mail, webhook, Telegram).
5. **Monitoring dashboard** — live annotated video, scene statistics, live alert feed with acknowledgement, event history with filters and CSV export, analytics, source switching / video upload.
6. **Evaluation** — accuracy, precision, recall, F1-score, AP@0.5, counting accuracy, FPS and latency for every algorithm, shown on the dashboard's *Performance metrics* page.

```
 video ──► detection + tracking ──► gender ──► activity ──► scene / interaction ──► threat ──► alerts
 (file,     YOLOv8-pose + ByteTrack   face DNN    rules /       zones, pairs, groups,     score,    dashboard (SSE),
  webcam,   (or HOG + IoU tracker)    body CNN    LSTM / TCN    objects, time context     level,    e-mail, webhook,
  RTSP)                                                                                   dedupe    SQLite history
```

---

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                              # (GPU: install the CUDA build of torch first)
python scripts/download_models.py                            # YOLOv8 weights, face + gender models, COCO128
```

## 2. Run

```bash
python run.py --source simulated                 # synthetic test scene - works without camera or models
python run.py --source data/videos/clip.mp4      # a video file
python run.py --source 0                         # webcam
python run.py --source rtsp://user:pw@ip/stream  # IP camera
python run.py --activity lstm                    # use the trained Bi-LSTM (after training, see §4)
```
Open **http://localhost:5000**. Pages: *Live monitor · Event history · Analytics · Performance metrics · Sources & settings*.
The source can also be switched, or a video uploaded, from *Sources & settings*.

All parameters (zones, thresholds, scores, notification channels) are in `config.yaml`.
Zones are polygons in normalised image coordinates, so they fit any resolution.

## 3. Project structure

```
run.py                       start pipeline + dashboard
config.yaml                  all settings
src/
  core.py                    Detection / SceneEvent / ThreatAlert data classes
  detection/detector.py      YOLOv8-pose + ByteTrack, HOG+SVM baseline, simulated detector
  tracking/                  IoU tracker (for HOG), per-person temporal state (TrackState)
  gender/                    face DNN (Levi-Hassner), body CNN (MobileNetV3), hybrid, per-track voting
  activity/                  features, rule-based classifier, Bi-LSTM & TCN models, dataset utils
  scene/analyzer.py          interaction + context analysis -> SceneEvents
  threat/assessment.py       scoring, levels, de-duplication, escalation
  threat/alerts.py           snapshots, SQLite, live push, e-mail / webhook / Telegram
  storage/db.py              event history + occupancy statistics (SQLite)
  pipeline/processor.py      real-time processing thread, stage timing
  pipeline/simulator.py      scripted synthetic scene with ground truth (self-test)
  evaluation/metrics.py      accuracy, P/R/F1, confusion matrix, AP, event matching
  dashboard/                 Flask app, templates, CSS, JS (no external CDN needed)
scripts/
  download_models.py         pretrained weights + COCO128
  extract_pose_dataset.py    videos -> pose-sequence dataset (.npz)
  train_activity.py          train Bi-LSTM / TCN
  train_gender.py            train body-gender CNN;  prepare_pa100k.py converts PA-100K
  evaluate.py                all evaluations -> results/metrics.json (dashboard)
tests/test_pipeline.py       unit + end-to-end tests  (python -m pytest -q)
docs/                        project report
```

## 4. Training the learned models

**Activity (Bi-LSTM / TCN).** Put clips in `data/activity_videos/<class>/` for the classes
`standing, walking, running, fighting, falling` (or supply frame-level annotations as CSV).
Good public sources: *Le2i Fall*, *UR Fall Detection* (falling); *RWF-2000*, *Hockey Fight*, *Surveillance Camera Fight* (fighting);
*KTH*, *UCF-101* or your own CCTV footage (walking, running, standing).
```bash
python scripts/extract_pose_dataset.py --videos data/activity_videos --out data/activity/dataset.npz
python scripts/train_activity.py --data data/activity/dataset.npz --model lstm
python scripts/train_activity.py --data data/activity/dataset.npz --model tcn
```
The split is per video, so no clip is in both training and test data.

**Gender (body CNN).** CCTV faces are often too small, so a full-body classifier is trained on PA-100K / PETA:
```bash
python scripts/prepare_pa100k.py --root /path/to/PA-100K --out data/gender
python scripts/train_gender.py --data data/gender
```

## 5. Evaluation (shown on the dashboard)

```bash
python scripts/evaluate.py detection --images data/eval/coco128 --algorithms yolov8n-pose,yolov8s-pose,hog
python scripts/evaluate.py gender    --images data/gender/test --algorithms face_dnn,body_cnn,hybrid
python scripts/evaluate.py activity  --data data/activity/dataset.npz --algorithms rule_based,lstm,tcn
python scripts/evaluate.py threat    --videos data/eval/threat --annotations data/eval/threat/events.json
python scripts/evaluate.py speed     --video data/videos/clip.mp4 --configs yolo+rule_based,yolo+lstm,yolo+tcn,hog+rule_based
```
Each command updates one section of `results/metrics.json`; reload the *Performance metrics* page.
Threat annotations format: `{"clip1.mp4": [{"type": "physical_altercation", "start": 12.0, "end": 19.5}, ...]}`.

**Results included in this package** (`results/metrics.json`):
* Detection — **HOG + SVM baseline on COCO128** (real images). YOLOv8 rows appear after you run the command above with ultralytics installed.
* Activity, threat events and speed — measured on the **synthetic self-test scene**, which checks that
  the code works end to end. These numbers do **not** measure real-world accuracy. Replace them by running the commands on real annotated footage.

## 6. Responsible use

Gender is *estimated from appearance*. It is binary, probabilistic, and may be biased. It is only shown as an aggregate
statistic and **is never used in threat scoring**. Surveillance analytics must comply with local privacy law
(for example India's DPDP Act 2023 or the GDPR). Show notices, limit how long footage is kept (snapshots and `data/events.db`), and keep a
human operator in the loop: alerts are decision *support*.
