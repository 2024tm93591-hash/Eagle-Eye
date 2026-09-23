# ThreatVision: AI-Based Scene Understanding and Threat Detection Using Video Analytics

ThreatVision watches a camera or a video file in real time and:

1. **Counts people and estimates their gender.** YOLOv8-pose finds people and ByteTrack follows them from frame to frame. Gender comes from a face DNN or a body CNN, with votes added up per person. The number of people, men and women is shown above the video and on the video itself.
2. **Recognises activities:** *standing, walking, running, fighting, falling*, from the body skeleton. There is a rule-based classifier, plus a Bi-LSTM with attention and a Temporal ConvNet you can train.
3. **Understands what is going on in the scene:** fights between people, chasing, loitering, a person down after a fall, groups gathering, crowds, panic running, bags left behind, and people present during quiet hours.
4. **Scores threats as they happen.** Each event gets a 0-100 score that explains how it was built, and a level (LOW / MEDIUM / HIGH / CRITICAL). Repeated alerts are merged, and alerts can escalate. Each alert saves a snapshot and can notify the dashboard (with sound), e-mail, a webhook or Telegram.
5. **Shows everything on a dashboard:** the live annotated video with an on/off switch, scene statistics, a live alert feed with acknowledgement, the event history with filters and CSV export, and source switching / video upload.
6. **Reports how well each algorithm works.** The *Analytics* page lists accuracy, precision, recall, F1 score, the confusion matrix and frames per second for each algorithm, as plain text.

```
 video --> detection + tracking --> gender --> activity --> scene analysis --> threat --> alerts
 (webcam,   YOLOv8-pose + ByteTrack   face DNN    rules /       pairs, groups,     score,    dashboard,
  file,     (or HOG + IoU tracker)    body CNN    LSTM / TCN    bags, time of day  level     e-mail, webhook,
  RTSP)                                                                                      SQLite history
```

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                              # for a GPU, install the CUDA build of torch first
python scripts/download_models.py                            # YOLOv8 weights, face + gender models, COCO128
```

`opencv-python` is pinned below version 5, because OpenCV 5 can no longer load the Caffe face and gender models.

## 2. Running it

```bash
python run.py                                    # webcam 0 (the default in config.yaml)
python run.py --source 1                         # another webcam
python run.py --source data/videos/clip.mp4      # a video file
python run.py --source rtsp://user:pw@ip/stream  # an IP camera
python run.py --activity lstm                    # use the trained Bi-LSTM (see section 4)
```

Open **http://localhost:5000**. The pages are *Live monitor*, *Event history*, *Analytics* and *Sources & settings*.
The **Turn feed off** button on the Live monitor releases the camera and stops the analysis until you turn the feed back on.
You can also switch the source or upload a video from *Sources & settings*.

Every setting (thresholds, scores, notification channels) is in `config.yaml`.

## 3. Project layout

```
run.py                       starts the pipeline and the dashboard
config.yaml                  all settings
src/
  core.py                    Detection / SceneEvent / ThreatAlert data classes
  detection/detector.py      YOLOv8-pose + ByteTrack, and the HOG + SVM baseline
  tracking/                  IoU tracker (for HOG) and what we remember per person (TrackState)
  gender/                    face DNN (Levi-Hassner), body CNN (MobileNetV3), hybrid
  activity/                  features, rule-based classifier, Bi-LSTM and TCN models, dataset tools
  scene/analyzer.py          turns people's activities and positions into scene events
  threat/assessment.py       scoring, levels, merging repeated alerts, escalation
  threat/alerts.py           snapshots, storage, live push, e-mail / webhook / Telegram
  storage/db.py              alert history (SQLite)
  pipeline/processor.py      the real-time processing thread and per-stage timing
  pipeline/draw.py           boxes, skeletons, counts and alerts drawn on the video
  evaluation/metrics.py      accuracy, precision / recall / F1, confusion matrix, AP, event matching
  evaluation/report.py       plain-text results used by the Analytics page and evaluate.py
  dashboard/                 Flask app, templates, CSS and JS (no external CDN needed)
scripts/
  download_models.py         pretrained weights + COCO128
  extract_pose_dataset.py    videos -> pose-sequence dataset (.npz)
  train_activity.py          trains the Bi-LSTM / TCN
  train_gender.py            trains the body-gender CNN; prepare_pa100k.py converts PA-100K for it
  evaluate.py                runs the evaluations and saves results/metrics.json
tests/test_pipeline.py       tests (python -m pytest -q)
docs/                        project report
```

## 4. Training the learned models

**Activity (Bi-LSTM / TCN).** Put clips in `data/activity_videos/<class>/` for the classes
`standing, walking, running, fighting, falling`, or label frames in a CSV file.
Good public sources: *Le2i Fall* and *UR Fall Detection* (falling); *RWF-2000*, *Hockey Fight* and *Surveillance Camera Fight* (fighting);
*KTH*, *UCF-101* or your own CCTV footage (walking, running, standing).
```bash
python scripts/extract_pose_dataset.py --videos data/activity_videos --out data/activity/dataset.npz
python scripts/train_activity.py --data data/activity/dataset.npz --model lstm
python scripts/train_activity.py --data data/activity/dataset.npz --model tcn
```
The data is split by video, so no clip ends up in both the training and the test set.

**Gender (body CNN).** CCTV faces are often too small to use, so a full-body classifier is trained on PA-100K or PETA:
```bash
python scripts/prepare_pa100k.py --root /path/to/PA-100K --out data/gender
python scripts/train_gender.py --data data/gender
```

## 5. Evaluation (shown on the Analytics page)

```bash
python scripts/evaluate.py detection --images data/eval/coco128 --algorithms yolov8n-pose,yolov8s-pose,hog
python scripts/evaluate.py gender    --images data/gender/test --algorithms face_dnn,body_cnn,hybrid
python scripts/evaluate.py activity  --data data/activity/dataset.npz --algorithms rule_based,lstm,tcn
python scripts/evaluate.py threat    --videos data/eval/threat --annotations data/eval/threat/events.json
python scripts/evaluate.py speed     --video data/videos/clip.mp4 --configs yolo+rule_based,hog+rule_based
```
Each command prints its results and updates one section of `results/metrics.json`. Reload the *Analytics* page to see them.
Threat annotations look like `{"clip1.mp4": [{"type": "physical_altercation", "start": 12.0, "end": 19.5}, ...]}`.
The Analytics page also shows the live frames per second of each stage while the system runs.

**Results included in `results/metrics.json`:**
* Person detection for YOLOv8n-pose, YOLOv8s-pose and HOG + SVM, measured on the 128 real images of COCO128.
  About a third of the people labelled there are very small (under 10% of the image height). This is why recall is low at the 0.40 confidence threshold.
* Gender, activity and threat detection need labelled test data (see section 4). Until you run those commands, the Analytics page shows how to produce them.

## 6. Responsible use

Gender is *estimated from appearance*. It is binary, probabilistic, and may be biased. It is only shown as a count
and **is never used to score threats**. Surveillance analytics must follow local privacy law
(for example India's DPDP Act 2023 or the GDPR). Put up notices, limit how long footage is kept (snapshots and `data/events.db`),
and keep a human operator in the loop: the alerts support decisions, they don't make them.
