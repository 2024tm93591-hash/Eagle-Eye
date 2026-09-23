#!/usr/bin/env python3
"""Download the pretrained weights used by ThreatVision into ./models

  YOLOv8 pose  (person boxes + 17 keypoints)  yolov8n-pose.pt, yolov8s-pose.pt
  YOLOv8 det   (bags for abandoned-object)    yolov8n.pt
  OpenCV res10 SSD face detector              face_deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
  Levi & Hassner gender CaffeNet              gender_deploy.prototxt, gender_net.caffemodel
  COCO128 sample images (detection eval)      data/eval/coco128

The body-gender CNN and the LSTM / TCN activity models are trained by you
(scripts/train_gender.py, scripts/train_activity.py).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models"
UL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
FILES = {
    "yolov8n-pose.pt": UL + "yolov8n-pose.pt",
    "yolov8s-pose.pt": UL + "yolov8s-pose.pt",
    "yolov8n.pt": UL + "yolov8n.pt",
    "face_deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel":
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel",
}
COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def fetch(url, dst: Path):
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  ok      {dst.name}")
        return True
    try:
        print(f"  get     {dst.name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, dst)
        print(f" {dst.stat().st_size / 1e6:.1f} MB")
        return True
    except Exception as exc:
        print(f" FAILED ({exc})")
        return False


def main():
    M.mkdir(exist_ok=True)
    ok = all([fetch(u, M / n) for n, u in FILES.items()])
    ev = ROOT / "data" / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    if not (ev / "coco128").exists():
        z = ev / "coco128.zip"
        if fetch(COCO128, z):
            zipfile.ZipFile(z).extractall(ev)
            z.unlink()
    print("\nDone." if ok else "\nSome downloads failed - download them manually into ./models (URLs above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
