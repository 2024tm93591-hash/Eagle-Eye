#!/usr/bin/env python3
"""Download the pretrained models into ./models, plus the COCO128 images used to test detection.

  YOLOv8 pose       person boxes + 17 keypoints        yolov8n-pose.pt, yolov8s-pose.pt
  YOLOv8 detection  bags, for the abandoned-object check yolov8n.pt
  OpenCV res10 SSD  face detector                      face_deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel
  Levi & Hassner    gender CaffeNet                    gender_deploy.prototxt, gender_net.caffemodel
  COCO128           sample images for evaluation       data/eval/coco128

The body-gender CNN and the LSTM / TCN activity models have to be trained yourself
(scripts/train_gender.py, scripts/train_activity.py).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"
GENDER_REPO = "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/"

MODELS = {
    "yolov8n-pose.pt": ULTRALYTICS + "yolov8n-pose.pt",
    "yolov8s-pose.pt": ULTRALYTICS + "yolov8s-pose.pt",
    "yolov8n.pt": ULTRALYTICS + "yolov8n.pt",
    "face_deploy.prototxt":
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel":
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel",
    "gender_deploy.prototxt": GENDER_REPO + "gender_deploy.prototxt",
    "gender_net.caffemodel": GENDER_REPO + "gender_net.caffemodel",
}
COCO128 = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def fetch(url, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  ok      {dest.name}")
        return True
    print(f"  get     {dest.name} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f" FAILED ({exc})")
        return False
    print(f" {dest.stat().st_size / 1e6:.1f} MB")
    return True


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    ok = all([fetch(url, MODEL_DIR / name) for name, url in MODELS.items()])

    eval_dir = ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not (eval_dir / "coco128").exists():
        archive = eval_dir / "coco128.zip"
        if fetch(COCO128, archive):
            zipfile.ZipFile(archive).extractall(eval_dir)
            archive.unlink()

    print("\nDone." if ok else "\nSome downloads failed - get them by hand into ./models (URLs above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
