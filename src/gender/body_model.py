"""Full-body gender classifier (MobileNetV3-Small, 2 classes: male, female).

Shared by the runtime classifier and scripts/train_gender.py.
Recommended training data: PA-100K or PETA pedestrian-attribute datasets (attribute 'Female'),
which contain low-resolution CCTV-style full-body crops where faces are rarely visible.
"""
from __future__ import annotations

INPUT_SIZE = (256, 128)          # H, W  (pedestrian aspect ratio)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASSES = ["male", "female"]


def build_body_model(pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASSES))
    return m


def preprocess_bgr(crops):
    """List of BGR uint8 crops -> normalised float tensor (N,3,H,W)."""
    import cv2
    import numpy as np
    import torch

    arr = []
    for c in crops:
        c = cv2.resize(c, (INPUT_SIZE[1], INPUT_SIZE[0]))[:, :, ::-1].astype(np.float32) / 255.0
        arr.append((c - MEAN) / STD)
    return torch.from_numpy(np.stack(arr).transpose(0, 3, 1, 2).astype(np.float32))
