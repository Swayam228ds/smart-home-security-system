"""
detector.py — YOLO Human Detector & MediaPipe Face Detector Wrappers
====================================================================
Encapsulates model initialization (loaded once on startup) and inference.
"""

import os
import urllib.request
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config

log = logging.getLogger(__name__)


class YoloHumanDetector:
    """Wraps YOLOv8 human detection model."""

    def __init__(self, model_path: str = config.YOLO_MODEL, conf: float = config.DETECTION_CONFIDENCE):
        self.model_path = model_path
        self.conf = conf
        log.info(f"Loading YOLO model: {model_path} ...")
        self.model = YOLO(model_path)
        log.info("YOLOv8 ready.")

    def detect_humans(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on frame to find humans.

        Returns:
            List of detected humans with bounding boxes and confidence.
        """
        results = self.model(
            frame_bgr,
            conf=self.conf,
            classes=[config.PERSON_CLASS_ID],
            verbose=False
        )
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                "bbox_xyxy": (x1, y1, x2, y2),
                "confidence": conf,
            })
        return detections


class MediaPipeFaceDetector:
    """Wraps MediaPipe Face Detection (Tasks API)."""

    def __init__(
        self,
        model_path: str = config.MEDIAPIPE_MODEL_PATH,
        min_confidence: float = 0.50
    ):
        self.model_path = model_path
        self.min_confidence = min_confidence

        self._ensure_model_exists()

        log.info(f"Initializing MediaPipe FaceDetector with {model_path} ...")
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        log.info("MediaPipe FaceDetector ready.")

    def _ensure_model_exists(self):
        """Download blaze_face_short_range.tflite if not already present."""
        path = Path(self.model_path)
        if not path.exists():
            log.info(f"Downloading MediaPipe face detection model to {path} ...")
            urllib.request.urlretrieve(config.MEDIAPIPE_MODEL_URL, str(path))
            log.info("MediaPipe model download complete.")

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run face detection on a BGR webcam frame.

        Returns:
            List of detected face dicts with:
                - 'confidence': float
                - 'bbox_px': (x, y, w, h)
                - 'bbox_norm': (xmin, ymin, width, height) in [0, 1]
                - 'keypoints': list of keypoints if present
        """
        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)
        faces = []

        for d in result.detections:
            conf = float(d.categories[0].score) if d.categories else 0.0
            box = d.bounding_box

            # Pixel bounding box clamped to image
            x = max(0, int(box.origin_x))
            y = max(0, int(box.origin_y))
            w = max(1, min(int(box.width), img_w - x))
            h = max(1, min(int(box.height), img_h - y))

            # Normalized bounding box [0.0, 1.0]
            xmin = max(0.0, float(box.origin_x) / img_w)
            ymin = max(0.0, float(box.origin_y) / img_h)
            w_norm = min(1.0 - xmin, float(box.width) / img_w)
            h_norm = min(1.0 - ymin, float(box.height) / img_h)

            faces.append({
                "confidence": conf,
                "bbox_px": (x, y, w, h),
                "bbox_norm": (xmin, ymin, w_norm, h_norm),
                "keypoints": d.keypoints if hasattr(d, "keypoints") else None
            })

        return faces

    def select_primary_face(
        self,
        faces: List[Dict[str, Any]],
        strategy: str = config.MULTI_FACE_STRATEGY
    ) -> Optional[Dict[str, Any]]:
        """
        Select a primary face when multiple faces are detected.
        Supported strategies:
          - 'largest': Area of bounding box (closest/primary person)
          - 'highest_confidence': Face with max confidence
          - 'first': First face detected
        """
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]

        if strategy == "largest":
            return max(
                faces,
                key=lambda f: f["bbox_norm"][2] * f["bbox_norm"][3]
            )
        elif strategy == "highest_confidence":
            return max(faces, key=lambda f: f["confidence"])
        else:
            return faces[0]
