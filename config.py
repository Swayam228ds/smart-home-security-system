"""
config.py — Centralized Configuration for Laptop Webcam Security System
=======================================================================
Configured exclusively for Laptop Webcam capture with YOLOv8n human detection
and MediaPipe Face Detection quality gating.
"""

from pathlib import Path

# ─── Server & Laptop Webcam Settings ─────────────────────────────────────────
HOST                 = "127.0.0.1"
PORT                 = 8000
CAMERA_INDEX         = 0        # 0 = primary laptop webcam; tries 1 automatically if needed
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
TARGET_FPS           = 30
CAMERA_RETRY_SECS    = 2.0      # Retry interval if laptop camera shutter is closed

# ─── Storage Settings ────────────────────────────────────────────────────────
DB_PATH              = "security.db"
IMAGE_SAVE_DIR       = Path("images")

# ─── YOLOv8 Human Detection Settings ─────────────────────────────────────────
YOLO_MODEL           = "yolov8n.pt"  # Automatically verified (~6 MB)
DETECTION_CONFIDENCE = 0.50          # YOLO confidence threshold for person
PERSON_CLASS_ID      = 0             # COCO class 0 = person
EVENT_COOLDOWN_SECS  = 30            # Seconds before a new visitor event can start

# ─── MediaPipe Face Detection Settings ───────────────────────────────────────
MEDIAPIPE_MODEL_PATH = "blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# ─── Face Quality / Range Gates (Laptop Webcam Proxies) ──────────────────────
# Note: These values represent image-based proxies for a suitable capture range,
# NOT an exact physical distance measurement.
FACE_MIN_CONFIDENCE         = 0.60  # Minimum face detection confidence
FACE_MIN_WIDTH_RATIO        = 0.15  # Minimum face box width relative to webcam frame width
FACE_MIN_HEIGHT_RATIO       = 0.15  # Minimum face box height relative to webcam frame height
FACE_CENTER_TOLERANCE       = 0.25  # Max allowed normalized offset from frame center
FACE_EDGE_MARGIN            = 0.05  # Minimum distance from frame boundaries (0.0 to 0.5)
FACE_STABLE_FRAMES          = 5     # Consecutive frames meeting criteria before capture
FACE_CAPTURE_TIMEOUT_SECS   = 30    # Max seconds to wait for suitable face
FALLBACK_CAPTURE_ON_TIMEOUT = True  # If True, save best frame available on timeout

# ─── Multi-Face & Cropping Settings ──────────────────────────────────────────
MULTI_FACE_STRATEGY         = "largest"  # "largest", "highest_confidence", "first"
SAVE_FACE_CROP              = True       # Save cropped face alongside full frame for recognition

# ─── Best Frame Scoring Weights ──────────────────────────────────────────────
# Composite quality score Q in [0.0, 1.0] computed from weighted components:
WEIGHT_CONFIDENCE           = 0.30
WEIGHT_SIZE                 = 0.30
WEIGHT_CENTER               = 0.20
WEIGHT_SHARPNESS            = 0.20
