# Smart Home Security Demo 🏠🔒 — Laptop Webcam Edition

> **Intelligent Edge Security Pipeline** — University Interim Demonstration  
> Two-stage computer vision pipeline designed for your **Laptop Webcam**, combining **YOLOv8n Human Detection** for intrusion sensing with **MediaPipe Face Detection** for quality-gated visitor image capture.

---

## Architecture Overview

```
                  Laptop Webcam
                        │
                        ▼
       YOLOv8n Human Detection (Intrusion Sensing)
                        │
                        ▼ (Person in frame)
      Visitor Event Created (Database `waiting_for_face`)
                        │
                        ▼
WAITING_FOR_FACE ───► MediaPipe Face Detection (BlazeFace Short-Range)
                        │
                        ▼
Face Quality & Range Gate (Size, Centering, Edge, Confidence, Sharpness)
                        │
                        ├─► Not ready? ──► Buffer Best Candidate Frame ──► Wait for Stability
                        │
                        ├─► Timeout (10s)? ──► Save Fallback Image & Mark `timeout` ──► Cooldown
                        │
                        ▼ (Stable for 5 consecutive frames)
STABLE_FACE ──► Select Best Frame ──► Save Full JPEG & Face Crop ──► Update SQLite
                        │
                        ▼
VISIT_RECORDED ──► Cooldown (30s duplicate prevention) ──► NO_VISITOR
```

---

## Core Pipeline Details & Answers

### 1. Why YOLO Alone is Insufficient for Visitor Image Capture
In typical security setups, capturing an image the exact millisecond YOLO detects a human produces poor-quality photos. Visitors are often:
- Walking towards the laptop from across the room (face is only a few blurry pixels).
- Looking down, turned sideways, or reaching for their phone/keys.
- Blurred due to walking motion.
- Partially occluded or at the very edge of the webcam's field of view.

Snapping an image immediately yields an unidentifiable photo. By gating capture with face verification, the system waits until the visitor is within optimal range and facing the webcam.

### 2. Why MediaPipe Face Detection Was Added
MediaPipe Face Detection (based on BlazeFace) is an ultra-lightweight, high-speed neural network optimized for sub-millisecond edge inference:
- **YOLOv8n** handles wide-area body detection and initiates the event lifecycle.
- **MediaPipe** verifies the presence, geometry, orientation, and resolution of the visitor's face before committing any image to disk or SQLite.

### 3. How the Capture-Range Logic Works
Rather than claiming to measure physical metric distance from a 2D laptop webcam without a depth sensor, the system uses **image-based geometric proxies**:
- Normalized bounding box width (`w / W_frame`) and height (`h / H_frame`).
- When a person sits or stands in front of a laptop webcam at comfortable conversational distance (1–1.5m), their face occupies at least 15% of the frame dimensions.
- If the visitor is farther away, the ratio is below 15%, and the live HUD displays: *"Visitor detected — move closer"*.

### 4. Face Confidence Threshold (`FACE_MIN_CONFIDENCE = 0.60`)
Filters out shadows, patterns on clothing, or background objects. MediaPipe must detect a face with confidence $\ge 0.60$ for the frame to be evaluated.

### 5. Minimum Face-Size Threshold (`FACE_MIN_WIDTH_RATIO = 0.15`, `FACE_MIN_HEIGHT_RATIO = 0.15`)
Ensures sufficient pixel density for downstream facial recognition. Faces occupying less than 15% width or height are rejected with the prompt *"Visitor detected — move closer"*.

### 6. Centering Threshold (`FACE_CENTER_TOLERANCE = 0.25`)
Ensures the face is reasonably centered in the webcam's optical sweet spot. If the normalized face center $(c_x, c_y)$ deviates by more than $0.25$ from the center $(0.5, 0.5)$, the HUD instructs: *"Face detected — center yourself in camera"*.

### 7. Stability Requirement (`FACE_STABLE_FRAMES = 5`)
Capturing the first instant a face passes the threshold risks motion blur if the visitor is still moving. The stability gate requires that the face passes **all** quality gates for $5$ consecutive frames. If the visitor moves out of range or turns away, the stability counter resets to zero.

### 8. Capture Timeout & Fallback (`FACE_CAPTURE_TIMEOUT_SECS = 30`)
If a human is detected but no suitable face is detected within 30 seconds (e.g., visitor is wearing a helmet or looking away):
- The state machine does not hang indefinitely.
- It records the visit with `capture_status = "timeout"`.
- If `FALLBACK_CAPTURE_ON_TIMEOUT = True`, it saves the best candidate frame observed so far so security operators still have a visual record.
- It transitions immediately to `COOLDOWN` to prevent repeated visit creation.

### 9. Best-Frame Selection
While waiting for stability, the system maintains a running candidate frame buffer and computes a composite quality score $Q \in [0.0, 1.0]$:
$$Q = 0.30 \cdot S_{\text{conf}} + 0.30 \cdot S_{\text{size}} + 0.20 \cdot S_{\text{center}} + 0.20 \cdot S_{\text{sharpness}}$$
- **Sharpness ($S_{\text{sharpness}}$)** is evaluated using the variance of the Laplacian (`cv2.Laplacian(face_gray, cv2.CV_64F).var()`).
- When the stability criterion is met, the system saves the candidate frame with the highest composite score $Q$.

### 10. Duplicate Visit Prevention
- Once an event starts, `current_visit_id` is assigned and subsequent frames update that active visit.
- After an image is captured (or times out), the system enters `COOLDOWN` for `EVENT_COOLDOWN_SECS = 30`.
- Even if the person remains in front of the camera, no new visits are generated during cooldown.
- A new visit can only trigger after the cooldown expires and the scene clears.

### 11. Laptop Webcam Setup & Troubleshooting
- **Auto-Discovery**: The pipeline auto-probes camera index `0` and `1` using DirectShow and default backends.
- **Physical Camera Shutter**: Many HP, Lenovo, and Dell laptops have a physical privacy slider over the lens or an **F9 / Fn+F9** camera kill key. If the video stream shows the diagnostic banner, slide open the camera shutter or press the camera key.
- **Auto-Recovery**: The pipeline automatically probes and reconnects every 2 seconds—as soon as you open the shutter, the live stream begins immediately without restarting the server!

---

## File Structure

```
smart-home-security-demo/
├── app.py                      ← FastAPI server + REST API endpoints
├── camera.py                   ← Laptop webcam coordinator with auto-recovery
├── config.py                   ← Centralized configuration & proxy thresholds
├── database.py                 ← SQLite layer with automatic schema migrations
├── detector.py                 ← YOLOv8 and MediaPipe FaceDetector wrappers
├── quality.py                  ← Face quality gates, distance proxies & Laplacian sharpness
├── rendering.py                ← Dual bounding boxes, center reticle & HUD overlay
├── index.html                  ← Live web dashboard with real-time feedback
├── requirements.txt            ← Python dependencies (FastAPI, OpenCV, YOLO, MediaPipe)
├── blaze_face_short_range.tflite ← MediaPipe face detection model (~230 KB)
├── yolov8n.pt                  ← YOLOv8 nano model (~6 MB)
├── security.db                 ← SQLite database
├── API.md                      ← REST API documentation
└── README.md                   ← Project documentation
```

---

## Running the Application

### 1. Start Server
```powershell
python app.py
```

### 2. Open Dashboard
Open your browser at:
```
http://127.0.0.1:8000
```

### 3. Demonstration Workflow
1. Stand in front of your laptop webcam.
2. YOLOv8n detects your body $\rightarrow$ **Green box** appears, Visit #1 is created.
3. System prompts: *"Visitor detected — move closer and face camera"*.
4. Move closer and center yourself $\rightarrow$ **Cyan box** appears: *"Face ready — hold still (1/5 ... 5/5)"*.
5. System captures the sharpest frame, saves it to `images/visit_0001.jpg` and face crop to `images/visit_0001_face.jpg`, and updates SQLite.
6. System enters cooldown $\rightarrow$ no duplicate visits while you remain seated in front of your laptop.
