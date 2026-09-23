"""
app.py — Smart Home Security Demo (Laptop Webcam Edition)
========================================================
FastAPI server + REST API endpoints + Computer Vision Pipeline:
  • Startup initialization of YOLOv8n & MediaPipe Face Detection
  • Laptop Webcam video pipeline with auto-recovery and shutter detection
  • Visitor State Machine tracking and quality-based capture
  • Live MJPEG streaming with dual bounding boxes and HUD
  • Web dashboard for live monitoring and history review

Run:
    python app.py
    Open: http://127.0.0.1:8000
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import uvicorn

import config
import database
from detector import YoloHumanDetector, MediaPipeFaceDetector
from state_machine import VisitorStateMachine
from camera import CameraPipeline
from quality import evaluate_face_quality
from rendering import draw_hud

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Pipeline Singletons ──────────────────────────────────────────────────────
human_detector: YoloHumanDetector = None  # type: ignore
face_detector: MediaPipeFaceDetector = None  # type: ignore
state_machine: VisitorStateMachine = None  # type: ignore
camera_pipeline: CameraPipeline = None  # type: ignore


# ─── Startup & Shutdown Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global human_detector, face_detector, state_machine, camera_pipeline
    config.IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    database.init_db()

    log.info("Initializing Computer Vision detectors...")
    human_detector = YoloHumanDetector(
        model_path=config.YOLO_MODEL,
        conf=config.DETECTION_CONFIDENCE
    )
    face_detector = MediaPipeFaceDetector(
        model_path=config.MEDIAPIPE_MODEL_PATH,
        min_confidence=config.FACE_MIN_CONFIDENCE
    )
    state_machine = VisitorStateMachine()
    camera_pipeline = CameraPipeline(
        human_detector=human_detector,
        face_detector=face_detector,
        state_machine=state_machine
    )

    camera_pipeline.start()
    log.info(f"System ready! Dashboard available at http://{config.HOST}:{config.PORT}")

    yield

    if camera_pipeline:
        camera_pipeline.stop()
    log.info("System shutdown complete.")


app = FastAPI(
    title="Smart Home Security Demo (Laptop Webcam)",
    description="Laptop webcam pipeline with YOLOv8n human sensing and MediaPipe face quality gating",
    version="2.1.0",
    lifespan=lifespan
)


# ─── Video Stream & Web Dashboard ─────────────────────────────────────────────

@app.get("/video_feed")
def video_feed():
    """Live MJPEG stream from the laptop webcam with dual bounding boxes and HUD."""
    if not camera_pipeline:
        return Response(status_code=503, content="Camera pipeline not initialized")
    return StreamingResponse(
        camera_pipeline.frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the main dashboard HTML page."""
    html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h3>index.html not found</h3>", status_code=404)
    return html_path.read_text(encoding="utf-8")


# ─── Status API ───────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    """
    Return comprehensive system state:
    Webcam connection status, human presence, MediaPipe face quality metrics,
    state machine progress, and active visit ID.
    """
    if not state_machine or not camera_pipeline:
        return {
            "camera": "stopped",
            "webcam_connected": False,
            "human_detected": False,
            "human_count": 0,
            "humans_detected": 0,
            "face_detected": False,
            "faces_detected": 0,
            "face_confidence": 0.0,
            "face_width_ratio": 0.0,
            "face_height_ratio": 0.0,
            "face_centered": False,
            "face_quality": 0.0,
            "capture_state": "stopped",
            "current_visit_id": None,
            "stability_progress": f"0/{config.FACE_STABLE_FRAMES}",
            "status": "System starting up…",
        }

    snapshot = state_machine.get_status_snapshot()
    snapshot["camera"] = "running" if camera_pipeline.is_running() else "stopped"
    snapshot["webcam_connected"] = camera_pipeline.is_camera_connected()
    snapshot["active_camera_index"] = camera_pipeline.active_camera_index
    # Backward compatibility alias
    snapshot["humans_detected"] = snapshot["human_count"]
    return snapshot


# ─── Visits API ───────────────────────────────────────────────────────────────

@app.get("/api/visits")
def api_get_visits():
    """Return all visit records, newest first."""
    return database.get_visits()


@app.get("/api/visits/{visit_id}")
def api_get_visit(visit_id: int):
    """Return a single visit record."""
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    return visit


@app.get("/api/visits/{visit_id}/image")
def api_visit_image(visit_id: int):
    """Serve the captured JPEG for a visit."""
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    img_path = Path(visit["image_path"])
    if not img_path.exists():
        return Response(status_code=404, content="Image file not found on disk")
    return FileResponse(str(img_path), media_type="image/jpeg")


@app.get("/api/visits/{visit_id}/face_image")
def api_visit_face_image(visit_id: int):
    """Serve the cropped face JPEG for a visit (for future face recognition)."""
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")
    img_path = Path(visit["image_path"])
    face_path = img_path.with_name(f"{img_path.stem}_face{img_path.suffix}")
    if not face_path.exists():
        return Response(status_code=404, content="Face crop not available for this visit")
    return FileResponse(str(face_path), media_type="image/jpeg")


@app.post("/api/process_frame")
async def api_process_frame(request: Request):
    """
    Accepts a live JPEG frame streamed from the laptop webcam via browser getUserMedia,
    runs YOLO human detection, MediaPipe face detection, advances the visitor state machine,
    draws dual bounding boxes & HUD, and returns the annotated frame with live metrics.
    """
    body = await request.body()
    if not body:
        return Response(status_code=400, content="Empty frame body")

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(status_code=400, content="Invalid image payload")

    # 1. YOLOv8 human detection
    humans = human_detector.detect_humans(frame) if human_detector else []

    # 2. MediaPipe face detection
    faces = []
    primary_face = None
    if face_detector:
        faces = face_detector.detect_faces(frame)
        primary_face = face_detector.select_primary_face(faces)

    # 3. Advance visitor state machine & candidate frame scoring
    if state_machine:
        status_snapshot = state_machine.process_frame(
            frame_bgr=frame,
            humans=humans,
            faces=faces,
            primary_face=primary_face
        )
    else:
        status_snapshot = {
            "human_detected": False,
            "human_count": 0,
            "face_detected": False,
            "status": "Detector offline"
        }

    # 4. Render dual bounding boxes & HUD overlay
    annotated = draw_hud(
        frame=frame,
        humans=humans,
        faces=faces,
        primary_face=primary_face,
        status_snapshot=status_snapshot
    )

    # 5. Compress annotated frame to base64
    _, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_str = base64.b64encode(enc).decode("ascii")

    return {
        "status": status_snapshot,
        "annotated_image": f"data:image/jpeg;base64,{b64_str}"
    }


@app.post("/api/visits")
async def api_create_visit(request: Request):
    """
    Accept an uploaded JPEG frame, run face-detection & quality checks,
    and record a visit.
    """
    body = await request.body()
    if not body:
        return Response(status_code=400, content="Empty image body")

    nparr = np.frombuffer(body, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response(status_code=400, content="Invalid JPEG payload")

    ts = datetime.now()

    face_det = 0
    face_conf = 0.0
    face_bbox_str = ""
    face_q = 0.0
    human_count = 1

    if face_detector:
        faces = face_detector.detect_faces(frame)
        primary = face_detector.select_primary_face(faces)
        if primary:
            q_res = evaluate_face_quality(frame, primary)
            face_det = 1
            face_conf = primary["confidence"]
            face_bbox_str = json.dumps(primary["bbox_px"])
            face_q = q_res["quality_score"]

    visit_id = database.create_visit(
        timestamp=ts,
        image_path="",
        human_count=human_count,
        capture_status="captured"
    )

    img_path = config.IMAGE_SAVE_DIR / f"visit_{visit_id:04d}.jpg"
    cv2.imwrite(str(img_path), frame)

    database.update_visit_capture(
        visit_id=visit_id,
        image_path=str(img_path),
        capture_status="captured",
        face_detected=face_det,
        face_confidence=face_conf,
        face_bbox=face_bbox_str,
        face_quality=face_q
    )

    log.info(f"[POST /api/visits] Created Visit #{visit_id} from uploaded frame.")
    return {
        "visit_id": visit_id,
        "image_path": str(img_path),
        "capture_status": "captured",
        "face_detected": bool(face_det),
        "face_confidence": face_conf,
        "face_quality": face_q
    }


@app.patch("/api/visits/{visit_id}")
async def api_update_visit(visit_id: int, request: Request):
    """Update visit fields (e.g. processing_status: pending -> reviewed)."""
    body = await request.json()
    visit = database.get_visit(visit_id)
    if not visit:
        return Response(status_code=404, content="Visit not found")

    new_status = body.get("processing_status")
    if new_status:
        import sqlite3
        with sqlite3.connect(database.DB_PATH) as conn:
            conn.execute(
                "UPDATE visits SET processing_status = ? WHERE visit_id = ?",
                (new_status, visit_id),
            )
            conn.commit()
    return database.get_visit(visit_id)


@app.delete("/api/visits")
def api_clear_history():
    """Delete ALL visit records from database and all image files from disk."""
    deleted = database.delete_all_visits()
    if state_machine:
        state_machine.reset_active_visit()
    return {"message": f"Cleared {deleted} visit(s) and their images."}


# ─── Camera Controls ───────────────────────────────────────────────────────────

@app.post("/api/camera/start")
def api_camera_start():
    if camera_pipeline:
        camera_pipeline.start()
    return {"message": "Camera started"}


@app.post("/api/camera/stop")
def api_camera_stop():
    if camera_pipeline:
        camera_pipeline.stop()
    return {"message": "Camera stopped"}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Smart Home Security Demo — Laptop Webcam Edition")
    print("  YOLOv8n Human Detection + MediaPipe Face Quality Gating")
    print("=" * 60)
    print(f"  Dashboard       → http://{config.HOST}:{config.PORT}")
    print(f"  Primary Camera  → Index {config.CAMERA_INDEX} (Auto-probes index 0 & 1)")
    print(f"  Face Min Conf   → {config.FACE_MIN_CONFIDENCE:.2f}")
    print(f"  Face Min Size   → {config.FACE_MIN_WIDTH_RATIO:.0%} W / {config.FACE_MIN_HEIGHT_RATIO:.0%} H")
    print(f"  Face Stability  → {config.FACE_STABLE_FRAMES} frames")
    print(f"  Capture Timeout → {config.FACE_CAPTURE_TIMEOUT_SECS}s")
    print(f"  Cooldown        → {config.EVENT_COOLDOWN_SECS}s")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host=config.HOST, port=config.PORT, reload=False)
