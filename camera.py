"""
camera.py — Laptop Webcam Pipeline Coordinator & Auto-Recovery
==============================================================
Specifically engineered for Laptop Webcams:
  • Multi-backend & multi-index camera discovery (DirectShow / Default)
  • Dynamic auto-recovery when camera shutter is opened/closed
  • Live diagnostic placeholder stream when webcam is inaccessible
  • Real-time YOLOv8n + MediaPipe Face quality pipeline execution
"""

import time
import logging
import threading
from typing import Optional, Generator, Tuple

import cv2
import numpy as np

import config
from detector import YoloHumanDetector, MediaPipeFaceDetector
from state_machine import VisitorStateMachine
from rendering import draw_hud

log = logging.getLogger(__name__)


def create_diagnostic_frame(title: str, lines: list, width: int = 640, height: int = 480) -> np.ndarray:
    """Create a clean visual diagnostic frame when the webcam is unavailable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 24, 32)  # Dark navy background

    # Border
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (50, 60, 75), 1)

    # Header banner
    cv2.rectangle(frame, (10, 10), (width - 10, 60), (35, 42, 54), -1)
    cv2.putText(
        frame, title, (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA
    )

    # Instruction lines
    y = 110
    for line in lines:
        cv2.putText(
            frame, line, (30, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 215, 230), 1, cv2.LINE_AA
        )
        y += 34

    # Timestamp indicator
    cur_time = time.strftime("%H:%M:%S")
    cv2.putText(
        frame, f"System Active | Probing Webcam... [{cur_time}]",
        (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 130, 160), 1, cv2.LINE_AA
    )

    return frame


class CameraPipeline:
    """Manages the laptop webcam lifecycle, inference, state machine, and streaming."""

    def __init__(
        self,
        human_detector: YoloHumanDetector,
        face_detector: MediaPipeFaceDetector,
        state_machine: VisitorStateMachine
    ):
        self.human_detector = human_detector
        self.face_detector = face_detector
        self.state_machine = state_machine

        self.running = False
        self.camera_connected = False
        self.active_camera_index = config.CAMERA_INDEX
        self._thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None

    def start(self):
        """Start the background camera pipeline loop."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="laptop-webcam-thread"
            )
            self._thread.start()
            log.info("Laptop webcam pipeline thread started.")

    def stop(self):
        """Signal the camera pipeline to stop."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Laptop webcam pipeline stopped.")

    def is_running(self) -> bool:
        return self.running

    def is_camera_connected(self) -> bool:
        return self.camera_connected

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Return the most recently rendered JPEG bytes for the MJPEG stream."""
        with self._frame_lock:
            return self._latest_jpeg

    def frame_generator(self) -> Generator[bytes, None, None]:
        """Yield MJPEG multipart stream boundary chunks."""
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            frame_bytes = self.get_latest_jpeg()
            if frame_bytes:
                yield boundary + frame_bytes + b"\r\n"
            time.sleep(1.0 / config.TARGET_FPS)

    def _probe_webcam(self) -> Optional[Tuple[cv2.VideoCapture, int]]:
        """
        Attempts to open the laptop webcam across indices and backends.
        Returns (cap, index) on success, or None.
        """
        candidates = [
            (config.CAMERA_INDEX, cv2.CAP_DSHOW),
            (config.CAMERA_INDEX, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW),
            (1, cv2.CAP_ANY),
        ]

        for idx, backend in candidates:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    # Test frame read
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                        log.info(f"✅ Laptop webcam successfully opened at index {idx} (backend={backend}).")
                        return cap, idx
                    cap.release()
            except Exception:
                pass
        return None

    def _set_placeholder(self, title: str, lines: list):
        """Encodes and publishes a placeholder diagnostic frame."""
        diag_frame = create_diagnostic_frame(title, lines, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        _, jpeg = cv2.imencode(".jpg", diag_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._frame_lock:
            self._latest_jpeg = jpeg.tobytes()

    def _run_loop(self):
        """Continuous webcam processing and auto-recovery loop."""
        frame_interval = 1.0 / config.TARGET_FPS

        while self.running:
            # ── Step 1: Probe or reconnect webcam ────────────────────────────
            cap_tuple = self._probe_webcam()

            if not cap_tuple:
                self.camera_connected = False
                log.warning(
                    "⚠️ Laptop webcam unavailable. Shutter may be closed or in use. Retrying in background..."
                )
                # Show helpful diagnostic frame to the user
                self._set_placeholder(
                    title="📷 Laptop Webcam Unavailable",
                    lines=[
                        "• Check if the physical camera shutter on your laptop is closed.",
                        "• Press the camera privacy key on your keyboard (often F9 or Fn+F9).",
                        "• Verify Windows Settings: Privacy & security -> Camera -> Allow.",
                        "• Ensure no other application (Zoom, Teams) is holding the camera.",
                        f"• Auto-reconnecting every {config.CAMERA_RETRY_SECS:.0f}s..."
                    ]
                )
                time.sleep(config.CAMERA_RETRY_SECS)
                continue

            cap, active_idx = cap_tuple
            self.camera_connected = True
            self.active_camera_index = active_idx

            # ── Step 2: Stream active frames ─────────────────────────────────
            while self.running:
                start_t = time.time()
                ok, frame = cap.read()

                if not ok or frame is None or frame.size == 0:
                    log.warning("Lost connection to laptop webcam — attempting recovery…")
                    break

                # 1. YOLOv8 Human Detection
                humans = self.human_detector.detect_humans(frame)

                # 2. MediaPipe Face Detection
                current_state = self.state_machine.state
                should_run_face = (
                    len(humans) > 0 or
                    current_state in ("WAITING_FOR_FACE", "FACE_DETECTED", "FACE_QUALITY_CHECK")
                )

                faces = []
                primary_face = None
                if should_run_face:
                    faces = self.face_detector.detect_faces(frame)
                    primary_face = self.face_detector.select_primary_face(faces)

                # 3. Advance State Machine
                status_snapshot = self.state_machine.process_frame(
                    frame_bgr=frame,
                    humans=humans,
                    faces=faces,
                    primary_face=primary_face
                )

                # 4. Render Dual Bounding Boxes + HUD Overlay
                annotated = draw_hud(
                    frame=frame,
                    humans=humans,
                    faces=faces,
                    primary_face=primary_face,
                    status_snapshot=status_snapshot
                )

                # 5. Store JPEG for live MJPEG stream
                _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self._frame_lock:
                    self._latest_jpeg = jpeg.tobytes()

                # Maintain target FPS
                elapsed = time.time() - start_t
                sleep_time = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_time)

            cap.release()
            self.camera_connected = False
            time.sleep(0.5)

        log.info("Laptop webcam pipeline loop exited.")
