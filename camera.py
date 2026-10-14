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
    """Manages the ESP32-CAM or laptop webcam lifecycle, inference, state machine, and streaming."""

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
        self.camera_source: str = getattr(config, "CAMERA_SOURCE", "esp32")
        self.esp32_url: str = getattr(config, "ESP32_CAM_URL", "http://192.168.1.50/stream")
        self.active_camera_index = self.esp32_url if self.camera_source == "esp32" else config.CAMERA_INDEX
        self._thread: Optional[threading.Thread] = None

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._active_cap: Optional[cv2.VideoCapture] = None
        self._reconnect_event = threading.Event()

    def configure(self, source: str, esp32_url: Optional[str] = None):
        """Dynamically switch camera source between 'esp32' and 'webcam'."""
        self.camera_source = source
        if esp32_url:
            self.esp32_url = esp32_url.strip()
        self.active_camera_index = self.esp32_url if self.camera_source == "esp32" else config.CAMERA_INDEX
        log.info(f"Camera reconfigured -> Source: {self.camera_source}, Target: {self.active_camera_index}")
        
        self._reconnect_event.set()
        if self._active_cap:
            try:
                self._active_cap.release()
            except Exception:
                pass

    def start(self):
        """Start the background camera pipeline loop."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="camera-pipeline-thread"
            )
            self._thread.start()
            log.info("Camera pipeline thread started.")

    def stop(self):
        """Signal the camera pipeline to stop."""
        self.running = False
        self._reconnect_event.set()
        if self._active_cap:
            try:
                self._active_cap.release()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        log.info("Camera pipeline stopped.")

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

    def _probe_esp32_cam(self) -> Optional[Tuple[cv2.VideoCapture, str]]:
        """Attempts to connect to the ESP32-CAM MJPEG stream."""
        from urllib.parse import urlparse
        raw_url = self.esp32_url.strip()
        if not raw_url.startswith("http://") and not raw_url.startswith("https://"):
            raw_url = "http://" + raw_url

        candidates = [raw_url]
        try:
            parsed = urlparse(raw_url)
            host = parsed.netloc.split(":")[0] if parsed.netloc else parsed.path.split("/")[0]
            candidates = [
                raw_url,
                f"http://{host}:81/stream",
                f"http://{host}/stream",
                f"http://{host}:81",
                f"http://{host}/mjpeg",
            ]
            # Deduplicate while preserving order
            seen = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        except Exception:
            candidates = [raw_url]

        for url in candidates:
            try:
                log.info(f"Connecting to ESP32-CAM stream: {url}...")
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        log.info(f"✅ ESP32-CAM stream successfully opened at {url}!")
                        return cap, url
                    cap.release()
            except Exception as e:
                log.debug(f"Failed candidate {url}: {e}")
        return None

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

    def _probe_camera(self) -> Optional[Tuple[cv2.VideoCapture, any]]:
        """Probes the currently selected camera source (ESP32-CAM or Laptop Webcam)."""
        if self.camera_source == "esp32":
            return self._probe_esp32_cam()
        else:
            return self._probe_webcam()

    def _set_placeholder(self, title: str, lines: list):
        """Encodes and publishes a placeholder diagnostic frame."""
        diag_frame = create_diagnostic_frame(title, lines, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        _, jpeg = cv2.imencode(".jpg", diag_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with self._frame_lock:
            self._latest_jpeg = jpeg.tobytes()

    def _run_loop(self):
        """Continuous camera processing and auto-recovery loop."""
        frame_interval = 1.0 / config.TARGET_FPS

        while self.running:
            self._reconnect_event.clear()

            # ── Step 1: Probe or reconnect camera (ESP32-CAM or Webcam) ────────
            cap_tuple = self._probe_camera()

            if not cap_tuple:
                self.camera_connected = False
                if self.camera_source == "esp32":
                    log.warning(
                        f"⚠️ ESP32-CAM unavailable at '{self.esp32_url}'. Retrying in background..."
                    )
                    self._set_placeholder(
                        title="📷 ESP32-CAM Stream Connecting / Offline",
                        lines=[
                            f"• Target URL: {self.esp32_url}",
                            "• Ensure ESP32-CAM is powered ON (5V with at least 2A recommended).",
                            "• Verify your computer and ESP32 are connected to the SAME Wi-Fi.",
                            "• Open the stream in your browser to check if it's broadcasting.",
                            "• Flash ESP32 with esp32_cam_stream.ino if you haven't uploaded it.",
                            f"• Auto-reconnecting every {config.CAMERA_RETRY_SECS:.0f}s..."
                        ]
                    )
                else:
                    log.warning(
                        "⚠️ Laptop webcam unavailable. Shutter may be closed or in use. Retrying in background..."
                    )
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

                # Wait before next retry, or wake up immediately if reconfigured
                self._reconnect_event.wait(timeout=config.CAMERA_RETRY_SECS)
                continue

            cap, active_target = cap_tuple
            self._active_cap = cap
            self.camera_connected = True
            self.active_camera_index = active_target

            # ── Step 2: Stream active frames ─────────────────────────────────
            while self.running and not self._reconnect_event.is_set():
                start_t = time.time()
                ok, frame = cap.read()

                if not ok or frame is None or frame.size == 0:
                    log.warning(f"Lost connection to {self.camera_source.upper()} ({active_target}) — attempting recovery…")
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

            try:
                cap.release()
            except Exception:
                pass
            self._active_cap = None
            self.camera_connected = False
            time.sleep(0.3)

        log.info("Camera pipeline loop exited.")
