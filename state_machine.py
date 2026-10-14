"""
state_machine.py — Visitor Event State Machine & Frame Selection
================================================================
Manages the visitor lifecycle:
  NO_VISITOR
  ↓
  HUMAN_DETECTED
  ↓
  WAITING_FOR_FACE
  ↓
  FACE_DETECTED
  ↓
  FACE_QUALITY_CHECK
  ↓
  STABLE_FACE
  ↓
  CAPTURE_IMAGE
  ↓
  VISIT_RECORDED
  ↓
  COOLDOWN
  ↓
  NO_VISITOR

Handles best-frame buffering, timeout fallback, stability counting,
and duplicate visit prevention.
"""

import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np

import config
import database
from quality import evaluate_face_quality

log = logging.getLogger(__name__)


class VisitorStateMachine:
    """Thread-safe state machine for visitor tracking and high-quality image capture."""

    def __init__(self):
        self._lock = threading.Lock()

        # State lifecycle
        self.state: str = "NO_VISITOR"
        self.status_message: str = "Waiting for visitor…"

        # Active visit tracking
        self.current_visit_id: Optional[int] = None
        self.event_start_time: float = 0.0
        self.last_visit_time: float = 0.0
        self.stable_frame_count: int = 0

        # Best candidate frame during active visitor event
        # Candidate structure: {"frame": ndarray, "face": dict, "quality": dict, "score": float}
        self.best_candidate: Optional[Dict[str, Any]] = None

        # Latest frame evaluation snapshot (for API & overlay)
        self.latest_snapshot: Dict[str, Any] = {
            "human_detected": False,
            "human_count": 0,
            "face_detected": False,
            "faces_detected": 0,
            "face_confidence": 0.0,
            "face_width_ratio": 0.0,
            "face_height_ratio": 0.0,
            "face_centered": False,
            "face_quality": 0.0,
            "capture_state": "no_visitor",
            "current_visit_id": None,
            "stability_progress": f"0/{config.FACE_STABLE_FRAMES}",
            "status": "Waiting for visitor…",
            "rejection_reason": "",
            "primary_face_bbox": None,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        humans: List[Dict[str, Any]],
        faces: List[Dict[str, Any]],
        primary_face: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes a single frame through the state machine.
        Executed in the camera thread.

        Returns:
            Dict containing annotation and state info for rendering and logging.
        """
        now = time.time()
        human_count = len(humans)
        faces_count = len(faces)
        face_present = primary_face is not None

        # Evaluate face quality if a primary face is detected
        quality_res: Optional[Dict[str, Any]] = None
        if face_present:
            quality_res = evaluate_face_quality(frame_bgr, primary_face)

        action_to_perform = None
        visit_to_finalize = None
        candidate_to_save = None

        with self._lock:
            time_since_last = now - self.last_visit_time

            # ── 1. COOLDOWN STATE ─────────────────────────────────────────────
            if self.state == "COOLDOWN":
                if time_since_last < config.EVENT_COOLDOWN_SECS:
                    if human_count > 0:
                        self.status_message = (
                            f"Active visit  |  Visit #{self.current_visit_id}  (cooldown)"
                        )
                    else:
                        remaining = int(config.EVENT_COOLDOWN_SECS - time_since_last)
                        self.status_message = (
                            f"Visit #{self.current_visit_id} ended  |"
                            f"  Next visit ready in {remaining}s"
                        )
                else:
                    # Cooldown expired
                    if human_count == 0:
                        self.state = "NO_VISITOR"
                        self.status_message = "Waiting for visitor…"
                    else:
                        # Human still lingering after cooldown; keep waiting until cleared
                        # or start a new visit if requested
                        self.status_message = "Visitor still in frame — step away to reset"

            # ── 2. NO_VISITOR STATE ──────────────────────────────────────────
            elif self.state == "NO_VISITOR":
                if human_count > 0 and time_since_last >= config.EVENT_COOLDOWN_SECS:
                    # Transition: NO_VISITOR -> HUMAN_DETECTED -> WAITING_FOR_FACE
                    self.state = "WAITING_FOR_FACE"
                    self.event_start_time = now
                    self.stable_frame_count = 0
                    self.best_candidate = None
                    self.status_message = "Visitor detected — move closer and face camera"
                    action_to_perform = "create_visit"
                else:
                    self.status_message = "Waiting for visitor…"

            # ── 3. WAITING_FOR_FACE & QUALITY CHECK ──────────────────────────
            elif self.state in ("WAITING_FOR_FACE", "FACE_DETECTED", "FACE_QUALITY_CHECK"):
                elapsed = now - self.event_start_time

                # Check for capture timeout
                if elapsed >= config.FACE_CAPTURE_TIMEOUT_SECS:
                    log.warning(
                        f"Timeout reached ({config.FACE_CAPTURE_TIMEOUT_SECS}s) "
                        f"for visit #{self.current_visit_id}."
                    )
                    self.state = "COOLDOWN"
                    self.last_visit_time = now
                    self.status_message = (
                        f"Visit #{self.current_visit_id} timeout — "
                        "face not detected or out of range"
                    )
                    action_to_perform = "handle_timeout"
                    visit_to_finalize = self.current_visit_id
                    candidate_to_save = self.best_candidate
                else:
                    if not face_present:
                        self.state = "WAITING_FOR_FACE"
                        self.stable_frame_count = 0
                        self.status_message = "Visitor detected — please face camera"
                    else:
                        self.state = "FACE_QUALITY_CHECK"
                        score = quality_res["quality_score"]

                        # Update best candidate frame if score is higher or first seen
                        if self.best_candidate is None or score > self.best_candidate["score"]:
                            self.best_candidate = {
                                "frame": frame_bgr.copy(),
                                "face": primary_face,
                                "quality": quality_res,
                                "score": score,
                            }

                        if quality_res["passed"]:
                            self.stable_frame_count += 1
                            self.status_message = (
                                f"Face ready — hold still "
                                f"({self.stable_frame_count}/{config.FACE_STABLE_FRAMES})"
                            )

                            if self.stable_frame_count >= config.FACE_STABLE_FRAMES:
                                # Transition: STABLE_FACE -> CAPTURE_IMAGE
                                self.state = "CAPTURE_IMAGE"
                                self.status_message = "Face verified — capturing best image…"
                                action_to_perform = "capture_visit"
                                visit_to_finalize = self.current_visit_id
                                candidate_to_save = self.best_candidate
                        else:
                            # Quality criteria not satisfied
                            self.stable_frame_count = 0
                            self.status_message = quality_res["rejection_reason"]

            # Build status snapshot under lock
            capture_state_lower = self.state.lower()
            stability_str = f"{self.stable_frame_count}/{config.FACE_STABLE_FRAMES}"

            self.latest_snapshot = {
                "human_detected":     human_count > 0,
                "human_count":        human_count,
                "face_detected":      face_present,
                "faces_detected":     faces_count,
                "face_confidence":    quality_res["confidence"] if quality_res else 0.0,
                "face_width_ratio":   quality_res["width_ratio"] if quality_res else 0.0,
                "face_height_ratio":  quality_res["height_ratio"] if quality_res else 0.0,
                "face_centered":      quality_res["is_centered"] if quality_res else False,
                "face_quality":       quality_res["quality_score"] if quality_res else 0.0,
                "capture_state":      capture_state_lower,
                "current_visit_id":   self.current_visit_id,
                "stability_progress": stability_str,
                "status":             self.status_message,
                "rejection_reason":   quality_res["rejection_reason"] if quality_res else "",
                "primary_face_bbox":  primary_face["bbox_px"] if primary_face else None,
            }

        # ── Heavy I/O outside lock ───────────────────────────────────────────
        if action_to_perform == "create_visit":
            ts = datetime.now()
            vid = database.create_visit(ts, "", human_count, capture_status="waiting_for_face")
            with self._lock:
                self.current_visit_id = vid
                self.latest_snapshot["current_visit_id"] = vid
            log.info(f"YOLO detected human → Started Visit #{vid} (waiting for suitable face)")

        elif action_to_perform == "capture_visit":
            self._save_captured_visit(visit_to_finalize, candidate_to_save, frame_bgr, primary_face, quality_res)

        elif action_to_perform == "handle_timeout":
            self._handle_timeout_save(visit_to_finalize, candidate_to_save, frame_bgr)

        return self.latest_snapshot

    def _save_captured_visit(
        self,
        visit_id: int,
        candidate: Optional[Dict[str, Any]],
        current_frame: np.ndarray,
        primary_face: Optional[Dict[str, Any]],
        quality_res: Optional[Dict[str, Any]]
    ):
        """Save the best candidate frame, optional face crop, and update database."""
        config.IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        img_path = config.IMAGE_SAVE_DIR / f"visit_{visit_id:04d}.jpg"

        # Determine best frame to save
        if candidate and "frame" in candidate:
            target_frame = candidate["frame"]
            face_meta = candidate["face"]
            q_meta = candidate["quality"]
        else:
            target_frame = current_frame
            face_meta = primary_face or {}
            q_meta = quality_res or {}

        # Save original full frame
        cv2.imwrite(str(img_path), target_frame)

        # Optionally save face crop
        bbox_px = face_meta.get("bbox_px")
        if config.SAVE_FACE_CROP and bbox_px:
            x, y, w, h = bbox_px
            ih, iw = target_frame.shape[:2]
            # Add 20% margin around face for recognition crop
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            crop_x1 = max(0, x - margin_x)
            crop_y1 = max(0, y - margin_y)
            crop_x2 = min(iw, x + w + margin_x)
            crop_y2 = min(ih, y + h + margin_y)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop = target_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_path = config.IMAGE_SAVE_DIR / f"visit_{visit_id:04d}_face.jpg"
                cv2.imwrite(str(crop_path), crop)
                
                # Recognize Face
                from recognizer import face_recognizer
                match = face_recognizer.recognize(crop)
                if match:
                    person_id, person_name, conf_dist = match
                    database.assign_person_to_visit(visit_id, person_id, person_name)
                    log.info(f"Recognized {person_name} (dist: {conf_dist:.1f})")

        conf = float(face_meta.get("confidence", 0.0))
        q_score = float(q_meta.get("quality_score", 0.0))
        bbox_str = json.dumps(bbox_px) if bbox_px else ""

        # Update database record
        database.update_visit_capture(
            visit_id=visit_id,
            image_path=str(img_path),
            capture_status="captured",
            face_detected=1,
            face_confidence=conf,
            face_bbox=bbox_str,
            face_quality=q_score,
        )

        with self._lock:
            self.state = "VISIT_RECORDED"
            self.last_visit_time = time.time()
            self.status_message = f"🚨 Visit #{visit_id} captured successfully!"
            # Transition to cooldown
            self.state = "COOLDOWN"

        log.info(
            f"\n{'═'*55}\n"
            f"  VISITOR IMAGE CAPTURED (BEST FRAME)\n"
            f"  Visit ID    : {visit_id}\n"
            f"  Confidence  : {conf:.2f}\n"
            f"  Quality Q   : {q_score:.2f}\n"
            f"  Image       : {img_path}\n"
            f"{'═'*55}"
        )

    def _handle_timeout_save(
        self,
        visit_id: int,
        candidate: Optional[Dict[str, Any]],
        current_frame: np.ndarray
    ):
        """Handle timeout event with configurable fallback capture."""
        if config.FALLBACK_CAPTURE_ON_TIMEOUT:
            config.IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            img_path = config.IMAGE_SAVE_DIR / f"visit_{visit_id:04d}.jpg"

            if candidate and "frame" in candidate:
                target_frame = candidate["frame"]
                face_meta = candidate.get("face") or {}
                q_meta = candidate.get("quality") or {}
                face_det = 1
                conf = float(face_meta.get("confidence", 0.0))
                q_score = float(q_meta.get("quality_score", 0.0))
                bbox_str = json.dumps(face_meta.get("bbox_px")) if face_meta.get("bbox_px") else ""
            else:
                target_frame = current_frame
                face_det = 0
                conf = 0.0
                q_score = 0.0
                bbox_str = ""

            cv2.imwrite(str(img_path), target_frame)

            database.update_visit_capture(
                visit_id=visit_id,
                image_path=str(img_path),
                capture_status="timeout",
                face_detected=face_det,
                face_confidence=conf,
                face_bbox=bbox_str,
                face_quality=q_score,
            )
            log.info(f"Visit #{visit_id} recorded with fallback image on timeout.")
        else:
            database.update_capture_status(visit_id=visit_id, capture_status="timeout")
            log.info(f"Visit #{visit_id} recorded as timeout (no image saved).")

    def get_status_snapshot(self) -> Dict[str, Any]:
        """Return a copy of the latest status snapshot for API endpoints."""
        with self._lock:
            return dict(self.latest_snapshot)

    def reset_active_visit(self):
        """Reset active visit state (e.g. if history was cleared)."""
        with self._lock:
            self.current_visit_id = None
            self.state = "NO_VISITOR"
            self.status_message = "Waiting for visitor…"
            self.stable_frame_count = 0
            self.best_candidate = None
