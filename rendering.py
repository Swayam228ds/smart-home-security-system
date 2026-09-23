"""
rendering.py — Live Video Annotation & HUD Overlays
===================================================
Draws dual bounding boxes (YOLO humans + MediaPipe faces),
center guide reticle, stability progress, and diagnostic status
directly onto the MJPEG video stream.
"""

from typing import List, Dict, Any, Optional
import cv2
import numpy as np

import config


def draw_hud(
    frame: np.ndarray,
    humans: List[Dict[str, Any]],
    faces: List[Dict[str, Any]],
    primary_face: Optional[Dict[str, Any]],
    status_snapshot: Dict[str, Any]
) -> np.ndarray:
    """
    Annotates the webcam frame with YOLO human boxes, MediaPipe face boxes,
    guide markers, and HUD state diagnostics.
    """
    annotated = frame.copy()
    img_h, img_w = annotated.shape[:2]

    # ── 1. Draw subtle center guidance box ────────────────────────────────────
    # Acceptable center zone defined by FACE_CENTER_TOLERANCE
    tol_x = int(img_w * config.FACE_CENTER_TOLERANCE)
    tol_y = int(img_h * config.FACE_CENTER_TOLERANCE)
    cx_mid = img_w // 2
    cy_mid = img_h // 2
    guide_x1 = cx_mid - tol_x
    guide_y1 = cy_mid - tol_y
    guide_x2 = cx_mid + tol_x
    guide_y2 = cy_mid + tol_y

    # Draw corner brackets for center zone
    bracket_len = 16
    color_bracket = (70, 70, 70)  # Subtle gray
    # Top-left
    cv2.line(annotated, (guide_x1, guide_y1), (guide_x1 + bracket_len, guide_y1), color_bracket, 1)
    cv2.line(annotated, (guide_x1, guide_y1), (guide_x1, guide_y1 + bracket_len), color_bracket, 1)
    # Top-right
    cv2.line(annotated, (guide_x2, guide_y1), (guide_x2 - bracket_len, guide_y1), color_bracket, 1)
    cv2.line(annotated, (guide_x2, guide_y1), (guide_x2, guide_y1 + bracket_len), color_bracket, 1)
    # Bottom-left
    cv2.line(annotated, (guide_x1, guide_y2), (guide_x1 + bracket_len, guide_y2), color_bracket, 1)
    cv2.line(annotated, (guide_x1, guide_y2), (guide_x1, guide_y2 - bracket_len), color_bracket, 1)
    # Bottom-right
    cv2.line(annotated, (guide_x2, guide_y2), (guide_x2 - bracket_len, guide_y2), color_bracket, 1)
    cv2.line(annotated, (guide_x2, guide_y2), (guide_x2, guide_y2 - bracket_len), color_bracket, 1)

    # ── 2. Draw YOLO human bounding boxes (Green) ────────────────────────────
    for h in humans:
        x1, y1, x2, y2 = h["bbox_xyxy"]
        conf = h["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"PERSON {conf:.2f}"
        cv2.rectangle(annotated, (x1, max(y1 - 20, 0)), (x1 + 105, max(y1, 20)), (0, 180, 0), -1)
        cv2.putText(
            annotated, label, (x1 + 4, max(y1 - 5, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

    # ── 3. Draw MediaPipe face bounding boxes ────────────────────────────────
    is_stable = status_snapshot.get("capture_state") in ("stable_face", "capture_image")
    face_quality = status_snapshot.get("face_quality", 0.0)
    stability_progress = status_snapshot.get("stability_progress", "0/5")
    rejection_code = ""

    for f in faces:
        x, y, w, h = f["bbox_px"]
        conf = f["confidence"]
        is_primary = (primary_face is not None and f["bbox_px"] == primary_face["bbox_px"])

        if is_primary:
            # Color code based on criteria satisfaction
            rejection_reason = status_snapshot.get("rejection_reason", "")
            if status_snapshot.get("capture_state") == "face_quality_check":
                if status_snapshot.get("face_width_ratio", 0) >= config.FACE_MIN_WIDTH_RATIO and status_snapshot.get("face_centered", False):
                    # Progressing toward stability
                    box_color = (0, 230, 255)  # Cyan
                    label = f"FACE {conf:.2f} | HOLD STILL ({stability_progress})"
                else:
                    box_color = (0, 165, 255)  # Orange
                    if not status_snapshot.get("face_centered", False):
                        label = f"FACE {conf:.2f} | MOVE TO CENTER"
                    else:
                        label = f"FACE {conf:.2f} | MOVE CLOSER"
            elif is_stable:
                box_color = (0, 255, 100)  # Bright Green
                label = f"FACE {conf:.2f} | READY (Q:{face_quality:.2f})"
            else:
                box_color = (0, 165, 255)  # Orange
                label = f"FACE {conf:.2f} | Q:{face_quality:.2f}"
            thickness = 2
        else:
            # Secondary face
            box_color = (180, 180, 180)  # Gray
            label = f"FACE {conf:.2f} (secondary)"
            thickness = 1

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)
        text_y = max(y - 8, 16)
        # Background pill for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (x, text_y - th - 3), (x + tw + 6, text_y + 3), (20, 20, 20), -1)
        cv2.putText(
            annotated, label, (x + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA
        )

    # ── 4. Top Status Header Overlay ─────────────────────────────────────────
    # Semi-transparent dark bar at the top
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, 54), (10, 14, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Status message
    status_text = status_snapshot.get("status", "Waiting for visitor…")
    capture_state = status_snapshot.get("capture_state", "no_visitor").upper()
    vid = status_snapshot.get("current_visit_id")
    vid_str = f" | Visit #{vid}" if vid else ""

    cv2.putText(
        annotated, f"[{capture_state}{vid_str}]  {status_text}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Metric pills
    h_count = status_snapshot.get("human_count", 0)
    f_count = status_snapshot.get("faces_detected", 0)
    metrics_text = (
        f"Humans: {h_count}  |  Faces: {f_count}  |  "
        f"Confidence: {status_snapshot.get('face_confidence', 0.0):.2f}  |  "
        f"Size: {status_snapshot.get('face_width_ratio', 0.0):.0%}  |  "
        f"Quality: {face_quality:.2f}"
    )
    cv2.putText(
        annotated, metrics_text,
        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA
    )

    return annotated
