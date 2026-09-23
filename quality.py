"""
quality.py — Face Quality Assessment & Frame Scoring
=====================================================
Evaluates whether a detected face meets range, size, position,
confidence, and sharpness criteria for high-quality capture.
Also calculates composite scores for selecting the best candidate frame.
"""

import math
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

import config


def compute_sharpness(image_bgr: np.ndarray, bbox_px: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    Compute image sharpness using the variance of the Laplacian.
    If bbox_px is given (x, y, w, h), compute sharpness on the face crop;
    otherwise compute across the whole frame.
    """
    if bbox_px:
        x, y, w, h = bbox_px
        img_h, img_w = image_bgr.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 > x1 and y2 > y1:
            roi = image_bgr[y1:y2, x1:x2]
        else:
            roi = image_bgr
    else:
        roi = image_bgr

    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(var)


def evaluate_face_quality(
    frame_bgr: np.ndarray,
    face_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates face metrics against configurable thresholds.

    Args:
        frame_bgr: Current BGR webcam frame
        face_data: Dict containing:
            - 'confidence': float (0.0 to 1.0)
            - 'bbox_px': (x, y, w, h)
            - 'bbox_norm': (xmin, ymin, width, height) relative to frame size

    Returns:
        Dict containing:
            - 'passed': bool (meets all quality gates)
            - 'quality_score': float (0.0 to 1.0 composite score)
            - 'confidence': float
            - 'width_ratio': float
            - 'height_ratio': float
            - 'offset_x': float
            - 'offset_y': float
            - 'is_centered': bool
            - 'in_range': bool
            - 'away_from_edge': bool
            - 'sharpness': float
            - 'rejection_reason': str (human readable)
            - 'rejection_code': str ('ok', 'low_confidence', 'too_far', 'off_center', 'near_edge')
    """
    confidence = face_data.get("confidence", 0.0)
    xmin, ymin, w_ratio, h_ratio = face_data["bbox_norm"]
    bbox_px = face_data.get("bbox_px")

    # Center coordinates in normalized [0, 1] space
    cx = xmin + (w_ratio / 2.0)
    cy = ymin + (h_ratio / 2.0)

    offset_x = abs(cx - 0.5)
    offset_y = abs(cy - 0.5)

    # 1. Confidence check
    conf_ok = confidence >= config.FACE_MIN_CONFIDENCE

    # 2. Size / Range check (proxy for physical distance)
    in_range = (
        w_ratio >= config.FACE_MIN_WIDTH_RATIO and
        h_ratio >= config.FACE_MIN_HEIGHT_RATIO
    )

    # 3. Centering check
    is_centered = (
        offset_x <= config.FACE_CENTER_TOLERANCE and
        offset_y <= config.FACE_CENTER_TOLERANCE
    )

    # 4. Edge proximity check
    xmax = xmin + w_ratio
    ymax = ymin + h_ratio
    away_from_edge = (
        xmin >= config.FACE_EDGE_MARGIN and
        xmax <= (1.0 - config.FACE_EDGE_MARGIN) and
        ymin >= config.FACE_EDGE_MARGIN and
        ymax <= (1.0 - config.FACE_EDGE_MARGIN)
    )

    # 5. Sharpness (variance of Laplacian on face region)
    raw_sharpness = compute_sharpness(frame_bgr, bbox_px)
    # Typical sharp webcam faces have variance 150-400+; normalize to [0, 1]
    norm_sharpness = min(raw_sharpness / 250.0, 1.0)

    # 6. Composite Quality Score Q in [0.0, 1.0]
    # S_size: scaled relative to ideal size ratio (e.g. 0.30)
    ideal_size = 0.30
    norm_size = min(((w_ratio + h_ratio) / 2.0) / ideal_size, 1.0)

    # S_center: 1.0 when perfectly centered, declining with distance
    max_dist = math.sqrt(0.5**2 + 0.5**2)
    dist = math.sqrt(offset_x**2 + offset_y**2)
    norm_center = max(0.0, 1.0 - (dist / max_dist))

    # S_conf: direct confidence
    norm_conf = max(0.0, min(confidence, 1.0))

    composite_score = (
        config.WEIGHT_CONFIDENCE * norm_conf +
        config.WEIGHT_SIZE       * norm_size +
        config.WEIGHT_CENTER     * norm_center +
        config.WEIGHT_SHARPNESS  * norm_sharpness
    )
    composite_score = round(float(composite_score), 4)

    # Determine pass/fail & diagnostic message
    rejection_code = "ok"
    rejection_reason = "Face ready — hold still"

    if not in_range:
        rejection_code = "too_far"
        rejection_reason = "Visitor detected — move closer"
    elif not is_centered:
        rejection_code = "off_center"
        rejection_reason = "Face detected — center yourself in camera"
    elif not away_from_edge:
        rejection_code = "near_edge"
        rejection_reason = "Face detected — step away from edge"
    elif not conf_ok:
        rejection_code = "low_confidence"
        rejection_reason = "Face detected — looking for clearer view"

    passed = (
        conf_ok and
        in_range and
        is_centered and
        away_from_edge
    )

    return {
        "passed":           passed,
        "quality_score":    composite_score,
        "confidence":       round(confidence, 4),
        "width_ratio":      round(w_ratio, 4),
        "height_ratio":     round(h_ratio, 4),
        "offset_x":         round(offset_x, 4),
        "offset_y":         round(offset_y, 4),
        "is_centered":      is_centered,
        "in_range":         in_range,
        "away_from_edge":   away_from_edge,
        "sharpness":        round(raw_sharpness, 2),
        "rejection_code":   rejection_code,
        "rejection_reason": rejection_reason,
    }
