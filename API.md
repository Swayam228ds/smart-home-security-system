# REST API Reference 📡

The Smart Home Security backend exposes a REST API along with an MJPEG video feed.

## Base URL
```
http://127.0.0.1:8000
```

---

## Endpoints

### 1. Dashboard UI
`GET /`
- **Description**: Returns the web dashboard HTML (`index.html`).
- **Response**: `200 OK` (text/html)

---

### 2. Live MJPEG Stream
`GET /video_feed`
- **Description**: Multipart JPEG stream with YOLO human bounding boxes, MediaPipe face bounding boxes, and HUD state diagnostics overlaid.
- **Response**: `200 OK` (`multipart/x-mixed-replace; boundary=frame`)

---

### 3. Pipeline Status
`GET /api/status`
- **Description**: Returns real-time system metrics, camera running status, detector stats, face quality metrics, and state machine progress.
- **Response**: `200 OK` (`application/json`)
- **Example Response**:
```json
{
  "camera": "running",
  "human_detected": true,
  "human_count": 1,
  "humans_detected": 1,
  "face_detected": true,
  "faces_detected": 1,
  "face_confidence": 0.912,
  "face_width_ratio": 0.241,
  "face_height_ratio": 0.315,
  "face_centered": true,
  "face_quality": 0.824,
  "capture_state": "waiting_for_stability",
  "current_visit_id": 14,
  "stability_progress": "3/5",
  "status": "Face ready — hold still (3/5)",
  "rejection_reason": "Face ready — hold still",
  "primary_face_bbox": [210, 140, 154, 151]
}
```

---

### 4. List Visits
`GET /api/visits`
- **Description**: Returns all visitor records ordered newest first.
- **Response**: `200 OK` (`application/json`)
- **Example Response**:
```json
[
  {
    "visit_id": 14,
    "timestamp": "2026-09-22 21:40:15",
    "image_path": "images/visit_0014.jpg",
    "human_detected": 1,
    "human_count": 1,
    "processing_status": "pending",
    "face_detected": 1,
    "recognized_person": null,
    "face_confidence": 0.912,
    "face_bbox": "[210, 140, 154, 151]",
    "face_quality": 0.824,
    "capture_status": "captured"
  }
]
```

---

### 5. Get Single Visit
`GET /api/visits/{id}`
- **Description**: Retrieves record for a specific visit ID.
- **Response**: `200 OK` or `404 Not Found`

---

### 6. Get Visit Image
`GET /api/visits/{id}/image`
- **Description**: Serves the captured full-resolution visitor JPEG.
- **Response**: `200 OK` (`image/jpeg`) or `404 Not Found`

---

### 7. Get Cropped Face Image
`GET /api/visits/{id}/face_image`
- **Description**: Serves the cropped face JPEG extracted for future face recognition.
- **Response**: `200 OK` (`image/jpeg`) or `404 Not Found`

---

### 8. Upload Visit Image (ESP32-CAM Endpoint)
`POST /api/visits`
- **Description**: Ingests a raw JPEG image from external hardware (e.g. ESP32-CAM), evaluates face presence and quality, stores image and DB record.
- **Headers**: `Content-Type: image/jpeg`
- **Body**: Raw binary JPEG bytes.
- **Response**: `200 OK` (`application/json`)
```json
{
  "visit_id": 15,
  "image_path": "images/visit_0015.jpg",
  "capture_status": "captured",
  "face_detected": true,
  "face_confidence": 0.88,
  "face_quality": 0.79
}
```

---

### 9. Update Visit Status
`PATCH /api/visits/{id}`
- **Description**: Updates review status of a visit.
- **Request Body**:
```json
{
  "processing_status": "reviewed"
}
```
- **Response**: `200 OK` with updated visit record.

---

### 10. Clear All Visits
`DELETE /api/visits`
- **Description**: Deletes all records from `security.db` and permanently removes all stored visitor images and face crops from `images/`.
- **Response**: `200 OK`
```json
{
  "message": "Cleared 14 visit(s) and their images."
}
```

---

### 11. Camera Controls
`POST /api/camera/start`
- **Description**: Starts background webcam capture thread.
- **Response**: `{"message": "Camera started"}`

`POST /api/camera/stop`
- **Description**: Stops background webcam capture thread.
- **Response**: `{"message": "Camera stopped"}`
