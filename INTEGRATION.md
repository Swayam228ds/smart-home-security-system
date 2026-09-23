# ESP32-CAM Hardware Integration Guide 🔌📷

This guide explains how to migrate from the laptop webcam prototype to physical **ESP32-CAM + PIR Motion Sensor** hardware without altering the core backend pipeline.

---

## 1. System Architecture

```
[ PIR Motion Sensor ]
        │  (Motion trigger / Wake pin)
        ▼
[ ESP32-CAM Module ]
        │  (Capture frame & Wi-Fi transmission)
        │
   ┌────┴───────────────────────────┐
   │                                │
Mode A: HTTP POST Ingestion     Mode B: Continuous MJPEG Stream
   │                                │
   ▼                                ▼
`POST /api/visits`              `CameraPipeline(url="http://esp32/stream")`
(Raw JPEG body)                 (Decoded frames in pipeline loop)
   │                                │
   └────────────────┬───────────────┘
                    ▼
   FastAPI Security Server
   ├── YOLOv8n Human Verification
   ├── MediaPipe Face Quality Gate
   ├── State Machine & Best Frame Selection
   ├── SQLite Database (`security.db`)
   └── Web Dashboard (`http://<SERVER_IP>:8000`)
```

---

## 2. Integration Modes

### Mode A: Event-Driven HTTP POST (Recommended for Battery / Low Power)
In this mode, the ESP32-CAM stays in deep sleep until the PIR motion sensor triggers a GPIO wake-up:
1. PIR senses motion $\rightarrow$ pulls ESP32 GPIO high.
2. ESP32 wakes up, initializes OV2640 camera, connects to Wi-Fi.
3. Captures JPEG frame.
4. Performs an HTTP POST request to:
   ```
   POST http://<BACKEND_IP>:8000/api/visits
   Content-Type: image/jpeg
   ```
5. The backend validates the human and face quality, records the visit in SQLite, and saves the image.
6. ESP32 returns to low-power deep sleep.

#### Arduino C++ Code Snippet for ESP32-CAM:
```cpp
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverUrl = "http://192.168.1.100:8000/api/visits";

void sendFrame() {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) return;

    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverUrl);
        http.addHeader("Content-Type", "image/jpeg");

        int httpResponseCode = http.POST(fb->buf, fb->len);
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.printf("Server response: %d - %s\n", httpResponseCode, response.c_str());
        }
        http.end();
    }
    esp_camera_fb_return(fb);
}
```

---

### Mode B: Direct Stream Ingestion (Full State Machine Mode)
If the ESP32-CAM is running a standard camera web server firmware that streams MJPEG (e.g. `http://<ESP32_IP>:81/stream`):
1. In `config.py`, change:
   ```python
   CAMERA_INDEX = "http://192.168.1.50:81/stream"
   ```
2. OpenCV's `cv2.VideoCapture` natively accepts HTTP/RTSP stream URLs identically to device indices (`0`).
3. The full state machine (YOLO human detection $\rightarrow$ waiting for face $\rightarrow$ MediaPipe quality/stability evaluation $\rightarrow$ best frame selection) operates directly on the incoming network stream.

---

## 3. Hardware Pinout Reference

| Component | ESP32-CAM Pin | Notes |
|---|---|---|
| **PIR Sensor VCC** | 5V / 3.3V | Match PIR operating voltage |
| **PIR Sensor GND** | GND | Common ground |
| **PIR Sensor OUT** | GPIO 13 (or GPIO 33) | Configured as wake-up interrupt |
| **OV2640 Camera** | Built-in FPC ribbon | 2MP sensor supporting UXGA/SVGA |
| **Power Supply** | 5V 2A external | Recommended to prevent Wi-Fi brownout resets |

---

## 4. Software Readiness Verification
The backend is already equipped with:
- `POST /api/visits` endpoint handling raw JPEG frames.
- Automatic image decoding and MediaPipe face verification on upload.
- Seamless compatibility with both event-driven uploads and live video streams.
