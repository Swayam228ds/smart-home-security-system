/*
 * ===================================================================================
 *  ESP32-CAM High-Performance MJPEG Video Streamer for Smart Home Security
 * ===================================================================================
 *  Target Hardware: AI-Thinker ESP32-CAM (OV2640 Camera Module)
 * 
 *  Features:
 *    • High-framerate MJPEG video streaming (/stream) for OpenCV & Web Dashboard
 *    • Single-frame JPEG capture (/capture)
 *    • Interactive web landing page (/) with live preview & flash LED toggle
 *    • Brownout detector disabled to prevent Wi-Fi power drops
 *    • Flash LED control on GPIO 4 for night illumination
 * 
 *  Arduino IDE Setup:
 *    1. Board: "AI Thinker ESP32-CAM" (under ESP32 Arduino)
 *    2. CPU Frequency: 240MHz (WiFi/BT)
 *    3. Flash Frequency: 80MHz
 *    4. Flash Mode: QIO
 *    5. Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 *    6. PSRAM: "Enabled"
 *    7. Upload Speed: 115200 (or 921600 if using a fast FTDI adapter)
 * 
 *  Flashing Instructions:
 *    • Connect GPIO 0 to GND before plugging in or resetting to enter flash mode.
 *    • Press the RST button on the bottom of the ESP32-CAM.
 *    • Click "Upload" in Arduino IDE.
 *    • When upload completes, DISCONNECT GPIO 0 from GND and press RST once more.
 * ===================================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ===================================================================================
//  1. Wi-Fi Configuration — CHANGE THESE TO YOUR WI-FI CREDENTIALS
// ===================================================================================
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ===================================================================================
//  2. AI-Thinker ESP32-CAM Pin Definitions
// ===================================================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4   // Onboard bright flash LED
bool flashState = false;

// HTTP Server Handle
httpd_handle_t stream_httpd = NULL;

// Multipart boundary for MJPEG
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===================================================================================
//  3. HTTP Request Handlers
// ===================================================================================

// Handler: Live MJPEG Video Stream (/stream)
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Camera capture failed");
            res = ESP_FAIL;
            break;
        }

        _jpg_buf_len = fb->len;
        _jpg_buf = fb->buf;

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }

        esp_camera_fb_return(fb);
        fb = NULL;
        _jpg_buf = NULL;

        if (res != ESP_OK) {
            break;
        }
    }
    return res;
}

// Handler: Single Frame Snapshot (/capture)
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t * fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

// Handler: Toggle Flash LED (/flash)
static esp_err_t flash_handler(httpd_req_t *req) {
    flashState = !flashState;
    digitalWrite(FLASH_LED_PIN, flashState ? HIGH : LOW);
    
    char resp[64];
    snprintf(resp, sizeof(resp), "{\"flash\": %s}", flashState ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

// Handler: Root Landing Page (/)
static esp_err_t index_handler(httpd_req_t *req) {
    static const char PROGMEM index_html[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>ESP32-CAM Smart Home Stream</title>
  <meta name='viewport' content='width=device-width,initial-scale=1'>
  <style>
    body{font-family:-apple-system,sans-serif;background:#0d1117;color:#e6edf3;text-align:center;padding:20px;margin:0}
    h2{color:#58a6ff;margin-bottom:6px}
    .badge{display:inline-block;padding:4px 12px;border-radius:12px;background:#238636;color:#fff;font-size:0.8rem;margin-bottom:14px}
    .frame-box{max-width:640px;margin:0 auto;border:2px solid #30363d;border-radius:8px;overflow:hidden;background:#000}
    img{width:100%;display:block}
    .btn-row{margin-top:16px;display:flex;justify-content:center;gap:12px}
    button{background:#1f6feb;border:none;color:#fff;padding:10px 20px;border-radius:6px;font-size:0.9rem;font-weight:600;cursor:pointer}
    button:hover{opacity:0.9}
    .info{margin-top:16px;font-size:0.85rem;color:#8b949e}
    code{background:#161b22;padding:2px 6px;border-radius:4px;color:#3fb950;font-family:monospace}
  </style>
</head>
<body>
  <h2>📷 ESP32-CAM Security Stream</h2>
  <div class='badge'>STREAM ONLINE</div>
  <div class='frame-box'><img src='/stream' alt='Live Stream'></div>
  <div class='btn-row'>
    <button onclick='fetch("/flash").then(()=>alert("Toggled Flash LED"))'>💡 Toggle Flash LED</button>
    <button onclick='window.open("/capture","_blank")'>📸 Capture Snapshot</button>
  </div>
  <div class='info'>
    <p>Direct Stream URL: <code id='stream-url'>Loading...</code></p>
    <p>VGA 640x480 | AI-Thinker OV2640 | Ready for YOLOv8 & MediaPipe</p>
  </div>
  <script>
    document.getElementById('stream-url').textContent = window.location.origin + '/stream';
  </script>
</body>
</html>
)rawliteral";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html, strlen(index_html));
}

// Handler: Status JSON (/status)
static esp_err_t status_handler(httpd_req_t *req) {
    char json[256];
    snprintf(json, sizeof(json),
        "{\"ip\":\"%s\",\"rssi\":%d,\"free_heap\":%u,\"psram\":%s,\"flash\":%s}",
        WiFi.localIP().toString().c_str(),
        WiFi.RSSI(),
        ESP.getFreeHeap(),
        psramFound() ? "true" : "false",
        flashState ? "true" : "false"
    );
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// ===================================================================================
//  4. Start HTTP Server
// ===================================================================================
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.ctrl_port = 32768;
    config.max_open_sockets = 7;

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t flash_uri = {
        .uri       = "/flash",
        .method    = HTTP_GET,
        .handler   = flash_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    Serial.printf("Starting camera server on port: '%d'\n", config.server_port);
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &index_uri);
        httpd_register_uri_handler(stream_httpd, &stream_uri);
        httpd_register_uri_handler(stream_httpd, &capture_uri);
        httpd_register_uri_handler(stream_httpd, &flash_uri);
        httpd_register_uri_handler(stream_httpd, &status_uri);
        Serial.println("HTTP Server successfully started with endpoints: /, /stream, /capture, /flash, /status");
    }
}

// ===================================================================================
//  5. Setup & Main Loop
// ===================================================================================
void setup() {
    // Disable brownout detector to prevent rebooting on Wi-Fi current spikes
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

    Serial.begin(115200);
    Serial.setDebugOutput(false);
    Serial.println("\n\n================================================");
    Serial.println("  ESP32-CAM Smart Home Security Video Streamer  ");
    Serial.println("================================================");

    // Initialize Flash LED pin
    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);

    // Camera Configuration
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Use VGA 640x480 — optimal for YOLOv8 & MediaPipe without high latency
    config.frame_size   = FRAMESIZE_VGA;
    config.jpeg_quality = 12;  // 10-12 provides sharp image with good compression
    config.fb_count     = 2;   // Double buffering for smooth streaming

    // Check for PSRAM (AI-Thinker modules have 4MB external PSRAM)
    if (psramFound()) {
        Serial.println("PSRAM detected! Using double buffering and high quality.");
        config.fb_count = 2;
    } else {
        Serial.println("Warning: PSRAM not detected. Lowering resolution to SVGA / single buffer.");
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = 14;
        config.fb_count = 1;
    }

    // Camera Initialization
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera initialization failed with error 0x%x\n", err);
        Serial.println("Troubleshooting tips:");
        Serial.println("  1. Verify camera ribbon cable is firmly seated in the FPC connector.");
        Serial.println("  2. Ensure power supply is 5V with at least 2A current capacity.");
        return;
    }
    Serial.println("OV2640 Camera successfully initialized!");

    // Sensor Settings Tweaks for crisp security surveillance
    sensor_t * s = esp_camera_sensor_get();
    if (s != NULL) {
        s->set_brightness(s, 1);     // -2 to 2
        s->set_contrast(s, 1);       // -2 to 2
        s->set_saturation(s, 0);     // -2 to 2
        s->set_special_effect(s, 0); // 0 = No Effect
        s->set_whitebal(s, 1);       // 0 = Disable, 1 = Enable
        s->set_awb_gain(s, 1);       // 0 = Disable, 1 = Enable
        s->set_wb_mode(s, 0);        // 0 = Auto
    }

    // Connect to Wi-Fi
    Serial.printf("Connecting to Wi-Fi: %s", ssid);
    WiFi.begin(ssid, password);
    WiFi.setSleep(false); // Disable Wi-Fi sleep for lowest latency streaming

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n================================================");
        Serial.println(" Wi-Fi Connected Successfully!");
        Serial.printf(" Local IP Address: http://%s\n", WiFi.localIP().toString().c_str());
        Serial.printf(" Live MJPEG Stream: http://%s/stream\n", WiFi.localIP().toString().c_str());
        Serial.printf(" Snapshot Capture:  http://%s/capture\n", WiFi.localIP().toString().c_str());
        Serial.println("================================================\n");
        Serial.println("Paste the Live MJPEG Stream URL into the Smart Home Security Dashboard to test!\n");

        // Start the web server
        startCameraServer();
    } else {
        Serial.println("\nFailed to connect to Wi-Fi! Please check SSID & Password.");
    }
}

void loop() {
    // Keep Wi-Fi alive
    if (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        WiFi.reconnect();
    }
    delay(1000);
}
