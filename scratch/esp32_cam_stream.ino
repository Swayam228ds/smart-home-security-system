/*
 * ===================================================================================
 *  ESP32-CAM Low-Latency MJPEG Streamer for the Smart Home Security Vision Server
 * ===================================================================================
 *  Target Hardware: AI-Thinker ESP32-CAM (OV2640)
 *
 *  Endpoints
 *    Port 81  /stream   MJPEG stream consumed by the vision server (one client at a time)
 *    Port 80  /         Landing page with preview
 *             /capture  Single JPEG (works while /stream is busy, used as a fallback)
 *             /status   JSON: ip, rssi, heap, frame size, fps
 *             /flash    Toggle the flash LED (GPIO 4)
 *             /control?var=<name>&val=<int>   framesize | quality | vflip | hmirror |
 *                                             brightness | gainceiling | flash
 *
 *  Why two ports: the ESP-IDF HTTP server handles one request per server task. With a
 *  single port, a running /stream blocked /capture and /status. The stream now has its
 *  own server (same layout as Espressif's CameraWebServer, so http://<ip>:81/stream).
 *
 *  Face-recognition friendly settings:
 *    • CAMERA_GRAB_LATEST: the driver always returns the newest frame, so the server
 *      never processes stale video
 *    • VGA 640x480 @ quality 10: faces at 1-2 m are ~90-160 px wide, enough for SFace
 *    • Auto exposure (AEC + AEC2/DSP), gain ceiling 16x, lens correction for dim porches
 *    • MOUNT_VFLIP / MOUNT_HMIRROR fix an upside-down module in the sensor itself.
 *      A sideways (90 deg) mount can't be fixed in the sensor; the server's
 *      CAMERA_ROTATION = "auto" (default) detects and corrects it.
 *
 *  Arduino IDE Setup:
 *    Board: "AI Thinker ESP32-CAM"  |  PSRAM: Enabled
 *    Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
 *    Power: 5V / 2A supply. A weak supply causes brownout resets and corrupted frames.
 *
 *  Flashing: GPIO 0 -> GND, press RST, upload, remove GPIO 0 jumper, press RST.
 * ===================================================================================
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ===================================================================================
//  1. Settings — CHANGE THESE
// ===================================================================================
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

#define STREAM_FRAMESIZE   FRAMESIZE_VGA   // VGA (640x480) recommended; SVGA if faces are far away
#define STREAM_QUALITY     10              // 0-63, lower = better quality / bigger frames
#define MOUNT_VFLIP        0               // 1 if the image is upside down...
#define MOUNT_HMIRROR      0               // ...set both to 1 for a 180 deg mount

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

#define FLASH_LED_PIN      4
bool flashState = false;

httpd_handle_t control_httpd = NULL;
httpd_handle_t stream_httpd  = NULL;

volatile float streamFps = 0.0f;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ===================================================================================
//  3. HTTP Handlers
// ===================================================================================

// Port 81: /stream
static esp_err_t stream_handler(httpd_req_t *req) {
    char part_buf[64];
    esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "X-Framerate", "15");

    int64_t fpsWindowStart = esp_timer_get_time();
    int frames = 0;
    int failures = 0;

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            // Transient capture failures happen under weak power; don't kill the stream
            if (++failures > 20) { res = ESP_FAIL; break; }
            delay(20);
            continue;
        }
        failures = 0;

        size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;  // client disconnected

        frames++;
        int64_t now = esp_timer_get_time();
        if (now - fpsWindowStart >= 2000000) {
            streamFps = frames * 1000000.0f / (now - fpsWindowStart);
            frames = 0;
            fpsWindowStart = now;
        }
    }
    streamFps = 0.0f;
    return res;
}

// Port 80: /capture
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
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

// Port 80: /flash
static esp_err_t flash_handler(httpd_req_t *req) {
    flashState = !flashState;
    digitalWrite(FLASH_LED_PIN, flashState ? HIGH : LOW);
    char resp[32];
    snprintf(resp, sizeof(resp), "{\"flash\":%s}", flashState ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

// Port 80: /control?var=...&val=...
static esp_err_t control_handler(httpd_req_t *req) {
    char query[96], var[32], val[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
        httpd_query_key_value(query, "val", val, sizeof(val)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "use /control?var=<name>&val=<int>");
        return ESP_FAIL;
    }
    int v = atoi(val);
    sensor_t *s = esp_camera_sensor_get();
    int res = 0;
    if (!strcmp(var, "framesize"))        res = s->set_framesize(s, (framesize_t)v);
    else if (!strcmp(var, "quality"))     res = s->set_quality(s, v);
    else if (!strcmp(var, "vflip"))       res = s->set_vflip(s, v);
    else if (!strcmp(var, "hmirror"))     res = s->set_hmirror(s, v);
    else if (!strcmp(var, "brightness"))  res = s->set_brightness(s, v);
    else if (!strcmp(var, "gainceiling")) res = s->set_gainceiling(s, (gainceiling_t)v);
    else if (!strcmp(var, "flash"))       { flashState = v; digitalWrite(FLASH_LED_PIN, v ? HIGH : LOW); }
    else res = -1;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    if (res != 0) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "unknown variable or invalid value");
        return ESP_FAIL;
    }
    return httpd_resp_send(req, "OK", 2);
}

// Port 80: /status
static esp_err_t status_handler(httpd_req_t *req) {
    sensor_t *s = esp_camera_sensor_get();
    char json[256];
    snprintf(json, sizeof(json),
        "{\"ip\":\"%s\",\"rssi\":%d,\"free_heap\":%u,\"psram\":%s,\"flash\":%s,"
        "\"framesize\":%d,\"quality\":%d,\"vflip\":%d,\"hmirror\":%d,\"stream_fps\":%.1f}",
        WiFi.localIP().toString().c_str(), WiFi.RSSI(), ESP.getFreeHeap(),
        psramFound() ? "true" : "false", flashState ? "true" : "false",
        s->status.framesize, s->status.quality, s->status.vflip, s->status.hmirror, streamFps);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json, strlen(json));
}

// Port 80: /
static esp_err_t index_handler(httpd_req_t *req) {
    static const char index_html[] = R"rawliteral(
<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-CAM</title>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<style>body{font-family:sans-serif;background:#0d1117;color:#e6edf3;text-align:center;margin:0;padding:16px}
img{max-width:100%;border:1px solid #30363d;border-radius:8px}button{margin:8px;padding:8px 16px}
code{color:#3fb950}</style></head><body>
<h3>ESP32-CAM Security Stream</h3>
<p>Preview uses the single stream slot. Close this page so the vision server can connect.</p>
<img id='s'><br>
<button onclick='fetch("/flash")'>Toggle flash</button>
<button onclick='window.open("/capture")'>Snapshot</button>
<p>Stream URL for the server: <code id='u'></code></p>
<script>const u='http://'+location.hostname+':81/stream';document.getElementById('u').textContent=u;
document.getElementById('s').src=u;</script></body></html>
)rawliteral";
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html, strlen(index_html));
}

// ===================================================================================
//  4. Start HTTP Servers
// ===================================================================================
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 8;
    config.server_port = 80;
    config.ctrl_port = 32768;

    httpd_uri_t index_uri   = { .uri = "/",        .method = HTTP_GET, .handler = index_handler,   .user_ctx = NULL };
    httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
    httpd_uri_t flash_uri   = { .uri = "/flash",   .method = HTTP_GET, .handler = flash_handler,   .user_ctx = NULL };
    httpd_uri_t status_uri  = { .uri = "/status",  .method = HTTP_GET, .handler = status_handler,  .user_ctx = NULL };
    httpd_uri_t control_uri = { .uri = "/control", .method = HTTP_GET, .handler = control_handler, .user_ctx = NULL };
    httpd_uri_t stream_uri  = { .uri = "/stream",  .method = HTTP_GET, .handler = stream_handler,  .user_ctx = NULL };

    if (httpd_start(&control_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(control_httpd, &index_uri);
        httpd_register_uri_handler(control_httpd, &capture_uri);
        httpd_register_uri_handler(control_httpd, &flash_uri);
        httpd_register_uri_handler(control_httpd, &status_uri);
        httpd_register_uri_handler(control_httpd, &control_uri);
    }

    config.server_port = 81;
    config.ctrl_port = 32769;
    if (httpd_start(&stream_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_httpd, &stream_uri);
    }
}

// ===================================================================================
//  5. Setup & Loop
// ===================================================================================
void setup() {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);  // avoid resets on Wi-Fi current spikes

    Serial.begin(115200);
    Serial.println("\n=== ESP32-CAM Smart Home Security Streamer ===");

    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);

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
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.grab_mode    = CAMERA_GRAB_LATEST;   // always hand out the newest frame

    if (psramFound()) {
        config.frame_size   = STREAM_FRAMESIZE;
        config.jpeg_quality = STREAM_QUALITY;
        config.fb_count     = 2;
        config.fb_location  = CAMERA_FB_IN_PSRAM;
    } else {
        Serial.println("Warning: no PSRAM, falling back to QVGA single buffer.");
        config.frame_size   = FRAMESIZE_QVGA;
        config.jpeg_quality = 12;
        config.fb_count     = 1;
        config.fb_location  = CAMERA_FB_IN_DRAM;
        config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed (0x%x). Check the ribbon cable and the 5V/2A supply.\n", err);
        delay(3000);
        ESP.restart();
    }

    sensor_t *s = esp_camera_sensor_get();
    s->set_brightness(s, 1);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_wb_mode(s, 0);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);                          // DSP auto-exposure: better faces against bright backgrounds
    s->set_gain_ctrl(s, 1);
    s->set_gainceiling(s, GAINCEILING_16X);     // brighter in dim light (more noise, still recognisable)
    s->set_lenc(s, 1);                          // lens shading correction (dark corners)
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
    s->set_dcw(s, 1);
    s->set_vflip(s, MOUNT_VFLIP);
    s->set_hmirror(s, MOUNT_HMIRROR);

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);  // modem sleep adds 100+ ms of jitter to the stream
    WiFi.begin(ssid, password);
    Serial.printf("Connecting to Wi-Fi '%s'", ssid);
    for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.print(".");
    }
    Serial.println();
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Wi-Fi failed. Check SSID/password (2.4 GHz only). Restarting...");
        delay(3000);
        ESP.restart();
    }

    startCameraServer();
    String ip = WiFi.localIP().toString();
    Serial.println("================================================");
    Serial.printf(" Stream (set ESP32_CAM_URL to this): http://%s:81/stream\n", ip.c_str());
    Serial.printf(" Snapshot: http://%s/capture   Status: http://%s/status\n", ip.c_str(), ip.c_str());
    Serial.println("================================================");
}

void loop() {
    static uint32_t lastCheck = 0;
    if (millis() - lastCheck > 5000) {
        lastCheck = millis();
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("Wi-Fi lost, reconnecting...");
            WiFi.reconnect();
        }
    }
    delay(100);
}
