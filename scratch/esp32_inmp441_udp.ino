#include <WiFi.h>
#include <WiFiUdp.h>
#include <driver/i2s.h>

// ---------------------------------------------------------
// Configuration
// ---------------------------------------------------------
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Server configuration (the IP of your laptop running the backend)
const char* server_ip = "192.168.1.100"; 
const int server_port = 5005;

// INMP441 I2S Pins
#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 2

#define I2S_PORT I2S_NUM_0

WiFiUDP udp;

// We will capture at 16kHz, 16-bit mono
const int sampleRate = 16000;
const int bufferSize = 1024; // Bytes
int8_t audioBuffer[bufferSize];

void setup() {
  Serial.begin(115200);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected.");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Configure I2S for INMP441
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = sampleRate,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  
  Serial.println("I2S initialized. Streaming to UDP...");
}

void loop() {
  size_t bytesRead = 0;
  
  // Read audio data from the microphone
  i2s_read(I2S_PORT, &audioBuffer, bufferSize, &bytesRead, portMAX_DELAY);

  if (bytesRead > 0) {
    // Stream raw PCM bytes over UDP
    udp.beginPacket(server_ip, server_port);
    udp.write((const uint8_t*)audioBuffer, bytesRead);
    udp.endPacket();
  }
}
