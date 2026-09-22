#include <driver/i2s.h>

// ---------------------------------------------------------
// Configuration
// ---------------------------------------------------------
// INMP441 I2S Pins
#define I2S_WS 15
#define I2S_SD 13
#define I2S_SCK 2

#define I2S_PORT I2S_NUM_0

// We will capture at 16kHz, 16-bit mono
const int sampleRate = 16000;
const int bufferSize = 1024; // Bytes
int8_t audioBuffer[bufferSize];

void setup() {
  // Use a high baud rate to support 32,000 bytes/sec audio stream
  Serial.begin(921600);
  
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
}

void loop() {
  size_t bytesRead = 0;
  
  // Read audio data from the microphone
  i2s_read(I2S_PORT, &audioBuffer, bufferSize, &bytesRead, portMAX_DELAY);

  if (bytesRead > 0) {
    // Write raw PCM bytes over Serial
    Serial.write((const uint8_t*)audioBuffer, bytesRead);
  }
}
