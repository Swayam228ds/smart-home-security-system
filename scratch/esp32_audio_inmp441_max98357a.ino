#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

// ---------------- PINS ----------------
// MAX98357A Speaker
#define I2S_SPK_BCLK  23
#define I2S_SPK_LRC   22
#define I2S_SPK_DOUT  5
#define I2S_SPK_PORT  I2S_NUM_0

// INMP441 Microphone
#define I2S_MIC_SCK   32
#define I2S_MIC_WS    27
#define I2S_MIC_SD    34
#define I2S_MIC_PORT  I2S_NUM_1

#define SAMPLE_RATE   16000
#define BUFFER_SIZE   256
#define PLAY_BLOCK    256   // mono samples (512 bytes) per block, matches Python

bool isRecording = false;

// ---------------- SPEAKER ----------------
void setupSpeaker() {
  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    // STEREO: the MAX98357A outputs (L+R)/2 when SD is floating.
    // Sending only LEFT gave half amplitude (-6 dB). We now send the
    // same sample on both channels so you get full volume.
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .mck_io_num = I2S_PIN_NO_CHANGE,
    .bck_io_num = I2S_SPK_BCLK,
    .ws_io_num = I2S_SPK_LRC,
    .data_out_num = I2S_SPK_DOUT,
    .data_in_num = I2S_PIN_NO_CHANGE
  };

  if (i2s_driver_install(I2S_SPK_PORT, &config, 0, NULL) != ESP_OK) {
    Serial.println("Speaker I2S installation failed");
    return;
  }
  if (i2s_set_pin(I2S_SPK_PORT, &pins) != ESP_OK) {
    Serial.println("Speaker pin configuration failed");
    return;
  }
  i2s_zero_dma_buffer(I2S_SPK_PORT);
  Serial.println("Speaker initialized");
}

// ---------------- MICROPHONE ----------------
void setupMicrophone() {
  i2s_config_t config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pins = {
    .mck_io_num = I2S_PIN_NO_CHANGE,
    .bck_io_num = I2S_MIC_SCK,
    .ws_io_num = I2S_MIC_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_SD
  };

  if (i2s_driver_install(I2S_MIC_PORT, &config, 0, NULL) != ESP_OK) {
    Serial.println("Microphone I2S installation failed");
    return;
  }
  if (i2s_set_pin(I2S_MIC_PORT, &pins) != ESP_OK) {
    Serial.println("Microphone pin configuration failed");
    return;
  }
  Serial.println("Microphone initialized");
}

// ---------------- CHIME ----------------
void playStartupChime() {
  const int totalSamples = 8000;
  int16_t stereo[BUFFER_SIZE * 2];
  size_t bytesWritten;

  for (int i = 0; i < totalSamples; i += BUFFER_SIZE) {
    int count = min(BUFFER_SIZE, totalSamples - i);
    for (int j = 0; j < count; j++) {
      float t = (float)(i + j) / SAMPLE_RATE;
      float env = 1.0f - ((float)(i + j) / totalSamples);
      int16_t s = (int16_t)(sin(2.0f * PI * 880.0f * t) * 12000.0f * env);
      stereo[2 * j] = s;
      stereo[2 * j + 1] = s;
    }
    i2s_write(I2S_SPK_PORT, stereo, count * 2 * sizeof(int16_t),
              &bytesWritten, portMAX_DELAY);
  }
}

// ---------------- MIC -> SERIAL ----------------
void processMicrophone() {
  int32_t micBuf[BUFFER_SIZE];
  int16_t pcmBuf[BUFFER_SIZE];
  size_t bytesRead = 0;

  esp_err_t result = i2s_read(I2S_MIC_PORT, micBuf, sizeof(micBuf),
                              &bytesRead, 10 / portTICK_PERIOD_MS);

  if (result == ESP_OK && bytesRead > 0) {
    int n = bytesRead / sizeof(int32_t);
    for (int i = 0; i < n; i++) pcmBuf[i] = (int16_t)(micBuf[i] >> 16);
    Serial.write((const uint8_t*)pcmBuf, n * sizeof(int16_t));
  }
}

// ---------------- SERIAL -> SPEAKER ----------------
// Flow-controlled: the ESP32 reads one block, plays it (i2s_write blocks
// until the DMA has room, so playback speed sets the pace), then sends
// a single 'K' byte. Python only keeps a few blocks in flight, so the
// serial buffer never overflows and the speaker never starves.
// No gain is applied here - do it in Python, where clipping is controlled.
void playAudioFromSerial(uint32_t totalBytes) {
  isRecording = false;

  if (totalBytes == 0 || totalBytes % 2 != 0) {
    Serial.println("PLAY_ERROR");
    return;
  }

  Serial.println("READY");
  Serial.flush();

  int16_t mono[PLAY_BLOCK];
  int16_t stereo[PLAY_BLOCK * 2];
  size_t bytesWritten;
  uint32_t received = 0;

  while (received < totalBytes) {
    size_t want = min((uint32_t)sizeof(mono), totalBytes - received);
    size_t got = Serial.readBytes((char*)mono, want);

    if (got != want) {
      i2s_zero_dma_buffer(I2S_SPK_PORT);
      Serial.println("PLAY_ERROR_TIMEOUT");
      return;
    }

    int n = got / sizeof(int16_t);
    for (int i = 0; i < n; i++) {
      stereo[2 * i] = mono[i];
      stereo[2 * i + 1] = mono[i];
    }

    i2s_write(I2S_SPK_PORT, stereo, n * 2 * sizeof(int16_t),
              &bytesWritten, portMAX_DELAY);

    received += got;
    Serial.write('K');   // ack: ready for another block
  }

  delay(300);                       // let the last DMA buffers play out
  i2s_zero_dma_buffer(I2S_SPK_PORT);
  Serial.println("PLAY_DONE");
  Serial.flush();
}

// ---------------- SETUP / LOOP ----------------
void setup() {
  Serial.setRxBufferSize(8192);
  Serial.begin(921600);
  Serial.setTimeout(1000);
  delay(500);

  setupSpeaker();
  setupMicrophone();
  delay(200);
  playStartupChime();

  Serial.println();
  Serial.println("[ESP32_AUDIO_READY]");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START_REC") {
      isRecording = true;
      Serial.println("REC_OK");
    } else if (cmd == "STOP_REC") {
      isRecording = false;
      Serial.println("REC_STOPPED");
    } else if (cmd.startsWith("PLAY:")) {
      uint32_t len = cmd.substring(5).toInt();
      if (len > 0) playAudioFromSerial(len);
    } else if (cmd == "PING") {
      Serial.println("PONG");
    } else if (cmd == "BEEP") {
      playStartupChime();
      Serial.println("BEEP_DONE");
    }
  }

  if (isRecording) processMicrophone();
  else delay(2);
}
