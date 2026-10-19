
import serial
import wave
import time

PORT = "COM10"  # Change this
BAUD = 921600

SAMPLE_RATE = 16000
DURATION = 10
OUTPUT_FILE = "microphone_recording.wav"

ser = serial.Serial(PORT, BAUD, timeout=0.2)

print("Waiting for microphone...")

marker = b"MIC_READY\r\n"
buffer = bytearray()

# Find MIC_READY even if binary data arrives before it
while marker not in buffer:
    data = ser.read(256)

    if data:
        buffer.extend(data)

    if len(buffer) > 100000:
        buffer = buffer[-len(marker):]

index = buffer.index(marker)

# Preserve audio bytes received immediately after MIC_READY
audio_data = bytearray(buffer[index + len(marker):])

print("Microphone connected!")
print("Recording for", DURATION, "seconds...")

start = time.monotonic()

while time.monotonic() - start < DURATION:
    data = ser.read(4096)
    if data:
        audio_data.extend(data)

ser.close()

# Remove incomplete 16-bit samples
audio_data = audio_data[:len(audio_data) // 2 * 2]

with wave.open(OUTPUT_FILE, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(audio_data)

print("Saved:", OUTPUT_FILE)
print("Duration:", len(audio_data) / (SAMPLE_RATE * 2), "seconds")
print("Size:", len(audio_data), "bytes")
