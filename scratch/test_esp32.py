from server.audio.esp32_receiver import esp32_receiver
import time

print("Starting recording...")
esp32_receiver.start_recording()
time.sleep(3)
data = esp32_receiver.stop_recording()
print(f"Captured {len(data)} bytes")
