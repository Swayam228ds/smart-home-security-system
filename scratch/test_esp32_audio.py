"""
Quick Hardware Diagnostic Script for ESP32 + INMP441 + MAX98357A
Tests Serial connection, plays a test tone, and records audio from the microphone.
Run with: .\ven\Scripts\python scratch/test_esp32_audio.py
"""

import math
import struct
import sys
import time
import serial
import serial.tools.list_ports

BAUD = 921600
SAMPLE_RATE = 16000

def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if any(chip in desc for chip in ["ch340", "cp210", "uart", "serial"]):
            return p.device
    if ports:
        return ports[0].device
    return None

def generate_tone(freq=880.0, duration_sec=1.5, amp=24000):
    """Generate 16-bit mono sine wave PCM."""
    samples = int(SAMPLE_RATE * duration_sec)
    data = bytearray()
    for i in range(samples):
        val = int(amp * math.sin(2 * math.pi * freq * (i / SAMPLE_RATE)))
        val = max(-32767, min(32767, val))
        data.extend(struct.pack('<h', val))
    return bytes(data)

def main():
    port = find_port()
    if not port:
        print("[-] Error: No COM port found. Please plug in your ESP32.")
        sys.exit(1)

    print(f"[+] Found ESP32 port: {port}")
    print(f"[+] Opening serial at {BAUD} baud...")
    
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(1)
    except Exception as e:
        print(f"[-] Failed to open {port}: {e}")
        print("[-] Make sure the Arduino Serial Monitor or other apps are CLOSED!")
        sys.exit(1)

    print("\n--- TEST 1: PING ESP32 ---")
    ser.reset_input_buffer()
    ser.write(b"PING\n")
    ser.flush()
    resp = ser.readline().decode('ascii', errors='ignore').strip()
    print(f"Response: '{resp}'")
    if "PONG" in resp or "[ESP32" in resp:
        print("[✓] ESP32 Firmware communication OK!")
    else:
        print("[!] No PONG received. Make sure you uploaded the updated esp32_audio.ino sketch!")

    print("\n--- TEST 2: PLAYING TEST TONE ON MAX98357A SPEAKER ---")
    tone_pcm = generate_tone(freq=660.0, duration_sec=2.0, amp=26000)
    total_bytes = len(tone_pcm)
    
    ser.reset_input_buffer()
    header = f"PLAY:{total_bytes}\n".encode("ascii")
    ser.write(header)
    ser.flush()
    
    # Wait for READY
    ready_line = ser.readline().decode('ascii', errors='ignore').strip()
    print(f"ESP32 status: '{ready_line}'")
    
    print("[+] Streaming tone to speaker with flow control...")
    block_size = 512
    blocks = [tone_pcm[i : i + block_size] for i in range(0, total_bytes, block_size)]
    
    window_size = 3
    in_flight = 0
    next_block = 0

    while next_block < len(blocks) and in_flight < window_size:
        ser.write(blocks[next_block])
        next_block += 1
        in_flight += 1
    ser.flush()

    while in_flight > 0:
        ack = ser.read(1)
        if ack == b'K':
            in_flight -= 1
            if next_block < len(blocks):
                ser.write(blocks[next_block])
                ser.flush()
                next_block += 1
                in_flight += 1
        elif not ack:
            break

    done_line = ser.readline().decode('ascii', errors='ignore').strip()
    print(f"[+] Playback finished: '{done_line}'")
    print("[✓] If you heard a loud, clear 2-second beep, MAX98357A speaker is WORKING!")

    print("\n--- TEST 3: RECORDING FROM INMP441 MICROPHONE (3 SECONDS) ---")
    print("[*] Speak into the INMP441 microphone now...")
    ser.reset_input_buffer()
    ser.write(b"START_REC\n")
    ser.flush()
    
    recorded = bytearray()
    start_time = time.time()
    while time.time() - start_time < 3.0:
        if ser.in_waiting:
            recorded.extend(ser.read(ser.in_waiting))
        time.sleep(0.01)
        
    ser.write(b"STOP_REC\n")
    ser.flush()
    time.sleep(0.1)
    
    print(f"[+] Recorded {len(recorded)} bytes of audio data.")
    if len(recorded) > 10000:
        import numpy as np
        audio = np.frombuffer(recorded, dtype=np.int16)
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        print(f"[✓] INMP441 Signal Peak: {peak}, RMS Energy: {rms:.1f}")
        if peak > 200:
            print("[✓] INMP441 Microphone is actively capturing voice/sound!")
        else:
            print("[!] Signal is very quiet. Check that INMP441 L/R pin is tied to GND!")
    else:
        print("[-] Insufficient audio received. Check INMP441 wiring (SCK=GPIO 2, WS=GPIO 15, SD=GPIO 13, L/R=GND, VDD=3.3V).")

    ser.close()
    print("\n[+] Diagnostic test complete!")

if __name__ == "__main__":
    main()
