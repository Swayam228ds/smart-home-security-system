"""
Microphone Recording & Testing Utility for ESP32 + INMP441
Records audio from the INMP441 microphone and saves it as a WAV file in 'mic testing/'

Usage:
    .\\ven\\Scripts\\python test_microphone.py
"""

import datetime
import math
import os
import struct
import sys
import threading
import time
import wave
import serial
import serial.tools.list_ports

BAUD_RATE = 921600
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM (2 bytes per sample)
OUTPUT_DIR = "mic testing"


def find_esp32_port():
    """Auto-detects the ESP32 USB serial COM port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = p.description.lower()
        if any(chip in desc for chip in ["cp210", "ch340", "uart", "serial", "usb"]):
            return p.device
    if ports:
        return ports[0].device
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("   🎙 ESP32 + INMP441 Microphone Recording Test Tool")
    print("=" * 60)

    # 1. Detect Port
    port = find_esp32_port()
    if not port:
        print("[-] Error: No COM port found. Please ensure ESP32 is plugged in via USB.")
        sys.exit(1)

    print(f"[+] Found ESP32 on port: {port}")
    print(f"[+] Connecting at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        ser.setDTR(False)
        ser.setRTS(False)
        time.sleep(1.0)
    except Exception as e:
        print(f"[-] Could not open {port}: {e}")
        print("[-] Please ensure the Arduino IDE Serial Monitor or other programs are closed!")
        sys.exit(1)

    # Ping check
    ser.reset_input_buffer()
    ser.write(b"PING\n")
    ser.flush()
    time.sleep(0.1)
    resp = ser.readline().decode("ascii", errors="ignore").strip()
    if "PONG" in resp or "[ESP32" in resp:
        print("[✓] ESP32 Firmware connection verified!")
    else:
        print("[*] Connected (firmware active).")

    print("\n" + "-" * 60)
    print("  Controls:")
    print("    1. Press [ENTER] to START recording.")
    print("    2. Speak into the INMP441 microphone.")
    print("    3. Press [ENTER] again to STOP recording.")
    print("-" * 60)

    try:
        input("\n>>> Press [ENTER] to start recording... ")
    except KeyboardInterrupt:
        ser.close()
        return

    # Start Recording
    print("\n[🔴 RECORDING IN PROGRESS...] Speak now! (Press [ENTER] to stop)")
    ser.reset_input_buffer()
    ser.write(b"START_REC\n")
    ser.flush()

    pcm_buffer = bytearray()
    stop_event = threading.Event()

    def reader_thread():
        while not stop_event.is_set():
            try:
                waiting = ser.in_waiting
                if waiting > 0:
                    data = ser.read(waiting)
                    pcm_buffer.extend(data)
                else:
                    time.sleep(0.005)
            except Exception:
                break

    t = threading.Thread(target=reader_thread, daemon=True)
    t.start()

    start_time = time.time()

    # Wait for user to press ENTER to stop
    try:
        input()
    except KeyboardInterrupt:
        pass

    stop_event.set()
    duration = time.time() - start_time

    # Tell ESP32 to stop streaming
    try:
        ser.write(b"STOP_REC\n")
        ser.flush()
        time.sleep(0.1)
    except Exception:
        pass

    t.join(timeout=1.0)

    print(f"\n[⏹ STOPPED] Recorded for {duration:.1f} seconds.")

    # Discard non-aligned odd byte if any
    if len(pcm_buffer) % 2 != 0:
        pcm_buffer = pcm_buffer[:-1]

    total_bytes = len(pcm_buffer)
    num_samples = total_bytes // 2
    print(f"[+] Total audio captured: {total_bytes:,} bytes ({num_samples:,} samples)")

    if total_bytes < 1000:
        print("[-] Warning: Very little audio captured. Check that INMP441 is properly wired:")
        print("    SCK -> GPIO 32, WS -> GPIO 27, SD -> GPIO 34, L/R -> GND, VDD -> 3.3V")
        ser.close()
        return

    # Calculate Volume & Energy Statistics
    samples = struct.unpack(f"<{num_samples}h", pcm_buffer)
    peak = max(abs(s) for s in samples)
    sum_squares = sum(s * s for s in samples)
    rms = math.sqrt(sum_squares / num_samples) if num_samples > 0 else 0

    # Save to WAV file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_filename = f"recording_{timestamp}.wav"
    wav_path = os.path.join(OUTPUT_DIR, wav_filename)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_buffer)

    abs_wav_path = os.path.abspath(wav_path)
    print("\n" + "=" * 60)
    print(f" [✓] Audio successfully saved!")
    print(f"     📁 File:     {abs_wav_path}")
    print(f"     ⏱  Duration: {duration:.2f} seconds")
    print(f"     📊 Peak Vol: {peak} / 32767 ({peak / 327.67:.1f}%)")
    print(f"     📈 RMS Vol:  {rms:.1f}")
    print("=" * 60)

    if peak > 1500:
        print("[✓] Excellent! Clear voice signal captured by the INMP441.")
    elif peak > 300:
        print("[✓] Sound detected, but signal is a bit quiet. Speak closer to the microphone hole.")
    else:
        print("[!] Very low signal detected. Verify that INMP441 L/R pin is firmly connected to GND.")

    # Option to playback through ESP32 MAX98357A speaker
    print("\nPlayback options:")
    print("  [1] Play through ESP32 MAX98357A Speaker")
    print("  [2] Open WAV in Windows Media Player")
    print("  [3] Exit")

    choice = input("Select an option (1/2/3, default=1): ").strip()

    if choice == "" or choice == "1":
        print("[+] Streaming audio to MAX98357A speaker with flow control...")
        header = f"PLAY:{total_bytes}\n".encode("ascii")
        ser.reset_input_buffer()
        ser.write(header)
        ser.flush()

        # Wait for READY
        ready_line = ser.readline().decode("ascii", errors="ignore").strip()
        block_size = 512
        blocks = [pcm_buffer[i : i + block_size] for i in range(0, total_bytes, block_size)]
        
        window_size = 3
        in_flight = 0
        next_block = 0

        # Send initial window
        while next_block < len(blocks) and in_flight < window_size:
            ser.write(blocks[next_block])
            next_block += 1
            in_flight += 1
        ser.flush()

        # Feed blocks as 'K' ACKs arrive
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

        done_line = ser.readline().decode("ascii", errors="ignore").strip()
        print("[✓] Playback through speaker finished!")

    elif choice == "2":
        try:
            os.startfile(abs_wav_path)
        except Exception as e:
            print(f"[-] Could not open file: {e}")

    ser.close()
    print("\n[+] Test complete. You can find all recordings in the 'mic testing/' folder.")


if __name__ == "__main__":
    main()
