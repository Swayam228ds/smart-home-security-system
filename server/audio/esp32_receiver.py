import asyncio
import logging
import serial
import serial.tools.list_ports
import threading
import time

logger = logging.getLogger(__name__)

class ESP32SerialReceiver:
    def __init__(self, baudrate=921600):
        self.baudrate = baudrate
        self.port = self._find_esp32_port()
        self.serial_conn = None
        self.is_recording = False
        self.buffer = bytearray()
        self._thread = None
        self._serial_lock = threading.Lock()
        
        if self.port:
            try:
                self.serial_conn = serial.Serial()
                self.serial_conn.port = self.port
                self.serial_conn.baudrate = self.baudrate
                self.serial_conn.timeout = 1
                self.serial_conn.setDTR(False)
                self.serial_conn.setRTS(False)
                self.serial_conn.open()
                logger.info(f"ESP32 Serial Audio Receiver connected on {self.port} at {self.baudrate} baud")
            except Exception as e:
                logger.error(f"Failed to open {self.port}: {e}")
                self.serial_conn = None

    def _find_esp32_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Common USB-to-UART bridge chips used on ESP32 boards
            desc = p.description.lower()
            if "ch340" in desc or "cp210" in desc or "uart" in desc or "serial" in desc:
                return p.device
        
        # Fallback to the first available port if none match specifically
        if ports:
            return ports[0].device
        return None

    def _read_loop(self):
        while True:
            try:
                if self.is_recording and self.serial_conn and self.serial_conn.is_open:
                    with self._serial_lock:
                        waiting = self.serial_conn.in_waiting
                        if waiting > 0:
                            data = self.serial_conn.read(waiting)
                            self.buffer.extend(data)
                time.sleep(0.005)
            except Exception as e:
                time.sleep(0.5)

    async def start_listening(self):
        if self.serial_conn:
            # Run the serial read loop in a background thread
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        else:
            logger.warning("ESP32 Serial Receiver not started (no COM port found)")

    def _ensure_connected(self) -> bool:
        """Attempts to reconnect if the COM port was disconnected or closed."""
        if self.serial_conn and self.serial_conn.is_open:
            return True
        self.port = self._find_esp32_port()
        if not self.port:
            return False
        try:
            self.serial_conn = serial.Serial()
            self.serial_conn.port = self.port
            self.serial_conn.baudrate = self.baudrate
            self.serial_conn.timeout = 1
            self.serial_conn.setDTR(False)
            self.serial_conn.setRTS(False)
            self.serial_conn.open()
            logger.info(f"Reconnected ESP32 Serial Audio Receiver on {self.port}")
            return True
        except Exception as e:
            logger.warning(f"Could not open ESP32 on {self.port}: {e}")
            self.serial_conn = None
            return False

    def start_recording(self):
        if not self._ensure_connected():
            logger.error("Cannot start recording: No ESP32 connected via Serial.")
            return False
            
        with self._serial_lock:
            self.buffer = bytearray()
            try:
                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(b"START_REC\n")
                self.serial_conn.flush()
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Failed to send START_REC: {e}")
            self.is_recording = True
            logger.info("Started recording from ESP32 INMP441 stream")
            return True

    def stop_recording(self) -> bytes:
        with self._serial_lock:
            self.is_recording = False
            try:
                self.serial_conn.write(b"STOP_REC\n")
                self.serial_conn.flush()
                time.sleep(0.08)
                waiting = self.serial_conn.in_waiting
                if waiting > 0:
                    data = self.serial_conn.read(waiting)
                    self.buffer.extend(data)
            except Exception as e:
                logger.error(f"Failed to send STOP_REC: {e}")

            captured_audio = bytes(self.buffer)
            self.buffer = bytearray()

            if captured_audio.endswith(b"REC_STOPPED\r\n"):
                captured_audio = captured_audio[:-13]
            elif captured_audio.endswith(b"REC_STOPPED\n"):
                captured_audio = captured_audio[:-12]

            if len(captured_audio) % 2 != 0:
                captured_audio = captured_audio[:-1]
            logger.info(f"Stopped recording from ESP32. Captured {len(captured_audio)} bytes.")
            return captured_audio

    def play_audio(self, pcm_data: bytes, boost: bool = True) -> bool:
        """
        Transmits audio over Serial to the ESP32 to be played through the MAX98357A speaker.
        Uses flow-controlled block transmission with 'K' ACK to match ESP32 DMA speed.
        """
        if not self._ensure_connected():
            logger.warning("ESP32 Serial not connected. Cannot play audio to MAX98357A.")
            return False

        if not pcm_data or len(pcm_data) < 100:
            return False

        if boost:
            pcm_to_send = boost_audio_loudness(pcm_data)
        else:
            pcm_to_send = pcm_data

        if len(pcm_to_send) % 2 != 0:
            pcm_to_send = pcm_to_send[:-1]

        total_bytes = len(pcm_to_send)
        if total_bytes == 0:
            return False

        with self._serial_lock:
            try:
                self.is_recording = False

                # 1. Clear any leftover input data from prior commands
                self.serial_conn.reset_input_buffer()

                # 2. Send playback header
                header = f"PLAY:{total_bytes}\n".encode("ascii")
                self.serial_conn.write(header)
                self.serial_conn.flush()

                # 3. Wait for ESP32 READY handshake (exact match so [ESP32_AUDIO_READY] doesn't trigger it)
                ready = False
                start_wait = time.time()
                while time.time() - start_wait < 3.0:
                    line = self.serial_conn.readline().decode("ascii", errors="ignore").strip()
                    if line == "READY":
                        ready = True
                        break

                if not ready:
                    logger.error(f"ESP32 did not respond with READY for PLAY:{total_bytes}")
                    return False

                # 4. Stream with flow control ('K' ACK per 512-byte block)
                block_size = 512
                blocks = [pcm_to_send[i : i + block_size] for i in range(0, total_bytes, block_size)]
                
                # Keep up to 3 blocks in flight
                window_size = 3
                in_flight = 0
                next_block = 0
                success = True

                # Send initial window
                while next_block < len(blocks) and in_flight < window_size:
                    self.serial_conn.write(blocks[next_block])
                    next_block += 1
                    in_flight += 1
                self.serial_conn.flush()

                # Feed blocks as 'K' ACKs arrive
                while in_flight > 0:
                    ack = self.serial_conn.read(1)
                    if ack == b'K':
                        in_flight -= 1
                        if next_block < len(blocks):
                            self.serial_conn.write(blocks[next_block])
                            self.serial_conn.flush()
                            next_block += 1
                            in_flight += 1
                    elif not ack:
                        logger.error(f"Timeout waiting for 'K' ack from ESP32. Sent {next_block}/{len(blocks)} blocks.")
                        success = False
                        break
                    else:
                        continue

                if not success:
                    return False

                # 5. Wait for PLAY_DONE
                start_done = time.time()
                while time.time() - start_done < 2.0:
                    line = self.serial_conn.readline().decode("ascii", errors="ignore").strip()
                    if "PLAY_DONE" in line:
                        break

                logger.info(f"🔊 Successfully played {total_bytes} bytes on ESP32 MAX98357A speaker.")
                return True
            except Exception as e:
                logger.error(f"Failed to play audio over ESP32 Serial: {e}")
                return False


def boost_audio_loudness(pcm_bytes: bytes) -> bytes:
    """
    Maximizes loudness for small 8-ohm 0.5W speakers without clipping distortion.
    Normalizes peak amplitude to full 16-bit range (up to 32,000).
    """
    if not pcm_bytes:
        return b""
    try:
        import numpy as np
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(audio) == 0:
            return pcm_bytes

        peak = np.max(np.abs(audio))
        if peak > 0:
            gain = min(32000.0 / peak, 5.0)
            boosted = np.clip(audio * gain, -32767.0, 32767.0)
            return boosted.astype(np.int16).tobytes()
        return pcm_bytes
    except Exception as e:
        logger.error(f"Error boosting audio loudness: {e}")
        return pcm_bytes


esp32_receiver = ESP32SerialReceiver()
