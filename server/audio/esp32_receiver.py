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
        if not self.serial_conn:
            return
            
        last_log = time.time()
        bytes_received = 0
            
        while True:
            try:
                if self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    bytes_received += len(data)
                    if self.is_recording:
                        self.buffer.extend(data)
                else:
                    time.sleep(0.005)
                    
                # Log stats every 5 seconds for debugging
                if time.time() - last_log > 5:
                    if bytes_received == 0:
                        logger.warning("ESP32 Serial: No data received in the last 5 seconds!")
                    else:
                        logger.info(f"ESP32 Serial: Received {bytes_received} bytes in the last 5 seconds")
                    bytes_received = 0
                    last_log = time.time()
                    
            except Exception as e:
                logger.error(f"Serial receive error: {e}")
                time.sleep(1)

    async def start_listening(self):
        if self.serial_conn:
            # Run the blocking serial read loop in a background thread
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
        else:
            logger.warning("ESP32 Serial Receiver not started (no COM port found)")

    def start_recording(self):
        if not self.serial_conn:
            logger.error("Cannot start recording: No ESP32 connected via Serial.")
            return
            
        self.buffer = bytearray()
        # Flush existing input buffer so we only get fresh audio
        self.serial_conn.reset_input_buffer()
        self.is_recording = True
        logger.info("Started recording from ESP32 Serial stream")

    def stop_recording(self) -> bytes:
        self.is_recording = False
        captured_audio = bytes(self.buffer)
        self.buffer = bytearray()
        logger.info(f"Stopped recording from ESP32. Captured {len(captured_audio)} bytes.")
        return captured_audio

esp32_receiver = ESP32SerialReceiver()
