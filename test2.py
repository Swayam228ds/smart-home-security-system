import serial
import time

ser = serial.Serial("COM10", 921600, timeout=2)  # Change COM3
time.sleep(2)

print("Connected")

for i in range(5):
    print(repr(ser.readline()))

ser.close()