import serial
import time
import sys
import hx711py.RPi.GPIO as GPIO
from hx711py.hx711 import HX711

def cleanAndExit():
    print("Cleaning...")

    GPIO.cleanup()
        
    print("Bye!")
    sys.exit()

referenceUnit = 1
hx = HX711(5, 6)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(referenceUnit)
hx.reset()
hx.tare()

# Define the serial port and baud rate
serial_port = '/dev/ttyUSB0'  # Update this with your actual serial port
baud_rate = 9600

try:
    # Initialize the serial connection
    ser = serial.Serial(serial_port, baud_rate)

    while True:
        # Read the float value (replace this with your data source)
        float_value = hx.get_weight(5)
        print(float_value)

        hx.power_down()
        hx.power_up()
        time.sleep(0.1)

        # Convert the float to a string and send it over serial
        ser.write(str(float_value).encode())
        time.sleep(1)  # Delay between sending values

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cleanAndExit()
    ser.close()
