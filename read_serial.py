import serial

# Define the serial port and baud rate (must match Raspberry Pi)
serial_port = '/dev/ttyUSB0'  # Update this with your actual serial port
baud_rate = 9600

try:
    # Initialize the serial connection
    ser = serial.Serial(serial_port, baud_rate)

    while True:
        # Read data from the serial port and decode it back to a float
        data = ser.readline().decode().strip()
        float_value = float(data)
        print(f"Received float value: {float_value}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    ser.close()
