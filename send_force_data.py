import time
import sys
import socket

import RPi.GPIO as GPIO
from hx711py.hx711 import HX711

referenceUnit = 1

def cleanAndExit():
    GPIO.cleanup() 
    print("Cleaned!")
    sys.exit()

hx = HX711(5, 6)
hx.set_reading_format("MSB", "MSB")
hx.set_reference_unit(referenceUnit)
hx.reset()
hx.tare()

# Define the IP address and port of the receiving device
server_ip = "10.10.10.50"  # Replace with the IP address of the receiving device
server_port = 8888   # Replace with the port number you want to use

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((server_ip, server_port))

while True:
    try:
        val = hx.get_weight(5)
        val_str = "{:.2f}".format(val)
        print(val_str)

        client_socket.send(val_str.encode())
        
        # hx.power_down()
        # hx.power_up()
        # time.sleep(0.01)

    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()
        print('ERROR')
        client_socket.close()

cleanAndExit()
client_socket.close()