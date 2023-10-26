import time
import socket

# Define the IP address and port of the receiving device
server_ip = "10.10.10.50"  # Replace with the IP address of the receiving device
server_port = 8888   # Replace with the port number you want to use

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((server_ip, server_port))

while True:
    client_socket.send(str(3.14159).encode())
    time.sleep(1)

client_socket.close()
