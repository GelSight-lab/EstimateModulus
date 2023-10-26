import socket

# Define the IP address and port to listen on
server_ip = "10.10.10.50"  # Listen on all available network interfaces
server_port = 8080   # The same port number you used on the Raspberry Pi

# Create a socket object and bind it to the specified address and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))

# Listen for incoming connections
server_socket.listen(1)  # You can adjust the number of allowed connections

print("Waiting for a connection...")

# Accept a connection from a client
client_socket, client_address = server_socket.accept()

print(f"Accepted connection from {client_address}")

# Receive the data from the client
received_data = client_socket.recv(1024)  # You may need to adjust the buffer size

# Decode the received data
float_str = received_data.decode()
received_float = float(float_str)

print(f"Received float value: {received_float}")

# Close the client socket
client_socket.close()

# Close the server socket
server_socket.close()
