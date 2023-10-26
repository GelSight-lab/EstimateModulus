import socket

# Define the IP address and port to listen on
server_ip = "0.0.0.0"  # Listen on all available network interfaces
server_port = 12345   # The same port number you used on the Raspberry Pi

# Create a socket object and bind it to the specified address and port
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_ip, server_port)

# Function to update the float_label with received data
def update_received_value():
    client_socket, client_address = server_socket.accept()
    
    while True:
        received_data = client_socket.recv(1024)  # You may need to adjust the buffer size
        if not received_data:
            break
        
        float_str = received_data.decode()
        received_float = float(float_str)

# Start receiving data
server_socket.listen(1)  # You can adjust the number of allowed connections
print("Waiting for a connection...")
update_received_value()

# Close the server socket
server_socket.close()