import socket
import pickle
import numpy as np

class ContactForce():
    '''
    Class to read and record contact force data sent from grasping gauge
    '''
    def __init__(self):
        self._IP = None
        self._socket = None
        self._forces = []

    # Clear all force measurements from the object
    def _reset_values(self):
        self.forces = []

    # Return array of force measurements
    def forces(self):
        return np.array(self._forces)

    # Open socket to begin streaming values
    def start_stream(self, IP, port=8888):
        self._IP = IP
        # Create a socket object and bind it to the specified address and port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(self._IP, port)
        self._socket.listen(1)
        return

    # Read force measurement from socket
    def _read_values(self):
        client_socket, _ = self._socket.accept()
        received_data = client_socket.recv(1024)
        if not received_data:
            raise ValueError()
        self._forces.append(float(received_data.decode()))
        return
    
    # Close socket when done measuring
    def end_stream(self, verbose=True):
        self._IP = None
        self._socket.close()
        if verbose: print('Done streaming.')
        return

    # Save array of force measurements to path
    def load(self, path_to_file):
        self._reset_values()
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'rb') as file:
            self._forces = pickle.load(file)
        return

    # Save array of force measurements to path
    def save(self, path_to_file):
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'wb') as file:
            pickle.dump(self.forces(), file)
        return