import socket
import pickle
import numpy as np
import time

from threading import Thread

class ContactForce():
    '''
    Class to read and record contact force data sent from grasping gauge
    '''
    def __init__(self, IP=None, port=8888):
        self._IP = IP           # IP address where force values are written to from Raspberry Pi
        self._port = port       # Port where force values are written to from Raspberry Pi

        self._socket = None                 # Grants access to data from URL
        self._client_socket = None          # Socket that we read from
        self._stream_thread = None          # Thread to receive measurements from sensor and update value
        self._stream_active = False         # Boolean of whether or not we're currently streaming
        self._latest_measurement = None     # Most recent reading from the sensor
        self._forces = []                   # Store force measurements from gaueg sequentially (in Newtons)

    # Clear all force measurements from the object
    def _reset_values(self):
        self._forces = []

    # Return array of force measurements
    def forces(self):
        return np.array(self._forces)

    # Return array of force measurements
    def clip(self, i_start, i_end):
        self._forces = self._forces[i_start:i_end]
        return

    # Open socket to begin streaming values
    def start_stream(self, IP=None, port=None, read_only=False, verbose=False):
        if IP != None:      self._IP = IP
        if port != None:    self._port = port
        assert self._IP != None and self._port != None

        self._reset_values()
        self._stream_active = True

        # Create a socket object and bind it to the specified address and port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((self._IP, self._port))
        self._socket.listen(1)
        
        self._client_socket, _ = self._socket.accept()

        self._stream_thread = Thread(target=self._stream, kwargs={'read_only': read_only, 'verbose': verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        return
    
    # Function to facilitate continuous reading of values from stream
    # (If read_only is True, will not save measurements to local force array.
    #   Instead, must directly execute _record_latest() to record the most recently recieved measurement.)
    def _stream(self, read_only=False, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming force measurements...')
            self._read_values()
            if not read_only:
                self._record_latest()
        return

    # Read force measurement from socket
    def _read_values(self, verbose=False):
        received_data = self._client_socket.recv(1024)
        if not received_data:
            raise ValueError()
        
        # Interpret data
        float_str = received_data.decode()
        if float_str.count('.') > 1:
            float_str = float_str[float_str.rfind('.', 0, float_str.rfind('.'))+3:]
        self._latest_measurement = float(float_str)*(0.002)*0.01
        if verbose: print(self._latest_measurement)
        return
    
    # Save the latest measurement from stream to local data
    def _record_latest(self, verbose=False):
        self._forces.append(self._latest_measurement)
        if verbose: print(self._latest_measurement)
        return
    
    # Close socket when done measuring
    def end_stream(self, verbose=False):
        self._stream_active = False
        self._IP = None
        self._stream_thread.join()
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
    

if __name__ == "__main__":
    # # Read data from source
    contact_force = ContactForce(IP="10.10.10.50")
    # contact_force.start_stream(verbose=True)
    # time.sleep(3)
    # contact_force.end_stream()

    # print(f'Read {len(contact_force.forces())} values in 3 seconds.')

    contact_force.load('./example.pkl')
    print(contact_force.forces())