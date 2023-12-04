import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from threading import Thread

class GripperWidth():
    '''
    Class to read and record contact gripper width over grasping
    '''
    def __init__(self, franka_arm=None):
        self._franka_arm = franka_arm       # Object to interface with the Panda arm
        self._stream_thread = None          # Thread to receive positions and update value
        self._stream_active = False         # Boolean of whether or not we're currently streaming

        self._widths = []                   # Gripper width in meters at times (after smoothing)
        self._widths_recorded = []          # Widths recorded at each respective time
        self._times_recorded = []           # Times when measurement recorded

    # Clear all width measurements from the object
    def _reset_values(self):
        self._widths_recorded = []
        self._times_recorded = []
        self._widths = []

    # Return array of width measurements
    def widths(self):
        return np.array(self._widths)

    # Clip measurements between provided indices
    def clip(self, i_start, i_end):
        i_start = max(0, i_start)
        i_end = min(i_end, len(self._widths))
        self._widths = self._widths[i_start:i_end]
        return

    # Open socket to begin streaming values
    def start_stream(self, verbose=False):
        self._reset_values()
        self._stream_active = True
        self._stream_thread = Thread(target=self._stream, kwargs={'verbose': verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        return
    
    # Function to facilitate continuous reading of values from stream
    def _stream(self, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming force measurements...')
            self._record_value()
        return
    
    # Save the latest measurement from stream to local data
    def _record_value(self):
        self._times_recorded.append(time.time())
        self._widths_recorded.append(self._franka_arm.get_gripper_width() - 0.0005)
        return
    
    # Smooth measurements based on time requested / recorded
    # Necessary because width measurement bandwidth is slower than video
    def _post_process_measurements(self, plot_smoothing=False):
        '''
        self._widths = []
        t_data = []
        for i in range(len(self._times_recorded)):
            t_data.append(self._times_recorded[i] - self._times_recorded[0])

        def sigmoid(x, L ,x0, k, b):
            y = L / (1 + np.exp(-k*(x-x0))) + b
            return (y)

        p0 = [max(self._widths_recorded), np.median(t_data), 1, min(self._widths_recorded)] # this is an mandatory initial guess

        popt, pcov = curve_fit(sigmoid, t_data, self._widths_recorded, p0, method='dogbox')

        # p = np.polyfit(t_data, self._widths_recorded, 4)
        # for t in t_data:
        #     w = 0
        #     for k in range(len(p)):
        #         w += p[k] * t**(4-k)
        #     self._widths.append(w)

        for t in t_data:
            self._widths.append(sigmoid(t, popt[0], popt[1], popt[2], popt[3]))

        if plot_smoothing:
            # Plot to check how the smoothing of data looks
            plt.plot(self._times_recorded, self._widths_recorded, '.')
            plt.plot(self._times_recorded, self._widths, '-')
            plt.show()
        '''

        self._widths = self._widths_recorded

        return
    
    # Close socket when done measuring
    def end_stream(self, verbose=False):
        self._stream_active = False
        self._stream_thread.join()
        self._post_process_measurements()
        if verbose: print('Done streaming.')
        return

    # Save array of width measurements to path
    def load(self, path_to_file):
        self._reset_values()
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'rb') as file:
            self._widths = pickle.load(file)
        return

    # Save array of width measurements to path
    def save(self, path_to_file):
        assert path_to_file[-4:] == '.pkl'
        with open(path_to_file, 'wb') as file:
            pickle.dump(self.widths(), file)
        return