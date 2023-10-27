import os
import time
import numpy as np

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce

from threading import Thread

class GraspRecorder():
    '''
    Class to streamline recording of data from Gelsight Wedge's / force gauge and package into training
    '''
    def __init__(self, wedge_video=GelsightWedgeVideo(), contact_force=ContactForce()):
        self.wedge_video = wedge_video          # Object containing all video data
        self.contact_force = contact_force      # Object containing all force measurments

        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._stream_active = False         # Boolean of whether or not we're currently streaming
    
    # Clear all data from the object
    def _reset_data(self):
        self.wedge_video._reset_frames()
        self.contact_force._reset_values()
    
    # Initiate streaming thread
    def start_stream(self, plot=False, plot_diff=False, plot_depth=False):
        pass
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self, verbose=True):
        while self._stream_active:
            if verbose: print('Streaming...')
            img_found = self.wedge_video._decode_image_from_stream()
            if img_found:
                self.contact_force._read_values()
        return

    # Terminate streaming thread
    def end_stream(self, verbose=True):
        self.wedge_video.end_stream(verbose=False)
        self.contact_force.end_stream(verbose=False)
        if verbose: print('Done streaming.')
        return
    
    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False):
        self.wedge_video.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Crop data to first press via thresholding
    def auto_crop(self, depth_threshold=0.5, diff_offset=15):
        i_start, i_end = self.wedge_video.auto_crop(depth_threshold=depth_threshold, diff_offset=diff_offset, return_indices=True)
        self.contact_force.crop(i_start, i_end)
        return

    # Read frames from a video file
    def upload(self, folder, file_name):
        self.wedge_video.upload(os.join(folder, file_name, '.avi'))
        self.contact_force.load(os.join(folder, file_name, '.pkl'))
        return

    # Save collected data to video and pickle files
    def save(self, folder, file_name):
        self.wedge_video.download(os.join(folder, file_name, '.avi'))
        self.contact_force.save(os.join(folder, file_name, '.pkl'))
        return
    

if __name__ == "__main__":
    # Typical data collection workflow might be...

    # Define streaming addresses
    wedge_video     =   GelsightWedgeVideo(IP="10.10.10.200", config_csv="./config.csv")
    contact_force   =   ContactForce(IP="10.10.10.50", port=8888)
    data_recorder   =   GraspRecorder(wedge_video=wedge_video, contact_force=contact_force)

    # Record example data and save
    data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True)
    time.sleep(10)
    data_recorder.end_stream()
    data_recorder.save('./', 'example')