import os
import cv2
import numpy as np

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce

from threading import Thread

class GraspRecorder():
    '''
    Class to streamline recording of data from Gelsight Wedge's / force gauge and package into training
    '''
    def __init__(self, wedge_video=GelsightWedgeVideo(), contact_force=ContactForce()):
        self.wedge_video = wedge_video
        self.contact_force = contact_force

        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._plotting = False              # Boolean of whether or not we're plotting during stream
        self._stream_active = False         # Boolean of whether or not we're currently streaming
    
    # Clear all data from the object
    def _reset_data(self):
        self.wedge_video._reset_frames()
        self.contact_force._reset_values()
    
    # Initiate streaming thread
    def start_stream(self, IP, plot=False, plot_diff=False, plot_depth=False):
        pass
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self):
        pass

    # Plot relevant images during streaming
    def _plot(self, plot_diff=False, plot_depth=False):
        pass

    # Terminate streaming thread
    def end_stream(self):
        pass
    
    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False):
        self.wedge_video.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Crop data to first press via thresholding
    def auto_crop(self, depth_threshold=0.5, diff_offset=15):
        pass

    # Read frames from a video file
    def upload(self, folder, file_name):
        self.wedge_video.upload(os.join(folder, file_name, '.avi'))
        self.contact_force.load(os.join(folder, file_name, '.pkl'))

    # Save collected data to video and pickle files
    def save(self, folder, file_name):
        self.wedge_video.download(os.join(folder, file_name, '.avi'))
        self.contact_force.save(os.join(folder, file_name, '.pkl'))
    

if __name__ == "__main__":
    pass