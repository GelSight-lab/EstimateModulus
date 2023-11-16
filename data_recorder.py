import os
import time
import numpy as np

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce

from threading import Thread

class DataRecorder():
    '''
    Class to streamline recording of data from Gelsight Wedge's / force gauge and package into training
    '''
    def __init__(self, wedge_video=GelsightWedgeVideo(), other_wedge_video=None, contact_force=ContactForce()):
        self.wedge_video = wedge_video              # Object containing all video data for wedge on force-sensing finger
        self.other_wedge_video = other_wedge_video  # Object containing all video data for wedge on other finger
        self.contact_force = contact_force          # Object containing all force measurments

        # How many wedge's are we streaming from?
        self._wedge_video_count = 1
        if self.other_wedge_video is not None:
            self._wedge_video_count += 1

        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._stream_active = False         # Boolean of whether or not we're currently streaming
        self._plotting = False              # Boolean of whether or not we're plotting during stream
    
    # Clear all data from the object
    def _reset_data(self):
        self.wedge_video._reset_frames()
        if self._wedge_video_count > 1:
            self.other_wedge_video._reset_frames()
        self.contact_force._reset_values()

    # Return forces
    def forces(self):
        return self.contact_force.forces()
    
    # Return depth images
    def depth_images(self, other_finger=False):
        if other_finger:
            assert self._wedge_video_count > 1
            return self.other_wedge_video.depth_images()
        return self.wedge_video.depth_images()
    
    # Initiate streaming thread
    def start_stream(self, verbose=True, plot=False, plot_other=False, plot_diff=False, plot_depth=False):
        self._reset_data()
        self._stream_active = True

        self.wedge_video._prepare_stream()
        if self._wedge_video_count > 1:
            self.other_wedge_video._prepare_stream()
        self.contact_force.start_stream(read_only=True)

        self._stream_thread = Thread(target=self._stream, kwargs={"verbose": verbose})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        time.sleep(1)

        # Argument plot_other determines which video is plotted
        if plot and not plot_other:
            self.wedge_video._start_plotting(plot_diff=plot_diff, plot_depth=plot_depth)
        elif plot and plot_other:
            assert self._wedge_video_count > 1          
            self.other_wedge_video._start_plotting(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self, verbose=False):
        while self._stream_active:
            if verbose: print('Streaming...')
            img_found = self.wedge_video._decode_image_from_stream()
            if self._wedge_video_count > 1:
                _ = self.other_wedge_video._decode_image_from_stream()
            if img_found:
                self.contact_force._request_value()
        return

    # Terminate streaming thread
    def end_stream(self, verbose=False):
        self._stream_active = False
        self._stream_thread.join()
        if self.wedge_video._plotting:
            self.wedge_video._stop_plotting()
        if self._wedge_video_count > 1 and self.other_wedge_video._plotting:
            self.other_wedge_video._stop_plotting()

        self.wedge_video._wipe_stream_info()
        if self._wedge_video_count > 1:
            self.other_wedge_video._wipe_stream_info()
        self.contact_force.end_stream(verbose=False)
        time.sleep(1)
        if verbose: print('Done streaming.')

        # Adjust by 2 frames for HDMI latency
        self.wedge_video.clip(2, len(self.wedge_video._raw_rgb_frames))
        self.contact_force.clip(0, len(self.contact_force.forces())-2)
        return
    
    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False, other_finger=False):
        if not other_finger:
            self.wedge_video.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        else:
            assert self._wedge_video_count > 1
            self.other_wedge_video.watch(plot_diff=plot_diff, plot_depth=plot_depth)
        return
    
    # Clip data to first press via thresholding
    def auto_clip(self, depth_threshold=0.5, diff_offset=15):
        i_start, i_end = self.wedge_video.auto_clip(depth_threshold=depth_threshold, diff_offset=diff_offset, return_indices=True)
        if self._wedge_video_count > 1:
            self.other_wedge_video.clip(i_start, i_end)
        self.contact_force.clip(i_start, i_end)
        return

    # Read frames from a video file
    def load(self, path_to_file):
        self.wedge_video.upload(path_to_file + '.avi')
        if self._wedge_video_count > 1:
            self.wedge_video.upload(path_to_file + '_other_finger.avi')
        self.contact_force.load(path_to_file + '.pkl')
        return

    # Save collected data to video and pickle files
    def save(self, path_to_file):
        self.wedge_video.download(path_to_file + '.avi')
        if self._wedge_video_count > 1:
            self.other_wedge_video.download(path_to_file + '_other_finger.avi')
        self.contact_force.save(path_to_file + '.pkl')
        return
    

if __name__ == "__main__":
    # Typical data collection workflow might be...

    # Define streaming addresses
    wedge_video         =   GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv") # Force-sensing finger
    # other_wedge_video   =   GelsightWedgeVideo(IP="10.10.10.200", config_csv="./config.csv") # Other finger
    contact_force       =   ContactForce(IP="10.10.10.50", port=8888)
    data_recorder       =   DataRecorder(wedge_video=wedge_video, contact_force=contact_force)

    # # Record example data and save
    # data_recorder.start_stream(verbose=True, plot=True, plot_diff=True, plot_depth=True)
    # time.sleep(10)
    # data_recorder.end_stream(verbose=True)
    # data_recorder.save('./example_data/example')

    data_recorder.load('./example_data/foam_brick_2')
    fb2 = data_recorder.forces()

    data_recorder.load('./example_data/foam_brick_3')
    fb3 = data_recorder.forces()

    data_recorder.load('./example_data/foam_brick_4')
    fb4 = data_recorder.forces()