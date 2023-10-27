import cv2
import numpy as np
import time
import urllib.request
import warnings

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce

from threading import Thread

# Measured from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)
PX_TO_MM = np.sqrt(WARPED_PX_TO_MM[0]**2 + WARPED_PX_TO_MM[1]**2)

# Derived from linear fit from max depth measured to known calibration ball diameter
DEPTH_TO_MM = 27.5

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
    def _reset_frames(self):
        self.wedge_video._reset_frames()
        self.contact_force._reset_values()
    
    # Initiate streaming thread
    def start_stream(self, IP, plot=False, plot_diff=False, plot_depth=False):
        self._IP = IP
        self._reset_frames()
        self._stream_active = True
        self._plotting = plot
        self._stream_thread = Thread(target=self._stream, kwargs={})
        self._stream_thread.daemon = True
        self._stream_thread.start()
        time.sleep(1)
        if plot:
            self._plot_thread = Thread(target=self._plot, kwargs={'plot_diff': plot_diff, 'plot_depth': plot_depth})
            self._plot_thread.daemon = True
            self._plot_thread.start()
        return
    
    # Facilitate streaming thread, read data from Raspberry Pi camera
    def _stream(self):
        stream = urllib.request.urlopen(self._IP)
        bytes = b''
        while self._stream_active:
            print('Streaming...')
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                self._curr_rgb_image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self._raw_rgb_frames.append(self._curr_rgb_image)
        return

    # Plot relevant images during streaming
    def _plot(self, plot_diff=False, plot_depth=False):
        if plot_depth:  Vis3D = ClassVis3D(n=self.warped_size[0], m=self.warped_size[1])
        while self._stream_active:
            cv2.imshow('raw_RGB', self._curr_rgb_image)

            # Plot difference image
            if plot_diff:
                diff_img = self.calc_diff_image(self.warp_image(self._raw_rgb_frames[0]), self.warp_image(self._curr_rgb_image))
                cv2.imshow('diff_img', diff_img)

            # Plot depth in 3D
            if plot_depth: 
                if not plot_diff:
                    diff_img = self.calc_diff_image(self.warp_image(self._raw_rgb_frames[0]), self.warp_image(self._curr_rgb_image))
                depth = self.img2depth(diff_img) / PX_TO_MM
                Vis3D.update(depth)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                break
            if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                break
        cv2.destroyAllWindows()
        return

    # Terminate streaming thread
    def end_stream(self):
        self._IP = ''
        self._stream_active = False
        self._stream_thread.join()
        if self._plotting:
            self._plot_thread.join()
        time.sleep(1)
        print('Done streaming.')
        return
    
    # Plot video for your viewing pleasure
    def watch(self, plot_diff=False, plot_depth=False):
        if plot_depth or plot_diff:
            diff_images = self._diff_images()
        if plot_depth:
            depth_images = self._depth_images()
            Vis3D = ClassVis3D(self.warped_size[1]//2, self.warped_size[1]//2)
        for i in range(len(self._raw_rgb_frames)):
            cv2.imshow('raw_RGB', self._raw_rgb_frames[i])
            if plot_diff:   cv2.imshow('diff_img', diff_images[i])
            if plot_depth:  Vis3D.update(depth_images[i] / PX_TO_MM)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                break
            if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                break
            time.sleep(1/self.wedge_video.FPS)
        cv2.destroyAllWindows()
        return
    
    # Crop frames to first press via threshold
    def auto_crop(self, depth_threshold=0.5, diff_offset=15):
        i_start, i_end = len(self._raw_rgb_frames), len(self._raw_rgb_frames)-1
        for i in range(len(self._raw_rgb_frames)):
            max_depth_i = self.depth_images()[i].max()
            if max_depth_i > depth_threshold and i <= i_start:
                i_start = i
            if max_depth_i < depth_threshold and i >= i_start and i <= i_end:
                i_end = i
            if max_depth_i < depth_threshold and i >= i_start: break

        if i_start >= i_end:
            warnings.warn("No press detected! Cannot crop.", Warning)            
        else:
            self._raw_rgb_frames = self._raw_rgb_frames[i_start - diff_offset:i_end + diff_offset]

    # Read frames from a video file
    def upload(self, path_to_file):
        self._reset_frames()
        cap = cv2.VideoCapture(path_to_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self._raw_rgb_frames.append(frame)
        cap.release()
        return

    # Write recorded frames to video file
    def download(self, path_to_file):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(path_to_file, fourcc, self.FPS, (self.image_size[1], self.image_size[0]))
        for frame in self._raw_rgb_frames:
            video_writer.write(frame)
        video_writer.release()
        return
    

if __name__ == "__main__":

