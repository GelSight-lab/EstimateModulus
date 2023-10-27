import cv2
import numpy as np
import time
import urllib.request
import csv
import warnings

from gelsight_wedge.src.gelsight.util.processing import warp_perspective
from gelsight_wedge.src.gelsight.util.fast_poisson import poisson_reconstruct
from gelsight_wedge.src.gelsight.util.helper import demark
from gelsight_wedge.src.gelsight.util.Vis3D import ClassVis3D

from threading import Thread

def read_csv(filename="config.csv"):
    rows = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        _ = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))
    return rows

def trim(img):
    img[img < 0] = 0
    img[img > 255] = 255

# Measured from calipers on surface in x and y directions
WARPED_PX_TO_MM = (11, 11)
RAW_PX_TO_MM = (12.5, 11)
PX_TO_MM = np.sqrt(WARPED_PX_TO_MM[0]**2 + WARPED_PX_TO_MM[1]**2)

# Derived from linear fit from max depth measured to known calibration ball diameter
DEPTH_TO_MM = 27.5

class GelsightWedgeVideo():
    '''
    Class to streamline processing of data from Gelsight Wedge's
    '''
    def __init__(self, config_csv="./config.csv", warped_size=(400, 300)):
        self.corners = read_csv(config_csv)     # CSV with pixel coordinates of mirror corners in the order (topleft,topright,bottomleft,bottomright)
        self.warped_size = warped_size          # The size of the image to output from warping process
        self.image_size = (480, 640)            # The size of original image from camera
        self.FPS = 30.0                         # Default FPS from Raspberry Pi camera

        self._IP = ''                       # IP address of Raspberry Pi stream via mjpg_streamer
        self._curr_rgb_image = None         # Current raw RGB image streamed from camera
        self._stream_thread = Thread        # Streaming thread
        self._plot_thread = Thread          # Plotting thread
        self._plotting = False              # Boolean of whether or not we're plotting during stream
        self._stream_active = False         # Boolean of whether or not we're currently streaming

        self._raw_rgb_frames = []           # List of raw RGB frames recorded from camera (in shape: [(640,480,3)])
        self._warped_rgb_frames = []        # Images from camera now cropped / warped to the mirror (in shape: [(640,480,3)])
        self._diff_images = []              # Difference images relative to first frame (after warping) (in shape: [(640,480,3)])
        self._grad_images = []              # Gradient images estimated from RGB (in shape: [(640,480,2)])
        self._depth_images = []             # Depth images calculated (in shape: [(640,480)])
    
    # Clear all video data from the object
    def _reset_frames(self):
        self._raw_rgb_frames = []
        self._forces = []
        self._warped_rgb_frames = []
        self._diff_images = []
        self._grad_images = []
        self._depth_images = []
        return

    # Return stored raw frames from Raspberry Pi camera
    def raw_RGB_frames(self):
        return np.stack(self._raw_rgb_frames, axis=0)
    
    # Return cropped and warped images, compute if necessary
    def warped_RGB_frames(self):
        if len(self._warped_rgb_frames) != len(self._raw_rgb_frames):
            self._warped_rgb_frames = []
            for img in self.raw_RGB_frames():
                self._warped_rgb_frames.append(self.warp_image(img))
        return np.stack(self._warped_rgb_frames, axis=0)

    # Return warped difference images from initial frame to rest of video
    def diff_images(self):
        if len(self._diff_images) != len(self._raw_rgb_frames):
            self._diff_images = []
            ref_img = self.warped_RGB_frames()[0]
            # ref_img = cv2.GaussianBlur(ref_img, (13, 13), 0)
            for img in self.warped_RGB_frames():
                self._diff_images.append(self.calc_diff_image(ref_img, img))
        return np.stack(self._diff_images, axis=0)
    
    # Return gradients across images, compute if necessary
    def grad_images(self):
        if len(self._grad_images) != len(self._raw_rgb_frames):
            self._grad_images = []
            for frame in self.diff_images():
                self._grad_images.append(self.img2grad(frame))
        return np.stack(self._grad_images, axis=0)
    
    # Return depth at each frame, compute if necessary
    def depth_images(self):
        if len(self._depth_images) != len(self._raw_rgb_frames):
            self._depth_images = []
            for frame in self.diff_images():
                self._depth_images.append(self.img2depth(frame))
        return np.stack(self._depth_images, axis=0)
    
    # Crop and warp a raw image to mirror shape based on config corners
    def warp_image(self, img):
        return warp_perspective(img, self.corners, self.warped_size)
    
    # Calculate difference image from reference frame
    def calc_diff_image(self, ref_img, img):
        # return img*1.0 - cv2.GaussianBlur(ref_img, (11, 11), 0)*1.0 + 127.0
        return (img * 1.0 - cv2.GaussianBlur(ref_img, (11, 11), 0) * 1.0) / 255 + 0.5

    # Calculate gradients from a cropped / warped difference image
    def img2grad(self, diff_img):
        dx = (diff_img[:, :, 1] - (diff_img[:, :, 0] + diff_img[:, :, 2]) * 0.5) # / 255.0
        dy = (diff_img[:, :, 0] - diff_img[:, :, 2]) # / 255.0
        dx = dx / (1 - dx ** 2) ** 0.5 / 128
        dy = dy / (1 - dy ** 2) ** 0.5 / 128
        return dx, dy
    
    # Calculate depth based on image gradients
    def grad2depth(self, diff_img, dx, dy):
        dx, dy = demark(diff_img, dx, dy)
        zeros = np.zeros_like(dx)
        unitless_depth = poisson_reconstruct(dy, dx, zeros)
        depth_in_mm = DEPTH_TO_MM * unitless_depth # Derived from linear fit of ball calibration
        return depth_in_mm

    # Calculate depth from a cropped / warped difference image
    def img2depth(self, diff_img):
        dx, dy = self.img2grad(diff_img)
        depth = self.grad2depth(diff_img, dx, dy)
        return depth
    
    # Return the maximum depth across all frames
    def max_depth(self):
        max_depth = 0
        for i in range(self.depth_images().shape[0]):
            depth = self.depth_images()[i,:,:]
            if depth.max() >= max_depth:    max_depth = depth.max()
        return max_depth

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
            time.sleep(1/self.FPS)
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
    # Typical data recording workflow might be...
    wedge_video = GelsightWedgeVideo(config_csv="./config.csv")
    IP_address = 'http://10.10.10.200:8080/?action=stream'
    # wedge_video.start_stream(IP_address, plot=True, plot_diff=True, plot_depth=True)
    # time.sleep(10)
    # wedge_video.end_stream()
    # print(wedge_video.max_depth())

    wedge_video.upload('./example.avi')
    wedge_video.auto_crop()
    wedge_video.download('./example_cropped.avi')

    wedge_video = GelsightWedgeVideo(config_csv="./config.csv")
    wedge_video.upload('./example_cropped.avi')
    wedge_video.watch()

