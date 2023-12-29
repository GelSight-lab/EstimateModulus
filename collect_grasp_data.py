import time
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from threading import Thread

from gelsight_wedge.src.gelsight.util.Vis3D import ClassVis3D

from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from gripper_width import GripperWidth
from grasp_data import GraspData

franka_arm = FrankaArm()

def open_gripper(_franka_arm): {
    _franka_arm.goto_gripper(
        0.08, # Maximum width in meters [m]
        speed=0.04, # Desired operation speed in [m/s]
        block=True,
        skill_desc="OpenGripper"
    )
}

def close_gripper(_franka_arm): {
    _franka_arm.goto_gripper(
        0.0, # Minimum width in meters [m]
        force=50, # Maximum force in Newtons [N]
        speed=0.0175, # Desired operation speed in [m/s]
        grasp=True,     
        block=True,
        skill_desc="CloseGripper"
    )
}

def collect_data_for_object(object_name, object_modulus, num_trials, folder_name=None, plot_collected_data=True):
    # Define streaming addresses
    wedge_video         =   GelsightWedgeVideo(IP="172.16.0.100", config_csv="./config_100.csv")
    # other_wedge_video   =   GelsightWedgeVideo(IP="172.16.0.200", config_csv="./config_200.csv")
    contact_force       =   ContactForce(IP="172.16.0.50", port=8888)
    gripper_width       =   GripperWidth(franka_arm=franka_arm)
    # grasp_data       =   GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, contact_force=contact_force, gripper_width=gripper_width)
    grasp_data       =   GraspData(wedge_video=wedge_video, contact_force=contact_force, gripper_width=gripper_width)

    if folder_name == None:
        # Choose folder name as YYYY-MM-DD by default
        folder_name = datetime.now().strftime('%Y-%m-%d')

    # Use these variables to keep force reading socket open over multiple trials
    _open_socket = True
    _close_socket = False

    # Dedicated plotting function for multi-trial plotting
    # Note: this became necessary because cv2.imshow() is not great with multiple threads
    def plot_stream(grasp_data=None, plot_diff=False, plot_depth=False, verbose=False):
        Vis3D = None
        while _plot_stream:
            if type(grasp_data.wedge_video._curr_rgb_image) != type(None):
                if verbose: print('Plotting...')

                if plot_diff or plot_depth:
                    diff_img = grasp_data.wedge_video.calc_diff_image(grasp_data.wedge_video.warp_image(grasp_data.wedge_video._raw_rgb_frames[0]), grasp_data.wedge_video.warp_image(grasp_data.wedge_video._curr_rgb_image))

                # Plot depth in 3D
                if plot_depth:
                    if no_data and type(Vis3D) == type(None): # Re-initialize 3D plotting
                        Vis3D = ClassVis3D(n=grasp_data.wedge_video.warped_size[0], m=grasp_data.wedge_video.warped_size[1])
                    Vis3D.update(grasp_data.wedge_video.img2depth(diff_img) / grasp_data.wedge_video.PX_TO_MM)

                if type(grasp_data.wedge_video._curr_rgb_image) != type(None):

                    # Plot raw RGB image
                    cv2.imshow('raw_RGB', grasp_data.wedge_video._curr_rgb_image)

                    # Plot difference image
                    if plot_diff:
                        cv2.imshow('diff_img', diff_img)

                    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit windows by pressing "q"
                        break
                    if cv2.waitKey(1) == 27: # Exit window by pressing Esc
                        break

                no_data = False

            else:
                if verbose: print('No data. Not plotting.')
                no_data = True; Vis3D = None
                cv2.destroyAllWindows()
            
        cv2.destroyAllWindows()
        return

    # TODO: Do not stream / plot video while recording training data!

    # _plot_stream = True
    # _stream_thread = Thread(target=plot_stream, kwargs={"grasp_data": grasp_data, "plot_diff": True, "plot_depth": True})
    # _stream_thread.daemon = True
    # _stream_thread.start()

    # Execute number of data collection trials
    for i in range(num_trials):

        # Start with gripper in open position
        open_gripper(franka_arm)

        # Start recording
        if i > 0: _open_socket = False
        print('Starting stream...')
        grasp_data.start_stream(plot=False, verbose=False, _open_socket=_open_socket)

        # Close gripper
        close_gripper(franka_arm)
        time.sleep(0.5)

        # Open gripper
        open_gripper(franka_arm)
        time.sleep(2)

        # Stop recording
        if i == num_trials - 1: _close_socket = True
        grasp_data.end_stream(_close_socket=_close_socket)
        print('Done streaming.')

        grasp_data.auto_clip()

        if plot_collected_data:
            plt.plot(abs(grasp_data.forces()) / abs(grasp_data.forces()).max(), label="Forces")
            plt.plot(grasp_data.gripper_widths() / grasp_data.gripper_widths().max(), label="Gripper Widths")
            plt.plot(grasp_data.max_depths() / grasp_data.max_depths().max(), label="Max Depths")
            plt.legend()
            plt.show()

        # Save
        grasp_data.save(f'./example_data/{folder_name}/{object_name}__E={str(object_modulus)}__t={str(i)}')

        print('Max depth in mm:', grasp_data.max_depths().max())
        print('Max force of N:', grasp_data.forces().max())
        print('Length of data:', len(grasp_data.forces()))

        # Reset data
        grasp_data._reset_data()

        # # User confirmation to continue
        # if i != num_trials-1:
        #     input('Press Enter to continue collecting data...')

    _plot_stream = False
    # _stream_thread.join()

    return True


if __name__ == "__main__":
    # Record grasp data for the given object
    OBJECT_NAME     = "rigid_strawberry"
    OBJECT_MODULUS  = 0.0
    NUM_TRIALS      = 3
    collect_data_for_object(
        OBJECT_NAME, \
        OBJECT_MODULUS, \
        NUM_TRIALS \
    )