import time
import matplotlib.pyplot as plt
from datetime import datetime

from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from gripper_width import GripperWidth
from data_recorder import DataRecorder

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
        force=40, # Maximum force in Newtons [N]
        speed=0.0175, # Desired operation speed in [m/s]
        grasp=True,     
        block=True,
        skill_desc="CloseGripper"
    )
}

def collect_data_for_object(object_name, object_modulus, num_trials, folder_name=None, plot=True):
    # Define streaming addresses
    wedge_video         =   GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv")
    # other_wedge_video   =   GelsightWedgeVideo(IP="10.10.10.200", config_csv="./config.csv")
    contact_force       =   ContactForce(IP="10.10.10.50", port=8888)
    # data_recorder       =   DataRecorder(wedge_video=wedge_video, other_wedge_video=other_wedge_video, contact_force=contact_force)
    gripper_width       =   GripperWidth(franka_arm=franka_arm)
    data_recorder       =   DataRecorder(wedge_video=wedge_video, contact_force=contact_force, gripper_width=gripper_width)

    if folder_name == None:
        # Choose folder name as YYYY-MM-DD by default
        folder_name = datetime.now().strftime('%Y-%m-%d')

    _open_socket = True
    _close_socket = False

    # Execute number of data collection trials
    for i in range(num_trials):

        # Start with gripper in open position
        open_gripper(franka_arm)

        # Start recording
        if i > 0: _open_socket = False
        data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True, verbose=False, _open_socket=_open_socket)

        ###################################################################
        # TODO: Do not plot depth while streaming to record training data #
        ###################################################################

        # Close gripper
        close_gripper(franka_arm)
        time.sleep(0.5)

        # Open gripper
        open_gripper(franka_arm)
        time.sleep(1)

        # Stop recording
        if i == num_trials - 1: _close_socket = True
        data_recorder.end_stream(_close_socket=_close_socket)

        # Save
        data_recorder.auto_clip()
        # data_recorder.save(f'./data/{folder_name}/{object_name}_t{str(i)}_E{str(object_modulus)}')
        data_recorder.save('./TEST')

        if plot:
            plt.plot(abs(data_recorder.forces()) / abs(data_recorder.forces()).max(), label="Forces")
            plt.plot(data_recorder.widths() / data_recorder.widths().max(), label="Gripper Width")
            plt.plot(data_recorder.max_depths() / data_recorder.max_depths().max(), label="Max Depth")
            plt.legend()
            plt.show()

        # Reset data
        data_recorder._reset_data()

        # User confirmation to continue
        if i != num_trials-1:
            input('Press Enter to continue collecting data...')

    return True


if __name__ == "__main__":

    OBJECT_NAME     = "TEST"
    OBJECT_MODULUS  = "0.0"
    NUM_TRIALS      = 2
    collect_data_for_object(
        OBJECT_NAME, \
        OBJECT_MODULUS, \
        NUM_TRIALS \
    )