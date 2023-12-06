import time
import paramiko
from datetime import datetime

from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
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
        speed=0.02, # Desired operation speed in [m/s]
        grasp=True,     
        block=True,
        skill_desc="CloseGripper"
    )
}

def collect_data_for_object(object_name, object_modulus, num_trials, folder_name=None):
    # Define streaming addresses
    wedge_video         =   GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv")
    other_wedge_video   =   GelsightWedgeVideo(IP="10.10.10.200", config_csv="./config.csv")
    contact_force       =   ContactForce(IP="10.10.10.50", port=8888)
    data_recorder       =   DataRecorder(wedge_video=wedge_video, other_wedge_video=other_wedge_video, contact_force=contact_force)

    if folder_name == None:
        # Choose folder name as YYYY-MM-DD by default
        folder_name = datetime.now().strftime('%Y-%m-%d')

    # Define the SSH connection parameters
    hostname = '10.10.10.100'
    port = 22
    username = 'pi'
    password = ' ' # DO NOT MODIFY

    # Open SSH client to start force sending
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password)

    # Execute number of data collection trials
    for _ in range(NUM_TRIALS):

        # Start with gripper in open position
        open_gripper(franka_arm)

        # Start recording
        data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True, verbose=False)

        # Close gripper
        close_gripper(franka_arm)
        time.sleep(0.5)

        # Open gripper
        open_gripper(franka_arm)

        time.sleep(1)

        # Stop recording
        data_recorder.end_stream()

        # Save
        data_recorder.auto_clip()
        data_recorder.save(f'./data/{folder_name}/{object_name}')

        # Reset data
        data_recorder._reset_data()

        # User confirmation to continue
        input('Press Enter to continue collecting data...')

    # End SSH session
    client.close()

    return True


if __name__ == "__main__":

    OBJECT_NAME     = ""
    OBJECT_MODULUS  = ""
    NUM_TRIALS      = 10
    collect_data_for_object(
        OBJECT_MODULUS, \
        OBJECT_MODULUS, \
        NUM_TRIALS \
    )