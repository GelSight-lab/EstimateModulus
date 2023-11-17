import time
import paramiko
from datetime import datetime

from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from data_recorder import DataRecorder

franka_arm = FrankaArm()

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
    password = ''

    # Open SSH client to start force sending
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password)

    # Execute number of data collection trials
    for i in range(NUM_TRIALS):

        # Start recording

        # Grasp

        # Wait briefly?

        # Stop recording

        # Open grasp

        # Crop

        # Save
        data_recorder.auto_clip()
        data_recorder.save(f'./data/{folder_name}/{object_name}')

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