import time
import threading

from frankapy import FrankaArm

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from gripper_width import GripperWidth
from data_recorder import DataRecorder

franka_arm = FrankaArm()

# Define streaming addresses
wedge_video     = GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv")
contact_force   = ContactForce(IP="10.10.10.50", port=8888)
gripper_width   = GripperWidth(franka_arm=franka_arm)
data_recorder   = DataRecorder(wedge_video=wedge_video, contact_force=contact_force, gripper_width=gripper_width)

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

# Start with gripper in open position
open_gripper(franka_arm)

print('Starting stream...')

# Start recording
data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True, verbose=False)

# Close gripper
print("Grasping...")
close_gripper(franka_arm)
print("Grasped.")
time.sleep(0.25)

# Open gripper
open_gripper(franka_arm)
print("Ungrasped.")

OBJECT_NAME = 'golf_ball_3'

time.sleep(1.5)

# Stop recording and save
data_recorder.end_stream()
print('Ended stream.')
data_recorder.auto_clip()
data_recorder.save(f'./example_data/{OBJECT_NAME}')

print('Done.')