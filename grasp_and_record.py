import time
import matplotlib.pyplot as plt
from datetime import datetime

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

# As a sanity check, can plot collected data before saving
PLOT = False

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

# Start with gripper in open position
open_gripper(franka_arm)

print('Starting stream...')

# Start recording
data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True, verbose=False)

# Close gripper
print("Grasping...")
close_gripper(franka_arm)
print("Grasped.")
time.sleep(0.5)

# Open gripper
open_gripper(franka_arm)
print("Ungrasped.")

OBJECT_NAME = 'TEST'

time.sleep(1)

# Stop recording
data_recorder.end_stream()
print('Ended stream.')

# Clip to grasp
data_recorder.auto_clip()

if PLOT:
    plt.plot(abs(data_recorder.forces()) / abs(data_recorder.forces()).max(), label="Forces")
    plt.plot(data_recorder.widths() / data_recorder.widths().max(), label="Gripper Width")
    plt.plot(data_recorder.max_depths() / data_recorder.max_depths().max(), label="Max Depth")
    plt.legend()
    plt.show()

# Save data
assert len(data_recorder.contact_force.forces()) == len(data_recorder.gripper_width.widths()) == len(data_recorder.wedge_video._raw_rgb_frames)
data_recorder.save(f'./example_data/{datetime.now().strftime("%Y-%m-%d")}/{OBJECT_NAME}')

print('Done.')