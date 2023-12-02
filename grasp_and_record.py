import time
import threading
from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from data_recorder import DataRecorder

franka_arm = FrankaArm()

# Define streaming addresses
wedge_video     =   GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv")
contact_force   =   ContactForce(IP="10.10.10.50", port=8888)
data_recorder   =   DataRecorder(wedge_video=wedge_video, contact_force=contact_force)

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

# Function to log gripper position
time_recorded = []
gripper_widths = []
record = True
def record_measurement():
    while record:
        raw_width = franka_arm.get_gripper_width()
        adj_width = 1.0 * raw_width - 0.0005
        gripper_widths.append(adj_width)
        time_recorded.append(time.time())
record_position_thread = threading.Thread(target=record_measurement)

# Start recording
data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True, verbose=False)

record_position_thread.start()

# Close gripper
print("Grasping...")
close_gripper(franka_arm)
print("Grasped.")
time.sleep(0.5)

# Open gripper
open_gripper(franka_arm)
print("Ungrasped.")

record = False
record_position_thread.join()

import matplotlib.pyplot as plt
plt.plot(time_recorded, gripper_widths, 'b-')
plt.show()

OBJECT_NAME = 'TEST'

# Stop recording and save
data_recorder.end_stream()
print('Ended stream.')
data_recorder.auto_clip()
data_recorder.save(f'./example_data/{OBJECT_NAME}')

print('Done.')