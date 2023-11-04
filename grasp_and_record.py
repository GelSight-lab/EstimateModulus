import time

from frankapy import FrankaArm
from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from data_recorder import DataRecorder

fa = FrankaArm()

# Define streaming addresses
wedge_video     =   GelsightWedgeVideo(IP="10.10.10.100", config_csv="./config.csv")
contact_force   =   ContactForce(IP="10.10.10.50", port=8888)
data_recorder   =   DataRecorder(wedge_video=wedge_video, contact_force=contact_force)

# Start recording
# DataRecorder()
# data_recorder.start_stream(plot=True, plot_diff=True, plot_depth=True)
# time.sleep(5)

# Close gripper
# fa.close_gripper()
fa.goto_gripper(0.08)
time.sleep(3)
fa.goto_gripper(0.03)
time.sleep(3)
fa.goto_gripper(0.08)

# Stop recording and save
# data_recorder.end_stream()
# data_recorder.save('./test_grasp')
