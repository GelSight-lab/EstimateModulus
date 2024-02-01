import os
import matplotlib.pyplot as plt

from wedge_video import GelsightWedgeVideo
from gripper_width import GripperWidth
from contact_force import ContactForce
from grasp_data import GraspData

def preprocess(path_to_file, grasp_data=GraspData(), auto_clip=False, num_frames_to_sample=5, max_num_augmentations=6):
    '''
    Preprocess recorded data for training...
        - Down sample frames
        - Create augmentations of frames
    '''

    # Load video and forces
    grasp_data._reset_data()
    grasp_data.load(path_to_file)
    if auto_clip:
        # Should already be auto-clipped from recording
        grasp_data.auto_clip()

    # Crop video and forces to loading sequence
    grasp_data.clip_to_press(pct_peak_threshold=0.9)
    if grasp_data.gripper_widths()[-1] > grasp_data.gripper_widths().min() + 0.0005:
        for i in range(len(grasp_data.gripper_widths()) - 1, -1, -1):
            if grasp_data.gripper_widths()[i] == grasp_data.gripper_widths().min():
                i_last_min_width = i
                break
        grasp_data.clip(0, i_last_min_width+1)
    assert len(grasp_data.depth_images()) == len(grasp_data.forces()) == len(grasp_data.gripper_widths())

    # Choose how many augmentations to downsample
    assert len(grasp_data.forces()) >= num_frames_to_sample
    num_augmentations = round((len(grasp_data.forces()) - 1)/ num_frames_to_sample) - 1
    num_augmentations = min(max_num_augmentations, num_augmentations)

    print('Total number of frames:', len(grasp_data.forces()))
    print('Total number of augmentations chosen:', num_augmentations)

    # Choose the 5 sets of 5 frames
    #   Save diff, depth, forces, and widths

    # PLOT THE THINGS BEING SAVED TO MAKE SURE THEY MAKE SENSE

    # Generate permutations
        # - Flip X  (torch.flip(tensor, dim=1))

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/f={n}/{object_name}_diff/depth
    #   flipped_horizontal/{object_name}__t={n}__f={n}

    return 0

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./config_200_markers.csv") # Other finger
    grasp_data          = GraspData(wedge_video=wedge_video, gripper_width=GripperWidth(), contact_force=ContactForce())

    # Loop through all data files
    DATA_DIR = "./data"
    for object_name in os.listdir(DATA_DIR):
        for file_name in os.listdir(f'{DATA_DIR}/{object_name}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            # Preprocess the file
            print(f'Processing {os.path.splitext(file_name)[0]}...')
            preprocess(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', grasp_data=grasp_data)