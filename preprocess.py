import os

from wedge_video import GelsightWedgeVideo
from gripper_width import GripperWidth
from contact_force import ContactForce
from grasp_data import GraspData

def preprocess(path_to_file, grasp_data=GraspData(), auto_clip=False, num_frames_to_sample=5, num_augmentation_sets=5):
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
    grasp_data.clip_to_press()        
    grasp_data.interpolate_gripper_widths(plot_result=True)
    assert len(grasp_data.depth_images()) == len(grasp_data.forces()) == len(grasp_data.gripper_widths())
    raise NotImplementedError() # CHECK THE THE INTERPOLATION WORKS

    print('Total number of frames:', grasp_data.diff_images().shape[0])
    assert grasp_data.diff_images().shape[0] >= num_frames_to_sample*num_augmentation_sets

    # Choose the 5 sets of 5 frames
    #   Save diff, depth, forces, and widths

    # PLOT THE THINGS BEING SAVED TO MAKE SURE THEY MAKE SENSE

    # Generate permutations
        # - Flip X  (torch.flip(tensor, dim=1))

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/f={n}/{object_name}_diff/depth
    #   flipped_horizontal/{object_name}__t={n}__f={n}

    return

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

            # Preporcess the file
            preprocess(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', grasp_data=grasp_data)