import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from wedge_video import GelsightWedgeVideo
from gripper_width import GripperWidth
from contact_force import ContactForce
from grasp_data import GraspData

'''
Preprocess recorded data for training...
    - Clip to the static loading sequence only
    - Down sample frames to small number
'''
def preprocess(path_to_file, grasp_data=GraspData(), destination_dir='./data/training_data', auto_clip=False, num_frames_to_sample=5, max_num_augmentations=6, plot_sampled_frames=False):
    _, file_name = os.path.split(path_to_file)
    object_name = file_name.split('__')[0]
    trial = int(file_name.split('__')[1][2:])

    # Make necessary directories
    object_dir      = f'{destination_dir}/{object_name}'
    trial_dir       = f'{object_dir}/t={str(trial)}'
    if not os.path.exists(object_dir):
        os.mkdir(object_dir)
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)

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

    # Choose the 5 sets of 5 frames
    #   Save diff, depth, forces, and widths
    L = len(grasp_data.forces()) - num_augmentations
    sample_indices = np.linspace(0, L, num_frames_to_sample, endpoint=True, dtype=int)
    for i in range(num_augmentations):

        # Get out all the sampled data
        diff_images = grasp_data.wedge_video.diff_images()[sample_indices + i, :, :]
        depth_images = grasp_data.wedge_video.depth_images()[sample_indices + i, :, :]
        other_diff_images = grasp_data.other_wedge_video.diff_images()[sample_indices + i, :, :]
        other_depth_images = grasp_data.other_wedge_video.depth_images()[sample_indices + i, :, :]
        forces = grasp_data.forces()[sample_indices + i]
        widths = grasp_data.gripper_widths()[sample_indices + i]

        # Plot chosen frames to make sure they look good
        if plot_sampled_frames:
            _, axs = plt.subplots(2, 3, figsize=(12, 8))
            for k in range(num_frames_to_sample):
                axs[i].imshow(diff_images[k])
                axs[i].axis('off')  # Turn off axis ticks and labels
                axs[i].set_title(f'Sampled Frame #{k + 1}')
            plt.tight_layout()
            plt.show()

        # Make necessary directories
        aug_dir = f'{trial_dir}/aug={i}'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)

        # Save to respective areas
        output_name_prefix = f'{object_name}__t={str(trial)}_aug={i}'
        output_path_prefix = f'{aug_dir}/{output_name_prefix}'
        with open(f'{output_path_prefix}_diff.pkl', 'wb') as file:
            pickle.dump(diff_images, file)
        with open(f'{output_path_prefix}_depth.pkl', 'wb') as file:
            pickle.dump(depth_images, file)
        with open(f'{output_path_prefix}_diff_other.pkl', 'wb') as file:
            pickle.dump(other_diff_images, file)
        with open(f'{output_path_prefix}_depth_other.pkl', 'wb') as file:
            pickle.dump(other_depth_images, file)
        with open(f'{output_path_prefix}_forces.pkl', 'wb') as file:
            pickle.dump(forces, file)
        with open(f'{output_path_prefix}_widths.pkl', 'wb') as file:
            pickle.dump(widths, file)

        # # Generate horizontal flip permutation and save
        # output_path_prefix = f'{aug_dir_flipped_horiz}/{output_name_prefix}'
        # with open(f'{output_path_prefix}_diff.pkl', 'wb') as file:
        #     pickle.dump(np.flip(diff_images, axis=1), file)
        # with open(f'{output_path_prefix}_depth.pkl', 'wb') as file:
        #     pickle.dump(np.flip(depth_images, axis=1), file)
        # with open(f'{output_path_prefix}_diff_other.pkl', 'wb') as file:
        #     pickle.dump(np.flip(other_diff_images, axis=1), file)
        # with open(f'{output_path_prefix}_depth_other.pkl', 'wb') as file:
        #     pickle.dump(np.flip(other_depth_images, axis=1), file)
        # with open(f'{output_path_prefix}_forces.pkl', 'wb') as file:
        #     pickle.dump(forces, file)
        # with open(f'{output_path_prefix}_widths.pkl', 'wb') as file:
        #     pickle.dump(widths, file)

    return

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./config_200_markers.csv") # Other finger
    grasp_data          = GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, gripper_width=GripperWidth(), contact_force=ContactForce())

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/aug={n}/{object_name}__t={n}_a={n}_diff/depth

    # Loop through all data files
    DATA_DIR = "./data/raw_data"
    for object_name in tqdm(os.listdir(DATA_DIR)):
        for file_name in os.listdir(f'{DATA_DIR}/{object_name}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            # Preprocess the file
            preprocess(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', grasp_data=grasp_data)