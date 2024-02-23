import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from wedge_video import GelsightWedgeVideo, DEPTH_THRESHOLD
from gripper_width import GripperWidth
from contact_force import ContactForce, FORCE_THRESHOLD
from grasp_data import GraspData

HARDDRIVE_DIR = '.'

'''
Preprocess recorded data for training...
    - Clip to the static loading sequence only
    - Down sample frames to small number
'''
def preprocess_grasp(path_to_file, grasp_data=GraspData(), destination_dir=f'{HARDDRIVE_DIR}/data/training_data', \
                     auto_clip=False, num_frames_to_sample=3, max_num_augmentations=5, plot_sampled_frames=True):
    
    _, file_name = os.path.split(path_to_file)
    object_name = file_name.split('__')[0]
    trial = int(file_name.split('__')[1][2:])

    # Make necessary directories
    object_dir      = f'{destination_dir}/{object_name}'
    trial_dir       = f'{object_dir}/t={str(trial)}'
    
    # if not os.path.exists(object_dir):
    #     os.mkdir(object_dir)
    # if not os.path.exists(trial_dir):
    #     os.mkdir(trial_dir)
    # else:
    #     return

    # Load video and forces
    grasp_data._reset_data()
    grasp_data.load(path_to_file)
    if auto_clip:
        # Should already be auto-clipped from recording
        grasp_data.auto_clip()
        
    i_start = np.argmax(grasp_data.forces() >= FORCE_THRESHOLD)
    i_peak = np.argmax(grasp_data.forces()) + 1

    if (i_peak - i_start + 1) < num_frames_to_sample:
        print('Skipping!')
        return
    
    max_num_augmentations = min(max_num_augmentations, round((i_peak - i_start)/ num_frames_to_sample) - 1)
    grasp_data.clip(i_start, i_peak + max_num_augmentations)

    L = i_peak - i_start - 1
    sample_indices = np.linspace(0, L, num_frames_to_sample, endpoint=True, dtype=int)
    sample_indices = np.maximum(sample_indices, 1)

    num_augmentations = max_num_augmentations
    for i in range(max_num_augmentations):
        if grasp_data.forces()[sample_indices[-1] + i] <= 0.9*grasp_data.forces().max():
            num_augmentations = i
            break

    if plot_sampled_frames:
        grasp_data.plot_grasp_data()

    for i in range(num_augmentations):

        # Make necessary directories
        aug_dir = f'{trial_dir}/aug={i}'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        output_name_prefix = f'{object_name}__t={str(trial)}_aug={i}'
        output_path_prefix = f'{aug_dir}/{output_name_prefix}'

        # Get out all the sampled data
        diff_images = grasp_data.wedge_video.diff_images()[sample_indices + i, :, :]
        depth_images = grasp_data.wedge_video.depth_images()[sample_indices + i, :, :]
        other_diff_images = grasp_data.other_wedge_video.diff_images()[sample_indices + i, :, :]
        other_depth_images = grasp_data.other_wedge_video.depth_images()[sample_indices + i, :, :]
        forces = grasp_data.forces()[sample_indices + i]
        widths = grasp_data.gripper_widths()[sample_indices + i]
            
        # Plot chosen frames to make sure they look good
        if plot_sampled_frames:
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'{object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.other_wedge_video.diff_images()[sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={sample_indices[j] + i})')
            plt.tight_layout()
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'{object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.other_wedge_video.depth_images()[sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={sample_indices[j] + i})')
            plt.tight_layout()
            plt.show()
            print('done')

        # # Save to respective areas
        # with open(f'{output_path_prefix}_diff.pkl', 'wb') as file:
        #     pickle.dump(diff_images, file)
        # with open(f'{output_path_prefix}_depth.pkl', 'wb') as file:
        #     pickle.dump(depth_images, file)
        # with open(f'{output_path_prefix}_diff_other.pkl', 'wb') as file:
        #     pickle.dump(other_diff_images, file)
        # with open(f'{output_path_prefix}_depth_other.pkl', 'wb') as file:
        #     pickle.dump(other_depth_images, file)
        # with open(f'{output_path_prefix}_forces.pkl', 'wb') as file:
        #     pickle.dump(forces, file)
        # with open(f'{output_path_prefix}_widths.pkl', 'wb') as file:
        #     pickle.dump(widths, file)

    return

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./wedge_config/config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./wedge_config/config_200_markers.csv") # Other finger
    grasp_data          = GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, gripper_width=GripperWidth(), contact_force=ContactForce())

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/aug={n}/{object_name}__t={n}_a={n}_diff/depth

    n_frames = 3
    DESTINATION_DIR = f'{HARDDRIVE_DIR}/data/training_data__Nframes={n_frames}__new'

    # Loop through all data files
    DATA_DIR = f'{HARDDRIVE_DIR}/data/raw_data'
    for object_name in tqdm(os.listdir(DATA_DIR)):
        for file_name in os.listdir(f'{DATA_DIR}/{object_name}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            # Preprocess the file
            preprocess_grasp(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', destination_dir=DESTINATION_DIR, \
                             num_frames_to_sample=n_frames, grasp_data=grasp_data)