import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from wedge_video import GelsightWedgeVideo, DEPTH_THRESHOLD
from gripper_width import GripperWidth
from contact_force import ContactForce, FORCE_THRESHOLD
from grasp_data import GraspData

HARDDRIVE_DIR = './'

'''
Preprocess recorded data for training...
    - Clip to the static loading sequence only
    - Down sample frames to small number
'''
def preprocess(path_to_file, grasp_data=GraspData(), destination_dir=f'{HARDDRIVE_DIR}/data/training_data__N=3', auto_clip=False, num_frames_to_sample=3, max_num_augmentations=5):
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
    else:
        return

    # Load video and forces
    grasp_data._reset_data()
    grasp_data.load(path_to_file)
    if auto_clip:
        # Should already be auto-clipped from recording
        grasp_data.auto_clip()
        
    i_start = np.argmax(grasp_data.forces() >= FORCE_THRESHOLD)
    i_peak = np.argmax(grasp_data.forces()) + 1

    assert (i_peak - i_start + 1) >= num_frames_to_sample
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
        
    # Check that peaks and force follow each other
    shift = 0
    other_shift = 0
    for i in range(num_augmentations):
        indices = sample_indices + i
        depth_images = grasp_data.wedge_video.depth_images()[indices, :, :]
        other_diff_images = grasp_data.other_wedge_video.diff_images()[indices, :, :]
        other_depth_images = grasp_data.other_wedge_video.depth_images()[indices, :, :]
        forces = grasp_data.forces()[indices]
        
        asked = False
        for k in range(len(indices)):
            if (depth_images[k-1,:,:].max() > 1.2*depth_images[k,:,:].max() and 1.2*forces[k-1] < forces[k]) or \
                (other_depth_images[k-1,:,:].max() > 1.2*other_depth_images[k,:,:].max() and 1.2*forces[k-1] < forces[k]):

                grasp_data.plot_grasp_data()
                _, axs = plt.subplots(1, 3, figsize=(12, 8))
                for j in range(num_frames_to_sample):
                    axs[j].imshow(grasp_data.other_wedge_video.diff_images()[indices[j], :, :])
                    axs[j].axis('off')  # Turn off axis ticks and labels
                    axs[j].set_title(f'Sampled Frame #{j + 1}')
                plt.tight_layout()
                plt.show()

                asked = True
                if input('Adjust? ').upper() == 'Y':
                    shift = int(input('Shift? '))
                    other_shift = int(input('Other shift? '))
                break
        if asked:
            break
    
    other_frame_sample_indices = np.maximum(sample_indices - other_shift, 1)
    frame_sample_indices = np.maximum(sample_indices - shift, 1)

    for i in range(num_augmentations):

        # Make necessary directories
        aug_dir = f'{trial_dir}/aug={i}'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        output_name_prefix = f'{object_name}__t={str(trial)}_aug={i}'
        output_path_prefix = f'{aug_dir}/{output_name_prefix}'

        # Get out all the sampled data
        diff_images = grasp_data.wedge_video.diff_images()[frame_sample_indices + i, :, :]
        depth_images = grasp_data.wedge_video.depth_images()[frame_sample_indices + i, :, :]
        other_diff_images = grasp_data.other_wedge_video.diff_images()[other_frame_sample_indices + i, :, :]
        other_depth_images = grasp_data.other_wedge_video.depth_images()[other_frame_sample_indices + i, :, :]
        forces = grasp_data.forces()[sample_indices + i]
        widths = grasp_data.gripper_widths()[sample_indices + i]

        # Save to respective areas
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

    return

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./wedge_config/config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./wedge_config/config_200_markers.csv") # Other finger
    grasp_data          = GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, gripper_width=GripperWidth(), contact_force=ContactForce())

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/aug={n}/{object_name}__t={n}_a={n}_diff/depth

    # Loop through all data files
    DATA_DIR = f'{HARDDRIVE_DIR}/data/raw_data'
    for object_name in tqdm(os.listdir(DATA_DIR)):
        for file_name in os.listdir(f'{DATA_DIR}/{object_name}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue

            # Preprocess the file
            preprocess(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', grasp_data=grasp_data)