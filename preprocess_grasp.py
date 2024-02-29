import os
import random
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from wedge_video import GelsightWedgeVideo, DEPTH_THRESHOLD
from gripper_width import GripperWidth
from contact_force import ContactForce, FORCE_THRESHOLD
from grasp_data import GraspData

HARDDRIVE_DIR = '.'
TRAINING_FORCE_THRESHOLD = 5 # [N]

# Read CSV files with objects and labels tabulated
object_to_modulus = {}
object_to_shape = {}
object_to_material = {}
csv_file_path = f'{HARDDRIVE_DIR}/data/objects_and_labels.csv'
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) # Skip title row
    for row in csv_reader:
        if row[14] != '':
            object_to_modulus[row[1]] = float(row[14])
            object_to_shape[row[1]] = row[2]
            object_to_material[row[1]] = row[3]

'''
Preprocess recorded data for training...
    - Clip to the static loading sequence only
    - Down sample frames to small number
'''
def preprocess_grasp(path_to_file, grasp_data=GraspData(), destination_dir=f'{HARDDRIVE_DIR}/data/training_data', \
                     auto_clip=False, num_frames_to_sample=3, max_num_aug=4, plot_sampled_frames=False):
    
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

    # Skip files with a low peak force
    if grasp_data.forces().max() < 10: return

    # Skip files where gripper width does not change
    if grasp_data.gripper_widths().max() == grasp_data.gripper_widths().min(): return
        






    i_start = np.argmax(grasp_data.forces() >= TRAINING_FORCE_THRESHOLD)
    i_end_peak = 0
    for i in range(len(grasp_data.forces())):
        if i > np.argmax(grasp_data.forces()) and grasp_data.forces()[i] <= 0.9*grasp_data.forces().max():
            i_end_peak = i
            break
    grasp_data.clip(i_start, i_end_peak)

    aug_i = 0
    for i in range(len(grasp_data.gripper_widths())):
        if grasp_data.forces()[i] >= 0.5*grasp_data.forces().max() and \
           grasp_data.max_depths()[i] >= 0.5*grasp_data.max_depths().max() and \
           grasp_data.max_depths(other_finger=True)[i] >= 0.5*grasp_data.max_depths(other_finger=True).max():
            
            # Make necessary directories
            aug_dir = f'{trial_dir}/aug={aug_i}'
            if not os.path.exists(aug_dir):
                os.mkdir(aug_dir)
            output_name_prefix = f'{object_name}__t={str(trial)}_aug={aug_i}'
            output_path_prefix = f'{aug_dir}/{output_name_prefix}'
            
            diff_image         = np.expand_dims(grasp_data.wedge_video.diff_images()[i, :, :], axis=0)
            depth_image        = np.expand_dims(grasp_data.wedge_video.depth_images()[i, :, :], axis=0)
            other_diff_image   = np.expand_dims(grasp_data.other_wedge_video.diff_images()[i, :, :], axis=0)
            other_depth_image  = np.expand_dims(grasp_data.other_wedge_video.depth_images()[i, :, :], axis=0)
            force              = np.expand_dims(grasp_data.forces()[i], axis=0)

            # Save to respective areas
            with open(f'{output_path_prefix}_diff.pkl', 'wb') as file:
                pickle.dump(diff_image, file)
            with open(f'{output_path_prefix}_depth.pkl', 'wb') as file:
                pickle.dump(depth_image, file)
            with open(f'{output_path_prefix}_diff_other.pkl', 'wb') as file:
                pickle.dump(other_diff_image, file)
            with open(f'{output_path_prefix}_depth_other.pkl', 'wb') as file:
                pickle.dump(other_depth_image, file)
            with open(f'{output_path_prefix}_force.pkl', 'wb') as file:
                pickle.dump(force, file)
            
            aug_i += 1

    if aug_i == 0: print(object_name)



    '''
    i_start = np.argmax(grasp_data.forces() >= TRAINING_FORCE_THRESHOLD) # 0.25*grasp_data.forces().max()) # FORCE_THRESHOLD)
    i_peak = np.argmax(grasp_data.forces() >= 0.975*grasp_data.forces().max()) + 1

    if (i_peak - i_start + 1) < num_frames_to_sample:
        print('Skipping!')
        return
    
    grasp_data.clip(i_start, i_peak + max_num_aug)

    force_sample_indices        = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    video_sample_indices        = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    other_video_sample_indices  = np.linspace(0, i_peak - i_start - 1, num_frames_to_sample, endpoint=True, dtype=int)
    
    # If depth does not peak at the force peak, adjust indices
    adjusted_peak = False
    if grasp_data.max_depths()[force_sample_indices[-1]] < 0.85*grasp_data.max_depths().max() \
        and np.argmax(grasp_data.max_depths()) < force_sample_indices[-1]:
        grasp_data.plot_grasp_data()
        if input('\nFix peak? ').lower() == 'y':
            adjusted_peak = True
            video_sample_indices = np.linspace(0, np.argmax(grasp_data.max_depths()), num_frames_to_sample, endpoint=True, dtype=int)

    if grasp_data.max_depths(other_finger=True)[force_sample_indices[-1]] < 0.85*grasp_data.max_depths(other_finger=True).max() \
        and np.argmax(grasp_data.max_depths(other_finger=True)) < force_sample_indices[-1]:
        grasp_data.plot_grasp_data()
        if input('\nFix other peak? ').lower() == 'y':
            adjusted_peak = True
            other_video_sample_indices = np.linspace(0, np.argmax(grasp_data.max_depths(other_finger=True)), num_frames_to_sample, endpoint=True, dtype=int)

    # Consider force and depth when limiting augmentations
    num_aug = 1
    for i in range(1, max_num_aug):
        if grasp_data.forces()[force_sample_indices[-1] + i] >= 0.95*grasp_data.forces().max():
            num_aug += 1
        else:
            break

    assert num_aug > 0

    contact_shift = -1
    for i in range(0, 5):
        if grasp_data.max_depths()[i] >= 0.075*grasp_data.max_depths().max():
            contact_shift = i
            break

    other_contact_shift = -1
    for i in range(0, 5):
        if grasp_data.max_depths(other_finger=True)[i] >= 0.075*grasp_data.max_depths(other_finger=True).max():
            other_contact_shift = i
            break

    if contact_shift == -1 and other_contact_shift == -1:
        contact_shift = 0
        other_contact_shift = 0
    elif contact_shift == -1:
        contact_shift = other_contact_shift
    elif other_contact_shift == -1:
        other_contact_shift = contact_shift

    force_sample_indices[0] += min(contact_shift, other_contact_shift)
    video_sample_indices[0] += contact_shift
    other_video_sample_indices[0] += other_contact_shift
    assert video_sample_indices[0] == contact_shift
    assert other_video_sample_indices[0] == other_contact_shift

    if video_sample_indices[1] > video_sample_indices[0]:
        np.linspace(video_sample_indices[0], video_sample_indices[-1], num_frames_to_sample, endpoint=True, dtype=int)
    if other_video_sample_indices[1] > other_video_sample_indices[0]:
        np.linspace(other_video_sample_indices[0], other_video_sample_indices[-1], num_frames_to_sample, endpoint=True, dtype=int)

    num_aug = min(num_aug, max(video_sample_indices[1] - video_sample_indices[0], \
                               other_video_sample_indices[1] - other_video_sample_indices[0], 2))
    if min(video_sample_indices[-1] - video_sample_indices[0], other_video_sample_indices[-1] - other_video_sample_indices[0]) <= num_frames_to_sample-1:
        num_aug = 1

    if plot_sampled_frames:
        grasp_data.plot_grasp_data()

    for i in range(num_aug):

        # Don't continue if the depth from the first frame is similar to the final
        if i > 0 and object_to_modulus[object_name] >= 5e5 and \
            grasp_data.wedge_video.depth_images()[video_sample_indices[0] + i, :, :].max() >= \
            0.8*grasp_data.wedge_video.depth_images()[video_sample_indices[-1] + i, :, :].max() and \
            grasp_data.other_wedge_video.depth_images()[other_video_sample_indices[0] + i, :, :].max() >= \
            0.8*grasp_data.other_wedge_video.depth_images()[other_video_sample_indices[-1] + i, :, :].max():
            break

        # Get out all the sampled data
        diff_images = grasp_data.wedge_video.diff_images()[video_sample_indices + i, :, :]
        depth_images = grasp_data.wedge_video.depth_images()[video_sample_indices + i, :, :]
        other_diff_images = grasp_data.other_wedge_video.diff_images()[other_video_sample_indices + i, :, :]
        other_depth_images = grasp_data.other_wedge_video.depth_images()[other_video_sample_indices + i, :, :]
        forces = grasp_data.forces()[force_sample_indices + i]
        widths = grasp_data.gripper_widths()[force_sample_indices + i]
            
        # Plot chosen frames to make sure they look good
        if (plot_sampled_frames and (i == 0 or i == max_num_aug-1)): # or adjusted_peak:
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'OTHER {object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.other_wedge_video.diff_images()[other_video_sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={other_video_sample_indices[j] + i})')
            plt.tight_layout()
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'OTHER {object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.other_wedge_video.depth_images()[other_video_sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={other_video_sample_indices[j] + i})')
            plt.tight_layout()
            
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'{object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.wedge_video.depth_images()[video_sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={video_sample_indices[j] + i})')
            plt.tight_layout()
            fig, axs = plt.subplots(1, 3, figsize=(12, 8))
            fig.suptitle(f'{object_name}/t={trial}/aug={i}')
            for j in range(num_frames_to_sample):
                axs[j].imshow(grasp_data.wedge_video.diff_images()[video_sample_indices[j] + i, :, :])
                axs[j].axis('off')  # Turn off axis ticks and labels
                axs[j].set_title(f'Sampled Frame #{j + 1} (index={video_sample_indices[j] + i})')
            plt.tight_layout()

            plt.show()
            print('done')
            
        # Make necessary directories
        aug_dir = f'{trial_dir}/aug={i}'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        output_name_prefix = f'{object_name}__t={str(trial)}_aug={i}'
        output_path_prefix = f'{aug_dir}/{output_name_prefix}'

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
    '''

    return

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./wedge_config/config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./wedge_config/config_200_markers.csv") # Other finger
    grasp_data          = GraspData(wedge_video=wedge_video, other_wedge_video=other_wedge_video, gripper_width=GripperWidth(), contact_force=ContactForce())

    # File structure...
    #   raw_data/{object_name}/{object_name}__t={n}
    #   training_data/{object_name}/t={n}/aug={n}/{object_name}__t={n}_a={n}_diff/depth

    n_frames = 1
    DESTINATION_DIR = f'{HARDDRIVE_DIR}/data/training_data__Nframes={n_frames}'

    # Loop through all data files
    DATA_DIR = f'{HARDDRIVE_DIR}/data/raw_data'
    object_folders = os.listdir(DATA_DIR)
    random.shuffle(object_folders)

    object_name_last = None
    for object_name in tqdm(object_folders):

        if object_name in ['silly_puty']: continue

        for file_name in os.listdir(f'{DATA_DIR}/{object_name}'):
            if os.path.splitext(file_name)[1] != '.avi' or file_name.count("_other") > 0:
                continue           

            # Preprocess the file
            preprocess_grasp(f'{DATA_DIR}/{object_name}/{os.path.splitext(file_name)[0]}', destination_dir=DESTINATION_DIR, \
                             num_frames_to_sample=n_frames, grasp_data=grasp_data, plot_sampled_frames=False) # (object_name != object_name_last))
            object_name_last = object_name