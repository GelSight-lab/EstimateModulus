import os

from wedge_video import GelsightWedgeVideo
from contact_force import ContactForce
from grasp_data import GraspData

def preprocess(path_to_file, grasp_data=GraspData()):
    '''
    Preprocess recorded data for training
    '''

    # Load video and forces

    # Crop video and forces

    # Choose the 5 frames

    # Precompute modulus estimate

    # Generate permutations
        # - Flip X  (torch.flip(tensor, dim=1))
        # - Flip Y  (torch.flip(tensor, dim=2))
        # - Mask random locations

    # Save ground truth
    # name = objectname_Ehat_groundtruth
    #   data/as_recorded
    #   data/flipped_x
    #   data/flipped_y
    #   data/random_masks

    return

if __name__ == "__main__":

    wedge_video         = GelsightWedgeVideo(config_csv="./config_100.csv") # Force-sensing finger
    other_wedge_video   = GelsightWedgeVideo(config_csv="./config_200.csv") # Other finger
    grasp_data       = GraspData(wedge_video=wedge_video, contact_force=ContactForce())

    DATA_DIR = "./data"
    for filename in os.listdir(DATA_DIR):
        preprocess(DATA_DIR + filename, grasp_data=grasp_data)