import os
import cv2
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from wedge_video import WARPED_CROPPED_IMG_SIZE

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

DATA_DIR = 'E:/data'
N_FRAMES = 5
WARPED_CROPPED_IMG_SIZE = WARPED_CROPPED_IMG_SIZE[::-1]

# Get the tree of all video files from a directory in place
def list_files(folder_path, file_paths, config):
    # Iterate through the list
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        marker_signal = '_other' if config['use_markers'] else '.avi'
        if os.path.isfile(item_path) and item.count(config['img_style']) > 0 and item.count(marker_signal) > 0:
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            list_files(item_path, file_paths, config)

def conv2D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
	return output_shape

def conv3D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
				  np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
	return output_shape 

class EncoderCNN(nn.Module):
    def __init__(self,
                 img_x=200,
                 img_y=200,
                 input_channels=1,
                 fc_hidden1=1024,
                 fc_hidden2=768,
                 drop_p=0.5,
                 CNN_embed_dim=512):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architectures
        self.ch1, self.ch2, self.ch3, self.ch4 = 8, 16, 32, 64
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2D kernel size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2D strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2D padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y),
                                                 self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2,
                                                 self.k2, self.s2)
        self.conv2_outshape = (int(self.conv2_outshape[0] / 2), int(self.conv2_outshape[1] / 2))
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3,
                                                 self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4,
                                                 self.k4, self.s4)
        self.conv4_outshape = (int(self.conv4_outshape[0] / 2), int(self.conv4_outshape[1] / 2))

        # Fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=self.ch1,
                      kernel_size=self.k1,
                      stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1,
                      out_channels=self.ch2,
                      kernel_size=self.k2,
                      stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2,
                      out_channels=self.ch3,
                      kernel_size=self.k3,
                      stride=self.s3,
                      padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3,
                      out_channels=self.ch4,
                      kernel_size=self.k4,
                      stride=self.s4,
                      padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.drop = nn.Dropout(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(
            self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1],
            self.fc_hidden1
        ) # Fully connected layer, output k classes
        self.fc2 = nn.Linear(
            self.fc_hidden1,
            self.CNN_embed_dim
        ) # Output = CNN embedding latent variables
        self.fc3 = nn.Linear(
            self.CNN_embed_dim,
            self.CNN_embed_dim
        ) # Output = CNN embedding latent variables

    def forward(self, x):
        # CNNs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv
        x = self.drop(x)
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x) # CNN embedding
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x) # CNN embedding
        return x

class DecoderFC(nn.Module):
    def __init__(self,
                CNN_embed_dim=512,
                FC_layer_nodes=[512, 512, 256],
                drop_p=0.5,
                output_dim=6):
        super(DecoderFC, self).__init__()

        self.FC_input_size = N_FRAMES * CNN_embed_dim
        self.FC_layer_nodes = FC_layer_nodes
        self.drop_p = drop_p
        self.output_dim = output_dim

        assert len(FC_layer_nodes) == 3

        self.fc1 = nn.Linear(self.FC_input_size, self.FC_layer_nodes[0])
        self.fc2 = nn.Linear(self.FC_layer_nodes[0], self.FC_layer_nodes[1])
        self.fc3 = nn.Linear(self.FC_layer_nodes[1], self.FC_layer_nodes[2])
        self.fc4 = nn.Linear(self.FC_layer_nodes[2], self.output_dim)
        self.drop = nn.Dropout(self.drop_p)

    def forward(self, x):
        x = torch.cat(x, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = 100 * torch.sigmoid(x)
        return x
 
class ForcesLinear(nn.Module):
    def __init__(self, hidden_size = 128, output_dim = 128):
        super(ForcesLinear, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(1, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
 
class WidthsLinear(nn.Module):
    def __init__(self, hidden_size = 128, output_dim = 128):
        super(WidthsLinear, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(1, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, config, paths_to_files, labels, \
                 video_frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
        ):
        # Data parameters 
        self.data_dir = config['data_dir']
        self.n_frames = config['n_frames']
        self.img_size = config['img_size']
        self.img_style = config['img_style']
        self.n_channels = config['n_channels']
        self.use_markers = config['use_markers']
        self.use_force = config['use_force']
        self.use_width = config['use_width']
        self.use_estimation = config['use_estimation']
        self.use_transformations  = config['use_transformations']
        self.exclude = config['exclude']

        # Define training parameters
        self.epochs         = config['epochs']
        self.batch_size     = config['batch_size']
        self.feature_size   = config['feature_size']
        self.val_pct        = config['val_pct']
        self.learning_rate  = config['learning_rate']
        self.random_state   = config['random_state']

        if self.use_transformations:
            self.input_paths = 2*paths_to_files
            self.modulus_labels = 2*labels
            self.flip_horizontal = [i > len(paths_to_files) for i in range(len(self.input_paths))]
        else:
            self.input_paths = paths_to_files
            self.modulus_labels = labels
            self.flip_horizontal = [False for i in range(len(self.input_paths))]

        # Define attributes to use to conserve memory
        self.cap = None
        self.ret = None
        self.frame = None
        self.x_frames = video_frame_tensor
        self.x_forces = force_tensor
        self.x_width = width_tensor
        self.data = []
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        self.x_frames = self.x_frames.zero_()
        self.x_forces = self.x_forces.zero_()
        self.x_width = self.x_width.zero_()

        # Read and store frames in the tensor
        self.cap = cv2.VideoCapture(self.input_paths[idx])
        for i in range(self.n_frames):
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break
            # Convert frame to tensor format
            self.x_frames[i] = torch.tensor(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        # Flip the data horizontally
        if self.use_transformations and self.flip_horizontal[idx]:
            self.x_frames = torch.flip(self.x_frames, dims=2)

        # Release the video capture object
        self.cap.release()

        # Unpack force measurements
        if self.use_force:
            self.x_forces = None
            raise NotImplementedError()

        # Unpack gripper width measurements
        if self.use_width:
            self.x_widths = None
            raise NotImplementedError()
        
        # Unpack gripper width measurements
        if self.use_width:
            self.x_widths = None
            raise NotImplementedError()

        return self.x_frames, self.x_forces, self.x_widths, self.modulus_labels[idx]
     
class ModulusModel():
    def __init__(self, config, device=None):

        self.model = None

        # Use GPU by default
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = device

        # Data parameters 
        self.data_dir = config['data_dir']
        self.n_frames = config['n_frames']
        self.img_size = config['img_size']
        self.img_style = config['img_style']
        self.n_channels = config['n_channels']
        self.use_markers = config['use_markers']
        self.use_force = config['use_force']
        self.use_width = config['use_width']
        self.use_estimation = config['use_estimation']
        self.use_transformations = config['use_transformations']
        self.exclude = config['exclude']

        # Define training parameters
        self.epochs         = config['epochs']
        self.batch_size     = config['batch_size']
        self.feature_size   = config['feature_size']
        self.val_pct        = config['val_pct']
        self.learning_rate  = config['learning_rate']
        self.random_state   = config['random_state']
        self.loss           = nn.MSELoss()
        self.params         = list(self.model.parameters())
        self.optimizer      = torch.optim.Adam(self.params, lr=self.learning_rate)

        self.use_wandb = True
        if self.use_wandb:
            wandb.init(
                # Set the wandb project where this run will be logged
                project="TrainModulus",
                
                # Track hyperparameters and run metadata
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "N_frames": self.n_frames,
                    "img_size": self.img_size,
                    "feature_size": self.feature_size,
                    "learning_rate": self.learning_rate,
                    "validation_pct": self.val_pct,
                    "random_state": self.random_state,
                    "architecture": "ENCODE_DECODE",
                    "num_params": len(self.params),
                    "optimizer": "Adam",
                    "loss": "MSE",
                    "scheduler": "None",
                }
            )
        
        # Log memory usage
        self.memory_allocated = torch.cuda.memory_allocated()
        self.memory_cached = torch.cuda.memory_reserved()
        if self.use_wandb:
            wandb.log({
                "epoch": 0,
                "memory_allocated": self.memory_allocated,
                "memory_reserved": self.memory_cached,
            })

        # Load data
        self._load_data_paths()

        return
    
    # Create data loaders based on configuration
    def _load_data_paths(self, labels_csv_name='objects_and_labels.csv', csv_modulus_column=14, training_data_folder_name='training_data'):
        # Read CSV files with objects and labels tabulated
        object_to_modulus = {}
        csv_file_path = f'{self.data_dir}/{labels_csv_name}'
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip title row
            for row in csv_reader:
                if row[csv_modulus_column] != '':
                    object_to_modulus[row[1]] = float(row[csv_modulus_column])

        # Extract object names as keys from data
        object_names = object_to_modulus.keys()
        object_names = [x for x in object_names if x not in self.exclude]

        # Extract elastic modulus labels for each object
        elastic_moduli = [object_to_modulus[x] for x in object_names]

        # Split objects into validation or training
        objects_train, objects_val, _, _ = train_test_split(object_names, elastic_moduli, test_size=self.val_pct, random_state=self.random_state)

        # Get all the paths to grasp data within directory
        paths_to_files = []
        list_files(f'{self.data_dir}/{training_data_folder_name}', paths_to_files, config)

        # Divide paths up into training and validation data
        x_train, x_val = [], []
        y_train, y_val = [], []
        for file_path in paths_to_files:
            file_name = os.path.basename(file_path)
            object_name = file_name.split('__')[0]
            if object_name in objects_train:
                x_train.append(file_path)
                y_train.append(object_to_modulus[object_name])
            elif object_name in objects_val:
                x_val.append(file_path)
                y_val.append(object_to_modulus[object_name])

        # Construct datasets
        kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
        self.train_dataset = CustomDataset(x_train, y_train, frame_tensor=torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=device), label_tensor=torch.zeros((1), device=device))
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_dataset = CustomDataset(x_val, y_val, frame_tensor=torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=device), label_tensor=torch.zeros((1), device=device))
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
        return

    def _train_epoch(self):
        self.model.train()
        raise NotImplementedError

    def _val_epoch(self):
        self.model.eval()
        raise NotImplementedError

    def _save_model(self):
        raise NotImplementedError

    def train(self):

        min_val_loss = 1e10

        for epoch in range(self.epochs):
            
            train_loss = self._train_epoch()
            val_loss, val_accuracy = self._val_epoch()

            if val_loss <= min_val_loss:
                self._save_model()

            # TODO: Add horizontal flipping here

            # Log information to W&B
            if self.use_wandb:
                self.memory_allocated = torch.cuda.memory_allocated()
                self.memory_cached = torch.cuda.memory_reserved()
                wandb.log({
                    "epoch": epoch,
                    "memory_allocated": self.memory_allocated,
                    "memory_reserved": self.memory_cached,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                })

        return
    

if __name__ == "__main__":

    # Training and model settings
    config = {
        # Data parameters
        'data_dir': DATA_DIR,
        'n_frames': N_FRAMES,
        'img_size': WARPED_CROPPED_IMG_SIZE,
        'img_style': 'diff',
        'use_markers': True,
        'use_force': True,
        'use_width': True,
        'use_estimation': True,
        'use_transformations': True,
        'exclude': ['playdoh', 'silly_putty'],

        # Training and model parameters
        'epochs'         : 250,
        'batch_size'     : 32,
        'feature_size'   : 512,
        'val_pct'        : 0.2,
        'learning_rate'  : 1e-5,
        'random_state'   : 40,
    }

    assert config['img_style'] in ['diff', 'depth']
    config['n_channels'] = 3 if config['img_style'] == 'diff' else 1

    # Train the model over some data
    train_modulus = ModulusModel(config, device=device)
    train_modulus.train()