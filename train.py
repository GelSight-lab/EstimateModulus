import os
import cv2
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

N_FRAMES = 5
WARPED_CROPPED_IMG_SIZE = (250, 350)

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
    def __init__(self, paths_to_files, labels):
        self.input_files = paths_to_files
        self.labels = labels
        # TODO: Add horizontal flipping here
        pass
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # TODO: Add horizontal flipping here
        raise NotImplementedError
    
     
class TrainModulus():
    def __init__(self, config, data_loader, use_cuda=True):

        self.model = None


        # TODO: Load in videos

        
        # Use GPU by default
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")

        # Define tactile input parameters
        self.n_frames       = N_FRAMES
        self.img_size       = WARPED_CROPPED_IMG_SIZE # TODO: Port these numbers over

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

        # TODO: Create data loader for training / val

        
        # Log memory usage
        self.memory_allocated = torch.cuda.memory_allocated()
        self.memory_cached = torch.cuda.memory_reserved()
        if self.use_wandb:
            wandb.log({
                "epoch": 0,
                "memory_allocated": self.memory_allocated,
                "memory_reserved": self.memory_cached,
            })

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

    IMAGE_STYLE = 'diff'
    assert IMAGE_STYLE in ['diff', 'depth']
    USE_MARKERS = True
    EXCLUDE = ['playdoh', 'silly_putty']

    # TODO: Add horizontal flipping here

    # Read CSV files with objects and labels tabulated
    object_info = {}
    csv_file_path = './data/objects_and_labels.csv'
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            object_info[row[1]] = row

    # Extract object names as keys from data
    object_names = object_info.keys()
    object_names = [x for x in object_names if x not in EXCLUDE]

    # Extract elastic modulus labels for each object
    elastic_moduli = []
    for object_name in object_names:
        E_object = object_info[object_name][14]
        elastic_moduli.append(E_object)

    # Training and model settings
    config = {        
        'epochs'         : 250,
        'batch_size'     : 32,
        'feature_size'   : 512,
        'val_pct'        : 0.2,
        'learning_rate'  : 1e-5,
        'random_state'   : 40,
    }

    objects_train, objects_val, E_train, E_val = train_test_split(object_names, elastic_moduli, test_size=config['val_pct'], random_state=config['random_state'])

    # TODO: Unpack files and sort paths based on val or not

    paths_to_files = []
    labels = []

    data_loader = CustomDataset(paths_to_files, labels)

    # kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
    # train_dataset = CustomDataset(X_train, y_train, frame_tensor=torch.zeros((N_FRAMES, 3, IMG_X, IMG_Y), device=device), label_tensor=torch.zeros((1), device=device))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # val_dataset = CustomDataset(X_val, y_val, frame_tensor=torch.zeros((N_FRAMES, 3, IMG_X, IMG_Y), device=device), label_tensor=torch.zeros((1), device=device))
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # Train the model over some data
    train_modulus = TrainModulus(config, data_loader)
    train_modulus.train()