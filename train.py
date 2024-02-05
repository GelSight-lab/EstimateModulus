import os
import pickle
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# from wedge_video import WARPED_CROPPED_IMG_SIZE

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

DATA_DIR = '/media/mike/Elements/data'
N_FRAMES = 5
WARPED_CROPPED_IMG_SIZE = (250, 350) # WARPED_CROPPED_IMG_SIZE[::-1]

# Get the tree of all video files from a directory in place
def list_files(folder_path, file_paths, config):
    # Iterate through the list
    for item in os.listdir(folder_path):
        item_path = f'{folder_path}/{item}'
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

        # 2D convolution output shapes
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
                input_dim=N_FRAMES * 512,
                FC_layer_nodes=[512, 512, 256],
                drop_p=0.5,
                output_dim=6):
        super(DecoderFC, self).__init__()
    
        self.FC_input_size = input_dim
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
 

class ForceFC(nn.Module):
    def __init__(self, hidden_size=16, output_dim=16):
        super(ForceFC, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(1, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
 
class WidthFC(nn.Module):
    def __init__(self, hidden_size=16, output_dim =16):
        super(WidthFC, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(1, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
 

class EstimationFC(nn.Module):
    def __init__(self, hidden_size=16, output_dim=16):
        super(EstimationFC, self).__init__()

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
                 frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
                 estimation_tensor=torch.zeros((N_FRAMES, 1)),
                 label_tensor=torch.zeros((1, 1)),
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
        self.base_name      = ''
        self.x_frames       = frame_tensor
        self.x_forces       = force_tensor
        self.x_widths       = width_tensor
        self.x_estimations  = estimation_tensor
        self.y_label        = label_tensor
    
    def __len__(self):
        return len(self.modulus_labels)
    
    def __getitem__(self, idx):
        self.x_frames       = self.x_frames.zero_()
        self.x_forces       = self.x_forces.zero_()
        self.x_widths       = self.x_widths.zero_()
        self.x_estimations  = self.x_estimations.zero_()
        self.y_label        = self.y_label.zero_()

        # Read and store frames in the tensor
        with open(self.input_paths[idx], 'rb') as file:
            self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)

        # Flip the data horizontally if desired
        if self.use_transformations and self.flip_horizontal[idx]:
            self.x_frames = torch.flip(self.x_frames, dims=(2,))

        # Unpack force measurements
        self.base_name = self.input_paths[idx][:self.input_paths[idx].find(self.img_style)-1] 
        if self.use_force:
            with open(self.base_name + '_forces.pkl', 'rb') as file:
                self.x_forces[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)

        # Unpack gripper width measurements
        if self.use_width:
            with open(self.base_name + '_widths.pkl', 'rb') as file:
                self.x_widths[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
        
        # Unpack modulus estimations
        if self.use_estimation:
            raise NotImplementedError()
        
        # Unpack label
        self.y_label[0] = self.modulus_labels[idx]

        return self.x_frames.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone()
     

class ModulusModel():
    def __init__(self, config, device=None):
        # Use GPU by default
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = device

        # Data parameters 
        self.data_dir               = config['data_dir']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.use_markers            = config['use_markers']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimation         = config['use_estimation']
        self.use_transformations    = config['use_transformations']
        self.exclude                = config['exclude']
        self.use_wandb              = config['use_wandb']

        # Define training parameters
        self.epochs         = config['epochs']
        self.batch_size     = config['batch_size']
        self.feature_size   = config['feature_size']
        self.val_pct        = config['val_pct']
        self.learning_rate  = config['learning_rate']
        self.random_state   = config['random_state']
        self.criterion      = nn.MSELoss()

        # Initialize models based on config
        self.video_encoder = EncoderCNN(img_x=self.img_size[0], img_y=self.img_size[1], input_channels=self.n_channels, CNN_embed_dim=self.feature_size)
        self.force_encoder = ForceFC() if self.use_force else None
        self.width_encoder = WidthFC() if self.use_width else None
        self.estimation_encoder = EstimationFC() if self.use_width else None

        # Compute the size of the input to the decoder based on config
        decoder_input_size = self.n_frames * self.feature_size
        if self.use_force: 
            decoder_input_size += self.n_frames * self.force_encoder.output_dim
        if self.use_width: 
            decoder_input_size += self.n_frames * self.width_encoder.output_dim
        if self.use_estimation: 
            decoder_input_size += self.n_frames * self.estimation_encoder.output_dim
        self.decoder = DecoderFC(input_dim=decoder_input_size, output_dim=1)

        # Send models to device
        self.video_encoder.to(self.device)
        if self.use_force:
            self.force_encoder.to(self.device)
        if self.use_width:
            self.width_encoder.to(self.device)
        if self.use_estimation:
            self.estimation_encoder.to(self.device)
        self.decoder.to(self.device)

        # Concatenate parameters of all models
        self.params         = list(self.video_encoder.parameters())
        if self.use_force: 
            self.params += list(self.force_encoder.parameters())
        if self.use_width: 
            self.params += list(self.width_encoder.parameters())
        if self.use_estimation: 
            self.params += list(self.estimation_encoder.parameters())
        self.params         += list(self.decoder.parameters())
        
        # Create optimizer, use Adam
        self.optimizer      = torch.optim.Adam(self.params, lr=self.learning_rate)

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

        # Create tensor's on device to send to dataset
        empty_frame_tensor        = torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=device)
        empty_force_tensor        = torch.zeros((self.n_frames, 1), device=device)
        empty_width_tensor        = torch.zeros((self.n_frames, 1), device=device)
        empty_estimation_tensor   = torch.zeros((self.n_frames, 1), device=device)
        empty_label_tensor        = torch.zeros((1), device=device)

        # Construct datasets
        kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
        self.train_dataset  = CustomDataset(config, x_train, y_train,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.val_dataset    = CustomDataset(config, x_val, y_val,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.train_loader   = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader     = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, **kwargs)
        return

    def _train_epoch(self):
        self.video_encoder.train()
        self.force_encoder.train()
        self.width_encoder.train()
        self.decoder.train()

        train_loss, batch_count = 0, 0
        for x_frames, x_forces, x_widths, x_estimations, y in self.train_loader:
            self.optimizer.zero_grad()

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):
                
                # Execute CNN on video frames
                features.append(self.video_encoder(x_frames[:, i, :, :, :].clone()))

                # Execute FC layers on other data and append
                if not (x_forces.max() == x_forces.min() == 0): # Force measurements
                    features.append(self.force_encoder(x_forces[:, i, :].clone()))
                if not (x_widths.max() == x_widths.min() == 0): # Width measurements
                    features.append(self.width_encoder(x_widths[:, i, :].clone()))
                if not (x_estimations.max() == x_estimations.min() == 0): # Precomputed modulus estimation
                    features.append(self.estimation_encoder(x_estimations[:, i, :].clone()))

            # Send aggregated features to the FC decoder
            outputs = self.decoder(features)

            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            batch_count += 1

        # Return loss
        train_loss /= (self.batch_size * batch_count)

        return train_loss

    def _val_epoch(self):
        self.video_encoder.eval()
        self.force_encoder.eval()
        self.width_encoder.eval()
        self.decoder.eval()

        val_loss, val_log_acc, batch_count = 0, 0, 0
        for x_frames, x_forces, x_widths, x_estimations, y in self.train_loader:

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):
                
                # Execute CNN on video frames
                features.append(self.video_encoder(x_frames[:, i, :, :, :].clone()))

                # Execute FC layers on other data and append
                if not (x_forces.max() == x_forces.min() == 0): # Force measurements
                    features.append(self.force_encoder(x_forces[:, i, :].clone()))
                if not (x_widths.max() == x_widths.min() == 0): # Width measurements
                    features.append(self.width_encoder(x_widths[:, i, :].clone()))
                if not (x_estimations.max() == x_estimations.min() == 0): # Precomputed modulus estimation
                    features.append(self.estimation_encoder(x_estimations[:, i, :].clone()))

            # Send aggregated features to the FC decoder
            outputs = self.decoder(features)

            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            val_loss += loss.item()
            val_log_acc += (torch.round(torch.log10(outputs)) == torch.round(torch.log10(y))).sum().item()
            batch_count += 1
        
        # Return loss and accuracy
        val_loss /= (self.batch_size * batch_count)
        val_log_acc /= (self.batch_size * batch_count)

        return val_loss, val_log_acc

    def _save_model(self):
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(self.video_encoder.state_dict(), f'{model_dir}/video_encoder.pth')
        torch.save(self.force_encoder.state_dict(), f'{model_dir}/force_encoder.pth')
        torch.save(self.width_encoder.state_dict(), f'{model_dir}/width_encoder.pth')
        torch.save(self.estimation_encoder.state_dict(), f'{model_dir}/estimation_encoder.pth')
        torch.save(self.decoder.state_dict(), f'{model_dir}/decoder.pth')
        return

    def train(self):
        min_val_loss = 1e10
        for epoch in range(self.epochs):

            # Train batch
            train_loss = self._train_epoch()

            # Validation statistics
            val_loss, val_log_acc = self._val_epoch()

            print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f},',
                f'Validation Loss: {val_loss:.4f},', 
                f'Validation Log Accuracy: {val_log_acc:.4f}',
                )

            # Save the best model based on validation loss
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                self._save_model()

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
                    "val_log_accuracy": val_log_acc,
                })

        if self.use_wandb: wandb.finish()

        return
    

if __name__ == "__main__":

    maybe_exclude = ['apple', 'orange', 'strawberry']

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
        'use_estimation': False,
        'use_transformations': True,
        'exclude': ['playdoh', 'silly_putty', 'racquetball', 'blue_sponge_dry', 'blue_sponge_wet',
                    'blue_foam_brick', 'green_foam_brick', 'red_foam_brick', 'yellow_foam_brick'],

        # Logging on/off
        'use_wandb': True,

        # Training and model parameters
        'epochs'         : 500,
        'batch_size'     : 32,
        'feature_size'   : 512,
        'val_pct'        : 0.2,
        'learning_rate'  : 1e-5,
        'random_state'   : 40,
    }
    assert config['img_style'] in ['diff', 'depth']
    config['n_channels'] = 3 if config['img_style'] == 'diff' else 1

    if config['use_estimation']: raise NotImplementedError()

    # Train the model over some data
    train_modulus = ModulusModel(config, device=device)
    train_modulus.train()