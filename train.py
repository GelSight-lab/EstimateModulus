import os
import sys
import pickle
import csv
import json
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import wandb

# from wedge_video import WARPED_CROPPED_IMG_SIZE
from nn_modules import *

from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

DATA_DIR = './data' # '/media/mike/Elements/data'
N_FRAMES = 3
WARPED_CROPPED_IMG_SIZE = (250, 350) # WARPED_CROPPED_IMG_SIZE[::-1]

# Get the tree of all video files from a directory in place
def list_files(folder_path, file_paths, config):
    # Iterate through the list
    for item in os.listdir(folder_path):
        item_path = f'{folder_path}/{item}'
        marker_signal = '_other' if config['use_markers'] else '.pkl'
        if os.path.isfile(item_path) and item.count(config['img_style']) > 0 and item.count(marker_signal) > 0:
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            list_files(item_path, file_paths, config)

class CustomDataset(Dataset):
    def __init__(self, config, paths_to_files, labels, normalization_values, \
                 frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
                 estimation_tensor=torch.zeros((3, 1)),
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
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.random_state       = config['random_state']

        self.normalization_values = normalization_values

        if self.use_transformations:
            self.input_paths = 4*paths_to_files
            self.normalized_modulus_labels = 4*labels
            self.flip_horizontal = [i > 2*len(paths_to_files) for i in range(len(self.input_paths))]
            self.flip_vertical = [i % 2 == 1 for i in range(len(self.input_paths))]
        else:
            self.input_paths = paths_to_files
            self.normalized_modulus_labels = labels
            self.flip_horizontal = [False for i in range(len(self.input_paths))]
            self.flip_vertical = [False for i in range(len(self.input_paths))]

        # Define attributes to use to conserve memory
        self.base_name      = ''
        self.x_frames       = frame_tensor
        # self.x_frames_other = frame_tensor
        self.x_forces       = force_tensor
        self.x_widths       = width_tensor
        self.x_estimations  = estimation_tensor
        self.y_label        = label_tensor
    
    def __len__(self):
        return len(self.normalized_modulus_labels)
    
    def __getitem__(self, idx):
        self.x_frames       = self.x_frames.zero_()
        self.x_forces       = self.x_forces.zero_()
        self.x_widths       = self.x_widths.zero_()
        self.x_estimations  = self.x_estimations.zero_()
        self.y_label        = self.y_label.zero_()

        object_name = os.path.basename(self.input_paths[idx]).split('__')[0]

        # Read and store frames in the tensor
        with open(self.input_paths[idx], 'rb') as file:
            if self.img_style == 'diff':
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)
            elif self.img_style == 'depth':
                self.x_frames[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(3).permute(0, 3, 1, 2)
                self.x_frames /= self.normalization_values['max_depth']

        # with open(self.input_paths[idx].replace('_other', ''), 'rb') as file:
        #     if self.img_style == 'diff':
        #         self.x_frames_other[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)
        #     elif self.img_style == 'depth':
        #         self.x_frames_other[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(3).permute(0, 3, 1, 2)
        #         self.x_frames_other /= self.normalization_values['max_depth']

        # Flip the data if desired
        if self.use_transformations and self.flip_horizontal[idx]:
            self.x_frames = torch.flip(self.x_frames, dims=(2,))
            # self.x_frames_other = torch.flip(self.x_frames_other, dims=(2,))
        if self.use_transformations and self.flip_vertical[idx]:
            self.x_frames = torch.flip(self.x_frames, dims=(3,))
            # self.x_frames_other = torch.flip(self.x_frames_other, dims=(3,))

        # Unpack force measurements
        self.base_name = self.input_paths[idx][:self.input_paths[idx].find(self.img_style)-1] 
        if self.use_force:
            with open(self.base_name + '_forces.pkl', 'rb') as file:
                self.x_forces[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_forces /= self.normalization_values['max_force']

        # Unpack gripper width measurements
        if self.use_width:
            with open(self.base_name + '_widths.pkl', 'rb') as file:
                self.x_widths[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_widths /= self.normalization_values['max_width'] # self.x_widths.max()
        
        # Unpack modulus estimations
        if self.use_estimation:
            t = self.base_name[self.base_name.find('t=')+2:self.base_name.find('/aug')]
            estimation_path = f'{DATA_DIR}/training_estimations/{object_name}/t={t}'
            with open(f'{estimation_path}/E.pkl', 'rb') as file:
                self.x_estimations[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
        
        # Unpack label
        self.y_label[0] = self.normalized_modulus_labels[idx]

        # return self.x_frames.clone(), self.x_frames_other.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name
        return self.x_frames.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name


class ModulusModel():
    def __init__(self, config, device=None):
        self.config = config

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
        self.run_name               = config['run_name']

        # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'max_modulus': 1e3,
            'min_modulus': 1e12,
            'max_depth': 7.0,
            'max_width': 0.08,
            'max_force': 60.0,
        }

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.dropout_pct        = config['dropout_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.lr_step_size       = config['lr_step_size']
        self.random_state       = config['random_state']
        self.criterion          = nn.MSELoss()

        # Initialize models based on config
        self.video_encoder = EncoderCNN(img_x=self.img_size[0], img_y=self.img_size[1], input_channels=self.n_channels, CNN_embed_dim=self.img_feature_size, dropout_pct=self.dropout_pct)
        # self.other_video_encoder = EncoderCNN(img_x=self.img_size[0], img_y=self.img_size[1], input_channels=self.n_channels, CNN_embed_dim=self.img_feature_size)
        self.force_encoder = ForceFC(hidden_size=self.fwe_feature_size, output_dim=self.fwe_feature_size) if self.use_force else None
        self.width_encoder = WidthFC(hidden_size=self.fwe_feature_size, output_dim=self.fwe_feature_size) if self.use_width else None
        self.estimation_encoder = EstimationFC(hidden_size=2*self.fwe_feature_size, output_dim=2*self.fwe_feature_size) if self.use_estimation else None

        # Compute the size of the input to the decoder based on config
        decoder_input_size = self.n_frames * self.img_feature_size
        # decoder_input_size += self.n_frames * self.img_feature_size
        if self.use_force: 
            decoder_input_size += self.n_frames * self.fwe_feature_size
        if self.use_width: 
            decoder_input_size += self.n_frames * self.fwe_feature_size
        if self.use_estimation: 
            decoder_input_size += 2 * self.fwe_feature_size
        self.decoder = DecoderFC(input_dim=decoder_input_size, output_dim=1, dropout_pct=self.dropout_pct)

        # Send models to device
        self.video_encoder.to(self.device)
        # self.other_video_encoder.to(self.device)
        if self.use_force:
            self.force_encoder.to(self.device)
        if self.use_width:
            self.width_encoder.to(self.device)
        if self.use_estimation:
            self.estimation_encoder.to(self.device)
        self.decoder.to(self.device)

        print('Summaries...')
        col_names = ("input_size", "output_size", "num_params", "params_percent")
        summary(self.video_encoder, (self.batch_size, self.n_channels,  self.img_size[0], self.img_size[1]), col_names=col_names, device=device)
        print('\nIn comparison, ResNet looks like this...')
        summary(torchvision.models.resnet18(), (self.batch_size, self.n_channels,  self.img_size[0], self.img_size[1]), col_names=col_names)
        if self.use_force:
            summary(self.force_encoder, (self.batch_size, 1), col_names=col_names, device=device)
        summary(self.decoder, (self.batch_size, decoder_input_size), col_names=col_names, device=device)
        print('Done.')

        # Concatenate parameters of all models
        self.params         = list(self.video_encoder.parameters())
        # self.params         += list(self.other_video_encoder.parameters())
        if self.use_force: 
            self.params += list(self.force_encoder.parameters())
        if self.use_width: 
            self.params += list(self.width_encoder.parameters())
        if self.use_estimation: 
            self.params += list(self.estimation_encoder.parameters())
        self.params         += list(self.decoder.parameters())
        
        # Create optimizer, use Adam
        self.optimizer      = torch.optim.Adam(self.params, lr=self.learning_rate)
        if self.gamma is not None:
            self.scheduler  = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.gamma)

        # Load data
        self.object_names = []
        self.object_to_modulus = {}
        self.train_object_performance = {}
        self.val_object_performance = {}
        self._load_data_paths()

        if self.use_wandb:
            wandb.init(
                # Set the wandb project where this run will be logged
                project="TrainModulus",
                name=self.run_name,
                
                # Track hyperparameters and run metadata
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "training_inputs": len(self.train_loader) * self.batch_size,
                    "val_inputs": len(self.val_loader) * self.batch_size,
                    "n_frames": self.n_frames,
                    "n_channels": self.n_channels,
                    "img_size": self.img_size,
                    "img_style": self.img_style,
                    "img_feature_size": self.img_feature_size,
                    "fwe_feature_size": self.fwe_feature_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "lr_step_size": self.lr_step_size,
                    "validation_pct": self.val_pct,
                    "dropout_pct": self.dropout_pct,
                    "random_state": self.random_state,
                    "architecture": "ENCODE_DECODE",
                    "num_params": len(self.params),
                    "optimizer": "Adam",
                    "loss": "MSE",
                    "scheduler": "StepLR",
                    "use_markers": self.use_markers,
                    "use_force": self.use_force,
                    "use_width": self.use_width,
                    "use_estimation": self.use_estimation,
                    "use_transformations": self.use_transformations,
                    "exclude": self.exclude,
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

        return
    
    # Normalize labels to maximum on log scale
    def log_normalize(self, x, x_max=None, x_min=None, use_torch=False):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        if use_torch:
            return (torch.log10(x) - torch.log10(self.x_min_cuda)) / (torch.log10(self.x_max_cuda) - torch.log10(self.x_min_cuda))
        return (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
    
    # Unnormalize labels from maximum on log scale
    def log_unnormalize(self, x_normal, x_max=None, x_min=None):
        if x_max is None: x_max = self.normalization_values['max_modulus']
        if x_min is None: x_min = self.normalization_values['min_modulus']
        return x_min * (x_max/x_min)**(x_normal)

    # Create data loaders based on configuration
    def _load_data_paths(self, labels_csv_name='objects_and_labels.csv', csv_modulus_column=14, training_data_folder_name=f'training_data__Nframes={N_FRAMES}'):
        # Read CSV files with objects and labels tabulated
        self.object_to_modulus = {}
        csv_file_path = f'{self.data_dir}/{labels_csv_name}'
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip title row
            for row in csv_reader:
                if row[csv_modulus_column] != '' and float(row[csv_modulus_column]) > 0:
                    modulus = float(row[csv_modulus_column])
                    self.object_to_modulus[row[1]] = modulus
                    self.normalization_values['max_modulus'] = max(self.normalization_values['max_modulus'], modulus)
                    self.normalization_values['min_modulus'] = min(self.normalization_values['min_modulus'], modulus)

        self.x_max_cuda = torch.Tensor([self.normalization_values['max_modulus']]).to(device)
        self.x_min_cuda = torch.Tensor([self.normalization_values['min_modulus']]).to(device)

        # Extract object names as keys from data
        object_names = self.object_to_modulus.keys()
        object_names = [x for x in object_names if x not in self.exclude]

        # Extract elastic modulus labels for each object
        elastic_moduli = [self.object_to_modulus[x] for x in object_names]

        # Split objects into validation or training
        objects_train, objects_val, _, _ = train_test_split(object_names, elastic_moduli, test_size=self.val_pct, random_state=self.random_state)
        del object_names, elastic_moduli

        # Get all the paths to grasp data within directory
        paths_to_files = []
        list_files(f'{self.data_dir}/{training_data_folder_name}', paths_to_files, self.config)

        # Create some data structures for tracking performance
        self.train_object_performance = {}
        self.val_object_performance = {}
        for object_name in objects_train:
            self.train_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }
        for object_name in objects_val:
            self.val_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }

        # Divide paths up into training and validation data
        x_train, x_val = [], []
        y_train, y_val = [], []
        self.object_names = []
        for file_path in paths_to_files:
            file_name = os.path.basename(file_path)
            object_name = file_name.split('__')[0]
            if object_name in self.exclude: continue

            if object_name in objects_train:
                self.object_names.append(object_name)
                x_train.append(file_path)
                y_train.append(self.log_normalize(self.object_to_modulus[object_name]))

            elif object_name in objects_val:
                self.object_names.append(object_name)
                x_val.append(file_path)
                y_val.append(self.log_normalize(self.object_to_modulus[object_name]))

        # Create tensor's on device to send to dataset
        empty_frame_tensor        = torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=device)
        empty_force_tensor        = torch.zeros((self.n_frames, 1), device=device)
        empty_width_tensor        = torch.zeros((self.n_frames, 1), device=device)
        empty_estimation_tensor   = torch.zeros((3, 1), device=device)
        empty_label_tensor        = torch.zeros((1), device=device)
    
        # Construct datasets
        kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
        self.train_dataset  = CustomDataset(self.config, x_train, y_train,
                                            self.normalization_values,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.val_dataset    = CustomDataset(self.config, x_val, y_val,
                                            self.normalization_values,
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
        if self.use_force:
            self.force_encoder.train()
        if self.use_width:
            self.width_encoder.train()
        if self.use_estimation:
            self.estimation_encoder.train()
        self.decoder.train()

        train_loss, train_log_acc, train_avg_log_diff, train_pct_with_100_factor_err, batch_count = 0, 0, 0, 0, 0
        for x_frames, x_forces, x_widths, x_estimations, y, object_names in self.train_loader:
            self.optimizer.zero_grad()

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):

                # Execute CNN on video frames
                features.append(self.video_encoder(x_frames[:, i, :, :, :]))

                # features.append(self.video_encoder(x_frames_other[:, i, :, :, :]))
                
                # Execute FC layers on other data and append
                if self.use_force: # Force measurements
                    features.append(self.force_encoder(x_forces[:, i, :]))
                if self.use_width: # Width measurements
                    features.append(self.width_encoder(x_widths[:, i, :]))

            if self.use_estimation: # Precomputed modulus estimation
                features.append(self.estimation_encoder(self.log_normalize(x_estimations[:, :, :], use_torch=True).squeeze()))

            # Send aggregated features to the FC decoder
            features = torch.cat(features, -1)
            outputs = self.decoder(features)
            
            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            train_avg_log_diff += abs_log_diff.sum()
            for i in range(self.batch_size):
                self.train_object_performance[object_names[i]]['total_log_diff'] += abs_log_diff[i][0]
                self.train_object_performance[object_names[i]]['count'] += 1
                if abs_log_diff[i] <= 0.5:
                    self.train_object_performance[object_names[i]]['total_log_acc'] += 1
                    train_log_acc += 1
                if abs_log_diff[i] >= 2:
                    self.train_object_performance[object_names[i]]['total_poorly_predicted'] += 1
                    train_pct_with_100_factor_err += 1
                    
        # Return loss
        train_loss /= batch_count
        train_log_acc /= (self.batch_size * batch_count)
        train_avg_log_diff /= (self.batch_size * batch_count)
        train_pct_with_100_factor_err /= (self.batch_size * batch_count)

        return train_loss, train_log_acc, train_avg_log_diff, train_pct_with_100_factor_err

    def _val_epoch(self):
        self.video_encoder.eval()
        if self.use_force:
            self.force_encoder.eval()
        if self.use_width:
            self.width_encoder.eval()
        if self.use_estimation:
            self.estimation_encoder.eval()
        self.decoder.eval()

        val_loss, val_log_acc, val_avg_log_diff, val_pct_with_100_factor_err, batch_count = 0, 0, 0, 0, 0
        for x_frames, x_forces, x_widths, x_estimations, y, object_names in self.val_loader:

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):
                
                # Execute CNN on video frames
                features.append(self.video_encoder(x_frames[:, i, :, :, :]))
                
                # features.append(self.video_encoder(x_frames_other[:, i, :, :, :]))

                # Execute FC layers on other data and append
                if self.use_force: # Force measurements
                    features.append(self.force_encoder(x_forces[:, i, :]))
                if self.use_width: # Width measurements
                    features.append(self.width_encoder(x_widths[:, i, :]))

            if self.use_estimation: # Precomputed modulus estimation
                features.append(self.estimation_encoder(self.log_normalize(x_estimations[:, :, :], use_torch=True).squeeze()))

            # Send aggregated features to the FC decoder
            features = torch.cat(features, -1)
            outputs = self.decoder(features)

            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            val_loss += loss.item()
            batch_count += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            val_avg_log_diff += abs_log_diff.sum()
            for i in range(self.batch_size):
                self.val_object_performance[object_names[i]]['total_log_diff'] += abs_log_diff[i][0]
                self.val_object_performance[object_names[i]]['count'] += 1
                if abs_log_diff[i] <= 0.5:
                    self.val_object_performance[object_names[i]]['total_log_acc'] += 1
                    val_log_acc += 1
                if abs_log_diff[i] >= 2:
                    self.val_object_performance[object_names[i]]['total_poorly_predicted'] += 1
                    val_pct_with_100_factor_err += 1

        # Return loss and accuracy
        val_loss /= batch_count
        val_log_acc /= (self.batch_size * batch_count)
        val_avg_log_diff /= (self.batch_size * batch_count)
        val_pct_with_100_factor_err /= (self.batch_size * batch_count)

        return val_loss, val_log_acc, val_avg_log_diff, val_pct_with_100_factor_err

    def _save_model(self):
        if not os.path.exists(f'./model/{self.run_name}'):
            os.mkdir(f'./model/{self.run_name}')
        else:
            if os.path.exists(f'./model/{self.run_name}/config.json'):
                os.remove(f'./model/{self.run_name}/config.json')
            if os.path.exists(f'./model/{self.run_name}/train_object_performance.json'):
                os.remove(f'./model/{self.run_name}/train_object_performance.json')
            if os.path.exists(f'./model/{self.run_name}/val_object_performance.json'):
                os.remove(f'./model/{self.run_name}/val_object_performance.json')
            if os.path.exists(f'./model/{self.run_name}/video_encoder.pth'):
                os.remove(f'./model/{self.run_name}/video_encoder.pth')
            if os.path.exists(f'./model/{self.run_name}/force_encoder.pth'):
                os.remove(f'./model/{self.run_name}/force_encoder.pth')
            if os.path.exists(f'./model/{self.run_name}/width_encoder.pth'):
                os.remove(f'./model/{self.run_name}/width_encoder.pth')
            if os.path.exists(f'./model/{self.run_name}/estimation_encoder.pth'):
                os.remove(f'./model/{self.run_name}/estimation_encoder.pth')
            if os.path.exists(f'./model/{self.run_name}/decoder.pth'):
                os.remove(f'./model/{self.run_name}/decoder.pth')

        # Save configuration dictionary and all files for the model(s)
        with open(f'./model/{self.run_name}/config.json', 'w') as json_file:
            json.dump(self.config, json_file)
        torch.save(self.video_encoder.state_dict(), f'./model/{self.run_name}/video_encoder.pth')
        if self.use_force: torch.save(self.force_encoder.state_dict(), f'./model/{self.run_name}/force_encoder.pth')
        if self.use_width: torch.save(self.width_encoder.state_dict(), f'./model/{self.run_name}/width_encoder.pth')
        if self.use_estimation: torch.save(self.estimation_encoder.state_dict(), f'./model/{self.run_name}/estimation_encoder.pth')
        torch.save(self.decoder.state_dict(), f'./model/{self.run_name}/decoder.pth')

        # Save performance data
        with open(f'./model/{self.run_name}/train_object_performance.json', 'w') as json_file:
            json.dump(self.train_object_performance, json_file)
        with open(f'./model/{self.run_name}/val_object_performance.json', 'w') as json_file:
            json.dump(self.val_object_performance, json_file)
        return

    def train(self):
        learning_rate = self.learning_rate 
        max_val_log_acc = 0.0
        for epoch in range(self.epochs):

            # Clean performance trackers
            for object_name in self.train_object_performance.keys():
                self.train_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }
            for object_name in self.val_object_performance.keys():
                self.val_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }

            # Train batch
            train_loss, train_log_acc, train_avg_log_diff, train_pct_with_100_factor_err = self._train_epoch()

            # Validation statistics
            val_loss, val_log_acc, val_avg_log_diff, val_pct_with_100_factor_err = self._val_epoch()

            # Increment learning rate
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            if self.gamma is not None:
                self.scheduler.step()

            print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f},',
                f'Validation Loss: {val_loss:.4f},',
                f'Validation Avg. Log Diff: {val_avg_log_diff:.4f}',
                f'Validation Log Accuracy: {val_log_acc:.4f}',
                '\n'
            )

            for name, param in self.video_encoder.named_parameters():
                print(f'{name}: {param.data}')
                break
            for name, param in self.force_encoder.named_parameters():
                print(f'{name}: {param.data}')
                break

            # Save the best model based on validation loss
            if val_log_acc >= max_val_log_acc:
                max_val_log_acc = val_log_acc
                self._save_model()

            # Log information to W&B
            if self.use_wandb:
                self.memory_allocated = torch.cuda.memory_allocated()
                self.memory_cached = torch.cuda.memory_reserved()
                wandb.log({
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "memory_allocated": self.memory_allocated,
                    "memory_reserved": self.memory_cached,
                    "train_loss": train_loss,
                    "train_avg_log_diff": train_avg_log_diff,
                    "train_log_accuracy": train_log_acc,
                    "train_pct_with_100_factor_err": train_pct_with_100_factor_err,
                    "val_loss": val_loss,
                    "val_avg_log_diff": val_avg_log_diff,
                    "val_log_accuracy": val_log_acc,
                    "val_pct_with_100_factor_err": val_pct_with_100_factor_err,
                })

        if self.use_wandb: wandb.finish()

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
        'exclude': ['playdoh', 'silly_puty', 'racquet_ball', 'blue_sponge_dry', 'blue_sponge_wet', \
                    'red_foam_brick', 'blue_foam_brick', 'yellow_foam_brick', 'green_foam_brick', 
                    'apple', 'orange', 'strawberry', 'lacrosse_ball', 'rubber_washer_stack'],

        # Logging on/off
        'use_wandb': True,
        'run_name': 'UseRealEstimations',   

        # Training and model parameters
        'epochs'            : 250,
        'batch_size'        : 32,
        'img_feature_size'  : 64,
        'fwe_feature_size'  : 8,
        'val_pct'           : 0.15,
        'dropout_pct'       : 0.3,
        'learning_rate'     : 1e-4,
        'gamma'             : None,
        'lr_step_size'      : None,
        'random_state'      : 40,
    }
    assert config['img_style'] in ['diff', 'depth']
    config['n_channels'] = 3 if config['img_style'] == 'diff' else 1

    base_run_name = config['run_name']
    for i in range(5):
        config['run_name'] = f'{base_run_name}__t={i}'

        # Train the model over some data
        train_modulus = ModulusModel(config, device=device)
        train_modulus.train()