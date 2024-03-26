import os
import math
import pickle
import csv
import json
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt

# from wedge_video import WARPED_CROPPED_IMG_SIZE
from nn_modules import *

from torchinfo import summary
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

DATA_DIR = '/media/mike/Elements/data'
ESTIMATION_DIR = 'training_estimations_nan_filtered'
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
        # if os.path.isfile(item_path) and item.count(config['img_style']) > 0:
        #     if item.count(marker_signal) > 0 or config['use_both_sides'] == False:
        #         file_paths.append(item_path)
        elif os.path.isdir(item_path):
            list_files(item_path, file_paths, config)

class CustomDataset(Dataset):
    def __init__(self, config, paths_to_files, labels, normalization_values, \
                 validation_dataset=False,
                 frame_tensor=torch.zeros((N_FRAMES, 3, WARPED_CROPPED_IMG_SIZE[0], WARPED_CROPPED_IMG_SIZE[1])),
                 force_tensor=torch.zeros((N_FRAMES, 1)),
                 width_tensor=torch.zeros((N_FRAMES, 1)),
                 estimation_tensor=torch.zeros((3, 1)),
                 label_tensor=torch.zeros((1, 1))
        ):
        # Data parameters 
        self.data_dir               = config['data_dir']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.use_both_sides         = config['use_both_sides']
        self.use_markers            = config['use_markers']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimation         = config['use_estimation']
        self.use_transformations    = config['use_transformations']
        self.use_width_transforms   = config['use_width_transforms']
        self.exclude                = config['exclude']

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.random_state       = config['random_state']

        self.validation_dataset = validation_dataset
        self.normalization_values = normalization_values
        self.input_paths = paths_to_files
        self.normalized_modulus_labels = labels

        if self.use_width_transforms:
            self.input_paths = 2*self.input_paths
            self.normalized_modulus_labels = 2*self.normalized_modulus_labels
            self.noise_force = [ i > len(self.input_paths)/2 and i % 2 == 1 for i in range(len(self.input_paths)) ]
            self.noise_width = [ i > len(self.input_paths)/2 for i in range(len(self.input_paths)) ]

        # Define attributes to use to conserve memory
        self.base_name      = ''
        self.x_frames       = frame_tensor
        self.x_frames_other = frame_tensor if self.use_both_sides else None
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
                
        if self.use_both_sides:
            with open(self.input_paths[idx].replace('_other', ''), 'rb') as file:
                if self.img_style == 'diff':
                    self.x_frames_other[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).permute(0, 3, 1, 2)
                elif self.img_style == 'depth':
                    self.x_frames_other[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(3).permute(0, 3, 1, 2)
                    self.x_frames_other /= self.normalization_values['max_depth']

        # Unpack force measurements
        self.base_name = self.input_paths[idx][:self.input_paths[idx].find(self.img_style)-1] 
        if self.use_force:
            with open(self.base_name + '_forces.pkl', 'rb') as file:
                self.x_forces[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_forces /= self.normalization_values['max_force']

        # Unpack gripper width measurements
        if self.use_width:
            with open(self.base_name + '_widths.pkl', 'rb') as file:
                # self.x_widths[:self.n_frames] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                # self.x_widths[self.n_frames:] = self.x_widths[:self.n_frames]
                # self.x_widths[self.n_frames:] /= self.x_widths.max()
                # self.x_widths[:self.n_frames] /= self.normalization_values['max_width'] 
                self.x_widths[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
                self.x_widths[:] /= self.normalization_values['max_width']

            if self.use_width_transforms:
                if self.noise_width[idx]:
                    noise_amplitude = min(
                        1 - self.x_widths.max(),
                        self.x_widths.min()
                    )
                    if random.random() > 0.5:
                        self.x_widths += noise_amplitude * random.random()
                    else:
                        self.x_widths -= noise_amplitude * random.random()
        
        # Unpack modulus estimations
        if self.use_estimation:
            t = self.base_name[self.base_name.find('t=')+2:self.base_name.find('/aug')]
            estimation_path = f'{DATA_DIR}/{ESTIMATION_DIR}/{object_name}/t={t}'
            with open(f'{estimation_path}/E.pkl', 'rb') as file:
                self.x_estimations[:] = torch.from_numpy(pickle.load(file).astype(np.float32)).unsqueeze(1)
        
        # Unpack label
        self.y_label[0] = self.normalized_modulus_labels[idx]

        if self.use_both_sides:
            return self.x_frames.clone(), self.x_frames_other.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name
        else:
            return self.x_frames.clone(), self.x_forces.clone(), self.x_widths.clone(), self.x_estimations.clone(), self.y_label.clone(), object_name


class ModulusModel():
    def __init__(self, config, device=None):
        self.config = config
        self._unpack_config(config)

        # Use GPU by default
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()
        else:
            self.device = device

        # Create max values for scaling
        self.normalization_values = { # Based on acquired data maximums
            'min_modulus': 1e3,
            'max_modulus': 1e12,
            'min_estimate': 1e2,
            'max_estimate': 1e14,
            'max_depth': 7.0,
            'max_width': 0.08,
            'max_force': 60.0,
        }

        # Initialize CNN based on config
        if self.pretrained_CNN:

            # Incorporate ViT pretrained CNN encoder
            self.pretrained_img_size = 224
            self.video_encoder = VisionTransformer(
                img_size=self.pretrained_img_size,
                patch_size=16,
                embed_dim=768,
                in_chans=self.n_channels,
                pre_norm=True
            )
            self.video_encoder.head = nn.Identity(device=self.device) # Remove head to retrain
            assert self.frozen_pretrained, 'Currently do not have the resources to retrain ViT.'
            self.video_encoder.eval()
            for param in self.video_encoder.parameters():
                param.requires_grad = False
            self.video_encoder_head = nn.Linear(768, self.img_feature_size, device=self.device)

            # self.video_encoder = ModifiedResNet18(CNN_embed_dim=self.img_feature_size, frozen=self.frozen_pretrained)
        else:
            self.video_encoder = EncoderCNN(
                img_x=self.img_size[0],
                img_y=self.img_size[1],
                input_channels=self.n_channels,
                CNN_embed_dim=self.img_feature_size,
                dropout_pct=self.dropout_pct
            )

        # Compute the size of the input to the decoder based on config
        self.decoder_input_size = self.n_frames * self.img_feature_size
        if self.use_both_sides:
            self.decoder_input_size += self.n_frames * self.img_feature_size
        if self.use_force: 
            self.decoder_input_size += self.fwe_feature_size
        if self.use_width: 
            self.decoder_input_size += self.fwe_feature_size
        self.decoder_output_size = 3 if self.use_estimation else 1

        # Initialize force, width, estimation based on config
        self.force_encoder = ForceFC(
                                    input_dim=self.n_frames,
                                    hidden_size=self.fwe_feature_size, 
                                    output_dim=self.fwe_feature_size, 
                                    dropout_pct=self.dropout_pct
                                ) if self.use_force else None
        self.width_encoder = WidthFC(
                                    input_dim=self.n_frames,
                                    hidden_size=self.fwe_feature_size,
                                    output_dim=self.fwe_feature_size,
                                    dropout_pct=self.dropout_pct
                                ) if self.use_width else None
        self.estimation_decoder = EstimationDecoderFC(
                                    input_dim=3 + self.decoder_output_size,
                                    FC_layer_nodes=config['est_decoder_size'],
                                    output_dim=1,
                                    dropout_pct=self.dropout_pct
                                ) if self.use_estimation else None
        self.decoder = DecoderFC(input_dim=self.decoder_input_size, FC_layer_nodes=config['decoder_size'], output_dim=self.decoder_output_size, dropout_pct=self.dropout_pct)

        # Send models to device
        self.video_encoder.to(self.device)
        if self.pretrained_CNN:
            self.video_encoder_head.to(self.device)
        if self.use_force:
            self.force_encoder.to(self.device)
        if self.use_width:
            self.width_encoder.to(self.device)
        if self.use_estimation:
            self.estimation_decoder.to(self.device)
        self.decoder.to(self.device)

        '''
        print('Summaries...')
        col_names = ("input_size", "output_size", "num_params", "params_percent")
        summary(self.video_encoder, (self.batch_size, self.n_channels,  self.img_size[0], self.img_size[1]), col_names=col_names, device=device)
        print('\nIn comparison, ResNet looks like this...')
        summary(torchvision.models.resnet18(), (self.batch_size, self.n_channels,  self.img_size[0], self.img_size[1]), col_names=col_names)
        if self.use_force:
            summary(self.force_encoder, (self.batch_size, self.n_frames), col_names=col_names, device=device)
        summary(self.decoder, (self.batch_size, self.decoder_input_size), col_names=col_names, device=device)
        summary(self.estimation_decoder, (self.batch_size, 6), col_names=col_names, device=device)
        print('Done.')
        '''

        # Concatenate parameters of all models
        if self.pretrained_CNN:
            self.params = list(self.video_encoder_head.parameters())
        else:
            self.params = list(self.video_encoder.parameters())
        if self.use_force: 
            self.params += list(self.force_encoder.parameters())
        if self.use_width: 
            self.params += list(self.width_encoder.parameters())
        if self.use_estimation: 
            self.params += list(self.estimation_decoder.parameters())
        self.params += list(self.decoder.parameters())
        
        # Create optimizer, use Adam
        self.optimizer      = torch.optim.Adam(self.params, lr=self.learning_rate)
        if self.gamma is not None:
            self.scheduler  = lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.gamma)

        # Normalize based on mean and std computed over the dataset
        if self.n_channels == 3:
            if self.pretrained_CNN:
                # Use the RGB mean and std for custom model
                self.image_normalization = torchvision.transforms.Normalize( \
                                                [0.485, 0.456, 0.406], \
                                                [0.229, 0.224, 0.225] \
                                            )
            else:
                # Use the RGB mean and std computed for our dataset
                self.image_normalization = torchvision.transforms.Normalize( \
                                                [0.49638007, 0.49770336, 0.49385751], \
                                                [0.04634926, 0.06181679, 0.07152624] \
                                            )

        # Apply random flipping transformations
        if self.use_transformations:
            self.random_transformer = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(0.5),
                    torchvision.transforms.RandomVerticalFlip(0.5),
                ])

        # Resize image as expected by thee pretrained network weights
        if self.pretrained_CNN:
            self.resize_transformer = torchvision.transforms.Resize(
                (self.pretrained_img_size, self.pretrained_img_size), 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
                antialias=True
            )

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
                    "num_params": len(self.params),
                    "optimizer": "Adam",
                    "loss_function": self.loss_function,
                    "scheduler": "StepLR",
                    "pretrained_CNN": self.pretrained_CNN,
                    "frozen_pretrained": self.frozen_pretrained,
                    "use_both_sides": self.use_both_sides,
                    "use_markers": self.use_markers,
                    "use_force": self.use_force,
                    "use_width": self.use_width,
                    "use_estimation": self.use_estimation,
                    "use_transformations": self.use_transformations,
                    "use_width_transforms": self.use_width_transforms,
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
    
    def _unpack_config(self, config):
        # Data parameters 
        self.data_dir               = config['data_dir']
        self.n_frames               = config['n_frames']
        self.img_size               = config['img_size']
        self.img_style              = config['img_style']
        self.n_channels             = config['n_channels']
        self.use_markers            = config['use_markers']
        self.use_both_sides         = config['use_both_sides']
        self.use_force              = config['use_force']
        self.use_width              = config['use_width']
        self.use_estimation         = config['use_estimation']
        self.use_transformations    = config['use_transformations']
        self.use_width_transforms   = config['use_width_transforms']
        self.loss_function          = config['loss_function']
        self.exclude                = config['exclude']
        
        self.use_wandb              = config['use_wandb']
        self.run_name               = config['run_name']

        # Define training parameters
        self.epochs             = config['epochs']
        self.batch_size         = config['batch_size']
        self.pretrained_CNN     = config['pretrained_CNN']
        self.frozen_pretrained  = config['frozen_pretrained']
        self.img_feature_size   = config['img_feature_size']
        self.fwe_feature_size   = config['fwe_feature_size']
        self.val_pct            = config['val_pct']
        self.dropout_pct        = config['dropout_pct']
        self.learning_rate      = config['learning_rate']
        self.gamma              = config['gamma']
        self.lr_step_size       = config['lr_step_size']
        self.random_state       = config['random_state']
        if self.loss_function == 'mse':
            self.criterion      = nn.MSELoss()
        elif self.loss_function == 'log_diff':
            self.criterion      = MSLogDiffLoss()
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
    def _load_data_paths(self, labels_csv_name='objects_and_labels.csv', csv_modulus_column=14, csv_shape_column=2, csv_material_column=3, \
                         training_data_folder_name=f'training_data__Nframes={N_FRAMES}__new'):
        # Read CSV files with objects and labels tabulated
        self.object_to_modulus = {}
        self.object_to_material = {}
        self.object_to_shape = {}
        csv_file_path = f'{self.data_dir}/{labels_csv_name}'
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) # Skip title row
            for row in csv_reader:
                if row[csv_modulus_column] != '' and float(row[csv_modulus_column]) > 0:
                    modulus = float(row[csv_modulus_column])
                    self.object_to_modulus[row[1]] = modulus
                    if not (row[1] in self.exclude):
                        self.normalization_values['max_modulus'] = max(self.normalization_values['max_modulus'], modulus)
                        self.normalization_values['min_modulus'] = min(self.normalization_values['min_modulus'], modulus)

                    self.object_to_material[row[1]] = row[csv_material_column]
                    self.object_to_shape[row[1]] = row[csv_shape_column]

        self.x_max_cuda = torch.Tensor([self.normalization_values['max_modulus']]).to(device)
        self.x_min_cuda = torch.Tensor([self.normalization_values['min_modulus']]).to(device)

        # Extract object names as keys from data
        object_names = self.object_to_modulus.keys()
        object_names = [x for x in object_names if (x not in self.exclude)]

        # Extract corresponding elastic modulus labels for each object
        elastic_moduli = [self.object_to_modulus[x] for x in object_names]

        # Split objects into validation or training
        self.objects_train, self.objects_val, _, _ = train_test_split(object_names, elastic_moduli, test_size=self.val_pct, random_state=self.random_state)
        del object_names, elastic_moduli

        # Get all the paths to grasp data within directory
        paths_to_files = []
        list_files(f'{self.data_dir}/{training_data_folder_name}', paths_to_files, self.config)
        self.paths_to_files = paths_to_files

        # Remove those with no estimation
        if self.use_estimation:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                file_prefix = file_path[:file_path.find('aug=')-1]
                file_prefix = file_prefix.replace(training_data_folder_name, ESTIMATION_DIR)
                if os.path.isfile(f'{file_prefix}/E.pkl'):
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Remove those with no force change
        if self.use_force:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                file_prefix = file_path[:file_path.find('_aug=')+6]
                with open(file_prefix + '_forces.pkl', 'rb') as file:
                    F = pickle.load(file)
                if F[-1] > F[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Remove those with no width change
        if self.use_width:
            clean_paths_to_files = []
            for file_path in self.paths_to_files:
                file_prefix = file_path[:file_path.find('_aug=')+6]
                with open(file_prefix + '_widths.pkl', 'rb') as file:
                    w = pickle.load(file)
                if w[-1] < w[0]:
                    clean_paths_to_files.append(file_path)
            self.paths_to_files = clean_paths_to_files

        # Create data loaders based on training / validation break-up
        self._create_data_loaders()
        return
    
    def _create_data_loaders(self):
        # Divide paths up into training and validation data
        x_train, x_val = [], []
        y_train, y_val = [], []
        self.object_names = []
        for file_path in self.paths_to_files:
            file_name = os.path.basename(file_path)
            object_name = file_name.split('__')[0]
            if object_name in self.exclude: continue

            if object_name in self.objects_train:
                self.object_names.append(object_name)
                x_train.append(file_path)
                y_train.append(self.log_normalize(self.object_to_modulus[object_name]))

            elif object_name in self.objects_val:
                self.object_names.append(object_name)
                x_val.append(file_path)
                y_val.append(self.log_normalize(self.object_to_modulus[object_name]))

        # Create some data structures for tracking performance
        self.train_object_performance = {}
        self.val_object_performance = {}
        for object_name in self.objects_train:
            self.train_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }
        for object_name in self.objects_val:
            self.val_object_performance[object_name] = {
                    'total_log_diff': 0,
                    'total_log_acc': 0,
                    'total_poorly_predicted': 0, # Off by factor of 100 or more
                    'total_very_poorly_predicted': 0, # Off by factor of 1000 or more
                    'count': 0
                }

        # Create tensor's on device to send to dataset
        empty_frame_tensor        = torch.zeros((self.n_frames, self.n_channels, self.img_size[0], self.img_size[1]), device=self.device)
        empty_force_tensor        = torch.zeros((self.n_frames, 1), device=self.device)
        empty_width_tensor        = torch.zeros((self.n_frames, 1), device=self.device)
        empty_estimation_tensor   = torch.zeros((3, 1), device=self.device)
        empty_label_tensor        = torch.zeros((1), device=self.device)
    
        # Construct datasets
        kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
        self.train_dataset  = CustomDataset(self.config, x_train, y_train,
                                            self.normalization_values,
                                            validation_dataset=False,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.val_dataset    = CustomDataset(self.config, x_val, y_val,
                                            self.normalization_values,
                                            validation_dataset=True,
                                            frame_tensor=empty_frame_tensor, 
                                            force_tensor=empty_force_tensor,
                                            width_tensor=empty_width_tensor,
                                            estimation_tensor=empty_estimation_tensor,
                                            label_tensor=empty_label_tensor)
        self.train_loader   = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)
        self.val_loader     = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)


        if not self.use_both_sides:
            if self.use_markers:
                x_pretrain = [ file_path.replace('_other', '') for file_path in x_train ]
            else:
                x_pretrain = [ file_path.replace(f'_{self.img_style}', '_{self.img_style}_other') for file_path in x_train ]
            y_pretrain = y_train
            self.pretrain_dataset = CustomDataset(self.config, x_pretrain, y_pretrain,
                                                    self.normalization_values,
                                                    validation_dataset=False,
                                                    frame_tensor=empty_frame_tensor, 
                                                    force_tensor=empty_force_tensor,
                                                    width_tensor=empty_width_tensor,
                                                    estimation_tensor=empty_estimation_tensor,
                                                    label_tensor=empty_label_tensor)
            self.pretrain_loader  = DataLoader(self.pretrain_dataset, batch_size=self.batch_size, shuffle=True, **kwargs)


        return

    def _train_epoch(self, train_loader=None):
        if self.pretrained_CNN:
            if not self.frozen_pretrained: 
                self.video_encoder.train()
            self.video_encoder_head.train()
        else:
            self.video_encoder.train()
        if self.use_force:
            self.force_encoder.train()
        if self.use_width:
            self.width_encoder.train()
        if self.use_estimation:
            self.estimation_decoder.train()
        self.decoder.train()

        train_stats = {
            'loss': 0,
            'log_acc': 0,
            'avg_log_diff': 0,
            'pct_w_100_factor_err': 0,
            'batch_count': 0,
        }
        if train_loader is None: train_loader = self.train_loader
        for batch_data in self.train_loader:
            self.optimizer.zero_grad()

            # Unpack data
            if self.use_both_sides:
                x_frames, x_frames_other, x_forces, x_widths, x_estimations, y, object_names = batch_data
            else:
                x_frames, x_forces, x_widths, x_estimations, y, object_names = batch_data
                
            x_frames = x_frames.view(-1, self.n_channels, self.img_size[0], self.img_size[1])
            if self.use_both_sides:
                x_frames_other = x_frames_other.view(-1, self.n_channels, self.img_size[0], self.img_size[1])

            # Normalize images
            if self.n_channels == 3:
                x_frames = self.image_normalization(x_frames)
                if self.use_both_sides:
                    x_frames_other = self.image_normalization(x_frames_other)

            # Apply random transformations for training
            if self.use_transformations:
                x_frames = self.random_transformer(x_frames) # Apply V/H flips
                if self.use_both_sides:
                    x_frames_other = self.random_transformer(x_frames_other)
                
            x_frames = x_frames.view(self.batch_size, self.n_frames, self.n_channels, self.img_size[0], self.img_size[1])
            if self.use_both_sides:
                x_frames_other = x_frames_other.view(self.batch_size, self.n_frames, self.n_channels, self.img_size[0], self.img_size[1])

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):
                
                # Execute CNN on video frames
                if self.pretrained_CNN:
                    features.append(self.video_encoder_head(self.video_encoder(x_frames[:, i, :, :, :])))
                    if self.use_both_sides:
                        features.append(self.video_encoder_head(self.video_encoder(x_frames_other[:, i, :, :, :])))
                else:
                    features.append(self.video_encoder(x_frames[:, i, :, :, :]))
                    if self.use_both_sides:
                        features.append(self.video_encoder(x_frames_other[:, i, :, :, :]))

            # Execute FC layers on other data and append
            if self.use_force: # Force measurements
                features.append(self.force_encoder(x_forces[:, :, :].squeeze()))
            if self.use_width: # Width measurements
                features.append(self.width_encoder(x_widths[:, :, :].squeeze()))
            
            features = torch.cat(features, -1)

            # Send aggregated features to the FC decoder
            outputs = self.decoder(features)

            # Send to decoder with deterministic estimations
            if self.use_estimation:
                x_estimations = torch.clamp(x_estimations, min=self.normalization_values['min_estimate'], max=self.normalization_values['max_estimate'])
                x_estimations = self.log_normalize(x_estimations, x_max=self.normalization_values['max_estimate'], x_min=self.normalization_values['min_estimate'], use_torch=True)
                outputs = self.estimation_decoder(torch.cat([outputs, x_estimations.squeeze(-1)], -1))
           
            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            
            # Add regularization to loss
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.params:
                l2_reg += torch.norm(param)
            if self.loss_function == 'mse':
                alpha = 0.00005
            elif self.loss_function == 'log_diff':
                alpha = 0.001
            loss += alpha * l2_reg

            loss.backward()
            self.optimizer.step()

            train_stats['loss'] += loss.item()
            train_stats['batch_count'] += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            train_stats['avg_log_diff'] += abs_log_diff.sum()
            for i in range(self.batch_size):
                self.train_object_performance[object_names[i]]['total_log_diff'] += abs_log_diff[i][0]
                self.train_object_performance[object_names[i]]['count'] += 1
                if abs_log_diff[i] <= 1:
                    self.train_object_performance[object_names[i]]['total_log_acc'] += 1
                    train_stats['log_acc'] += 1
                if abs_log_diff[i] >= 2:
                    self.train_object_performance[object_names[i]]['total_poorly_predicted'] += 1
                    train_stats['avg_log_diff'] += 1
                    
        # Return loss
        train_stats['loss']                     /= train_stats['batch_count']
        train_stats['log_acc']                  /= (self.batch_size * train_stats['batch_count'])
        train_stats['avg_log_diff']             /= (self.batch_size * train_stats['batch_count'])
        train_stats['pct_w_100_factor_err']     /= (self.batch_size * train_stats['batch_count'])

        return train_stats

    def _val_epoch(self, track_predictions=False):
        if self.pretrained_CNN:
            if not self.frozen_pretrained: 
                self.video_encoder.eval()
            self.video_encoder_head.eval()
        else:
            self.video_encoder.eval()
        if self.use_force:
            self.force_encoder.eval()
        if self.use_width:
            self.width_encoder.eval()
        if self.use_estimation:
            self.estimation_decoder.eval()
        self.decoder.eval()

        if track_predictions:
            predictions = { obj : [] for obj in self.objects_val }

        val_stats = {
            'loss': 0,
            'log_acc': 0,
            'soft_log_acc': 0,
            'hard_log_acc': 0,
            'avg_log_diff': 0,
            'soft_avg_log_diff': 0,
            'hard_avg_log_diff': 0,
            'pct_w_100_factor_err': 0,
            'pct_w_1000_factor_err': 0,
            'soft_count': 0,
            'hard_count': 0,
            'batch_count': 0,
        }
        for batch_data in self.val_loader:

            # Unpack data
            if self.use_both_sides:
                x_frames, x_frames_other, x_forces, x_widths, x_estimations, y, object_names = batch_data
            else:
                x_frames, x_forces, x_widths, x_estimations, y, object_names = batch_data
                
            x_frames = x_frames.view(-1, self.n_channels, self.img_size[0], self.img_size[1])
            if self.use_both_sides:
                x_frames_other = x_frames_other.view(-1, self.n_channels, self.img_size[0], self.img_size[1])

            # Normalize images
            if self.n_channels == 3:
                x_frames = self.image_normalization(x_frames)
                if self.use_both_sides:
                    x_frames_other = self.image_normalization(x_frames_other)
                
            x_frames = x_frames.view(self.batch_size, self.n_frames, self.n_channels, self.img_size[0], self.img_size[1])
            if self.use_both_sides:
                x_frames_other = x_frames_other.view(self.batch_size, self.n_frames, self.n_channels, self.img_size[0], self.img_size[1])

            # Concatenate features across frames into a single vector
            features = []
            for i in range(N_FRAMES):
                
                # Execute CNN on video frames
                if self.pretrained_CNN:
                    features.append(self.video_encoder_head(self.video_encoder(x_frames[:, i, :, :, :])))
                    if self.use_both_sides:
                        features.append(self.video_encoder_head(self.video_encoder(x_frames_other[:, i, :, :, :])))
                else:
                    features.append(self.video_encoder(x_frames[:, i, :, :, :]))
                    if self.use_both_sides:
                        features.append(self.video_encoder(x_frames_other[:, i, :, :, :]))
                    print('CNN Outputs:', self.video_encoder(x_frames[:, i, :, :, :])[0:5,:])

            # Execute FC layers on other data and append
            if self.use_force: # Force measurements
                features.append(self.force_encoder(x_forces[:, :, :].squeeze()))
                print('Force Inputs:', x_forces[0:3, :, :])
                print('Force Outputs:', self.force_encoder(x_forces[:, :, :].squeeze())[0:3,:])

            if self.use_width: # Width measurements
                features.append(self.width_encoder(x_widths[:, :, :].squeeze()))
                print('Width Inputs:', x_widths[0:3, :, :])
                print('Width Outputs:', self.width_encoder(x_widths[:, :, :].squeeze())[0:3,:])

            # Send aggregated features to the FC decoder
            features = torch.cat(features, -1)
            outputs = self.decoder(features)

            print('Decoder Outputs:', outputs[0:5,:])

            # Send to decoder with deterministic estimations
            if self.use_estimation:
                x_estimations = torch.clamp(x_estimations, min=self.normalization_values['min_estimate'], max=self.normalization_values['max_estimate'])
                x_estimations = self.log_normalize(x_estimations, x_max=self.normalization_values['max_estimate'], x_min=self.normalization_values['min_estimate'], use_torch=True)
                outputs = self.estimation_decoder(torch.cat([outputs, x_estimations.squeeze(-1)], -1))

            print('Final Outputs:', outputs[0:5,:])

            loss = self.criterion(outputs.squeeze(1), y.squeeze(1))
            
            # Add regularization to loss
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.params:
                l2_reg += torch.norm(param)
            if self.loss_function == 'mse':
                alpha = 0.00005
            elif self.loss_function == 'log_diff':
                alpha = 0.001
            loss += alpha * l2_reg

            val_stats['loss'] += loss.item()
            val_stats['batch_count'] += 1

            # Calculate performance metrics
            abs_log_diff = torch.abs(torch.log10(self.log_unnormalize(outputs.cpu())) - torch.log10(self.log_unnormalize(y.cpu()))).detach().numpy()
            val_stats['avg_log_diff'] += abs_log_diff.sum()
            for i in range(self.batch_size):
                self.val_object_performance[object_names[i]]['total_log_diff'] += abs_log_diff[i][0]
                self.val_object_performance[object_names[i]]['count'] += 1
                if self.object_to_modulus[object_names[i]] < 1e8:
                    val_stats['soft_avg_log_diff'] += abs_log_diff[i]
                else:
                    val_stats['hard_avg_log_diff'] += abs_log_diff[i]

                if abs_log_diff[i] <= 1:
                    self.val_object_performance[object_names[i]]['total_log_acc'] += 1
                    val_stats['log_acc'] += 1
                    if self.object_to_modulus[object_names[i]] < 1e8:
                        val_stats['soft_log_acc'] += 1
                    else:
                        val_stats['hard_log_acc'] += 1

                if abs_log_diff[i] >= 2:
                    self.val_object_performance[object_names[i]]['total_poorly_predicted'] += 1
                    val_stats['pct_w_100_factor_err'] += 1

                if abs_log_diff[i] >= 3:
                    self.val_object_performance[object_names[i]]['total_very_poorly_predicted'] += 1
                    val_stats['pct_w_1000_factor_err'] += 1

                if self.object_to_modulus[object_names[i]] < 1e8: val_stats['soft_count'] += 1
                else: val_stats['hard_count'] += 1

                if track_predictions:
                    predictions[object_names[i]].append(self.log_unnormalize(outputs[i][0].cpu()).detach().numpy())
            

        # Return loss and accuracy
        val_stats['loss']                   /= val_stats['batch_count']
        val_stats['log_acc']                /= (self.batch_size * val_stats['batch_count'])
        val_stats['avg_log_diff']           /= (self.batch_size * val_stats['batch_count'])
        val_stats['pct_w_100_factor_err']   /= (self.batch_size * val_stats['batch_count'])
        val_stats['pct_w_1000_factor_err']  /= (self.batch_size * val_stats['batch_count'])
        val_stats['soft_log_acc']           /= val_stats['soft_count']
        val_stats['soft_avg_log_diff']      /= val_stats['soft_count']
        if val_stats['hard_count'] > 0:
            val_stats['hard_log_acc']           /= val_stats['hard_count']
            val_stats['hard_avg_log_diff']      /= val_stats['hard_count']

        if track_predictions:
            return predictions
        else:
            return val_stats
        
    # Try pretraining on images without markers
    def pretrain(self, epochs=10):
        for epoch in range(epochs):
            train_stats = self._train_epoch(train_loader=self.pretrain_loader)
            if self.gamma is not None:
                self.scheduler.step()
            print(f'Pretain Epoch: {epoch}, Training Loss: {train_stats["loss"]:.4f},\n')

        return

    def train(self):
        learning_rate = self.learning_rate 
        max_val_log_acc = 0.0
        min_val_loss = 1e10
        min_val_outlier_pct = 1e10

        # if not self.use_both_sides:
        #     self.pretrain()
        #     self.optimizer.param_groups[0]['lr'] = self.learning_rate

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
                    'total_very_poorly_predicted': 0, # Off by factor of 100 or more
                    'count': 0
                }

            # Train batch
            train_stats = self._train_epoch()

            # Validation statistics
            val_stats = self._val_epoch()

            # Increment learning rate
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
            if self.gamma is not None:
                self.scheduler.step()

            print(f'Epoch: {epoch}, Training Loss: {train_stats["loss"]:.4f},',
                f'Validation Loss: {val_stats["loss"]:.4f},',
                f'Validation Avg. Log Diff: {val_stats["avg_log_diff"]:.4f}',
                f'Validation Log Accuracy: {val_stats["log_acc"]:.4f}',
                '\n'
            )

            # for name, param in self.video_encoder.named_parameters():
            #     print(f'{name} : {param.data} : {name}')
            #     break
            # for name, param in self.force_encoder.named_parameters():
            #     print(f'{name} : {param.data} : {name}')
            #     break

            # Save the best model based on validation loss and accuracy
            if val_stats['loss'] <= min_val_loss:
                min_val_loss = val_stats['loss']
                self.save_model(method='by_loss')
            if val_stats['log_acc'] >= max_val_log_acc:
                max_val_log_acc = val_stats['log_acc']
                self.save_model(method='by_acc')
            if val_stats['pct_w_100_factor_err'] <= min_val_outlier_pct:
                min_val_outlier_pct = val_stats['pct_w_100_factor_err']
                self.save_model(method='by_outliers')

            # Log information to W&B
            if self.use_wandb:
                self.memory_allocated = torch.cuda.memory_allocated()
                self.memory_cached = torch.cuda.memory_reserved()
                wandb.log({
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "memory_allocated": self.memory_allocated,
                    "memory_reserved": self.memory_cached,
                    "train_loss": train_stats['loss'],
                    "train_avg_log_diff": train_stats['avg_log_diff'],
                    "train_log_accuracy": train_stats['log_acc'],
                    "train_pct_with_100_factor_err": train_stats['pct_w_100_factor_err'],
                    "val_loss": val_stats['loss'],
                    "val_avg_log_diff": val_stats['avg_log_diff'],
                    "val_log_accuracy": val_stats['log_acc'],
                    "val_pct_with_100_factor_err": val_stats['pct_w_100_factor_err'],
                    "val_pct_with_1000_factor_err": val_stats['pct_w_1000_factor_err'],
                    "val_soft_avg_log_diff": val_stats['soft_avg_log_diff'],
                    "val_soft_log_acc": val_stats['soft_log_acc'],
                    "val_hard_avg_log_diff": val_stats['hard_avg_log_diff'],
                    "val_hard_log_acc": val_stats['hard_log_acc'],
                })

        if self.use_wandb: wandb.finish()

        return
    
    def save_model(self, method='by_loss'):
        assert method in ['by_loss', 'by_acc', 'by_outliers']
        model_save_dir = f'./model/{method}/{self.run_name}'
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        else:
            [os.remove(f'{model_save_dir}/{filename}') for filename in os.listdir(model_save_dir)]

        # Save configuration dictionary and all files for the model(s)
        with open(f'./model/{method}/{self.run_name}/config.json', 'w') as json_file:
            json.dump(self.config, json_file)
        if not (self.frozen_pretrained and self.pretrained_CNN): 
            torch.save(self.video_encoder.state_dict(), f'{model_save_dir}/video_encoder.pth')
        if self.pretrained_CNN: 
            torch.save(self.video_encoder_head.state_dict(), f'{model_save_dir}/video_encoder_head.pth')
        if self.use_force: 
            torch.save(self.force_encoder.state_dict(), f'{model_save_dir}/force_encoder.pth')
        if self.use_width: 
            torch.save(self.width_encoder.state_dict(), f'{model_save_dir}/width_encoder.pth')
        if self.use_estimation: 
            torch.save(self.estimation_decoder.state_dict(), f'{model_save_dir}/estimation_decoder.pth')
        torch.save(self.decoder.state_dict(), f'{model_save_dir}/decoder.pth')

        # Save performance data
        with open(f'{model_save_dir}/train_object_performance.json', 'w') as json_file:
            json.dump(self.train_object_performance, json_file)
        with open(f'{model_save_dir}/val_object_performance.json', 'w') as json_file:
            json.dump(self.val_object_performance, json_file)
        return
    
    def load_model(self, folder_path):
        with open(f'{folder_path}/config.json', 'r') as file:
            config = json.load(file)
        self._unpack_config(config)

        if not (self.frozen_pretrained and self.pretrained_CNN): 
            self.video_encoder.load_state_dict(torch.load(f'{folder_path}/video_encoder.pth', map_location=self.device))
        if self.pretrained_CNN: 
            self.video_encoder_head.load_state_dict(torch.load(f'{folder_path}/video_encoder_head.pth', map_location=self.device))
        if self.use_force: 
            self.force_encoder.load_state_dict(torch.load(f'{folder_path}/force_encoder.pth', map_location=self.device))
        if self.use_width: 
            self.width_encoder.load_state_dict(torch.load(f'{folder_path}/width_encoder.pth', map_location=self.device))
        if self.use_estimation: 
            self.estimation_decoder.load_state_dict(torch.load(f'{folder_path}/estimation_decoder.pth', map_location=self.device))
        self.decoder.load_state_dict(torch.load(f'{folder_path}/decoder.pth', map_location=self.device))

        with open(f'{folder_path}/train_object_performance.json', 'r') as file:
            train_object_performance = json.load(file)
        with open(f'{folder_path}/val_object_performance.json', 'r') as file:
            val_object_performance = json.load(file)
        self.objects_train = list(train_object_performance.keys())
        self.objects_val = list(val_object_performance.keys())

        # Create data loaders based on original training distinctions
        self._create_data_loaders()
        return

    def make_performance_plot(self):
        material_to_color = {
            'Foam': 'firebrick',
            'Plastic': 'forestgreen',
            'Wood': 'goldenrod',
            'Paper': 'yellow',
            'Glass': 'darkgray',
            'Ceramic': 'pink',
            'Rubber': 'slateblue',
            'Metal': 'royalblue',
            'Food': 'darkorange',
        }
        material_prediction_data = {
            mat : [] for mat in material_to_color.keys()
        }
        material_label_data = {
            mat : [] for mat in material_to_color.keys()
        }

        # Run validation epoch to get out all predictions
        predictions = self._val_epoch(track_predictions=True)
        log_diff_predictions = { key:[] for key in predictions.keys() }

        # Turn predictions into plotting data
        count = 0
        log_acc_count = 0
        for obj in predictions.keys():
            if len(predictions[obj]) == 0: continue
            if obj in self.exclude: continue
            assert obj in self.object_to_material.keys()
            mat = self.object_to_material[obj]
            for E in predictions[obj]:
                log_diff = np.log10(E) - np.log10(self.object_to_modulus[obj])
                log_diff_predictions[obj].append(log_diff)

                if E > 0 and not math.isnan(E):
                    assert not math.isnan(E)

                    if abs(np.log10(E) - np.log10(self.object_to_modulus[obj])) <= 1:
                        log_acc_count += 1

                    material_prediction_data[mat].append(float(E))
                    material_label_data[mat].append(float(self.object_to_modulus[obj]))
                    count += 1

        print(log_diff_predictions)
        print('\n', log_acc_count / count)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.float32):
                    return float(obj)
                return json.JSONEncoder.default(self, obj)
    
        if not os.path.exists(f'./plotting_data/{self.run_name}'):
            os.mkdir(f'./plotting_data/{self.run_name}')
        with open(f'./plotting_data/{self.run_name}/obj_log_diff.json', 'w') as json_file:
            json.dump(log_diff_predictions, json_file, indent=4, sort_keys=True, cls=NumpyEncoder)
        with open(f'./plotting_data/{self.run_name}/predictions.json', 'w') as json_file:
            json.dump(material_prediction_data, json_file, cls=NumpyEncoder)
        with open(f'./plotting_data/{self.run_name}/labels.json', 'w') as json_file:
            json.dump(material_label_data, json_file, cls=NumpyEncoder)
    
        # Create plot
        mpl.rcParams['font.family'] = ['serif']
        mpl.rcParams['font.serif'] = ['Times New Roman']
        plt.figure()
        plt.plot([100, 10**12], [100, 10**12], 'k--', label='_')
        plt.fill_between([100, 10**12], [10**1, 10**11], [10**3, 10**13], color='gray', alpha=0.2)
        plt.xscale('log')
        plt.yscale('log')

        for mat in material_to_color.keys():
            plt.plot(material_label_data[mat], material_prediction_data[mat], '.', markersize=10, color=material_to_color[mat], label=mat)

        plt.xlabel("Ground Truth Modulus ($E$) [$\\frac{N}{m^2}$]", fontsize=12)
        plt.ylabel("Predicted Modulus ($\\tilde{E}$) [$\\frac{N}{m^2}$]", fontsize=12)
        plt.xlim([100, 10**12])
        plt.ylim([100, 10**12])
        plt.title('Neural Network', fontsize=14)

        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.25)
        plt.tick_params(axis='both', which='both', labelsize=10)

        plt.savefig('./figures/nn.png')
        plt.show()
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
        'use_both_sides': False,
        'use_force': True,
        'use_width': True,
        'use_estimation': True,
        'use_transformations': False,
        'use_width_transforms': True,
        'loss_function': 'mse',
        'exclude': [
                    'playdoh', 'silly_puty', 'racquet_ball', 'blue_sponge_dry', # 'blue_sponge_wet', \
                    'blue_foam_brick', 'green_foam_brick', # 'yellow_foam_brick', 'red_foam_brick', 
                    'apple', 'orange', 'strawberry', 'ripe_banana', 'unripe_banana', 'tennis_ball', 
                    'lacrosse_ball', 'cork', 'rubber_spatula', # 'fake_washer_stack',
                    'baseball', 'plastic_measuring_cup', 'whiteboard_eraser', 'cutting_board',
                    'plastic_knife', 'plastic_fork', 'plastic_spoon', 'plastic_fork_white',
                    
                    # Decrease to 200
                    'bowl_small_plastic', 'bowl_big_plastic', 'bowl_ceramic', 'plate_small', 'plate_big',
                    'gel', 'gel_big', 'gel_double_wide', 'wooden_spoon', 'metal_fork', 'metal_spoon', 'metal_knife',
                    'key_ring', 'ring', 'white_bottle_cap', 'blue_bottle_cap', 'red_foam_brick', 'buckle', 'peeler',
                    'insole', 'pi_usb_cable', 'hdmi_adapter', 'mechanical_pencil', 'red_electrical_piece',
                    'heat_insert', 'iphone_brick', 'rubber_band', 'rubber_band_bundle', 'molded_rectangle', 'molded_cylinder_wide',
                    'motorcycle_eraser', 'tennis_ball', 'mousepad', 'charger_cable', 'power_cable', 'wooden_sheet', 'chopstick', 
                    'orange_elastic_ball', 'rubber_pancake', 'magnetic_whiteboard_eraser', 'paper_towel_bundle', 'half_rose_eraser',
                    'fake_half_rose', 'half_bumpy_ball_eraser', 'golf_ball', 'watermelon_eraser', 'strawberry_eraser',
                    'lion_eraser', 'crab_eraser', 'zebra_eraser', 'fox_eraser', 'bear_eraser', 'bee_eraser', 'banana_eraser', 'frog_eraser',
                    'scotch_brite', 'lifesaver_hard', 'blue_sponge_wet', 'fake_washer_stack'

                    # 'bowl_small_plastic', 'bowl_big_plastic', 'bowl_ceramic', 'plate_small', 'plate_big',
                    # 'wooden_spoon', 'metal_spoon', 'metal_knife',
                    # 'red_foam_brick', 'molded_rectangle', 'molded_cylinder_wide', 'wooden_sheet',
                    # 'mousepad', 'chopstick', 'rubber_band_bundle',  
                ],

        # Logging on/off
        'use_wandb': True,
        'run_name': 'NoPretrain_NoFW_NoTransforms_ExcludeTo200',

        # Training and model parameters
        'epochs'            : 70,
        'batch_size'        : 32,
        'pretrained_CNN'    : False,
        'frozen_pretrained' : False,
        'img_feature_size'  : 128,
        'fwe_feature_size'  : 32,
        'val_pct'           : 0.175,
        'dropout_pct'       : 0.3,
        'learning_rate'     : 3e-6, # 5e-6, # 1e-5,
        'gamma'             : 0.975,
        'lr_step_size'      : 1,
        'random_state'      : 27,
        'decoder_size'      : [512, 512, 128, 16],
        'est_decoder_size'  : [64, 64, 32],
    }
    assert config['img_style'] in ['diff', 'depth']
    assert config['loss_function'] in ['mse', 'log_diff']
    config['n_channels'] = 3 if config['img_style'] == 'diff' else 1

    if config['frozen_pretrained'] and not config['pretrained_CNN']:
        raise ValueError('Frozen option is only necessary when training with a pretrained CNN.')
    
    ARCHITECTURES = {
        # 'Base':         ([256, 256, 64], [64, 64, 32]),
        # 'FatDecoder':   ([512, 512, 128], [64, 64, 32]),
        # 'FatEst':       ([256, 256, 64], [128, 128, 64]),
        # 'FatBoth':      ([512, 512, 128], [128, 128, 64]),
        # 'LeanBoth':     ([128, 128, 32], [32, 32, 16]),
        # 'DeepBase':     ([256, 256, 64, 64], [64, 64, 32]),
        # 'DeepFat':      ([512, 512, 128, 128], [128, 128, 64]),
        # 'DeepLean':     ([128, 128, 64, 32], [32, 32, 16]),
    }

    # Train the model over some data
    base_run_name = config['run_name']
    chosen_random_states = [27, 60, 24] # , 16, 12] # [27, 60, 74, 24, 16, 12, 4, 8]

    for lr in ['5e-6', '3e-6']:
        for i in range(len(chosen_random_states)):
            config['run_name'] = f'LR={lr}_{base_run_name}__t={i}'
            config['learning_rate'] = float(lr)
            
            if i < len(chosen_random_states):
                config['random_state'] = chosen_random_states[i]
            else:
                config['random_state'] = random.randint(1, 100)

            train_modulus = ModulusModel(config, device=device)
            train_modulus.train()

    # for run_name in ['Layer4Decoder_Normalized_ExcludeTo200__t=0']:
    #     train_modulus = ModulusModel(config, device=device)
    #     train_modulus.load_model(f'./model/by_loss/{run_name}')
    #     train_modulus.make_performance_plot()