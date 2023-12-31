import cv2
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

'''
Gelsight Mini Hardness Model
- Implement and train a machine learning model to replicate the results of W. Yuan et al. hardness paper.
- Uses data from Gelsight Mini recorded over human controlled press on various shapes and materials.
- Architecture has been updated to encoder-decoder from original paper's CNN-LSTM.

Dataset is publicly available at http://people.csail.mit.edu/yuan_wz/hardnessdataset/
Paper is available at https://arxiv.org/abs/1704.03955
'''

# TODO: Make this whole thing object oriented

TRAIN = True
USE_WANDB = True
AVI_DIR = "./hardness_dataset/robotData"

N_FRAMES        = 5
IMG_X, IMG_Y    = 180, 240

batch_size      = 32
feature_size    = 512
epochs          = 250
learning_rate   = 1e-5
validation_pct  = 0.2
random_state    = 42

# Read data folder
video_files = sorted(os.listdir(AVI_DIR)) # [:50]
hardness = [int(avi_file.split('_')[1]) for avi_file in video_files]

assert len(video_files) == len(hardness)

def conv2D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
	return output_shape


def conv3D_output_size(img_size, padding, kernel_size, stride):
	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
				  np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
	return output_shape 

# # Define a preprocessing transform to match CNN's requirements
# preprocess = transforms.Compose([
#     transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], # Normalization mean values for ImageNet
#         std=[0.229, 0.224, 0.225]   # Normalization standard deviation values
#     ),
# ])

if USE_WANDB:
    wandb.init(
        # Set the wandb project where this run will be logged
        project="MINI_HARDNESS",
        
        # Track hyperparameters and run metadata
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "N_frames": N_FRAMES,
            "img_size": (IMG_X, IMG_Y),
            "feature_size": feature_size,
            "learning_rate": learning_rate,
            "validation_pct": validation_pct,
            "random_state": random_state,
            "architecture": "ENCODE_DECODE",
            "optimizer": "Adam",
            "loss": "MSE",
            "scheduler": "None",
        }
    )

class CustomDataset(Dataset):
    def __init__(self, video_files, labels, frame_tensor=torch.zeros((N_FRAMES, 3, IMG_X, IMG_Y)), label_tensor=torch.zeros((1))):
        self.video_files = video_files
        self.labels = labels
        self.intensity = []
        self.indices = []
        self.i = 0
        self.cap = None
        self.ret, self.frame = None, None
        self.sampled_frames = None

        self.frame_tensor = frame_tensor
        self.label_tensor = label_tensor
        self.index_cache = {}

    def __len__(self):
        return len(self.labels)
    
    def extract_frames_from_path(self, avi_path):
        self.cap = cv2.VideoCapture(avi_path)
        self.sampled_frames = []
        self.intensity = []

        if avi_path in self.index_cache:
            self.indices = self.index_cache[avi_path]
        else:
            while True:
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    break
                self.intensity.append(np.mean(self.frame))
                self.ret, self.frame = None, None
            self.cap.release()
            cv2.destroyAllWindows()

            # Choose frames based on intensity (as proxy for force)
            self.intensity = np.array(self.intensity)
            self.indices = np.linspace(np.argmax(self.intensity > np.mean(self.intensity)), np.where(self.intensity > np.mean(self.intensity))[0][-1], num=N_FRAMES, dtype=int).tolist()
            self.index_cache[avi_path] = self.indices
            self.intensity = []

        # Get frames
        self.i = 0
        self.cap = cv2.VideoCapture(avi_path)
        while True:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break
            self.unsized_frame = self.frame
            self.frame = cv2.resize(self.frame, (self.frame.shape[1]//4, self.frame.shape[0]//4))
            if self.i in self.indices:
                self.sampled_frames.append(np.array(self.frame))
            self.ret, self.frame = None, None
            self.i += 1

        assert len(self.sampled_frames) == N_FRAMES
        self.sampled_frames = np.array(self.sampled_frames)        

        # Clear data
        self.indices = []; self.i = None
        self.cap.release()
        cv2.destroyAllWindows()

        return self.sampled_frames

    def __getitem__(self, idx):
        self.frame_tensor = self.frame_tensor.zero_()
        # TODO: Optimize memory here
        self.extracted_frames = self.extract_frames_from_path(os.path.join(AVI_DIR, self.video_files[idx]))
        for i in range(self.frame_tensor.shape[0]):
            self.frame_tensor[i,:,:,:] = torch.movedim(torch.Tensor(self.extracted_frames[i,:,:,:] / 255), 2, 0)

        self.label_tensor = self.label_tensor.zero_()
        self.label_tensor[0] = self.labels[idx]

        return self.frame_tensor.clone(), self.label_tensor.clone()

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
    
encoder = EncoderCNN(img_x=IMG_X, img_y=IMG_Y, input_channels=3,
                        CNN_embed_dim=feature_size)
encoder = encoder.to(device)

decoder = DecoderFC(CNN_embed_dim=feature_size, output_dim=1)
decoder = decoder.to(device)

# # Parallelize model to multiple GPUs
# if torch.cuda.device_count() > 0:
#     print("Using", torch.cuda.device_count(), "GPUs!")
#     if torch.cuda.device_count() > 1:
#         encoder = nn.DataParallel(encoder)
#         decoder = nn.DataParallel(decoder)

X_train, X_val, y_train, y_val = train_test_split(video_files, hardness, test_size=validation_pct, random_state=random_state)

kwargs = {'num_workers': 0, 'pin_memory': False, 'drop_last': True}
train_dataset = CustomDataset(X_train, y_train, frame_tensor=torch.zeros((N_FRAMES, 3, IMG_X, IMG_Y), device=device), label_tensor=torch.zeros((1), device=device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
val_dataset = CustomDataset(X_val, y_val, frame_tensor=torch.zeros((N_FRAMES, 3, IMG_X, IMG_Y), device=device), label_tensor=torch.zeros((1), device=device))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

def train(_encoder, _decoder):
    criterion = nn.MSELoss() # nn.HuberLoss()
    params = list(_encoder.parameters()) + list(_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if USE_WANDB: wandb.init()
    
    features = []
    outputs = torch.zeros((batch_size), device=device)

    loss = None
    min_val_loss = 1e10
    train_loss_over_epoch, train_batch_count = 0, 0
    val_loss_over_epoch, val_batch_count = 0, 0
    
    memory_allocated = torch.cuda.memory_allocated()
    memory_cached = torch.cuda.memory_reserved()
    if USE_WANDB: wandb.log({"memory_allocated": memory_allocated, "memory_reserved": memory_cached})

    for epoch in range(epochs):

        # TRAIN EPOCH

        train_loss_over_epoch, train_batch_count = 0, 0
        _encoder.train(); _decoder.train()

        for x, labels in train_loader:
            optimizer.zero_grad()

            features = []
            for i in range(N_FRAMES):
                features.append(_encoder(x[:, i, :, :, :].clone()))
            outputs = _decoder(features)

            loss = criterion(outputs.squeeze(1), labels.squeeze(1))
            loss.backward()
            optimizer.step()

            train_loss_over_epoch += loss.item()
            train_batch_count += 1

        # TODO: Validate during training

        # VALIDATE

        val_loss_over_epoch, val_num_correct, val_batch_count = 0, 0, 0
        _encoder.eval(); _decoder.eval()

        for x, labels in val_loader:

            features = []
            for i in range(N_FRAMES):
                features.append(_encoder(x[:, i, :, :, :].clone()))
            outputs = _decoder(features)
            loss = criterion(outputs.squeeze(1), labels.squeeze(1))
            
            val_loss_over_epoch += loss.item()
            val_num_correct += (torch.round(outputs) == torch.round(labels)).sum().item()
            val_batch_count += 1

        print(f'Epoch: {epoch}, Training Loss: {train_loss_over_epoch / train_batch_count:.4f},',
              f'Validation Loss: {val_loss_over_epoch / val_batch_count:.4f},', 
              f'Validation Accuracy: {val_num_correct / val_batch_count:.4f}',
            )

        # Save model with the best validation loss over training
        if val_loss_over_epoch / val_batch_count < min_val_loss:
            min_val_loss = val_loss_over_epoch / val_batch_count
            torch.save(_encoder.state_dict(), './encoder.pth')
            torch.save(_decoder.state_dict(), './decoder.pth')

        memory_allocated = torch.cuda.memory_allocated()
        memory_cached = torch.cuda.memory_reserved()

        # Log memory usage to WandB
        if USE_WANDB: wandb.log({
            "memory_allocated": memory_allocated,
            "memory_reserved": memory_cached,
            "train_loss": (train_loss_over_epoch / train_batch_count),
            "val_loss": (val_loss_over_epoch / val_batch_count),
            "val_accuracy": (val_num_correct / (batch_size*val_batch_count)),
            "epoch": epoch,
        })
    
    if USE_WANDB: wandb.finish()

    return _encoder, _decoder


if TRAIN == True:
    print('Training... \t\t\t\t', datetime.now().strftime("%H:%M:%S %m/%d/%Y"))
    encoder, decoder = train(encoder, decoder)
    print('Done. \t\t\t\t', datetime.now().strftime("%H:%M:%S %m/%d/%Y"))