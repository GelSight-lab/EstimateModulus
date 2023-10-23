import cv2
import os
import numpy as np
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

TRAIN   = True
AVI_DIR = "./hardness_dataset/robotData"

N_FRAME_SAMPLES  = 5
LSTM_INPUT_SIZE  = 1000 # Feature length per frame
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS  = 2
NUM_CLASSES      = 1
BATCH_SIZE       = 32

print('Loading in data... \t\t', datetime.now().strftime("%H:%M:%S %m/%d/%Y"))

video_files = os.listdir(AVI_DIR)[:50]
labels = []

# Unpack data
for i in tqdm(range(len(video_files))):
    avi_file = video_files[i]
    if avi_file.endswith(".avi"):
        # Add labels based on your dataset structure
        hardness = torch.zeros([1])
        hardness[0] = int(avi_file.split('_')[1])
        # hardness = hardness.to(device)
        labels.append(hardness)
        del hardness

# Define a preprocessing transform to match CNN's requirements
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # Normalization mean values for ImageNet
        std=[0.229, 0.224, 0.225]       # Normalization standard deviation values
    ),
])

class CustomDataset(Dataset):
    def __init__(self, video_files, labels, transform=None, frame_sequence=torch.zeros((N_FRAME_SAMPLES, 3, 360, 480))):
        self.video_files = video_files
        self.labels = labels

        self.video_frames = []
        self.intensity = []
        self.cap = None
        self.ret, self.frame = None, None
        self.sampled_frames = None

        self.frames = torch.zeros((5, 360, 480, 3))
        self.transform = transform
        self.frame_sequence = frame_sequence

    def __len__(self):
        return len(self.labels)
    
    def extract_frames_from_path(self, avi_path):
        self.cap = cv2.VideoCapture(avi_path)
        self.video_frames = []
        self.intensity = []

        while True:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break
            self.frame = cv2.resize(self.frame, (self.frame.shape[1]//2, self.frame.shape[0]//2))
            self.video_frames.append(np.array(self.frame))
            self.intensity.append(np.mean(self.frame))
            del self.ret, self.frame

        # Choose frames based on intensity (as proxy for force)
        self.intensity = np.array(self.intensity)
        self.sampled_frames = [torch.from_numpy(self.video_frames[i]) for i in np.linspace(np.argmax(self.intensity > np.mean(self.intensity)), np.where(self.intensity > np.mean(self.intensity))[0][-1], num=N_FRAME_SAMPLES, dtype=int).tolist()]
        
        self.frames = []
        self.intensity = []

        self.cap.release()
        cv2.destroyAllWindows()
        return self.sampled_frames

    def __getitem__(self, idx):
        self.frames = torch.stack(self.extract_frames_from_path(os.path.join(AVI_DIR, self.video_files[idx])), dim=0)
        self.frame_sequence = self.frame_sequence.zero_()
        for i in range(self.frames.shape[0]):
            self.frame_sequence[i,:,:,:] = preprocess(self.frames[i,:,:,:].numpy())
        self.frames = None
        return self.frame_sequence, self.labels[idx]

X_train, X_val, y_train, y_val = train_test_split(video_files, labels, test_size=0.2, random_state=42)
del video_files, labels

kwargs = {'num_workers': 0, 'pin_memory': True}
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = models.mobilenet_v2(pretrained=True).features

    def forward(self, x):
        return self.features(x)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.out = None

    def forward(self, x):
        self.out, _ = self.lstm(x)
        self.out = self.fc(self.out[:, -1, :])
        return self.out

class VideoProcessingModel(nn.Module):
    def __init__(self, cnn, lstm):
        super(VideoProcessingModel, self).__init__()
        self.cnn = cnn
        self.lstm = lstm
        self.features_sequence = torch.zeros((BATCH_SIZE, N_FRAME_SAMPLES, 1000))

    def forward(self, x):
        self.features_sequence = torch.zeros((BATCH_SIZE, N_FRAME_SAMPLES, 1000))  # To store features for each frame
        for i in range(N_FRAME_SAMPLES):
            self.features_sequence[:, i, :] = self.cnn(x[:, i, :, :, :])
        del x

        # Stack the features along batch dim and pass through LSTM
        return self.lstm(self.features_sequence)

CNN = CNNModel()
LSTM = LSTMModel(LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, NUM_CLASSES)
model = VideoProcessingModel(CNN, LSTM)
model.to(device)

# from torchinfo import summary
# summary(model, input_data=torch.zeros((1, 1, 3, 360, 480), device=device))

def train(_model):
    criterion = nn.HuberLoss()
    optimizer = optim.SGD(_model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    _model.train()

    outputs = None
    loss = None

    num_epochs = 3 # 100
    for epoch in range(num_epochs):

        for frame_tensor, labels in train_loader:
            optimizer.zero_grad()

            outputs = _model(frame_tensor.to(device))
            loss = criterion(outputs.squeeze(1), labels.squeeze(1).to(device))
            loss.backward()
            optimizer.step()

            del frame_tensor, labels, outputs

        scheduler.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    return _model

if TRAIN == True:
    print('Training... \t\t\t\t', datetime.now().strftime("%H:%M:%S %m/%d/%Y"))
    model = train(model)
    torch.save(model.state_dict(), './model.pth')
    print('Done. \t\t\t\t', datetime.now().strftime("%H:%M:%S %m/%d/%Y"))
else:
    model.load_state_dict(torch.load('./lstm_model.pth'))
model.eval()

del X_train, y_train, train_dataset, train_loader 

val_dataset = CustomDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, **kwargs)

correct = 0
error = 0
total = 0

with torch.no_grad():
    for frame_tensor, labels in val_loader:
        outputs = model(frame_tensor)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        error += torch.abs(predicted - labels.squeeze(1)).sum().item()
        correct += (predicted == labels.squeeze(1)).sum().item()

print(f"Validation Accuracy: ~{int(100 * correct / total)}%")
print(f"Avg. Validation Error: {error / total}")
with open('validation_accuracy.txt', 'w') as f:
    f.write(str(correct / total))
with open('avg_validation_error.txt', 'w') as f:
    f.write(str(error / total))