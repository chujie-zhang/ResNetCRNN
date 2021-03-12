import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

## ---------------------- Dataloaders ---------------------- ##

# for cnn and rnn
class process_dataset(data.Dataset):

    def __init__(self, data_path, folders, labels, frames, transform=None):
        
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        
        X = []      
        numberOfFrames=os.listdir(path+selected_folder)
        n=0
        for num in numberOfFrames:
            n+=1
        
        selected_frames=np.arange(n//2-11, n//2+13, 1).tolist()
        #print(selected_frames)
        for i in selected_frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:d}.png'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)
        
        

        return X

    def __getitem__(self, index):
        
        # get one sample
        folder = self.folders[index]

        X = self.read_images(self.data_path, folder, self.transform)     
        y = torch.LongTensor([self.labels[index]])               
        return X, y



## ------------------------ resnet and lstm module ---------------------- #

# resnet for 2d cnn
class ResNetEncoder(nn.Module):
    def __init__(self, hidden1=512, hidden2=512, drop_out=0.3, CNN_DIM=300):
        super(ResNetEncoder, self).__init__()

        self.hidden1, self.hidden2 = hidden1, hidden2
        self.drop_out = drop_out

        resnet = models.resnet152(pretrained=True)
        #resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]     
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1, momentum=0.01)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2, momentum=0.01)
        self.fc3 = nn.Linear(hidden2, CNN_DIM)
        
    def forward(self, x_3d):
        CNN_SEQUENCE = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)            

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_out, training=self.training)
            x = self.fc3(x)

            CNN_SEQUENCE.append(x)

        
        CNN_SEQUENCE = torch.stack(CNN_SEQUENCE, dim=0).transpose_(0, 1)
        

        return CNN_SEQUENCE


class RNNDecoder(nn.Module):
    def __init__(self, CNN_DIM=300, RNNLayers=3, h_RNN=256, h_Dim=128, drop_out=0.3, num_classes=50):
        super(RNNDecoder, self).__init__()

        self.RNN_input = CNN_DIM
        self.RNNLayers = RNNLayers   
        self.h_RNN = h_RNN                 
        self.h_Dim = h_Dim
        self.drop_out = drop_out
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input,
            hidden_size=self.h_RNN,        
            num_layers=RNNLayers,       
            batch_first=True,       
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_Dim)
        self.fc2 = nn.Linear(self.h_Dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
    

       
        x = self.fc1(RNN_out[:, -1, :])   
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.fc2(x)

        return x

