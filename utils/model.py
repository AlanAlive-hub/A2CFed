#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import torch
from torch import nn
import torch.nn.functional as F


class FedAvgMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
        self.act = nn.ReLU(True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

# McMahan et al., 2016; 1,663,370 parameters
class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()

        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(32 * 2) * (7 * 7), out_features=256, bias=False)
        self.fc2 = nn.Linear(in_features=256, out_features=10, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = F.log_softmax(self.fc2(x),dim=1)
        return x


class CNNCifar(nn.Module):
    def __init__(self, class_num = 10):
        super(CNNCifar, self).__init__()
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        # self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(in_features=(32 * 2) * (8 * 8), out_features=394, bias=False)
        self.fc2 = nn.Linear(in_features=394, out_features=192, bias=False)
        self.out = nn.Linear(in_features=192, out_features=class_num, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
    
        x = self.activation(self.fc1(x))
        # x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        # x = self.dropout2(x)
        
        return self.out(x)

    
class CNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=False),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=False),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 394), 
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(394, 192), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out

# ====================================================================================================================
    

class CNNFemnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.out(x)


    
    
def getModel(args,device):
    
    model = None
    
    if "MNIST" in args.dataset:
        model = FedAvgMLP().to(device)
    elif args.dataset == "CIFAR10":
        model = CNN(3, 10, 1600).to(device)
    elif args.dataset == "CIFAR100":
        model = CNN(3, 100, 1600).to(device)


    return model