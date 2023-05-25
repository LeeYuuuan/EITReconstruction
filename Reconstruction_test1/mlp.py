import scipy.io as scio
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 576)
        )
        
    def forward(self, x):
        x = self.flatten()
        x = nn.functional.normalize(x)
        x = self.linear_relu_stack(x)

