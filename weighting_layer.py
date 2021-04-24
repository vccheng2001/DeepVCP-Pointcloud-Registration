import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU, Softplus
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.linear import Linear

class WeightingLayer(nn.Module):
    def __init__(self):
        super(WeightingLayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32, 16, True)
            nn.BatchNorm1d(16)
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 8, True)
            nn.BatchNorm1d(8)
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(8, 1, True)
            nn.Softplus()
        )
    
    def forward(self, X, K = 512):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)

        return torch.topk(X, K, dim = 2)
