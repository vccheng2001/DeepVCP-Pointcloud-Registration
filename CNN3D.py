import torch
import torch.nn.functional as F
import os
import numpy as np
from torch import nn
from utils import *

class CNN3D(nn.Module):

    def __init__(self):
        super(CNN3D, self).__init__()
        # three 3D conv
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=3, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=1)
        # softmax 
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.softmax(x)

net = CNN3D()
print(net)