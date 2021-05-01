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
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3,stride=1,padding=1,)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        # softmax 
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.softmax(x)
        return x 

net = CNN3D()
print(net)
B = 10 
N = 100
C = 20
test = torch.randn(B,1,N,C,32)
pred = net(test)
pred = pred.squeeze()

yprime = torch.randn(B,N,C,32)

yi = torch.mul(pred,yprime)
sumproduct = torch.sum(yi,-1)
sumweight = torch.sum(pred,-1)
yi = sumproduct/sumweight
idx = torch.argmax(yi,dim = -1)
print(yi.shape)
yC = torch.randn(B,N,C,3)

out = torch.index_select(yC,2,idx)
print(out.shape)

test = torch.randn([N,C,32])

