import torch
import torch.nn.functional as F
import os
import numpy as np
from torch import nn
from utils import *

''' 3D convolution for CPG layer tolearns similarities between 
    target/source features, and suppresses noise 
    @param input: (B x 1 x N x C x 32)
    @return out: weights (B x 1 x N x C x 32) 
'''
class CNN3D(nn.Module):

    def __init__(self):
        super(CNN3D, self).__init__()
        # three 3D conv
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3,stride=1,padding=1,)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        # softmax 
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = self.softmax(x)
        return out


''' 
    Corresponding point generation layer
    Uses 3DCNN + Softmax to pick best candidate for each point 
    @param inp:     input to 3DCNN       (B x 1 x N x C x 32)
           yprime:  candidate points      (B x N x C x 32)
           y:       points to select from (B x N x C x 3)
    @return out: B x N x 3 (virtual corresponding points)
'''
def cpg_layer(inp, yprime, y):
    net = CNN3D()
    # output of softmax 
    pred = net(inp).squeeze()
    # weighted sum to get probabilities: B x N x C
    yi = torch.sum(torch.mul(pred,yprime), -1) 
    yi /= torch.sum(pred, -1)
    # get index of largest weighted candidate: B x N 
    idx = torch.argmax(yi,dim = -1)
    # B N 1 1, repeat last dim 3 times to gather x,y,z
    idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)
    # index into C axis to get best candidates for each point
    out = torch.gather(y, -2, idx).squeeze(-2)
    # B x N x 3
    return out 


net = CNN3D()
B = 2
N = 3
C = 4
torch.manual_seed(0)
# input to 3DCNN: (num_batch, channel_in, depth, height, width)
inp = torch.randn(B,1,N,C,32)
# candidate corresponding points to dot prod with weight 
yprime = torch.randn(B,N,C,32)
# points to select from 
y = torch.randn(B,N,C,3) 
# output: B x N x 3 
out = cpg_layer(inp, yprime, y)
print(f'Output: {out.shape}')