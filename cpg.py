import torch
import torch.nn.functional as F
import os
import numpy as np
from torch import nn
from utils import *

''' 
    Learn similarity in DFE between source/transformed points
    Select best candidate for each source keypoint to get virtual corresponding points
    in the transformed point cloud 

    @param  source keypoints DFE: (B x N x 1 x 32)
            transformed keypoints DFE: (B x N x 32 x C)
            candidates: B x N x C x 3
    @return target  (B x N x 3)
'''
class cpg(nn.Module):
    def __init__(self):
        super(cpg, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
     
    def forward(self, source, transformed, candidates):
        print(f"Source: {source}")
        print(f"Transformed: {transformed}")

        # measure similarity of each source keypoint with its 
        # possible corresponding candidates using dot prod on 32-dim FE descriptor
        # (B x N x 1 x 32) @ (B x N x 32 x C) = (B x N x 1 x C)
        similarity = torch.matmul(source, transformed)   # B x N x 1 x C 
        # convert to probability
        w = self.softmax(similarity)                     # B x N x 1 x C 
        w = w.permute(0,1,3,2)                           # B x N x C x 1
   
        # weighted sum (multiply scalar weight across candidate xyz)
        wy = torch.mul(w, candidates)                    # B x N x C x 3
        # sum all candidates for each point
        wy_sum = torch.sum(wy, -2)                       # B x N x 3
        
        # weighted sum over candidates to get virtual corresponding points
        vcp = torch.mul(1/(torch.sum(w, -2)), wy_sum)    # B x N x 3 
        print('Virtual corresponding points', vcp.shape)
        return vcp 

net = cpg()
B = 2
N = 64
C = 50
feature_dim = 32
out = torch.manual_seed(0)

# randomize
source = torch.randn(B,N,1,feature_dim)       # FE descriptor for source keypoints
transformed = torch.randn(B,N,feature_dim,C)  # FE descriptor for transformed keypoint's candidates
candidates = torch.randn(B,N,C,3)             # candidates xyz 
# run cpg to get vcp
vcp = net(source, transformed, candidates)
