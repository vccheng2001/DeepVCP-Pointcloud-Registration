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
        self.conv1 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1,)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        # softmax 
        self.softmax = nn.Softmax(dim=-1)
     
    # src_dfe_feat:  # (B, K_topk, 1, 32)
    # tgt_dfe_feat:  # (B, K_topk, 32, C)
    def forward(self, src_dfe_feat, tgt_dfe_feat, candidates, r, s):
        B, N, C, _ = candidates.shape
        grid_size = int((2*r)/s + 1 )
        assert C == grid_size * grid_size * grid_size
        # B x N x GS x GS x GS x 32
        src_dfe_feat_cost_volume = src_dfe_feat.reshape(B, N, 1, 1, 1, 32).repeat(1, 1, grid_size, grid_size, grid_size, 1)
        # B x N x GS x GS x GS x 32 
        tgt_dfe_feat_cost_volume = tgt_dfe_feat.reshape(B, N,grid_size, grid_size, grid_size, 32)
        # B x N x GS x GS x GS x 32
        cost_volume = torch.square(src_dfe_feat_cost_volume - tgt_dfe_feat_cost_volume)
        assert cost_volume.shape == (B, N, grid_size, grid_size, grid_size, 32)
        
        # permute -> B x N x 32 x GS x GS x GS
        x = cost_volume.permute(0, 1, 5, 2, 3, 4)
        # compress B,N -> BN x 32 x GS x GS x GS
        x = x.flatten(start_dim=0,end_dim=1)

        # Conv3D: 32 -> 16 -> 4 -> 1 channel
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # output of conv3D: BN x 32 x grid_size x grid_size x grid_size

        # reshape to B x N x C 
        x = x.reshape(B, N, C)
        # softmax over weights: B x N x C 
        weights = self.softmax(x)
        # B x N x C x 3 (repeat weight for each coordinate xyz)
        weights = weights.unsqueeze(-1).repeat(1,1,1,3)
        # weights,  candidates: B x N x C x 3 
        vcp = torch.sum(torch.mul(weights,candidates), -2) # sum over candidates
        vcp /= torch.sum(weights, -2)
        # vcp: B x N x 3

        print('returned vcp from cpg', vcp.shape)
        return vcp

if __name__ == "__main__":
    net = cpg()
    B = 3
    N = 64
    C = 216
    feature_dim = 32
    out = torch.manual_seed(0)

    # randomize
    source = torch.randn(B,N,1,feature_dim)       # FE descriptor for source keypoints
    transformed = torch.randn(B,N,feature_dim,C)  # FE descriptor for transformed keypoint's candidates
    candidates = torch.randn(B,N,C,3)             # candidates xyz 
    # run cpg to get vcp
    r = 1 
    s = 0.4


    vcp = net(source, transformed, candidates, r, s)



