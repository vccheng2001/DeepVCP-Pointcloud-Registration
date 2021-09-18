import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

class feat_extraction_layer(nn.Module):
    def __init__(self, use_normal = True):
        super(feat_extraction_layer, self).__init__()
        in_channel = 6 if use_normal else 3
        self.use_normal = use_normal
        self.sa1 = PointNetSetAbstraction(npoint = 10000, # num subsampled pts after FPS
                                          radius = 0.1,   # search radius in ball
                                          nsample = 4096, # num points in local region
                                          in_channel = in_channel, # num in_channels
                                          mlp = [16, 32], # output size of each MLP layer
                                          group_all = False)
        self.sa2 = PointNetSetAbstraction(npoint=10000,
                                          radius=0.2,
                                          nsample=1024,
                                          in_channel = 32,
                                          mlp=[32, 64],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=10000,
                                          radius=0.4,
                                          nsample=256,
                                          in_channel=64,
                                          mlp=[64, 64],
                                          group_all=False)
        self.fc = nn.Linear(64, 32)
        self.dropoout = nn.Dropout(0.3) # keeping prob: 0.7

    def forward(self, pts):
        B, _, _, = pts.shape
        if self.use_normal:
            normal = pts[:, 3:, :]
            xyz = pts[:, :3, :]
        else:
            normal = None
            xyz = pts
        
        output_xyz, output_pts = self.sa1(xyz, normal)
        output_xyz, output_pts = self.sa2(output_xyz, normal)
        output_xyz, output_pts = self.sa3(output_xyz, normal)
        output_xyz = output_xyz.permute(0, 2, 1)
        output_pts = output_pts.permute(0, 2, 1)

        return output_xyz, output_pts
