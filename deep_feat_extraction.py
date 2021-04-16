import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

class feat_extraction_layer(nn.Module):
    def __init__(self, use_normal = True):
        super(feat_extraction_layer, self).__init__()
        in_channel = 6 if use_normal else 3
        self.use_normal = use_normal
        self.sa1 = PointNetSetAbstraction(npoint = 512, radius = 0.1, nsample = 16, in_channel = in_channel, mlp = [16, 16, 32], group_all = False)

    def forward(self, pts):
        B, _, _, = pts.shape
        if self.use_normal:
            normal = pts[:, 3:, :]
            xyz = pts[:, :3, :]
        else:
            normal = None
        output_xyz, output_pts = self.sa1(xyz, normal)
        
        return output_xyz, output_pts