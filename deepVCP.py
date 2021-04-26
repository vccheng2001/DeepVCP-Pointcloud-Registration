import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import sample_and_group
from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer

class DeepVCP(nn.Module):
    def __init__(self): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer()
        self.WL = weighting_layer()
    
    def forward(self, src_pts, tgt_pts):
        src_deep_feat_xyz, src_deep_feat_pts = self.FE1(src_pts)
        src_keypts_idx = self.WL(src_deep_feat_pts)
        src_keypts = src_pts[:, :, src_keypts_idx]
        src_keypts = src_keypts.permute(0, 2, 1)

        src_keypts_grouped_xyz, src_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = src_keypts[:, :, :3], points = src_keypts[:, :, 3:])
        
        return src_keypts