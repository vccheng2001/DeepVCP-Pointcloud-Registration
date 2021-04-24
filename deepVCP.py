import torch.nn as nn
import torch.nn.functional as F

from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer

class DeepVCP(nn.Module):
    def __init__(self): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer()
        self.WL = weighting_layer()
    
    def forward(self, pts):
        deep_feat_xyz, deep_feat_pts = self.FE1(pts)
        keypts = self.WL(deep_feat_pts)

        return keypts