import torch.nn as nn
import torch.nn.functional as F

from deep_feat_extraction import feat_extraction_layer

class DeepVCP(nn.Module):
    def __init__(self): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer()
    
    def forward(self, pts):
        output_xyz, output_pts = self.FE1(pts)

        return output_xyz, output_pts