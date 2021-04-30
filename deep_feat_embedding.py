import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

'''
1) For each of N keypoints, collect K neighboring points within radius d
    if < K, duplicate them 
2) Normalize local coordinates by searching radius d
Input: NxKx36
Output Nx32
'''

class feat_embedding_layer(nn.Module):
    def __init__(self, use_normal = True):
        super(feat_embedding_layer, self).__init__()
        # three FC layers + maxpool to obtain feature descriptor 
        self.fc1 = nn.Linear(32, 32, True)
        self.fc2 = nn.Linear(32, 32, True)
        self.fc3 = nn.Linear(32, 32, True)
        self.max_pool = nn.MaxPool2d(kernel_size=4)  
    
    def forward(self, X):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.max_pool(X)
        return X
