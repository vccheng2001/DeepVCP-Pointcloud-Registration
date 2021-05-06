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
    def __init__(self, K_nsample = 32):
        super(feat_embedding_layer, self).__init__()
        # three FC layers + maxpool to obtain feature descriptor 
        self.K_nsample = 32
        self.fc1 = nn.Linear(35, 32, True)
        self.fc2 = nn.Linear(32, 32, True)
        self.fc3 = nn.Linear(32, 32, True)
        self.max_pool = nn.MaxPool1d(kernel_size = self.K_nsample)
    
    def forward(self, X, src = True):
        '''
        If it's src (B x N x K x out_channel), pool along the 3rd dim
        If it's tgt (B x N x C x K x out_channel), pool along the 4th dim  
        '''
        out_channel = 32
        X = X.float()
        if src:
            B, N, K, _ = X.shape
            X = self.fc1(X)
            X = self.fc2(X)
            X = self.fc3(X)

            # transposing X: (B x N x out_channel x K)
            # flatten X: (B x (N x out_channel) x K)
            # pooling: (B x (N x out_channel) x 1)
            # reshape: (B x N x out_channel x 1)
            # squeeze: (B x N x out_channel)
            X = X.permute(0, 1, 3, 2)
            X = torch.flatten(X, start_dim = 1, end_dim = 2)
            X = self.max_pool(X)
            X = X.view(B, N, out_channel, 1).squeeze(3)

        else:
            B, N, C, K, _ = X.shape
            X = self.fc1(X)
            X = self.fc2(X)
            X = self.fc3(X)

            # transposing X: (B x N x C x out_channel x K)
            # flatten X: (B x (N x C x out_channel) x K)
            # pooling: (B x (N x C x out_channel) x 1)
            # reshape: (B x N x C x out_channel x 1)
            # squeeze: (B x N x out_channel)
            X = X.permute(0, 1, 2, 4, 3)
            X = torch.flatten(X, start_dim = 1, end_dim = 3)
            X = self.max_pool(X)
            X = X.view(B, N, C, out_channel, 1).squeeze(4)
        return X
