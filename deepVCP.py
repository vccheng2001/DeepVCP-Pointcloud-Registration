import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

from pointnet2_utils import sample_and_group
from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer
from voxelize import voxelize

class DeepVCP(nn.Module):
    def __init__(self): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer()
        self.WL = weighting_layer()
    
    def forward(self, src_pts, tgt_pts):
        B, _, _ = src_pts.shape
        # deep feature extration for src
        src_deep_feat_xyz, src_deep_feat_pts = self.FE1(src_pts)
        # obtain the top k indices for src point clouds
        src_keypts_idx = self.WL(src_deep_feat_pts)
        src_keypts = src_pts[:, :, src_keypts_idx]
        src_keypts = src_keypts.permute(0, 2, 1)
        # group the keypoints
        src_keypts_grouped_xyz, src_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = src_keypts[:, :, :3], points = src_keypts[:, :, 3:])
        
        
        tgt_deep_feat_xyz, tgt_deep_feat_pts = self.FE1(tgt_pts)
        # get candidate points for corresponding points of the keypts in src
        candidate_pts = voxelize(src_keypts)
        # reshape the candidate_pts from B * N * C * 3 to B * (N * C) * 3 to perform knn
        candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2)
        
        # use KNN to find nearest neighbors of the candidates in tgt_pts 
        k_nn = 32
        knn = KNN(k = 10, transpose_mode = True)
        query_pts = candidate_pts
        ref_pts = tgt_pts.repeat(B, 1, 1)
        dist, indx = knn(ref_pts.cuda(), query_pts.cuda())

        # normalize the deep features based on distance
        dist_sum = torch.sum(dist, dim = 2, keepdim = True, dtype = float)
        dist_normalize = dist / dist_sum
        print(dist_normalize.shape)
        feat_weight_map = (dist_normalize.repeat(1, 1, k_nn)).repeat(B, 1, 1, 1)
        

        # obtain the top k indices for tgt point clouds
        tgt_keypts_idx = self.WL(tgt_deep_feat_pts)
        tgt_keypts = tgt_pts[:, :, tgt_keypts_idx]
        tgt_keypts = tgt_keypts.permute(0, 2, 1)
        # group the keypoints
        tgt_keypts_grouped_xyz, tgt_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = tgt_keypts[:, :, :3], points = tgt_keypts[:, :, 3:])
        return src_keypts