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
        src_deep_feat_xyz, src_deep_feat_pts = self.FE1(src_pts)
        # obtain the top k indices for src point clouds
        src_keypts_idx = self.WL(src_deep_feat_pts)
        src_keypts = src_pts[:, :, src_keypts_idx]
        src_keypts = src_keypts.permute(0, 2, 1)
        # group the keypoints
        src_keypts_grouped_xyz, src_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = src_keypts[:, :, :3], points = src_keypts[:, :, 3:])
        
        tgt_pts_xyz = tgt_pts[:, :3, :]
        tgt_pts_xyz = tgt_pts_xyz.permute(0, 2, 1)
        tgt_deep_feat_xyz, tgt_deep_feat_pts = self.FE1(tgt_pts)
        print("tgt_deep_feat_pts: ", tgt_deep_feat_pts.shape)
        # get candidate points for corresponding points of the keypts in src
        r = 2.0
        s = 0.4
        ###########################
        # seems to not taking batch size in voxelize.py
        ###########################
        # candidate_pts = voxelize(src_keypts, r, s)
        candidate_pts = torch.randn(B, src_keypts.shape[1], 552, 3)
        # reshape the candidate_pts from (B, N, C, 3) to (B, (N * C), 3) to perform knn
        candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2)
        
        # use KNN to find nearest neighbors of the candidates in tgt_pts 
        k_nn = 32
        knn = KNN(k = k_nn, transpose_mode = True)
        query_pts = candidate_pts_flat
        ref_pts = tgt_pts_xyz.repeat(B, 1, 1)
        dist, idx = knn(ref_pts.cuda(), query_pts.cuda())

        # normalize the deep features based on distance
        dist_sum = torch.sum(dist, dim = 2, keepdim = True, dtype = float)
        dist_normalize = dist / dist_sum

        # stacking normalized distance into a map for deep features
        # feat_weight_map: (B, N * C, K, C_deep_feat) C_deep_feat is number of channels for deep features 
        feat_weight_map = dist_normalize.unsqueeze(3).repeat(1, 1, 1, k_nn)
        print("feat_weight_map: ", feat_weight_map.shape)

        # pick deep features of tgt_pts with idx
        N_keypts = src_keypts.shape[1]
        C_candidates = candidate_pts.shape[2]
        C_deep_feat = tgt_deep_feat_pts.shape[2]
        N_idx = idx.shape[1]
        C_idx = idx.shape[2]
        idx_1_mask = torch.arange(B)
        idx_1_mask = idx_1_mask.unsqueeze(1).repeat(1, B)
        idx_1_mask = idx_1_mask.flatten()
        idx_2_mask = idx.flatten()
        tgt_feat_picked = tgt_deep_feat_pts[idx_1_mask, idx_2_mask, :].view(B, N_keypts, C_candidates, k_nn, C_deep_feat)
        
        # reshape the normalizing feature weight map
        feat_weight_map = feat_weight_map.view(B, N_keypts, C_candidates, k_nn, C_deep_feat)

        # normalize the picked deep features from tgt_pts
        tgt_feat_norm = tgt_feat_picked * feat_weight_map
        

        # obtain the top k indices for tgt point clouds
        tgt_keypts_idx = self.WL(tgt_deep_feat_pts)
        tgt_keypts = tgt_pts[:, :, tgt_keypts_idx]
        tgt_keypts = tgt_keypts.permute(0, 2, 1)
        # group the keypoints
        tgt_keypts_grouped_xyz, tgt_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = tgt_keypts[:, :, :3], points = tgt_keypts[:, :, 3:])
        return src_keypts