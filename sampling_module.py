import torch
import torch.nn as nn

from pointnet2_utils import sample_and_group

from knn_cuda import KNN


class Sampling_Module(nn.Module):
    def __init__(self):
        super(Sampling_Module, self).__init__()

    def forward(self, candidate_pts, src_keypts, tgt_pts_xyz, tgt_deep_feat_pts):
        B, _, _ = src_keypts.shape
        candidate_pts_reshape = candidate_pts.view(B, candidate_pts.shape[1] * candidate_pts.shape[2], \
                                                   candidate_pts.shape[3])
        
        # sample and group the candidate points
        candidate_pts_grouped_xyz, candidate_pts_grouped_pts = sample_and_group(npoint = candidate_pts_reshape.shape[1], \
                                                                                radius = 1, nsample = 32, \
                                                                                xyz = candidate_pts_reshape, points = candidate_pts_reshape)
        candidate_pts_grouped_xyz = candidate_pts_grouped_pts[:, :, :, :3]
        candidate_pts_grouped_xyz = candidate_pts_grouped_xyz.view(B, candidate_pts.shape[1], candidate_pts.shape[2], 32, \
                                                                   candidate_pts.shape[3])
        
        # reshape the candidate_pts from B x N x C x 3 to B x (N x C) x 3 to perform knn
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

        # pick deep features of tgt_pts with idx
        N_keypts = src_keypts.shape[1]
        C_candidates = candidate_pts.shape[2]
        C_deep_feat = tgt_deep_feat_pts.shape[2]

        idx_1_mask = torch.arange(B)
        idx_1_mask = idx_1_mask.unsqueeze(1).repeat(1, B)
        idx_1_mask = idx_1_mask.flatten()
        idx_2_mask = idx.flatten()
        tgt_feat_picked = tgt_deep_feat_pts[idx_1_mask, idx_2_mask, :].view(B, N_keypts, \
                                                                            C_candidates, k_nn, C_deep_feat)

        # reshape the normalizing feature weight map
        feat_weight_map = feat_weight_map.view(B, N_keypts, C_candidates, k_nn, C_deep_feat)

        # normalize the picked deep features from tgt_pts
        tgt_feat_norm = tgt_feat_picked * feat_weight_map
        tgt_feat_cat = torch.cat((candidate_pts_grouped_xyz.cuda(), tgt_feat_norm), dim = 4)

        return tgt_feat_cat