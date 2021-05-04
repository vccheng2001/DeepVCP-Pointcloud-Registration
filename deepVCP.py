import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

from pointnet2_utils import sample_and_group, index_points
from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer
from voxelize import voxelize
from sampling_module import Sampling_Module

class DeepVCP(nn.Module):
    def __init__(self): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer()
        self.WL = weighting_layer()
    
    def forward(self, src_pts, tgt_pts):
        B, _, _ = src_pts.shape

        # deep features exrtacted from FE layer: B x N x 32
        src_deep_feat_xyz, src_deep_feat_pts = self.FE1(src_pts)
        print("src_deep_feat_pts: ", src_deep_feat_pts.shape)

        # obtain the top k indices for src point clouds
        K_topk = 64
        src_keypts_idx = self.WL(src_deep_feat_pts)
        print("src_keypts_idx: ", src_keypts_idx.shape)
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()

        # indexing the src_pts to get keypts: B x K_topk x 6
        src_keypts = src_pts[batch_mask, :, src_keypts_idx].view(B, K_topk, src_pts.shape[1])
        print("src_keypts: ", src_keypts.shape)
        # src_keyfeats = src_deep_feat_pts[]

        # group the keypoints 
        # src_keypts_grouped_pts: B x K_topk x nsample x 6
        # picked_idx: B x K_topk x nsample
        src_keypts_grouped_xyz, src_keypts_grouped_pts, picked_idx = sample_and_group(npoint = 64, radius = 1, nsample = 32, \
                                                                                      xyz = src_keypts[:, :, :3], \
                                                                                      points = None, returnidx = True)
        
        # pick the deep feature corresponding to src_keypts_grouped
        # src_keyfeats: B x K_topk x nsample x num_feat
        num_feat = 32
        src_keyfeats = index_points(src_deep_feat_pts, picked_idx)
        
        # normalize src_deep_feat_pts with distance between src point and its k nearest neighbors
        # repeat src_keypts n sample times to normalize features and get local coordinates after grouping
        # src_keypts_k: B x K_topk x nsample x 3
        nsample = 32
        src_keypts_k = src_keypts[:, :, :3].unsqueeze(2).repeat(1, 1, nsample, 1)
        pdist = nn.PairwiseDistance(p = 2, keepdim = True)
        src_dist = pdist(torch.flatten(src_keypts_k, start_dim = 0, end_dim = 2), \
                         torch.flatten(src_keypts_grouped_pts[:, :, :, :3], start_dim = 0, end_dim = 2))
        src_dist = src_dist.view(B, K_topk, nsample).unsqueeze(3).repeat(1, 1, 1, num_feat)

        src_keypts_grouped_local = src_keypts_grouped_pts[:, :, :, :3] - src_keypts_k
        src_keyfeats_normalized = src_keyfeats * src_dist

        src_keyfeats_cat = torch.cat((src_keypts_grouped_local, src_keyfeats_normalized), dim = 3)

        tgt_pts_xyz = tgt_pts[:, :3, :]
        tgt_pts_xyz = tgt_pts_xyz.permute(0, 2, 1)
        tgt_deep_feat_xyz, tgt_deep_feat_pts = self.FE1(tgt_pts)

        # get candidate points for corresponding points of the keypts in src
        r = 2.0
        s = 0.4
        ###########################
        # seems to not taking batch size in voxelize.py
        ###########################
        # candidate_pts = voxelize(src_keypts, r, s)
        candidate_pts = torch.randn(B, src_keypts.shape[1], 552, 3)

        # group the tgt_pts to feed into DFE layer
        sm = Sampling_Module()
        tgt_pts_grouped = sm(candidate_pts, src_keypts, tgt_pts_xyz, tgt_deep_feat_pts)
        print("tgt_pts_grouped", tgt_pts_grouped.shape)

        # obtain the top k indices for tgt point clouds
        tgt_keypts_idx = self.WL(tgt_deep_feat_pts)
        tgt_keypts = tgt_pts[:, :, tgt_keypts_idx]
        tgt_keypts = tgt_keypts.permute(0, 2, 1)
        # group the keypoints
        tgt_keypts_grouped_xyz, tgt_keypts_grouped_pts = sample_and_group(npoint = 64, radius = 1, nsample = 32, xyz = tgt_keypts[:, :, :3], points = tgt_keypts[:, :, 3:])
        return src_keypts