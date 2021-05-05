import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

from pointnet2_utils import sample_and_group, index_points
from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer
from voxelize import voxelize
from get_cat_feat_tgt import Get_Cat_Feat_Tgt
from get_cat_feat_src import Get_Cat_Feat_Src

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        src_gcf = Get_Cat_Feat_Src()
        src_keyfeats_cat = src_gcf(src_keypts, src_keypts_grouped_pts, src_keyfeats)
        print("src_keyfeats_cat: ", src_keyfeats_cat.shape)

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
        candidate_pts = torch.randn(B, src_keypts.shape[1], 552, 3).to(device)

        # group the tgt_pts to feed into DFE layer
        tgt_gcf = Get_Cat_Feat_Tgt()
        src_keyfeats_cat = tgt_gcf(candidate_pts, src_keypts, tgt_pts_xyz, tgt_deep_feat_pts)
        print("src_keyfeats_cat", src_keyfeats_cat.shape)

        return src_keypts