import torch
import torch.nn as nn
import time

from pointnet2_utils import sample_and_group, index_points
from deep_feat_extraction import feat_extraction_layer
from weighting_layer import weighting_layer
from voxelize import voxelize
from get_cat_feat_tgt import Get_Cat_Feat_Tgt
from get_cat_feat_src import Get_Cat_Feat_Src
from deep_feat_embedding import feat_embedding_layer
from cpg import cpg

device = 'cpu'
# device =  'cuda' if torch.cuda.is_available() else 'cpu'

class DeepVCP(nn.Module):
    def __init__(self, use_normal): 
        super(DeepVCP, self).__init__()
        self.FE1 = feat_extraction_layer(use_normal=use_normal)
        self.WL = weighting_layer()
        self.DFE = feat_embedding_layer()
        self.cpg = cpg()
    
    def forward(self, src_pts, tgt_pts, R_init, t_init):
        B, _, _ = src_pts.shape

        # deep features extracted from FE layer: B x N x 32
        fe_start_time = time.time()
        src_deep_feat_xyz, src_deep_feat_pts = self.FE1(src_pts)
        print("feature extraction time: ", time.time() - fe_start_time)

        # obtain the top k indices for src point clouds
        K_topk = 64
        wl_start_time = time.time()
        src_keypts_idx = self.WL(src_deep_feat_pts)
        # print("src_keypts_idx: ", src_keypts_idx.shape)
        # print("src_pts: ", src_pts.shape)
        # print("weighting layer time: ", time.time() - wl_start_time)
        batch_mask = torch.arange(B)
        batch_mask = batch_mask.unsqueeze(1).repeat(1, B)
        batch_mask = batch_mask.flatten()

        # indexing the src_pts to get keypts: B x K_topk x 6
        src_keypts_idx_unsqueezed = (src_keypts_idx.unsqueeze(0)).unsqueeze(1).repeat(1, 6, 1)
        # print("src_keypts_idx_unsqueezed: ", src_keypts_idx_unsqueezed.shape)
        src_keypts = torch.gather(src_pts, 2, src_keypts_idx_unsqueezed).view(B, K_topk, src_pts.shape[1])
        # print("src_keypts: ", src_keypts.shape)
        # src_keypts = src_pts[batch_mask, :, src_keypts_idx].view(B, K_topk, src_pts.shape[1])
        
        # group the keypoints 
        # src_keypts_grouped_pts: B x K_topk x nsample x 6
        # picked_idx: B x K_topk x nsample
        group_start_time = time.time()
        src_keypts_grouped_xyz, src_keypts_grouped_pts, picked_idx = sample_and_group(npoint = 64, radius = 1, nsample = 32, \
                                                                                      xyz = src_keypts[:, :, :3], \
                                                                                      points = None, returnidx = True)
        print("Grouping keypoints time: ", time.time() - group_start_time)
        
        # pick the deep feature corresponding to src_keypts_grouped
        # src_keyfeats: B x K_topk x nsample x num_feat
        num_feat = 32
        src_keyfeats = index_points(src_deep_feat_pts, picked_idx)
        
        # normalize src_deep_feat_pts with distance between src point and its k nearest neighbors
        get_cat_feat_src_start_time = time.time()
        src_gcf = Get_Cat_Feat_Src()
        src_keyfeats_cat = src_gcf(src_keypts, src_keypts_grouped_pts, src_keyfeats)
        # print('src_keyfeats_cat', src_keyfeats_cat.shape)
        print("get_cat_feat_src time: ", time.time() - get_cat_feat_src_start_time)

        tgt_pts_xyz = tgt_pts[:, :3, :]
        tgt_pts_xyz = tgt_pts_xyz.permute(0, 2, 1)
        tgt_deep_feat_xyz, tgt_deep_feat_pts = self.FE1(tgt_pts)

        # rotate and translate the src_keypts with R_init and t_init
        # get candidate points by voxelization
        r = 2.0
        s = 0.4

        # there should be (2*2/0.4 + 1, 2*2/0.4 2*2/0.4) = 16x16x16 = 216 candidates

        # t_init: (1 x 3)
        # R_init: (1 x 3 x 3)
        # t_init_rep: (B x K_topk x 3)
        # R_init_rep: (B x 3 x 3)
        # src_keypts_T: (B x 3 x K_topk)
        # src_transformed: (B x 3 x 3) @ (B x 3 x K_topk) = (B x 3 x K_topk)
        # src_transformed_T: (B x K_topk x 3)
        t_init_rep = t_init.unsqueeze(0).repeat(B, 1, 1)
        R_init_rep = R_init.repeat(B, 1, 1)
        src_keypts_T = src_keypts.permute(0, 2, 1)
        src_keypts_T = src_keypts_T[:, :3, :]
        src_transformed = torch.matmul(R_init_rep, src_keypts_T.double())
        src_transformed_T = src_transformed.permute(0, 2, 1)
        candidate_pts = voxelize(src_transformed_T, r, s)

        # group the tgt_pts to feed into DFE layer
        get_cat_feat_tgt_start_time = time.time()
        tgt_gcf = Get_Cat_Feat_Tgt()
        tgt_keyfeats_cat = tgt_gcf(candidate_pts, src_keypts, tgt_pts_xyz, tgt_deep_feat_pts)
        # print('tgt_keyfeats_cat', tgt_keyfeats_cat.shape)

        print("get_cat_feat_tgt time: ", time.time() - get_cat_feat_tgt_start_time)

        # deep feature embedding
        src_dfe_feat = self.DFE(src_keyfeats_cat, src = True)
        tgt_dfe_feat = self.DFE(tgt_keyfeats_cat, src = False)

        # print('src_dfe_feat', src_dfe_feat.shape) # (B, K_topk, 32)
        # print('tgt_dfe_feat', tgt_dfe_feat.shape) # (B, K_topk, C, 32)
        
        # similarity learning
        src_dfe_feat = src_dfe_feat.unsqueeze(2) # (B, K_topk, 1, 32)
        tgt_dfe_feat = tgt_dfe_feat.permute(0, 1, 3, 2) # (B, K_topk, 32, C)
        
        tgt_vcp = self.cpg(src_dfe_feat, tgt_dfe_feat, candidate_pts, r, s)

        return src_keypts[:, :, :3], tgt_vcp
