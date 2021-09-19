import torch
import torch.nn as nn

from utils import knn
from pointnet2_utils import query_ball_point, index_points
'''
Get concatenated local coordinates and normalized features for candidate points

    1) Find K nearest neighbors for candidate_pts in tgt_pts_xyz
    2) Convert the xyz coordinates for KNN into local w.r.t. candidate_pts
    3) Normalize the deep features in src_keyfeats based on distance between
       src_keypts with its K nearest neighbors
'''
class Get_Cat_Feat_Tgt(nn.Module):
    def __init__(self):
        super(Get_Cat_Feat_Tgt, self).__init__()

# dfe should get B x K_topK x C x 32 for target, then get input into 3D CNN
    def forward(self, candidate_pts,  src_keypts, tgt_pts_xyz, tgt_deep_feat_pts):
        """
        Input:
            candidate_pts: candidate corresponding points (B x K_topk x C x 3)
                for each transformed keypt, get C candidates 
            src_keypts: keypoints in src point cloud (B x K_topk x 3)
            tgt_pts_xyz: original points in target point cloud (B x N2 x 3)
            tgt_deep_feat_pts: deep features for tgt point cloud (B x N2 x num_feats)
            
        Output: 
            tgt_keyfeats_cat: concatenated local coordinates of candidate points and 
                              normalized deep features (B x K_topk x C x nsample x (3 + num_feats))
        """
        B, K_topk, C, _ = candidate_pts.shape
        # print('B:', B)
        # print('K_topk:',K_topk)
        # print("C: ", C)
        B, N2, _ = tgt_pts_xyz.shape
        # print('N2:', N2)

        # extract DFE descriptors for target pts
        

        ''' 1. Collect K neighboring points within a
            certain radius d of each keypoint, duplicate if < K
        '''

        D_radius = 2 # 1 meter search radius
        K_knn = 32 
        # print('K_knn:', K_knn)

        candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2) # (B x (K_topk x C) x 3)

                
        # Search among all original N2 target points for each candidate's nearest neighbors 
        # nn_idx: (B x (K_topk x C) x K_knn)
        nn_idx = query_ball_point(radius=D_radius,
                                    nsample=K_knn, 
                                    xyz=tgt_pts_xyz.float(),             # (B x N2 x 3) 
                                    new_xyz=candidate_pts_flat.float())  # (B x (K_topk x C) x 3)

        
        # Index into tgt_pts_xyz   
        # (B x K_topk x C x K_knn x 3)
        nn_idx_xyz = nn_idx.reshape(B, K_topk, C, K_knn).unsqueeze(-1).repeat(1,1,1,1,3)
        # print('nn_idx_xyz', nn_idx_xyz.shape)
        # [1, 64, 216, 32, 3])

        # (B x N2 x C x K_knn x 3 )
        tgt_pts_xyz = tgt_pts_xyz.unsqueeze(-2).unsqueeze(-2).repeat(1,1,C, K_knn,1)
        # print('tgt_pts_xyz', tgt_pts_xyz.shape)

        # [1, 10000, 216, 32, 3])
        # (B x K_topk x C x K_knn x 3)
        nn_candidate_pts = torch.gather(tgt_pts_xyz, dim=1, index=nn_idx_xyz)
        nn_candidate_pts_norm = nn_candidate_pts / D_radius


        # Index into tgt_deep_feat_pts
        nn_idx_deep_feat_pts = nn_idx.reshape(B, K_topk, C, K_knn).unsqueeze(-1).repeat(1,1,1,1,32)
        # (B x N2 x C x K_knn x num_feats)
        tgt_deep_feat_pts = tgt_deep_feat_pts.unsqueeze(-2).unsqueeze(-2).repeat(1,1,C,K_knn, 1)
        nn_tgt_deep_feat_pts = torch.gather(tgt_deep_feat_pts, dim=1, index=nn_idx_deep_feat_pts)

        


        # (B x K_topk x C x K_knn x 35)
        tgt_keyfeats_cat = torch.cat((nn_candidate_pts_norm, nn_tgt_deep_feat_pts), dim = 4)

        # print('tgt_keyfeats_cat', tgt_keyfeats_cat.shape)

        return tgt_keyfeats_cat








        
        # # sample and group the candidate points
        # # candidate_pts_grouped_xyz: B x K_topk x C x nsample x 3
        # nsample = 32
    
        # # reshape the candidate_pts from B x N x C x 3 to B x (N x C) x 3 to perform knn
        # candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2)



        # # use KNN to find nearest neighbors of the candidates in tgt_pts 
        # # dist: (B x (K_topk x C) x k_nn)
        # # idx: (B x (K_topk x C) x k_nn)
        # # candidate_pts_k: (B x K_topk x C x nsample x 3)
        # k_nn = 32
        # query_pts = candidate_pts_flat
        # print("tgt_pts_xyz: ", tgt_pts_xyz.shape)
        # # previous
        # # ref_pts = tgt_pts_xyz.repeat(B, 1, 1)
        # ref_pts = tgt_pts_xyz
        # print("ref_pts: ", ref_pts.shape)
        # dist, idx = knn(ref_pts.cuda(), query_pts.cuda(), k=k_nn)
        # candidate_pts_k = candidate_pts.unsqueeze(3).repeat(1, 1, 1, nsample, 1)

        # # normalize the deep features based on distance
        # # dist_normalize: (B x (K_topk x C) x num_feat)
        # dist_sum = torch.sum(dist, dim = 2, keepdim = True, dtype = float)
        # dist_normalize = dist / dist_sum
        # print("dist_normalize: ", dist_normalize.shape)

        # # stacking normalized distance into a map for deep features
        # # feat_weight_map: (B x (K_topk x C) x k_nn x num_feat) 
        # # previous
        # # feat_weight_map = dist_normalize.unsqueeze(3).repeat(1, 1, 1, k_nn)
        # feat_weight_map = dist_normalize.unsqueeze(2).repeat(1, 1, k_nn, 1)
        # print("feat_weight_map: ", feat_weight_map.shape)
        
        # # pick deep features of tgt_pts with idx
        # N_keypts = src_keypts.shape[1]
        # C_candidates = candidate_pts.shape[2]
        # C_deep_feat = tgt_deep_feat_pts.shape[2]

        # # indexing tgt_deep_feat_pts and normalize
        # # convert tgt_pts_picked to local
        # # tgt_feat_picked: (B x K_topk x C x k_nn x num_feat)
        # # tgt_pts_picked: (B x K_topk x C x k_nn x 3)
        # # tgt_keyfeats_cat: (B x K_topk x C x k_nn x (3 + num_feat))
        # idx_1_mask = torch.arange(B)
        # idx_1_mask = idx_1_mask.unsqueeze(1).repeat(1, B)
        # print("idx_1_mask: ", idx_1_mask)
        # idx_1_mask = idx_1_mask.flatten()
        # print("idx_1_mask_flatten: ", idx_1_mask)
        # idx_2_mask = idx.flatten()
        # print("idx_2_mask: ", idx_2_mask)
        # tgt_feat_picked = tgt_deep_feat_pts[idx_1_mask, idx_2_mask, :].view(B, N_keypts, \
        #                                                                     C_candidates, k_nn, C_deep_feat)
        # tgt_pts_picked = tgt_pts_xyz[idx_1_mask, idx_2_mask, :].view(B, N_keypts, \
        #                                                              C_candidates, k_nn, 3)
        # candidates_grouped_local = tgt_pts_picked - candidate_pts_k

        # # reshape the normalizing feature weight map
        # feat_weight_map = feat_weight_map.view(B, N_keypts, C_candidates, k_nn, C_deep_feat)

        # # normalize the picked deep features from tgt_pts
        # tgt_feat_norm = tgt_feat_picked * feat_weight_map
        # tgt_keyfeats_cat = torch.cat((candidates_grouped_local, tgt_feat_norm), dim = 4)

        # return tgt_keyfeats_cat

