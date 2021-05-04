import torch
import torch.nn as nn


"""
Get concatenated local coordinates and normalized features

    1) Convert the xyz coordinates in src_keypts_grouped_pts into local
    2) Normalize the deep features in src_keyfeats based on distance between
       src_keypts with its K nearest neighbors
"""
class Get_Cat_Feat_Src(nn.Module):
    def __init__(self):
        super(Get_Cat_Feat_Src, self).__init__()

    def forward(self, src_keypts, src_keypts_grouped_pts, src_keyfeats):
        ''' 
        Input:
            src_keypts: Extracted keypts in src point cloud with normals (B x K_topk x 6)
            src_keypts_grouped_pts: 
                Grouped src_keypts with sample_and_group (B x K_topk x nsample x 6)
            src_keyfeats: 
                Deep features corresponding to src_keypts_grouped_pts (B x K_topk x nsample x num_feat)

        Return:
            src_keyfeats_cat (B x K_topk x nsample x (3 + num_feat)):
                Concatenated local coordinates of src_keypts and normalized src_keyfeats
        '''
        B, K_topk, nsample, num_feat = src_keyfeats.shape

        # get distance between k nearest neighbors and the point itself
        src_keypts_k = src_keypts[:, :, :3].unsqueeze(2).repeat(1, 1, nsample, 1)
        pdist = nn.PairwiseDistance(p = 2, keepdim = True)
        src_dist = pdist(torch.flatten(src_keypts_k, start_dim = 0, end_dim = 2), \
                         torch.flatten(src_keypts_grouped_pts[:, :, :, :3], start_dim = 0, end_dim = 2))
        src_dist = src_dist.view(B, K_topk, nsample, 1)
        src_dist_sum = torch.sum(src_dist, dim = 2, keepdim = True)
        src_dist_norm = src_dist / src_dist_sum
        src_dist_norm = src_dist_norm.view(B, K_topk, nsample).unsqueeze(3).repeat(1, 1, 1, num_feat)

        # get local coordinates of the k nearest neighbors and normalize deep features based on src_dist
        # src_keypts_grouped_local: B x K_topk x nsmaple x 3
        # src_keyfeats_normalized: B x K_topk x nsample x num_feat
        src_keypts_grouped_local = src_keypts_grouped_pts[:, :, :, :3] - src_keypts_k
        src_keyfeats_normalized = src_keyfeats * src_dist_norm

        # src_keyfeats_cat: B x K_topk x nsample x (3 + num_feat)
        src_keyfeats_cat = torch.cat((src_keypts_grouped_local, src_keyfeats_normalized), dim = 3)

        return src_keyfeats_cat