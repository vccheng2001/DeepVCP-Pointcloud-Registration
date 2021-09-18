import torch
from pointnet2_utils import index_points

# index_pts doesn't index correctly for this shape 
# should be (B,K_topK, K_Knn,3)
B = 5
K_topK = 64
K_knn=32
C = 15 # 
candidate_pts = torch.rand(B,K_topK,C,3)
candidate_pts_flat = torch.flatten(candidate_pts, start_dim = 1, end_dim = 2) # (B x (K_topk x C) x 3)

nn_idx = torch.rand(B,K_topK, K_knn)
nn_idx = nn_idx.long()
pts  = index_points(candidate_pts_flat, nn_idx) 
print(pts.shape)


# torch.gather(candidate_ptsnn_idx)

# import torch

# # print('qrs', qrs)
# # print('pts', pts)

# # D=torch.cdist(qrs, pts, p=2) # Euclidean
# # print("D", D.shape, D)
# # K=2 #number of neighbors
# # dist, idx = D.topk(k=K, dim=-1, largest=False)
# # print('dist', dist)
# # print('idx', idx)
# # nn_pts = torch.stack([pts[n][i,:] for n, i in enumerate(torch.unbind(idx, dim = 0))], dim = 0) # [batch, queries_number, K, dim]

# # print('nn_pts', nn_pts)


# def knn(qry, ref, K):
#     # qry: B x P x M
#     # ref: B x R x M 

#     # B x P x R 
#     D = torch.cdist(qry, ref, p=2)
#     dist, idx = D.topk(k=K, dim=-1, largest=False)

#     # for each point in qry, returns the K nearest neighbors 
#     # nn_pts: B, qry_num, K, dims
#     nn_pts = torch.stack([ref[n][i,:] for n, i in enumerate(torch.unbind(idx, dim = 0))], dim = 0) 
#     return nn_pts 

# import timeit

# start = timeit.timeit()
# qry = torch.rand(1,10000,3)
# ref = torch.rand(1,10000,3)
# K = 32
# nn_pts = knn(qry, ref, K)
# print('nn_pts', nn_pts.shape)
# end = timeit.timeit()
# print('Time taken:', (end - start), 'seconds')
