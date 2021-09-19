''' Utility functions '''
import itertools
import torch
import numpy as np
import math 


def knn(qry, ref, K, return_ref_pts=True):
    # qry: B x P x M
    # ref: B x R x M 
    print('q', qry.shape)
    print('ref', ref.shape)

    # D: B x P x R 
    D = torch.cdist(qry, ref, p=2)
    dist, idx = D.topk(k=K, dim=-1, largest=False)

    # for each point in qry, returns the K nearest neighbors 
    # nn_pts: B, qry_num, K, dims
    if return_ref_pts:
        nn_pts = torch.stack([ref[n][i,:] for n, i in enumerate(torch.unbind(idx, dim = 0))], dim = 0) 
        return dist, idx, nn_pts 

    else:
        return dist, idx

# rotation about x axis
def RotX(theta):
    Rx = np.matrix([[ 1,            0           , 0     ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
    return Rx
  
# rotation about y axis
def RotY(theta):
    Ry =  np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1,          0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
    return Ry

# rotation about z axis
def RotZ(theta):
    Rz = np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
    return Rz

# euclidean distance between two points
def euclidean_dist(a,b):
    return torch.linalg.norm(a-b)
