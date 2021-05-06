import os
import torch
from torch import nn
from utils import *
import matplotlib.pyplot as plt
import random
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

''' Calls voxelize_point on each point
    @param xp: batch of transformed point clouds 
           r:  search radius, as described in paper
           s:  voxel edge length, as described in paper 
    @returns output: BxNxCx3 
'''
# input: B x N x 3 
# output:B x N x C x 3
def voxelize(point_clouds, r, s):
    out = []
    B, N, _= point_clouds.shape
    # flatten into B x N 3-element tensors
    points = point_clouds.view(-1, 3)  
    for point in points:
        out.append(voxelize_point(point, r, s))
    # reshape into B x N x C x 3
    out = torch.stack((out))
    out = torch.reshape(out, (B, N, -1, 3))
    return out


''' Voxelizing neighborhood space for one point (shape: [3])

    Given search radius r and voxel length, voxelize search space around point 
    1) create bounding box around sphere with radius r
    2) voxelize space within bounding box
    3) reject any point that falls outside of the sphere

    @param point: a transformed point in target point cloud xp
           search radius of neighborhood 
           voxel_len: edge length of each individual voxel
    @return list of C candidates
'''
def voxelize_point(point, search_radius, voxel_len):
    # coordinate of point
    bbox_center = point
    cx, cy, cz = bbox_center

    # enclose search space in bounding box
    tx = torch.tensor([cx-search_radius, cx+search_radius]).to(device)
    ty = torch.tensor([cy-search_radius, cy+search_radius]).to(device)
    tz = torch.tensor([cz-search_radius, cz+search_radius]).to(device)
    bbox_vertices = torch.cartesian_prod(tx,ty,tz)

    # min, max xyz coords of bounding box
    search_min = torch.min(bbox_vertices, axis=0)
    search_max = torch.max(bbox_vertices, axis=0)    

    # list of candidates 
    candidates = []
    # create 3D grid for each of x,y,z
    xrange = torch.arange(search_min.values[0], search_max.values[0], voxel_len)
    yrange = torch.arange(search_min.values[1], search_max.values[1], voxel_len)
    zrange = torch.arange(search_min.values[2], search_max.values[2], voxel_len)

    # create 3d mesh grid 
    xgrid, ygrid, zgrid = torch.meshgrid(xrange, yrange, zrange)
    xgrid = xgrid.to(device)
    ygrid = ygrid.to(device)
    zgrid = zgrid.to(device)
    grid3D = torch.stack((xgrid, ygrid, zgrid), axis=-1)   

    # reject points that lie outside of sphere (radius > search radius)
    for coord in grid3D.reshape(-1, grid3D.shape[-1]):
        dist = euclidean_dist(coord+voxel_len/2, bbox_center)
        if dist <= search_radius: 
            candidates.append(coord)
    
    candidates = torch.stack((candidates))

    # visualize points 
    # visualize_voxelization(bbox_center, bbox_vertices,candidates) 
    return candidates


def visualize_voxelization(bbox_center, bbox_vertices, candidates):
    cx, cy, cz = bbox_center
    vx, vy, vz = bbox_vertices[:,0], bbox_vertices[:,1], bbox_vertices[:,2]
    px, py, pz = candidates[:,0], candidates[:,1], candidates[:,2]

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # plot bounding box vertices
    ax.scatter(vx,vy,vz, cmap='Greens', linewidth=1)
    # plot center point of search space 
    plt.plot(cx,cy,cz,'ro') 
    # plot candidates 
    ax.scatter(px, py, pz, c=pz, cmap='viridis', linewidth=1)
    plt.show()

if __name__ == "__main__":
    # setup parameters 
    b = 10 # batch size
    n = 64  # number of points
    r = 2.0 # search radius, as described in paper
    s = 0.4 # voxel edge length, as described in paper 

    maxval = 10
    minval = -10

    point_clouds = (maxval - minval) * torch.rand((b, n, 3)) + minval
    out = voxelize(point_clouds, r, s)
    print(out.shape)
