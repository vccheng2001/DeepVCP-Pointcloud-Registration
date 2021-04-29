import os
import numpy as np
import torch
from torch import nn
from utils import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
import time


''' Calls voxelize_point on each point xp_i 
    @param xp: transformed point cloud
           r:  search radius, as described in paper
           s:  voxel edge length, as described in paper 
    @returns output: NxCx3 
'''
def voxelize(xp, r, s):
    output = []
    for xp_i in xp:
        candidates = voxelize_point(xp_i, r, s)
        output.append(candidates)

    output = np.array(output)#,dtype=object) # N x C 
    print(f"Output shape: {output.shape}")
    return output 


''' Voxelizing neighborhood space for one point xp_i

    Given search radius r and voxel length, voxelize search space around xp_i 
    1) create bounding box around sphere with radius r
    2) voxelize space within bounding box
    3) reject any point that falls outside of the sphere

    @param xp_i: a transformed point in target point cloud xp
           search radius of neighborhood 
           voxel_len: edge length of each individual voxel
    @return list of C candidates
'''
def voxelize_point(xp_i, search_radius, voxel_len):
    # coordinate of point
    bbox_center = xp_i
    cx, cy, cz = bbox_center

    # enclose search space in bounding box
    bbox_vertices = np.asarray(cart_prod([[cx-search_radius, cx+search_radius],
                                        [cy-search_radius, cy+search_radius],
                                        [cz-search_radius, cz+search_radius]]))

    # min, max xyz coords of bounding box
    search_min = np.min(bbox_vertices, axis=0)
    search_max = np.max(bbox_vertices, axis=0)    

    # list of candidates 
    candidates = []

    # create 3D grid for each of x,y,z
    xrange = np.arange(search_min[0], search_max[0], voxel_len)
    yrange = np.arange(search_min[1], search_max[1], voxel_len)
    zrange = np.arange(search_min[2], search_max[2], voxel_len)

    # create 3d mesh grid 
    xgrid, ygrid, zgrid=  np.meshgrid(xrange, yrange, zrange)
    grid3D = np.stack((xgrid, ygrid, zgrid), axis=-1)   

    # reject points that lie outside of sphere (radius > search radius)
    for coord in grid3D.reshape(-1, grid3D.shape[-1]):
        dist = euclidean_dist(coord+voxel_len/2, bbox_center)
        if dist <= search_radius: 
            candidates.append(coord)
    
    candidates = np.array(candidates)
    # visualize points 
    visualize_voxelization(bbox_centerbox_vertices,candidates) 
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
    start = time.process_time()

    N = 64  # number of point
    r = 2.0 # search radius, as described in paper
    s = 0.4 # voxel edge length, as described in paper 

    # Generate N random points 
    xrange = (-15.0, 15.0)
    yrange = (-15.0, 15.0)
    zrange = (-15.0, 15.0)
    xp = []
    [xp.append((random.uniform(*xrange), random.uniform(*yrange),\
         random.uniform(*zrange))) for i in range(N) ]

    voxelize(xp, r, s)

