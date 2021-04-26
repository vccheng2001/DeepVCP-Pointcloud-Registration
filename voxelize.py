import os
import numpy as np
import torch
from torch import nn
from utils import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random


# defines a voxel object with a center, edge length, and vertices 
class Voxel():
    def __init__(self, center, length):
        self.center = center 
        self.cx = center[0]
        self.cy = center[1]
        self.cz = center[2]
        self.length = length # edge length
        self.center
        # 8 coordinates defining cube vertices
        self.vertices = np.asarray(cart_prod([[self.cx-length/2, self.cx+length/2],
                                  [self.cy-length/2, self.cy+length/2],
                                  [self.cz-length/2, self.cz+length/2]]))


''' Calls voxelize_point on each point xp_i 
    @param xp: transformed point cloud
           r:  search radius, as described in paper
           s:  voxel edge length, as described in paper 
    @returns output: NxC output 
'''
def voxelize(xp, r, s):
    output = []
    for xp_i in xp:
        candidates = voxelize_point(xp_i, r, s)
        output.append(candidates)

    output = np.array(output) # N x C 
    print(f"Output shape: {output.shape}")
    return output 


''' Voxelizing neighborhood space for one point xp_i

    Given search radius r, divide search space around xp_i 
    into (2r/s + 1, 2r/s + 1, 2r/s + 1) voxels

    1) create bounding box around sphere with radius r
    2) voxelize space within bounding box
    3) measure Euclidean distance of each voxel center to origin
       if voxel center falls outside of search radius, reject

    @param xp_i: a transformed point in target point cloud xp
           search radius of neighborhood 
           voxel_len: edge length of each individual voxel
    @return list of C candidates
'''
def voxelize_point(xp_i, search_radius, voxel_len):
    # enclose search space in bounding box (think of it as a voxel of edge length 2r)
    search_space = Voxel(xp_i, 2*search_radius)     
    ss_vertices = search_space.vertices

    # min, max xyz coords of bounding box
    ss_minx, ss_miny, ss_minz  = np.min(ss_vertices, axis=0)
    ss_maxx, ss_maxy, ss_maxz = np.max(ss_vertices, axis=0)

    # number of voxels along each axes

    num_voxels = np.ceil((np.max(ss_vertices, axis=0) - np.min(ss_vertices, axis=0))/voxel_len)
    # print(f"voxelize into {num_voxels} voxels along each axis ")
    # store voxels in sparse representation 
    voxels = {} 
    x,y,z = [], [], []

    # voxelize search space/cube 
    for xi in np.arange(ss_minx, ss_maxx, voxel_len):
        for yi in np.arange(ss_miny,ss_maxy,voxel_len):
            for zi in np.arange(ss_minz,ss_maxz,voxel_len):
                # define center of voxel
                cx,cy,cz = (xi+voxel_len/2, yi+voxel_len/2, zi+voxel_len/2)
                # create voxel object at center point with edge length <voxel_len>
                vox = Voxel([cx,cy,cz], 2*voxel_len)
                # store voxel in dictionary
                voxels[vox] = 1
                
    # measure distance from each voxel to center of search space
    # reject voxels whose distance >= search radius
    for vox in voxels.keys():
        dist = euclidean_dist(vox.center, search_space.center)
        if dist >= search_radius: 
            # print(f'Rejecting point {vox.center} which is {dist} from center')
            voxels[vox] = 0

    #####* * * * * this part is for visualization purposes only * * * #######
        else:
            x.append(vox.center[0])
            y.append(vox.center[1])
            z.append(vox.center[2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plot voxel centroids that are within search radius
    ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=1)
    # plot bounding box vertices
    ax.scatter(ss_vertices[:,0], ss_vertices[:,1], ss_vertices[:,2], cmap='Greens', linewidth=1);
    plt.plot(search_space.center[0], search_space.center[1], search_space.center[2],'ro') #
    plt.show()
    ########################################################################

    candidates = {k:v for (k,v) in voxels.items() if v != 0}
    candidates = np.array(list(candidates.keys()))
    return candidates


if __name__ == "__main__":
    
    # setup parameters 
    
    N = 64  # number of point
    r = 2.0 # search radius, as described in paper
    s = 0.4 # voxel edge length, as described in paper 

    # Generate N random points 
    xrange = (-10.0, 10.0)
    yrange = (-10.0, 10.0)
    zrange = (-10.0, 10.0)
    xp = []
    [xp.append((random.uniform(*xrange), random.uniform(*yrange),\
         random.uniform(*zrange))) for i in range(N) ]

    voxelize(xp, r, s)

